from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download, snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mpq.eval_utils import (
    eval_ppl_from_tokens,
    eval_protocol_for_dataset,
    load_or_build_c4_tokens,
    load_or_build_wikitext2_tokens,
)


def _load_tokenizer(model_ref: str):
    last_exc: Exception | None = None
    try:
        return AutoTokenizer.from_pretrained(model_ref, use_fast=True)
    except Exception as exc:
        last_exc = exc

    try:
        model_path = Path(model_ref)
        if model_path.exists():
            tokenizer_cfg_path = model_path / "tokenizer_config.json"
        else:
            tokenizer_cfg_path = Path(hf_hub_download(model_ref, "tokenizer_config.json"))
        tokenizer_cfg = json.loads(Path(tokenizer_cfg_path).read_text(encoding="utf-8"))
    except Exception:
        tokenizer_cfg = {}

    tokenizer_class = str(tokenizer_cfg.get("tokenizer_class", "") or "")
    if tokenizer_class == "Qwen2Tokenizer":
        from transformers import Qwen2Tokenizer

        return Qwen2Tokenizer.from_pretrained(model_ref)

    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"failed to load tokenizer for {model_ref}")


def _load_model(model_ref: str):
    """Load a standard HuggingFace causal LM and fail fast if weights are missing."""
    try:
        return _load_model_checked(model_ref, trust_remote_code=False)
    except Exception:
        return _load_model_checked(model_ref, trust_remote_code=True)


def _load_model_checked(model_ref: str, *, trust_remote_code: bool):
    # `output_loading_info=True` helps prevent silent random init which produces
    # meaningless PPL numbers (we want the run to fail instead).
    try:
        out = AutoModelForCausalLM.from_pretrained(
            model_ref,
            torch_dtype="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=bool(trust_remote_code),
            output_loading_info=True,
        )
    except TypeError:
        # Older transformers: no loading info available.
        return AutoModelForCausalLM.from_pretrained(
            model_ref,
            torch_dtype="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=bool(trust_remote_code),
        )

    if isinstance(out, tuple) and len(out) == 2 and isinstance(out[1], dict):
        model, loading_info = out
    else:
        model, loading_info = out, {}

    missing = list(loading_info.get("missing_keys", []) or [])
    mismatched = list(loading_info.get("mismatched_keys", []) or [])
    errors = list(loading_info.get("error_msgs", []) or [])
    if missing or mismatched or errors:
        raise RuntimeError(
            "from_pretrained did not load a complete checkpoint; refusing to eval.\n"
            f"missing_keys={len(missing)} mismatched_keys={len(mismatched)} error_msgs={len(errors)}"
        )
    return model


def _resolve_model_path(model_ref: str) -> str:
    path = Path(model_ref)
    if path.exists() and (path / "config.json").exists():
        return str(path)
    return str(
        snapshot_download(
            model_ref,
            allow_patterns=["*.json", "*.safetensors", "tokenizer*", "merges.txt", "vocab.json"],
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_path", required=True, type=str)
    parser.add_argument(
        "--quip",
        action="store_true",
        help="Load QUIP# hfized checkpoints via third_party/quip-sharp model loader (legacy Llama path).",
    )
    parser.add_argument(
        "--tokenizer-ref",
        type=str,
        default=None,
        help="Optional HF repo/path to load the tokenizer from (recommended for QUIP# hfized Llama checkpoints).",
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--seqlen", default=2048, type=int)
    parser.add_argument("--datasets", nargs="*", default=["wikitext2", "c4"])
    parser.add_argument("--max_eval_tokens", type=int, default=0, help="0 => paper-style defaults: full Wikitext2, C4 = 256*seqlen")
    parser.add_argument("--json_out", type=str, default=None)
    args = parser.parse_args()

    random.seed(int(args.seed))
    torch.random.manual_seed(int(args.seed))

    model_path = _resolve_model_path(str(args.hf_path))
    if bool(args.quip):
        quip_root = str(REPO_ROOT / "third_party" / "quip-sharp")
        if quip_root not in sys.path:
            sys.path.insert(0, quip_root)
        from lib.utils.unsafe_import import model_from_hf_path

        device_map = "cuda:0" if torch.cuda.is_available() else None
        model, model_ref = model_from_hf_path(
            model_path,
            use_cuda_graph=False,
            use_flash_attn=False,
            device_map=device_map,
        )
        tokenizer_source = str(args.tokenizer_ref) if args.tokenizer_ref else str(model_ref)
        tokenizer = _load_tokenizer(tokenizer_source)
    else:
        tokenizer_source = str(args.tokenizer_ref) if args.tokenizer_ref else str(model_path)
        tokenizer = _load_tokenizer(tokenizer_source)
        model = _load_model(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    if not bool(args.quip):
        model.to("cuda:0" if torch.cuda.is_available() else "cpu")

    ppls = {}
    requested_max_eval_tokens = int(args.max_eval_tokens)
    eval_protocol = {}
    for dataset in args.datasets:
        dataset_norm = str(dataset).strip().lower()
        eval_protocol[dataset_norm] = eval_protocol_for_dataset(
            dataset_norm,
            seqlen=int(args.seqlen),
            requested_max_eval_tokens=requested_max_eval_tokens,
        )
        if dataset_norm == "wikitext2":
            tokens = load_or_build_wikitext2_tokens(
                tokenizer,
                max_eval_tokens=requested_max_eval_tokens,
                seqlen=int(args.seqlen),
                seed=int(args.seed),
                output_dir=str(Path(model_path)),
            )
        elif dataset_norm == "c4":
            tokens = load_or_build_c4_tokens(
                tokenizer,
                repo_root=str(REPO_ROOT),
                model_ref=str(model_path),
                max_eval_tokens=requested_max_eval_tokens,
                seqlen=int(args.seqlen),
                seed=int(args.seed),
                output_dir=str(Path(model_path)),
            )
        else:
            raise ValueError(f"unsupported eval dataset: {dataset_norm!r}")
        ppl = eval_ppl_from_tokens(model, tokens, seqlen=int(args.seqlen))
        print(f"{dataset_norm} perplexity: {ppl}", flush=True)
        ppls[dataset_norm] = float(ppl)

    if args.json_out:
        payload = {
            "seed": int(args.seed),
            "seqlen": int(args.seqlen),
            "datasets": list(args.datasets),
            "max_eval_tokens": requested_max_eval_tokens,
            "eval_protocol": eval_protocol,
            "ppl": ppls,
        }
        Path(args.json_out).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
