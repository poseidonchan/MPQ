from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from datasets import load_dataset
from transformers import AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]


def _git_head(path: Path) -> str:
    proc = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stdout)
    return proc.stdout.strip()


def _apply_patches() -> None:
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "patches" / "apply_patches.py")],
        cwd=str(REPO_ROOT),
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError("apply_patches.py failed")


def _apply_awq_preprocess(
    *,
    qmodel,
    calibration_blocks,
    bits: int,
    group_size: int,
    sym: bool,
    apply_clip: bool,
    batch_size: int,
) -> None:
    """Apply AWQ scale/(optional clip) in-place without replacing linear modules."""
    from gptqmodel import BACKEND
    from gptqmodel.looper.awq_processor import AWQProcessor
    from gptqmodel.looper.loop_processor import LoopProcessor
    from gptqmodel.looper.module_looper import ModuleLooper
    from gptqmodel.quantization.config import FORMAT, METHOD, QuantizeConfig

    class _AWQPreprocessProcessor(AWQProcessor):
        def apply_quant(self, named_linears, scales_list):
            for named_module in named_linears.values():
                named_module.module.cpu()

        def submodule_finalize(self, module, model, **kwargs):
            return

        def finalize(self, model, **kwargs):
            LoopProcessor.finalize(self, model=model, **kwargs)

        @classmethod
        def name(cls) -> str:  # pragma: no cover
            return "awq-preprocess"

    device = getattr(qmodel.quantize_config, "device", None)
    awq_qcfg = QuantizeConfig(
        bits=int(bits),
        group_size=int(group_size),
        sym=bool(sym),
        desc_act=False,
        act_group_aware=False,
        quant_method=METHOD.AWQ,
        format=FORMAT.GEMM,
        offload_to_disk=False,
        device=device,
    )
    processor = _AWQPreprocessProcessor(
        tokenizer=qmodel.tokenizer,
        qcfg=awq_qcfg,
        calibration=calibration_blocks,
        prepare_dataset_func=qmodel.prepare_dataset,
        calibration_concat_size=None,
        calibration_sort=None,
        batch_size=int(batch_size),
        gptq_model=qmodel,
        model=qmodel.model,
        calculate_w_wq_diff=False,
    )
    processor.apply_clip = bool(apply_clip)

    looper = ModuleLooper(qmodel, processors=[processor])
    looper.loop(backend=BACKEND, failsafe=awq_qcfg.failsafe)


@torch.inference_mode()
def _eval_cache_path(output_dir: str, tokenizer, max_eval_tokens: int) -> str:
    cache_dir = os.path.join(output_dir, "_eval_cache")
    os.makedirs(cache_dir, exist_ok=True)
    tokenizer_id = str(getattr(tokenizer, "name_or_path", "tokenizer"))
    digest = hashlib.sha256(f"{tokenizer_id}|wikitext2|{max_eval_tokens}|eval_v1".encode("utf-8")).hexdigest()[:16]
    tokenizer_label = Path(tokenizer_id).name.replace("/", "_") or "tokenizer"
    return os.path.join(cache_dir, f"wikitext2_{tokenizer_label}_{max_eval_tokens}_{digest}.pt")


def _load_or_build_wikitext2_tokens(tokenizer, *, max_eval_tokens: int, output_dir: str) -> torch.Tensor:
    cache_path = _eval_cache_path(output_dir, tokenizer, max_eval_tokens)
    if os.path.exists(cache_path):
        tokens = torch.load(cache_path, map_location="cpu")
        return tokens[:, :max_eval_tokens].contiguous()

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    tokens = tokenizer(text, return_tensors="pt", truncation=False).input_ids[:, :max_eval_tokens].cpu().contiguous()
    torch.save(tokens, cache_path)
    return tokens


def eval_ppl_wikitext2(
    model,
    tokenizer,
    *,
    seqlen: int = 2048,
    max_eval_tokens: int = 131072,
    output_dir: str,
) -> float:
    runtime_device = next(model.parameters()).device
    tokens = _load_or_build_wikitext2_tokens(tokenizer, max_eval_tokens=int(max_eval_tokens), output_dir=str(output_dir))
    tokens = tokens.to(runtime_device)

    nlls = []
    nsamples = tokens.shape[1] // int(seqlen)
    for i in range(nsamples):
        batch = tokens[:, i * seqlen : (i + 1) * seqlen]
        outputs = model(batch)
        shift_logits = outputs.logits[:, :-1, :].contiguous()
        shift_labels = batch[:, 1:].contiguous()
        loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        nlls.append(loss.float() * (int(seqlen) - 1))

    total_nll = torch.stack(nlls).sum()
    total_tokens = nsamples * (int(seqlen) - 1)
    return float(torch.exp(total_nll / total_tokens).item())


def _build_quantize_config(method: str, args) -> Any:
    from gptqmodel import QuantizeConfig

    device = str(args.device) if args.device else ("cuda:0" if torch.cuda.is_available() else "cpu")
    qcfg = QuantizeConfig(
        bits=int(args.bits),
        group_size=int(args.group_size),
        sym=bool(args.sym),
        device=device,
        offload_to_disk=bool(args.offload_to_disk),
        rotation=str(args.rotation) if args.rotation else None,
        desc_act=bool(args.desc_act),
    )

    if method in {"mpgq", "rtn"}:
        qcfg.mpgq.enabled = True
        qcfg.mpgq.iters = 0 if method == "rtn" else int(args.mpgq_iters)
        qcfg.mpgq.damping = float(args.mpgq_damping)
        qcfg.mpgq.prior_tau = 0.0 if method == "rtn" else float(args.mpgq_prior_tau)
        qcfg.mpgq.x1_mode = str(args.mpgq_x1_mode)
        qcfg.mpgq.polish_sweeps = 0 if method == "rtn" else int(args.mpgq_polish_sweeps)
        qcfg.mpgq.em_obs_precision = False if method == "rtn" else bool(args.mpgq_em_obs_precision)
        qcfg.mpgq.block_rows = int(args.mpgq_block_rows)
        qcfg.mpgq.rotation_seed = int(args.mpgq_rotation_seed)
    else:
        qcfg.mpgq.enabled = False

    if method == "ojbkq":
        from mpgq.baselines.ojbkq import OJBKQConfig

        qcfg.desc_act = False
        qcfg.act_group_aware = False
        qcfg.ojbkq = OJBKQConfig(
            enabled=True,
            random_trials=int(args.ojbkq_random_trials),
            alpha_mode=str(args.ojbkq_alpha_mode),
            alpha_override=None if args.ojbkq_alpha_override is None else float(args.ojbkq_alpha_override),
            alpha_scale=float(args.ojbkq_alpha_scale),
            sample_topk=None if args.ojbkq_sample_topk is None else int(args.ojbkq_sample_topk),
        )

    return qcfg


def run(args) -> Dict[str, Any]:
    _apply_patches()

    from gptqmodel import GPTQModel

    requested_method = str(args.method).strip().lower()
    supported = {"gptq", "gptq-strong", "mpgq", "rtn", "ojbkq", "awq+gptq", "awq+mpgq"}
    if requested_method not in supported:
        raise ValueError(f"unsupported method={requested_method!r} for backend_gptq")

    device = torch.device(args.device) if args.device else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    awq_preprocess = False
    awq_apply_clip = False
    awq_time_s = 0.0

    # Resolve method -> base solver + canonical defaults.
    if requested_method in {"gptq", "gptq-strong"}:
        method = "gptq"
        args.desc_act = True
        args.rotation = "hadamard"
    elif requested_method == "mpgq":
        method = "mpgq"
        args.desc_act = False
        args.rotation = "hadamard"
    elif requested_method == "rtn":
        method = "rtn"
        args.desc_act = False
        args.rotation = "hadamard"
    elif requested_method == "ojbkq":
        method = "ojbkq"
        args.desc_act = False
        args.rotation = None
    elif requested_method == "awq+gptq":
        method = "gptq"
        awq_preprocess = True
        awq_apply_clip = True
        args.sym = False
        args.desc_act = True
        args.rotation = "hadamard"
    elif requested_method == "awq+mpgq":
        method = "mpgq"
        awq_preprocess = True
        awq_apply_clip = False
        args.sym = False
        args.desc_act = False
        args.rotation = "hadamard"
    else:
        raise RuntimeError("unreachable")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)

    from mpgq.data import load_or_build_calibration_blocks

    calib_blocks = load_or_build_calibration_blocks(
        tokenizer=tokenizer,
        dataset_name=str(args.calibration_dataset),
        split=str(args.calibration_split),
        limit=int(args.calibration_limit),
        seqlen=int(args.calibration_seqlen),
        seed=int(args.seed),
        output_dir=str(args.output_dir),
        log_fn=lambda s: print(s, flush=True),
    )

    qcfg = _build_quantize_config(method, args)
    t_load = time.time()
    model = GPTQModel.load(args.model_id, qcfg)
    load_time_s = float(time.time() - t_load)
    if awq_preprocess:
        t_awq = time.time()
        _apply_awq_preprocess(
            qmodel=model,
            calibration_blocks=calib_blocks,
            bits=int(args.bits),
            group_size=int(args.group_size),
            sym=bool(args.sym),
            apply_clip=bool(awq_apply_clip),
            batch_size=int(args.awq_batch_size),
        )
        awq_time_s = float(time.time() - t_awq)
    t_quant = time.time()
    model.quantize(calib_blocks)
    quant_time = float(time.time() - t_quant)

    # Evaluate in-process.
    hf_model = getattr(model, "model", model)
    hf_model.eval()
    hf_model.to(device)
    ppl = eval_ppl_wikitext2(
        hf_model,
        tokenizer,
        seqlen=int(args.eval_seqlen),
        max_eval_tokens=int(args.eval_max_eval_tokens),
        output_dir=str(args.output_dir),
    )

    return {
        "backend": "gptqmodel",
        "requested_method": requested_method,
        "resolved_method": method,
        "baseline_family": "gptq" if requested_method in {"gptq", "gptq-strong"} else requested_method,
        "bits": int(args.bits),
        "group_size": int(args.group_size),
        "sym": bool(args.sym),
        "rotation": "none" if args.rotation in {None, "", "none"} else str(args.rotation),
        "desc_act": bool(args.desc_act),
        "seed": int(args.seed),
        "model_id": str(args.model_id),
        "awq_preprocess": {
            "enabled": bool(awq_preprocess),
            "apply_clip": bool(awq_apply_clip),
            "batch_size": int(args.awq_batch_size),
            "time_s": float(awq_time_s),
        },
        "calibration": {
            "dataset": str(args.calibration_dataset),
            "split": str(args.calibration_split),
            "limit": int(args.calibration_limit),
            "seqlen": int(args.calibration_seqlen),
        },
        "eval": {
            "dataset": "wikitext2",
            "seqlen": int(args.eval_seqlen),
            "max_eval_tokens": int(args.eval_max_eval_tokens),
            "ppl": float(ppl),
        },
        "timing": {
            "awq_preprocess_s": float(awq_time_s),
            "load_s": float(load_time_s),
            "quantize_s": float(quant_time),
        },
        "quantization_path": {
            "dense": bool(getattr(qcfg, "mpgq", None) and bool(getattr(qcfg.mpgq, "enabled", False)) and str(getattr(qcfg, "rotation", None)) == "hadamard"),
            "packed": not bool(getattr(qcfg, "mpgq", None) and bool(getattr(qcfg.mpgq, "enabled", False)) and str(getattr(qcfg, "rotation", None)) == "hadamard"),
        },
        "meta": {
            "repo_sha": _git_head(REPO_ROOT),
            "submodules": {
                "GPTQModel": _git_head(REPO_ROOT / "third_party" / "GPTQModel"),
                "quip-sharp": _git_head(REPO_ROOT / "third_party" / "quip-sharp"),
            },
        },
        "mpgq": {
            "enabled": bool(getattr(qcfg, "mpgq", None) and getattr(qcfg.mpgq, "enabled", False)),
            "iters": int(getattr(qcfg.mpgq, "iters", 0)) if getattr(qcfg, "mpgq", None) else 0,
            "prior_tau": float(getattr(qcfg.mpgq, "prior_tau", 0.0)) if getattr(qcfg, "mpgq", None) else 0.0,
            "damping": float(getattr(qcfg.mpgq, "damping", 0.0)) if getattr(qcfg, "mpgq", None) else 0.0,
            "x1_mode": str(getattr(qcfg.mpgq, "x1_mode", "")) if getattr(qcfg, "mpgq", None) else "",
            "polish_sweeps": int(getattr(qcfg.mpgq, "polish_sweeps", 0)) if getattr(qcfg, "mpgq", None) else 0,
            "em_obs_precision": bool(getattr(qcfg.mpgq, "em_obs_precision", False)) if getattr(qcfg, "mpgq", None) else False,
            "block_rows": int(getattr(qcfg.mpgq, "block_rows", 0)) if getattr(qcfg, "mpgq", None) else 0,
            "rotation_seed": int(getattr(qcfg.mpgq, "rotation_seed", 0)) if getattr(qcfg, "mpgq", None) else 0,
        },
        "ojbkq": {
            "enabled": bool(getattr(qcfg, "ojbkq", None) is not None and bool(getattr(qcfg.ojbkq, "enabled", False))),
            "random_trials": int(getattr(qcfg.ojbkq, "random_trials", 0)) if getattr(qcfg, "ojbkq", None) else 0,
            "alpha_mode": str(getattr(qcfg.ojbkq, "alpha_mode", "")) if getattr(qcfg, "ojbkq", None) else "",
            "alpha_override": getattr(qcfg.ojbkq, "alpha_override", None) if getattr(qcfg, "ojbkq", None) else None,
            "alpha_scale": float(getattr(qcfg.ojbkq, "alpha_scale", 0.0)) if getattr(qcfg, "ojbkq", None) else 0.0,
            "sample_topk": getattr(qcfg.ojbkq, "sample_topk", None) if getattr(qcfg, "ojbkq", None) else None,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        required=True,
        choices=["gptq", "gptq-strong", "mpgq", "rtn", "ojbkq", "awq+gptq", "awq+mpgq"],
    )
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--sym", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--offload-to-disk", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--seed", type=int, default=0)

    # Calibration (Gate A defaults)
    parser.add_argument("--calibration-dataset", type=str, default="c4")
    parser.add_argument("--calibration-split", type=str, default="train")
    parser.add_argument("--calibration-limit", type=int, default=128)
    parser.add_argument("--calibration-seqlen", type=int, default=2048)

    # Eval
    parser.add_argument("--eval-seqlen", type=int, default=2048)
    parser.add_argument("--eval-max-eval-tokens", type=int, default=131072)

    # AWQ preprocess
    parser.add_argument("--awq-batch-size", type=int, default=1)

    # MPGQ hyperparams (Gate A defaults)
    parser.add_argument("--mpgq-iters", type=int, default=10)
    parser.add_argument("--mpgq-damping", type=float, default=0.5)
    parser.add_argument("--mpgq-prior-tau", type=float, default=0.2)
    parser.add_argument("--mpgq-x1-mode", type=str, default="mmse", choices=["mmse", "map"])
    parser.add_argument("--mpgq-polish-sweeps", type=int, default=3)
    parser.add_argument("--mpgq-em-obs-precision", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mpgq-block-rows", type=int, default=32)
    parser.add_argument("--mpgq-rotation-seed", type=int, default=0)

    # OJBKQ baseline hyperparams
    parser.add_argument("--ojbkq-random-trials", type=int, default=5)
    parser.add_argument("--ojbkq-alpha-mode", type=str, default="paper", choices=["paper", "fixed"])
    parser.add_argument("--ojbkq-alpha-override", type=float, default=None)
    parser.add_argument("--ojbkq-alpha-scale", type=float, default=1.0)
    parser.add_argument("--ojbkq-sample-topk", type=int, default=None)

    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--json-out", type=str, required=True)

    args = parser.parse_args()

    result = run(args)

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(f"WROTE {out_path}")


if __name__ == "__main__":
    main()
