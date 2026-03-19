from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mpq.eval_utils import eval_ppl_from_tokens, load_or_build_c4_tokens, load_or_build_wikitext2_tokens


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_path", required=True, type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--seqlen", default=2048, type=int)
    parser.add_argument("--datasets", nargs="*", default=["wikitext2", "c4"])
    parser.add_argument("--max_eval_tokens", type=int, default=0)
    parser.add_argument("--json_out", type=str, default=None)
    args = parser.parse_args()

    random.seed(int(args.seed))
    torch.random.manual_seed(int(args.seed))

    tokenizer = AutoTokenizer.from_pretrained(args.hf_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.hf_path, torch_dtype="auto", low_cpu_mem_usage=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval().to("cuda:0" if torch.cuda.is_available() else "cpu")

    ppls = {}
    max_eval_tokens = int(args.max_eval_tokens) if int(args.max_eval_tokens) > 0 else 131072
    for dataset in args.datasets:
        dataset_norm = str(dataset).strip().lower()
        if dataset_norm == "wikitext2":
            tokens = load_or_build_wikitext2_tokens(
                tokenizer,
                max_eval_tokens=max_eval_tokens,
                seqlen=int(args.seqlen),
                seed=int(args.seed),
                output_dir=str(Path(args.hf_path)),
            )
        elif dataset_norm == "c4":
            tokens = load_or_build_c4_tokens(
                tokenizer,
                repo_root=str(REPO_ROOT),
                model_ref=str(args.hf_path),
                max_eval_tokens=max_eval_tokens,
                seqlen=int(args.seqlen),
                seed=int(args.seed),
                output_dir=str(Path(args.hf_path)),
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
            "max_eval_tokens": max_eval_tokens,
            "ppl": ppls,
        }
        Path(args.json_out).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
