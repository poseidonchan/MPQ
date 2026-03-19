#!/usr/bin/env python3
"""Standalone lm-evaluation-harness runner for both QUIP# and GPTQ backends.

Usage:
    python scripts/eval_lm_eval.py --model-path /path/to/model --tasks arc_challenge,hellaswag
    python scripts/eval_lm_eval.py --model-path /path/to/quip-model --quip --tasks arc_challenge
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_quip_model(model_path: str):
    quip_root = str(REPO_ROOT / "third_party" / "quip-sharp")
    if quip_root not in sys.path:
        sys.path.insert(0, quip_root)
    from lib.utils.unsafe_import import model_from_hf_path

    model, tokenizer = model_from_hf_path(
        model_path, use_cuda_graph=False, use_flash_attn=False,
    )
    return model, tokenizer


def _load_hf_model(model_path: str, *, device: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map=device,
    )
    return model, tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--tasks", required=True, help="Comma-separated task list")
    parser.add_argument("--batch-size", default="1")
    parser.add_argument("--num-fewshot", type=int, default=0)
    parser.add_argument("--json-out", required=True)
    parser.add_argument("--quip", action="store_true")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    if not tasks:
        raise ValueError("no tasks specified")

    if args.quip:
        model, tokenizer = _load_quip_model(args.model_path)
    else:
        model, tokenizer = _load_hf_model(args.model_path, device=args.device)

    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM

    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=args.batch_size)
    import os
    os.environ["HF_ALLOW_CODE_EVAL"] = "1"
    eval_results = simple_evaluate(
        model=lm, tasks=tasks, batch_size=args.batch_size, num_fewshot=args.num_fewshot,
        confirm_run_unsafe_code=True,
    )

    results = {}
    for task_name, task_metrics in eval_results.get("results", {}).items():
        results[task_name] = {
            k: v for k, v in task_metrics.items() if k != "alias"
        }

    output = {"tasks": tasks, "results": results}
    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, sort_keys=True))
    print(f"WROTE {out_path}")


if __name__ == "__main__":
    main()
