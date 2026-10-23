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
    from transformers import AutoTokenizer
    from transformers.generation.utils import GenerationMixin

    model, model_str = model_from_hf_path(
        model_path, use_cuda_graph=False, use_flash_attn=False,
    )
    # quip-sharp's LlamaForCausalLM does NOT inherit GenerationMixin, so
    # .generate() is missing — which breaks gsm8k / mbpp (generate_until tasks).
    # Monkey-patch the class to mix in GenerationMixin at runtime and attach
    # a default generation_config from the model config.
    cls = type(model)
    if not isinstance(model, GenerationMixin):
        new_cls = type(cls.__name__, (cls, GenerationMixin), {})
        model.__class__ = new_cls
    if getattr(model, "generation_config", None) is None:
        from transformers import GenerationConfig
        model.generation_config = GenerationConfig.from_model_config(model.config)
    # Disable KV cache — quip-sharp's Llama forward was written against an older
    # transformers API (get_max_length, non-lazy DynamicCache). With KV cache
    # off, the forward path doesn't touch those code paths.
    model.generation_config.use_cache = False
    model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(model_str, use_fast=True)
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
    parser.add_argument("--num-fewshot", type=int, default=-1,
                        help="if negative, use per-task yaml defaults (gsm8k=5, mbpp=3, etc.)")
    parser.add_argument("--json-out", required=True)
    parser.add_argument("--quip", action="store_true")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--include-path", default=None,
                        help="extra task yaml directory to register (e.g. custom gpqa_main_mc_local)")
    args = parser.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    if not tasks:
        raise ValueError("no tasks specified")

    import os
    os.environ["HF_ALLOW_CODE_EVAL"] = "1"

    from lm_eval.evaluator import simple_evaluate
    from lm_eval.models.huggingface import HFLM
    from lm_eval.tasks import TaskManager

    if args.quip:
        model, tokenizer = _load_quip_model(args.model_path)
    else:
        model, tokenizer = _load_hf_model(args.model_path, device=args.device)
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=args.batch_size)
    if args.quip:
        # HFLM._model_generate hardcodes `use_cache=True`, but quip-sharp's custom
        # LlamaForCausalLM forward was written against older transformers and chokes
        # on the new DynamicCache API. Override _model_generate to force use_cache=False.
        import torch as _torch
        from lm_eval.models.utils_hf import stop_sequences_criteria as _stop
        def _model_generate_no_cache(self, context, max_length, stop, **generation_kwargs):
            generation_kwargs["temperature"] = generation_kwargs.get("temperature", 0.0)
            do_sample = generation_kwargs.get("do_sample")
            if generation_kwargs.get("temperature") == 0.0 and do_sample is None:
                generation_kwargs["do_sample"] = do_sample = False
            if do_sample is False and generation_kwargs.get("temperature") == 0.0:
                generation_kwargs.pop("temperature", None)
            stopping_criteria = _stop(self.tokenizer, stop, context.shape[1], context.shape[0])
            return self.model.generate(
                input_ids=context,
                max_length=max_length,
                stopping_criteria=stopping_criteria,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=False,
                **generation_kwargs,
            )
        import types as _types
        lm._model_generate = _types.MethodType(_model_generate_no_cache, lm)
    nfs_kw = {} if args.num_fewshot < 0 else {"num_fewshot": args.num_fewshot}
    tm_kw = {"task_manager": TaskManager(include_path=args.include_path)} if args.include_path else {}
    eval_results = simple_evaluate(
        model=lm, tasks=tasks, batch_size=args.batch_size,
        confirm_run_unsafe_code=True, **nfs_kw, **tm_kw,
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
