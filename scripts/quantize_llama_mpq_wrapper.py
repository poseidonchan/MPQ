from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
QUIP_ROOT = REPO_ROOT / "third_party" / "quip-sharp"
if str(QUIP_ROOT) not in sys.path:
    sys.path.insert(0, str(QUIP_ROOT))
from quantize_llama import quantize_finetune_llama as quip_quant  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--mpq-method", type=str, default="vamp")
    parser.add_argument("--mpq-solver", type=str, default=None)
    parser.add_argument("--mpq-polish-mode", type=str, default=None)
    parser.add_argument("--mpq-topk-k", type=int, default=16)
    parser.add_argument("--mpq-best-iterate", "--mpq_best_iterate", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--mpq-compile-prior", "--mpq_compile_prior", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--quant-backend", type=str, default=None)
    parser.add_argument("--quant-codebook", type=str, default=None)
    parser.add_argument("--quant-preprocess", type=str, default=None)
    parser.add_argument("--quant-rotation", type=str, default=None)
    parser.add_argument("--quant-solver", type=str, default=None)
    parser.add_argument("--quant-polish-mode", type=str, default=None)
    parser.add_argument("--quant-topk-source", type=str, default=None)
    parser.add_argument("--quant-topk-k", type=int, default=None)
    wrapper_args, passthrough = parser.parse_known_args()

    torch.set_grad_enabled(False)
    try:
        quip_quant.mp.set_start_method("spawn")
    except RuntimeError:
        pass
    quip_quant.mp.set_sharing_strategy("file_system")
    args = quip_quant.parser.parse_args(passthrough)
    torch.manual_seed(int(args.seed))
    os.makedirs(args.save_path, exist_ok=True)
    setattr(args, "mpq_method", str(wrapper_args.mpq_method))
    if wrapper_args.quant_backend is not None:
        setattr(args, "quant_backend", str(wrapper_args.quant_backend))
    if wrapper_args.quant_codebook is not None:
        setattr(args, "quant_codebook", str(wrapper_args.quant_codebook))
    if wrapper_args.quant_preprocess is not None:
        setattr(args, "quant_preprocess", str(wrapper_args.quant_preprocess))
    if wrapper_args.quant_rotation is not None:
        setattr(args, "quant_rotation", str(wrapper_args.quant_rotation))
    if wrapper_args.quant_solver is not None:
        setattr(args, "quant_solver", str(wrapper_args.quant_solver))
    if wrapper_args.quant_polish_mode is not None:
        setattr(args, "quant_polish_mode", str(wrapper_args.quant_polish_mode))
    if wrapper_args.quant_topk_source is not None:
        setattr(args, "quant_topk_source", str(wrapper_args.quant_topk_source))
    if wrapper_args.quant_topk_k is not None:
        setattr(args, "quant_topk_k", int(wrapper_args.quant_topk_k))
    if wrapper_args.mpq_solver is not None:
        setattr(args, "mpq_solver", str(wrapper_args.mpq_solver))
    if wrapper_args.mpq_polish_mode is not None:
        setattr(args, "mpq_polish_mode", str(wrapper_args.mpq_polish_mode))
    setattr(args, "mpq_topk_k", int(wrapper_args.mpq_topk_k))
    setattr(args, "mpq_best_iterate", bool(wrapper_args.mpq_best_iterate))
    setattr(args, "mpq_compile_prior", bool(wrapper_args.mpq_compile_prior))
    quip_quant.main(args)


if __name__ == "__main__":
    main()
