from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
QUIP_ROOT = REPO_ROOT / "third_party" / "quip-sharp"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(QUIP_ROOT) not in sys.path:
    sys.path.insert(0, str(QUIP_ROOT))

from mpq.quip_generic import quantize_model_generic


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--calibration_seed", default=None, type=int)
    parser.add_argument("--num_cpu_threads", default=8, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--devset_size", default=384, type=int)
    parser.add_argument("--ctx_size", default=4096, type=int)
    parser.add_argument("--calibration_dataset", default="rp1t", type=str)
    parser.add_argument("--calibration_split", default="train", type=str)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--hessian_path", type=str, required=True)
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--sigma_reg", default=1e-2, type=float)
    parser.add_argument("--sigma_reg2", default=1e-2, type=float)
    parser.add_argument("--incoh_mode", default="had", type=str, choices=["had", "kron"])
    parser.add_argument("--lora_rank", default=0, type=int)
    parser.add_argument("--scale_override", default=-1, type=float)
    parser.add_argument("--resid_scale_override", default=-1, type=float)
    parser.add_argument("--codebook", type=str, required=True)
    parser.add_argument("--quip_tune_iters", default=10, type=int)
    parser.add_argument("--use_fp64", action="store_true")
    parser.add_argument("--full_svd", action="store_true")
    parser.add_argument("--no_use_buffered", action="store_true")
    parser.add_argument("--rescale_WH", action="store_true")
    parser.add_argument("--sample_proc", default=1, type=int)
    parser.add_argument("--lowmem_ldlq", action="store_true")
    parser.add_argument("--ft_lr", default=5e-5, type=float)
    parser.add_argument("--ft_susv_lr", default=5e-4, type=float)
    parser.add_argument("--ft_bs", default=4, type=int)
    parser.add_argument("--ft_update_freq", default=2, type=int)
    parser.add_argument("--ft_epochs", default=5, type=int)
    parser.add_argument("--ft_valid_freq", default=1, type=int)
    parser.add_argument("--ft_valid_size", default=128, type=float)
    parser.add_argument("--ft_early_stop", default=3, type=int)
    parser.add_argument("--ft_train_mode", action="store_true")
    parser.add_argument("--ft_grad_ckpt", action="store_true")
    parser.add_argument("--solver", default="quip", type=str, choices=["quip", "mpq"])
    parser.add_argument("--quant-backend", default="quip", type=str)
    parser.add_argument("--quant-codebook", default="e8p12", type=str)
    parser.add_argument("--quant-preprocess", default="none", type=str)
    parser.add_argument("--quant-rotation", default="had", type=str)
    parser.add_argument("--quant-solver", default=None, type=str)
    parser.add_argument("--quant-polish-mode", default="none", type=str)
    parser.add_argument("--quant-topk-source", default="aux_vamp", type=str)
    parser.add_argument("--quant-topk-k", default=16, type=int)
    parser.add_argument("--mpq_init", default="random", type=str, choices=["random", "rtn", "ldlq"])
    parser.add_argument("--mpq_iters", default=4, type=int)
    parser.add_argument("--mpq_damping", default=0.5, type=float)
    parser.add_argument("--mpq_prior_tau", default=0.0, type=float)
    parser.add_argument("--mpq_prior_tau_scale_mode", default="absmax", type=str)
    parser.add_argument("--mpq_gamma_start", default=1.0, type=float)
    parser.add_argument("--mpq_gamma_retry_factor", default=4.0, type=float)
    parser.add_argument("--mpq_gamma_retry_steps", default=3, type=int)
    parser.add_argument("--mpq_obs_precision", default=0.0, type=float)
    parser.add_argument("--mpq_compile_prior", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mpq_em_obs_precision", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mpq_em_obs_precision_damping", default=0.3, type=float)
    parser.add_argument("--mpq_em_obs_precision_min", default=1e-4, type=float)
    parser.add_argument("--mpq_em_obs_precision_max", default=1e8, type=float)
    parser.add_argument("--mpq_blocks_per_chunk", default=16, type=int)
    parser.add_argument("--mpq_block_rows", default=128, type=int)
    parser.add_argument("--mpq_polish_sweeps", default=3, type=int)
    parser.add_argument("--mpq_polish_metric", default="gram", type=str, choices=["gram", "isotropic", "diag"])
    parser.add_argument("--mpq_polish_max_blocks", default=0, type=int)
    parser.add_argument("--mpq_metric_rows_per_chunk", default=0, type=int)
    parser.add_argument("--mpq-method", dest="mpq_method", default="vamp", type=str, choices=["vamp", "vamp-topk-polish"])
    parser.add_argument("--mpq-topk-k", dest="mpq_topk_k", default=16, type=int)
    parser.add_argument("--mpq_best_iterate", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--ojbkq-random-trials", dest="ojbkq_random_trials", default=5, type=int)
    parser.add_argument("--ojbkq-alpha-mode", dest="ojbkq_alpha_mode", default="paper", type=str, choices=["paper", "fixed"])
    parser.add_argument("--ojbkq-alpha-override", dest="ojbkq_alpha_override", default=None, type=float)
    parser.add_argument("--ojbkq-alpha-scale", dest="ojbkq_alpha_scale", default=1.0, type=float)
    parser.add_argument("--ojbkq-sample-topk", dest="ojbkq_sample_topk", default=None, type=int)
    parser.add_argument("--model-arch", default="auto", type=str)
    parser.add_argument("--start-layer", default=None, type=int)
    parser.add_argument("--stop-after-layer", default=None, type=int)
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    torch.manual_seed(int(args.seed))
    os.makedirs(args.save_path, exist_ok=True)
    quantize_model_generic(args)


if __name__ == "__main__":
    main()
