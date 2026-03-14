from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM


REPO_ROOT = Path(__file__).resolve().parents[1]
QUIP_ROOT = REPO_ROOT / "third_party" / "quip-sharp"


def _setup_pythonpath() -> None:
    import sys

    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(QUIP_ROOT))


def _maybe_reexec_with_conda_ld_library_path() -> None:
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix:
        return
    lib_dir = str(Path(conda_prefix) / "lib")
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    paths = [p for p in existing.split(":") if p]
    if lib_dir in paths:
        return
    if os.environ.get("MPGQ_REEXEC_LD_LIBRARY_PATH") == "1":
        return
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = f"{lib_dir}:{existing}" if existing else lib_dir
    env["MPGQ_REEXEC_LD_LIBRARY_PATH"] = "1"
    os.execvpe(sys.executable, [sys.executable, *sys.argv], env)


def _extract_llama_weights(model: Any, layer: int, suffix: str) -> List[torch.Tensor]:
    suffix = str(suffix).strip().lower()
    layers = None
    for attr in ("model", "transformer"):
        root = getattr(model, attr, None)
        if root is not None and hasattr(root, "layers"):
            layers = root.layers
            break
    if layers is None:
        raise RuntimeError("could not locate Llama layers at model.model.layers")
    block = layers[int(layer)]
    if suffix == "qkv":
        attn = block.self_attn
        return [attn.q_proj.weight, attn.k_proj.weight, attn.v_proj.weight]
    if suffix == "o":
        return [block.self_attn.o_proj.weight]
    if suffix == "down":
        return [block.mlp.down_proj.weight]
    if suffix == "up":
        # Match QUIP# convention: stack gate + up for the 'up' module.
        return [block.mlp.gate_proj.weight, block.mlp.up_proj.weight]
    raise ValueError(f"unsupported suffix={suffix!r}; expected qkv|o|up|down")


@torch.inference_mode()
def _load_hessian_sym(path: Path) -> torch.Tensor:
    data = torch.load(str(path), map_location=torch.device("cpu"))
    if not isinstance(data, dict) or "flatH" not in data:
        raise RuntimeError(f"invalid Hessian file: {path}")

    from lib import utils as quip_utils

    H = quip_utils.flat_to_sym(data["flatH"], int(data["n"]))
    mu = data["mu"]
    H.add_(mu[None, :] * mu[:, None])
    return H


def _build_quip_args(base: argparse.Namespace) -> Any:
    # Reuse QUIP#'s own parser defaults so we don't miss required fields.
    from quantize_llama.quantize_finetune_llama import parser as quip_parser

    args = quip_parser.parse_args([])
    args.seed = int(base.seed)
    args.sigma_reg = float(base.sigma_reg)
    args.sigma_reg2 = float(base.sigma_reg2)
    args.incoh_mode = str(base.incoh_mode)
    args.scale_override = float(base.scale_override)
    args.resid_scale_override = float(base.resid_scale_override)
    args.use_fp64 = bool(base.use_fp64)
    args.full_svd = bool(base.full_svd)
    args.no_use_buffered = bool(base.no_use_buffered)
    args.lowmem_ldlq = bool(base.lowmem_ldlq)
    args.lora_rank = int(base.lora_rank)
    args.rescale_WH = bool(base.rescale_wh)
    # mpgq knobs
    args.solver = "mpgq"
    args.mpgq_init = str(base.mpgq_init)
    args.mpgq_iters = int(base.mpgq_iters)
    args.mpgq_damping = float(base.mpgq_damping)
    args.mpgq_prior_tau = float(base.mpgq_prior_tau)
    args.mpgq_prior_tau_scale_mode = str(base.mpgq_prior_tau_scale_mode)
    args.mpgq_gamma_start = float(base.mpgq_gamma_start)
    args.mpgq_gamma_retry_factor = float(base.mpgq_gamma_retry_factor)
    args.mpgq_gamma_retry_steps = int(base.mpgq_gamma_retry_steps)
    args.mpgq_obs_precision = float(base.mpgq_obs_precision)
    args.mpgq_obs_precision_end = base.mpgq_obs_precision_end
    args.mpgq_em_obs_precision = bool(base.mpgq_em_obs_precision)
    args.mpgq_em_obs_precision_damping = float(base.mpgq_em_obs_precision_damping)
    args.mpgq_em_obs_precision_min = float(base.mpgq_em_obs_precision_min)
    args.mpgq_em_obs_precision_max = float(base.mpgq_em_obs_precision_max)
    args.mpgq_vamp_x1_mode = str(base.mpgq_vamp_x1_mode)
    args.mpgq_vamp_map_mix = float(base.mpgq_vamp_map_mix)
    args.mpgq_denoiser_metric = str(base.mpgq_denoiser_metric)
    args.mpgq_adaptive_accept_reject = bool(base.mpgq_adaptive_accept_reject)
    args.mpgq_adaptive_retry_steps = int(base.mpgq_adaptive_retry_steps)
    args.mpgq_adaptive_accept_tol = float(base.mpgq_adaptive_accept_tol)
    args.mpgq_adaptive_damping_min = float(base.mpgq_adaptive_damping_min)
    args.mpgq_adaptive_obs_backoff = float(base.mpgq_adaptive_obs_backoff)
    args.mpgq_best_iterate_export = bool(base.mpgq_best_iterate_export)
    args.mpgq_blocks_per_chunk = int(base.mpgq_blocks_per_chunk)
    args.mpgq_block_vamp = bool(base.mpgq_block_vamp)
    args.mpgq_block_rows = int(base.mpgq_block_rows)
    args.mpgq_polish_sweeps = int(base.mpgq_polish_sweeps)
    args.mpgq_polish_metric = str(base.mpgq_polish_metric)
    args.mpgq_polish_max_blocks = int(base.mpgq_polish_max_blocks)
    args.mpgq_metric_rows_per_chunk = int(base.mpgq_metric_rows_per_chunk)
    return args


def _override(obj: Any, updates: Dict[str, Any]) -> Any:
    clone = copy.deepcopy(obj)
    for k, v in updates.items():
        setattr(clone, k, v)
    return clone


@torch.inference_mode()
def _quip_prepare_processed(
    *,
    H: torch.Tensor,
    weights: List[torch.Tensor],
    cb_name: str,
    args: Any,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, Any, torch.Tensor, torch.Tensor, torch.Tensor]:
    from lib import codebook as quip_codebook
    from lib import utils as quip_utils
    from lib.algo import quip

    H_in = H
    dtype_ = torch.float64 if bool(args.use_fp64) else torch.float32
    scales = [w.to(dtype_).square().mean().sqrt() for w in weights]
    W = torch.vstack([weights[i].to(dtype_) / scales[i] for i in range(len(weights))]).to(dtype_)

    n = int(H_in.shape[0])
    H_work = H_in.to(dtype_).clone()
    H_work = quip_utils.regularize_H(H_work, n, float(args.sigma_reg))

    H = H_work.to(device)
    W = W.to(device)
    cb = quip_codebook.get_codebook(str(cb_name))
    cb = cb.to(dtype_).to(device)

    incoh_out = quip.incoherence_preprocess(H, W, args)
    if incoh_out is None and not bool(args.use_fp64):
        args64 = _override(args, {"use_fp64": True})
        return _quip_prepare_processed(H=H_in, weights=weights, cb_name=cb_name, args=args64, device=device)
    if incoh_out is None:
        raise RuntimeError("quip.incoherence_preprocess failed (even in fp64).")

    Lhr, Hr, Wr, SU, SV, scaleWH = incoh_out
    if int(getattr(args, "lora_rank", 0)) > 0:
        Wr, Hr = quip.low_rank_preprocess(Wr, Hr, Lhr, args)

    block_LDL_out = quip_utils.block_LDL(Hr, int(cb.codesz))
    if block_LDL_out is None and not bool(args.use_fp64):
        args64 = _override(args, {"use_fp64": True})
        return _quip_prepare_processed(H=H_in, weights=weights, cb_name=cb_name, args=args64, device=device)
    if block_LDL_out is None:
        raise RuntimeError("quip block_LDL failed (even in fp64).")

    L, D = block_LDL_out

    Wscale = Wr.square().mean().sqrt()
    if float(args.scale_override) > 0:
        Wscale = Wscale / float(args.scale_override)
    else:
        Wscale = Wscale / float(cb.opt_scale)
    Wr = Wr / Wscale

    return Wr, Hr, cb, L, D, Wscale


@torch.inference_mode()
def _run_ldlq(
    *,
    Wr: torch.Tensor,
    Hr: torch.Tensor,
    L: torch.Tensor,
    D: torch.Tensor,
    cb: Any,
    args: Any,
) -> Tuple[torch.Tensor, torch.Tensor]:
    from lib.algo import quip

    if bool(getattr(args, "no_use_buffered", False)):
        hatWr, Qidxs = quip.LDLQ(Wr, Hr, L, D, cb, args)
    elif bool(getattr(args, "lowmem_ldlq", False)) or bool(getattr(args, "use_fp64", False)):
        hatWr, Qidxs = quip.LDLQ_buffered_lowmem(Wr, Hr, L, D, cb, args, buf_cols=128)
    else:
        hatWr, Qidxs = quip.LDLQ_buffered(Wr, Hr, L, D, cb, args, buf_cols=128)
    return hatWr, Qidxs


@torch.inference_mode()
def _run_mpgq_refine_cached(
    *,
    Wr: torch.Tensor,
    Hr: torch.Tensor,
    hatWr0: torch.Tensor,
    Qidxs0: torch.Tensor,
    cb: Any,
    args: Any,
    module_name: str,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    from mpgq.adapters.quip_sharp import (
        QuipAdapterConfig,
        _exact_surrogate_chunk_means,
        _random_e8p12_init,
        _rtn_e8p12_init,
        _zero_e8p12_init,
        _vamp_hessian_eigen,
    )
    from mpgq.solver import E8P12VampConfig, solve_e8p12_rankspace_vamp

    device = Wr.device
    m, n = Wr.shape
    codesz = int(getattr(cb, "codesz", 8))
    nblocks = n // codesz

    init_mode = str(getattr(args, "mpgq_init", "rtn") or "rtn").strip().lower()
    seed = int(getattr(args, "seed", getattr(args, "mpgq_seed", 0)) or 0)
    resid_scale_override = getattr(args, "resid_scale_override", None)
    if resid_scale_override is not None and float(resid_scale_override) < 0:
        resid_scale_override = None

    if init_mode == "random":
        hatWr_init, Qidxs_init = _random_e8p12_init(
            cb=cb,
            rows=m,
            nblocks=nblocks,
            seed_key=f"{module_name}:mpgq_init",
            base_seed=seed,
            device=device,
            dtype=Wr.dtype,
        )
    elif init_mode == "rtn":
        hatWr_init, Qidxs_init = _rtn_e8p12_init(Wr=Wr, cb=cb, resid_scale_override=resid_scale_override)
    elif init_mode == "zero":
        hatWr_init, Qidxs_init = _zero_e8p12_init(Wr=Wr, cb=cb, resid_scale_override=resid_scale_override)
    elif init_mode == "ldlq":
        hatWr_init = hatWr0.to(dtype=torch.float32)
        Qidxs_init = Qidxs0.to(dtype=getattr(cb, "idx_dtype", torch.int64))
    else:
        raise ValueError(f"unknown mpgq_init={init_mode!r}")

    adapter_cfg = QuipAdapterConfig()
    eigenvalues, V, rank, eigen_floor = _vamp_hessian_eigen(
        Hr,
        jitter_base=float(adapter_cfg.jitter_base),
        floor_scale=float(adapter_cfg.eigen_floor_scale),
    )
    V = V.to(device=device, dtype=torch.float32)
    Vh = V.T.contiguous()
    eigenvalues = eigenvalues.to(device=device, dtype=torch.float32)
    S = torch.sqrt(eigenvalues).contiguous()
    Vs = V * S.unsqueeze(0)

    requested_obs = float(getattr(args, "mpgq_obs_precision", 0.0) or 0.0)
    if requested_obs > 0.0:
        obs_precision = requested_obs
    else:
        diff_u = (Wr - hatWr_init).matmul(Vs)
        mse = float(diff_u.square().mean().clamp_min_(1e-12).item())
        obs_precision = 1.0 / mse

    vamp_cfg = E8P12VampConfig(
        iters=int(getattr(args, "mpgq_iters", 4) or 4),
        damping=float(getattr(args, "mpgq_damping", 0.5) or 0.5),
        x1_mode=str(getattr(args, "mpgq_vamp_x1_mode", "map") or "map"),
        map_mix=float(getattr(args, "mpgq_vamp_map_mix", 0.5) or 0.5),
        denoiser_metric=str(getattr(args, "mpgq_denoiser_metric", "isotropic") or "isotropic"),
        gamma_start=float(getattr(args, "mpgq_gamma_start", 1.0) or 1.0),
        gamma_retry_factor=float(getattr(args, "mpgq_gamma_retry_factor", 4.0) or 4.0),
        gamma_retry_steps=int(getattr(args, "mpgq_gamma_retry_steps", 3) or 3),
        prior_tau=float(getattr(args, "mpgq_prior_tau", 0.0) or 0.0),
        prior_tau_scale_mode=str(getattr(args, "mpgq_prior_tau_scale_mode", "absmax") or "absmax"),
        obs_precision=float(requested_obs),
        obs_precision_end=(
            None
            if getattr(args, "mpgq_obs_precision_end", None) is None
            else float(getattr(args, "mpgq_obs_precision_end", None))
        ),
        em_obs_precision=bool(getattr(args, "mpgq_em_obs_precision", True)),
        em_obs_precision_damping=float(getattr(args, "mpgq_em_obs_precision_damping", 0.3) or 0.3),
        em_obs_precision_min=float(getattr(args, "mpgq_em_obs_precision_min", 1e-4) or 1e-4),
        em_obs_precision_max=float(getattr(args, "mpgq_em_obs_precision_max", 1e8) or 1e8),
        best_iterate_export=bool(getattr(args, "mpgq_best_iterate_export", True)),
        adaptive_accept_reject=bool(getattr(args, "mpgq_adaptive_accept_reject", True)),
        adaptive_retry_steps=int(getattr(args, "mpgq_adaptive_retry_steps", 3) or 3),
        adaptive_accept_tol=float(getattr(args, "mpgq_adaptive_accept_tol", 1e-4) or 1e-4),
        adaptive_damping_min=float(getattr(args, "mpgq_adaptive_damping_min", 0.1) or 0.1),
        adaptive_obs_precision_backoff=float(getattr(args, "mpgq_adaptive_obs_backoff", 0.5) or 0.5),
        blocks_per_chunk=int(getattr(args, "mpgq_blocks_per_chunk", 16) or 16),
        block_vamp=bool(getattr(args, "mpgq_block_vamp", False)),
        polish_sweeps=int(getattr(args, "mpgq_polish_sweeps", 3)),
        polish_metric=str(getattr(args, "mpgq_polish_metric", "gram") or "gram"),
        polish_metric_rows_per_chunk=int(getattr(args, "mpgq_metric_rows_per_chunk", 0) or 0),
        polish_max_blocks=int(getattr(args, "mpgq_polish_max_blocks", 0) or 0),
    )

    block_rows = min(int(m), max(int(getattr(args, "mpgq_block_rows", 32) or 32), 1))
    exact_ldlq, _ = _exact_surrogate_chunk_means(Wr, hatWr0.to(dtype=torch.float32), Hr, block_rows=block_rows)
    exact_before, _ = _exact_surrogate_chunk_means(Wr, hatWr_init, Hr, block_rows=block_rows)

    t0 = time.time()
    final_hatWr = hatWr_init.clone()
    final_Qidxs = Qidxs_init.clone()
    obs_precision_block = float(obs_precision)
    for row_start in range(0, int(m), block_rows):
        row_end = min(row_start + block_rows, int(m))
        rb = slice(row_start, row_end)
        hat_chunk, idx_chunk, _err_chunk, stats_chunk = solve_e8p12_rankspace_vamp(
            Wr=Wr[rb],
            hatWr_init=final_hatWr[rb],
            Qidxs_init=final_Qidxs[rb],
            cb=cb,
            V=V,
            Vh=Vh,
            S=S,
            Vs=Vs,
            cfg=vamp_cfg,
            module_name=module_name,
            obs_precision_init=obs_precision_block,
            log_stats=True,
        )
        final_hatWr[rb] = hat_chunk
        final_Qidxs[rb] = idx_chunk
        if bool(vamp_cfg.em_obs_precision) and float(vamp_cfg.obs_precision) <= 0.0:
            obs_precision_block = float(stats_chunk.get("obs_precision_final", obs_precision_block))

    exact_after, _ = _exact_surrogate_chunk_means(Wr, final_hatWr, Hr, block_rows=block_rows)
    stats = {
        "module": module_name,
        "rank": int(rank),
        "eigen_floor": float(eigen_floor),
        "obs_precision_initial": float(obs_precision),
        "obs_precision_final": float(obs_precision_block),
        "exact_surrogate_ldlq_mean": float(exact_ldlq),
        "exact_surrogate_before_mean": float(exact_before),
        "exact_surrogate_after_mean": float(exact_after),
        "exact_surrogate_gain_mean": float(exact_before - exact_after),
        "solver_time_s": float(time.time() - t0),
        "vamp": stats_chunk,
    }
    return final_hatWr, final_Qidxs, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-module QUIP# proxy benchmark (LDLQ vs RTN+MPGQ).")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--hessian-dir", type=str, default=None)
    parser.add_argument("--hessian-path", type=str, default=None)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--suffix", type=str, default="qkv", choices=["qkv", "o", "up", "down"])
    parser.add_argument("--codebook", type=str, default="E8P12")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sigma-reg", type=float, default=1e-2)
    parser.add_argument("--sigma-reg2", type=float, default=1e-2)
    parser.add_argument("--incoh-mode", type=str, default="had", choices=["had", "kron"])
    parser.add_argument("--scale-override", type=float, default=0.9)
    parser.add_argument("--resid-scale-override", type=float, default=-1.0)
    parser.add_argument("--lora-rank", type=int, default=0)
    parser.add_argument("--use-fp64", action="store_true")
    parser.add_argument("--full-svd", action="store_true")
    parser.add_argument("--no-use-buffered", action="store_true")
    parser.add_argument("--lowmem-ldlq", action="store_true")
    parser.add_argument("--rescale-wh", action="store_true")

    parser.add_argument("--mpgq-init", type=str, default="rtn", choices=["random", "rtn", "zero", "ldlq"])
    parser.add_argument("--mpgq-iters", type=int, default=4)
    parser.add_argument("--mpgq-damping", type=float, default=0.5)
    parser.add_argument("--mpgq-prior-tau", type=float, default=0.0)
    parser.add_argument(
        "--mpgq-prior-tau-scale-mode",
        type=str,
        default="absmax",
        choices=["none", "absmax", "rms", "mean_abs"],
    )
    parser.add_argument("--mpgq-gamma-start", type=float, default=1.0)
    parser.add_argument("--mpgq-gamma-retry-factor", type=float, default=4.0)
    parser.add_argument("--mpgq-gamma-retry-steps", type=int, default=3)
    parser.add_argument("--mpgq-obs-precision", type=float, default=0.0)
    parser.add_argument("--mpgq-obs-precision-end", type=float, default=None)
    parser.add_argument("--mpgq-em-obs-precision", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mpgq-em-obs-precision-damping", type=float, default=0.3)
    parser.add_argument("--mpgq-em-obs-precision-min", type=float, default=1e-4)
    parser.add_argument("--mpgq-em-obs-precision-max", type=float, default=1e8)
    parser.add_argument("--mpgq-vamp-x1-mode", type=str, default="map", choices=["mean", "map", "mix"])
    parser.add_argument("--mpgq-vamp-map-mix", type=float, default=0.5)
    parser.add_argument(
        "--mpgq-denoiser-metric",
        type=str,
        default="isotropic",
        choices=["isotropic", "gram"],
        help="E8P12 x1/MMSE metric: isotropic or gram (full block Hessian).",
    )
    parser.add_argument("--mpgq-denoiser-diag-hessian-clamp", type=float, default=10.0)
    parser.add_argument("--mpgq-adaptive-accept-reject", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mpgq-adaptive-retry-steps", type=int, default=3)
    parser.add_argument("--mpgq-adaptive-accept-tol", type=float, default=1e-4)
    parser.add_argument("--mpgq-adaptive-damping-min", type=float, default=0.1)
    parser.add_argument("--mpgq-adaptive-obs-backoff", type=float, default=0.5)
    parser.add_argument("--mpgq-best-iterate-export", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mpgq-blocks-per-chunk", type=int, default=16)
    parser.add_argument("--mpgq-block-vamp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--mpgq-block-rows", type=int, default=32)
    parser.add_argument("--mpgq-polish-sweeps", type=int, default=3)
    parser.add_argument("--mpgq-polish-metric", type=str, default="gram", choices=["gram", "isotropic", "diag"])
    parser.add_argument("--mpgq-polish-max-blocks", type=int, default=0)
    parser.add_argument("--mpgq-metric-rows-per-chunk", type=int, default=0)
    parser.add_argument("--mpgq-ldlq-tune-sweeps", type=int, default=0,
                        help="Post-VAMP LDLQ-style L2 BCD tune sweeps (fast alternative to Gram polish)")

    parser.add_argument("--json-out", type=str, default=None)
    args = parser.parse_args()

    _maybe_reexec_with_conda_ld_library_path()
    _setup_pythonpath()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    hessian_path = None
    if args.hessian_path:
        hessian_path = Path(args.hessian_path).expanduser().resolve()
    elif args.hessian_dir:
        hessian_path = Path(args.hessian_dir).expanduser().resolve() / f"{int(args.layer)}_{args.suffix}.pt"
    if hessian_path is None or not hessian_path.exists():
        raise RuntimeError("provide --hessian-path or --hessian-dir (with existing {layer}_{suffix}.pt).")

    model = AutoModelForCausalLM.from_pretrained(
        str(args.base_model),
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map=None,
    )
    model.eval()
    weights = _extract_llama_weights(model, int(args.layer), str(args.suffix))
    weights = [w.detach().cpu() for w in weights]
    del model

    H = _load_hessian_sym(hessian_path)
    quip_args = _build_quip_args(args)

    Wr, Hr, cb, L, D, Wscale = _quip_prepare_processed(
        H=H,
        weights=weights,
        cb_name=str(args.codebook),
        args=quip_args,
        device=device,
    )

    t0 = time.time()
    hatWr0, Qidxs0 = _run_ldlq(Wr=Wr, Hr=Hr, L=L, D=D, cb=cb.to(device), args=quip_args)
    ldlq_time = float(time.time() - t0)

    # Also measure LDLQ without tune for ablation
    saved_tune = quip_args.quip_tune_iters
    quip_args.quip_tune_iters = 0
    hatWr0_notune, _ = _run_ldlq(Wr=Wr, Hr=Hr, L=L, D=D, cb=cb.to(device), args=quip_args)
    quip_args.quip_tune_iters = saved_tune

    mpgq_hatWr, mpgq_Qidxs, mpgq_stats = _run_mpgq_refine_cached(
        Wr=Wr,
        Hr=Hr,
        hatWr0=hatWr0,
        Qidxs0=Qidxs0,
        cb=cb.to(device),
        args=quip_args,
        module_name=f"{int(args.layer)}_{args.suffix}",
    )

    # Optional LDLQ-style tune (fast L2 BCD) as post-VAMP polish
    ldlq_tune_sweeps = int(getattr(args, "mpgq_ldlq_tune_sweeps", 0) or 0)
    if ldlq_tune_sweeps > 0:
        from mpgq.adapters.quip_sharp import _exact_surrogate_chunk_means as _esc
        _br = min(Wr.shape[0], max(int(getattr(args, "mpgq_block_rows", 32) or 32), 1))
        mpgq_hatWr = mpgq_hatWr.clone()
        mpgq_Qidxs = mpgq_Qidxs.clone()
        _e_pre, _ = _esc(Wr, mpgq_hatWr.float(), Hr, block_rows=_br)
        codesz_t = int(getattr(cb, "codesz", 8))
        nblocks_t = Wr.shape[1] // codesz_t
        resid_scale = float(getattr(quip_args, "resid_scale_override", 0.9) or 0.9)
        Hr_f = Hr.float()
        Hr_blk_invs = [torch.linalg.inv(Hr_f[codesz_t*k:codesz_t*(k+1), codesz_t*k:codesz_t*(k+1)].double()).float() for k in range(nblocks_t)]
        for _sw in range(ldlq_tune_sweeps):
            for k in reversed(range(nblocks_t)):
                s, e = codesz_t * k, codesz_t * (k + 1)
                res = (Wr - mpgq_hatWr.to(dtype=Wr.dtype)).float()
                tgt = mpgq_hatWr[:, s:e].float() + res.matmul(Hr_f[:, s:e]).matmul(Hr_blk_invs[k])
                mpgq_hatWr[:, s:e], mpgq_Qidxs[:, k] = cb.quantize(tgt, resid_scale_override=resid_scale)
        _e_post, _ = _esc(Wr, mpgq_hatWr.float(), Hr, block_rows=_br)
        print(f"[ldlq-tune] {ldlq_tune_sweeps} sweeps: {float(_e_pre):.2f} -> {float(_e_post):.2f} (delta {float(_e_post-_e_pre):+.2f})")
        mpgq_stats["exact_surrogate_after_mean"] = float(_e_post)
        mpgq_stats["ldlq_tune_sweeps"] = ldlq_tune_sweeps

    out = {
        "base_model": str(args.base_model),
        "layer": int(args.layer),
        "suffix": str(args.suffix),
        "codebook": str(args.codebook),
        "device": str(device),
        "hessian_path": str(hessian_path),
        "ldlq_time_s": float(ldlq_time),
        "mpgq": mpgq_stats,
    }

    # Compute LDLQ no-tune error for ablation
    from mpgq.adapters.quip_sharp import _exact_surrogate_chunk_means
    block_rows = min(Wr.shape[0], max(int(getattr(args, "mpgq_block_rows", 32) or 32), 1))
    ldlq_notune_mean, _ = _exact_surrogate_chunk_means(Wr, hatWr0_notune.to(dtype=torch.float32), Hr, block_rows=block_rows)
    out["ldlq_notune_proxy"] = float(ldlq_notune_mean)

    json_out = (
        Path(args.json_out).resolve()
        if args.json_out
        else (REPO_ROOT / "runs" / f"module_proxy_{int(time.time())}.json")
    )
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    print(f"WROTE {json_out}")
    ldlq_mean = float(mpgq_stats["exact_surrogate_ldlq_mean"])
    after_mean = float(mpgq_stats["exact_surrogate_after_mean"])
    print(f"LDLQ no-tune proxy: {float(ldlq_notune_mean):.6g}")
    print(f"LDLQ mean proxy: {ldlq_mean:.6g}")
    print(f"MPGQ mean proxy: {after_mean:.6g} (delta {after_mean-ldlq_mean:+.6g})")
    ratio = float(mpgq_stats["solver_time_s"]) / max(ldlq_time, 1e-12)
    print(f"time ratio mpgq/ldlq ~= {ratio:.2f}")


if __name__ == "__main__":
    main()
