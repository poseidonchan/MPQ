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
    if os.environ.get("MPQ_REEXEC_LD_LIBRARY_PATH") == "1":
        return
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = f"{lib_dir}:{existing}" if existing else lib_dir
    env["MPQ_REEXEC_LD_LIBRARY_PATH"] = "1"
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
    if getattr(base, "quip_tune_iters", None) is not None and int(base.quip_tune_iters) >= 0:
        args.quip_tune_iters = int(base.quip_tune_iters)
    # mpq knobs
    args.solver = "mpq"
    args.mpq_init = str(base.mpq_init)
    args.mpq_iters = int(base.mpq_iters)
    args.mpq_damping = float(base.mpq_damping)
    args.mpq_prior_tau = float(base.mpq_prior_tau)
    args.mpq_prior_tau_scale_mode = str(base.mpq_prior_tau_scale_mode)
    args.mpq_gamma_start = float(base.mpq_gamma_start)
    args.mpq_gamma_retry_factor = float(base.mpq_gamma_retry_factor)
    args.mpq_gamma_retry_steps = int(base.mpq_gamma_retry_steps)
    args.mpq_obs_precision = float(base.mpq_obs_precision)
    args.mpq_em_obs_precision = bool(base.mpq_em_obs_precision)
    args.mpq_compile_prior = bool(getattr(base, "mpq_compile_prior", True))
    args.mpq_em_obs_precision_damping = float(base.mpq_em_obs_precision_damping)
    args.mpq_em_obs_precision_min = float(base.mpq_em_obs_precision_min)
    args.mpq_em_obs_precision_max = float(base.mpq_em_obs_precision_max)
    args.mpq_blocks_per_chunk = int(base.mpq_blocks_per_chunk)
    args.mpq_block_rows = int(base.mpq_block_rows)
    args.mpq_polish_sweeps = int(base.mpq_polish_sweeps)
    args.mpq_polish_metric = str(base.mpq_polish_metric)
    args.mpq_polish_max_blocks = int(base.mpq_polish_max_blocks)
    args.mpq_metric_rows_per_chunk = int(base.mpq_metric_rows_per_chunk)
    # Optional solver selection (consumed by MPQ wrapper, not QUIP# proper).
    args.mpq_method = str(getattr(base, "mpq_method", "vamp"))
    args.mpq_topk_k = int(getattr(base, "mpq_topk_k", 8))
    args.mpq_debug_export = getattr(base, "mpq_debug_export", None)
    args.mpq_debug_summary = bool(getattr(base, "mpq_debug_summary", False))
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

    # QUIP# uses randomized incoherence preprocessing (Hadamard signs / butterfly factors).
    # Seed torch so proxy runs are reproducible for a fixed `--seed`.
    torch.manual_seed(int(getattr(args, "seed", 0)))

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
def _run_mpq_refine_cached(
    *,
    Wr: torch.Tensor,
    Hr: torch.Tensor,
    hatWr0: torch.Tensor,
    Qidxs0: torch.Tensor,
    cb: Any,
    args: Any,
    module_name: str,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    from mpq.adapters.quip_sharp import (
        QuipAdapterConfig,
        _exact_surrogate_chunk_means,
        _random_e8p12_init,
        _rtn_e8p12_init,
        _zero_e8p12_init,
        _vamp_hessian_eigen,
    )
    from mpq.solver import (
        E8P12VampConfig,
        solve_e8p12_rankspace_vamp,
    )
    from mpq.solvers.vamp_topk_polish import solve_e8p12_rankspace_vamp_topk_polish

    device = Wr.device
    m, n = Wr.shape
    codesz = int(getattr(cb, "codesz", 8))
    nblocks = n // codesz

    init_mode = str(getattr(args, "mpq_init", "rtn") or "rtn").strip().lower()
    seed = int(getattr(args, "seed", getattr(args, "mpq_seed", 0)) or 0)
    resid_scale_override = getattr(args, "resid_scale_override", None)
    if resid_scale_override is not None and float(resid_scale_override) < 0:
        resid_scale_override = None

    if init_mode == "random":
        hatWr_init, Qidxs_init = _random_e8p12_init(
            cb=cb,
            rows=m,
            nblocks=nblocks,
            seed_key=f"{module_name}:mpq_init",
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
        raise ValueError(f"unknown mpq_init={init_mode!r}")

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

    requested_obs = float(getattr(args, "mpq_obs_precision", 0.0) or 0.0)
    if requested_obs > 0.0:
        obs_precision = requested_obs
    else:
        diff_u = (Wr - hatWr_init).matmul(Vs)
        mse = float(diff_u.square().mean().clamp_min_(1e-12).item())
        obs_precision = 1.0 / mse

    method = str(getattr(args, "mpq_method", "vamp") or "vamp").strip().lower()
    if method not in {"vamp", "vamp-topk-polish"}:
        raise ValueError(f"unknown mpq_method={method!r}")

    vamp_cfg = E8P12VampConfig(
        iters=int(getattr(args, "mpq_iters", 16)),
        damping=float(getattr(args, "mpq_damping", 0.5)),
        gamma_start=float(getattr(args, "mpq_gamma_start", 1.0)),
        gamma_retry_factor=float(getattr(args, "mpq_gamma_retry_factor", 4.0)),
        gamma_retry_steps=int(getattr(args, "mpq_gamma_retry_steps", 3)),
        prior_tau=float(getattr(args, "mpq_prior_tau", 0.25)),
        prior_tau_scale_mode=str(getattr(args, "mpq_prior_tau_scale_mode", "absmax")),
        compile_prior=bool(getattr(args, "mpq_compile_prior", True)),
        obs_precision=float(requested_obs),
        em_obs_precision=bool(getattr(args, "mpq_em_obs_precision", True)),
        em_obs_precision_damping=float(getattr(args, "mpq_em_obs_precision_damping", 0.3)),
        em_obs_precision_min=float(getattr(args, "mpq_em_obs_precision_min", 1e-4)),
        em_obs_precision_max=float(getattr(args, "mpq_em_obs_precision_max", 1e8)),
        blocks_per_chunk=int(getattr(args, "mpq_blocks_per_chunk", 16)),
        polish_sweeps=int(getattr(args, "mpq_polish_sweeps", 3)),
        polish_metric=str(getattr(args, "mpq_polish_metric", "gram")),
        polish_metric_rows_per_chunk=int(getattr(args, "mpq_metric_rows_per_chunk", 0)),
        polish_max_blocks=int(getattr(args, "mpq_polish_max_blocks", 0)),
    )

    block_rows = min(int(m), max(int(getattr(args, "mpq_block_rows", 32) or 32), 1))
    exact_ldlq, _ = _exact_surrogate_chunk_means(Wr, hatWr0.to(dtype=torch.float32), Hr, block_rows=block_rows)
    exact_before, _ = _exact_surrogate_chunk_means(Wr, hatWr_init, Hr, block_rows=block_rows)

    t0 = time.time()
    final_hatWr = hatWr_init.clone()
    final_Qidxs = Qidxs_init.clone()
    obs_precision_block = float(obs_precision)

    debug_export = getattr(args, "mpq_debug_export", None)
    debug_summary = bool(getattr(args, "mpq_debug_summary", False))
    debug_chunks: List[Dict[str, Any]] = []

    for row_start in range(0, int(m), block_rows):
        row_end = min(row_start + block_rows, int(m))
        rb = slice(row_start, row_end)
        if method == "vamp":
            dbg: Dict[str, Any] | None = {} if (debug_export or debug_summary) else None
            hat_chunk, idx_chunk, _err_chunk, stats_chunk = solve_e8p12_rankspace_vamp(
                Wr=Wr[rb],
                hatWr_init=final_hatWr[rb],
                cb=cb,
                V=V,
                Vh=Vh,
                S=S,
                Vs=Vs,
                cfg=vamp_cfg,
                module_name=module_name,
                obs_precision_init=obs_precision_block,
                log_stats=True,
                debug_out=dbg,
            )
            final_hatWr[rb] = hat_chunk
            final_Qidxs[rb] = idx_chunk
            if bool(vamp_cfg.em_obs_precision) and float(vamp_cfg.obs_precision) <= 0.0:
                obs_precision_block = float(stats_chunk.get("obs_precision_final", obs_precision_block))
            if dbg is not None:
                dbg["row_start"] = int(row_start)
                dbg["row_end"] = int(row_end)
                dbg["method"] = "vamp"
                debug_chunks.append(dbg)
        elif method == "vamp-topk-polish":
            dbg = {} if (debug_export or debug_summary) else None
            hat_chunk, idx_chunk, _err_chunk, stats_chunk = solve_e8p12_rankspace_vamp_topk_polish(
                Wr=Wr[rb],
                hatWr_init=final_hatWr[rb],
                cb=cb,
                V=V,
                Vh=Vh,
                S=S,
                Vs=Vs,
                cfg=vamp_cfg,
                candidate_k=int(getattr(args, "mpq_topk_k", 8) or 8),
                module_name=module_name,
                obs_precision_init=obs_precision_block,
                log_stats=True,
                debug_out=dbg,
            )
            final_hatWr[rb] = hat_chunk
            final_Qidxs[rb] = idx_chunk
            if bool(vamp_cfg.em_obs_precision) and float(vamp_cfg.obs_precision) <= 0.0:
                obs_precision_block = float(stats_chunk.get("obs_precision_final", obs_precision_block))
            if dbg is not None:
                dbg["row_start"] = int(row_start)
                dbg["row_end"] = int(row_end)
                dbg["method"] = "vamp-topk-polish"
                debug_chunks.append(dbg)
    exact_after, _ = _exact_surrogate_chunk_means(Wr, final_hatWr, Hr, block_rows=block_rows)

    if debug_chunks and (debug_export or debug_summary):
        payload = {
            "meta": {
                "module": str(module_name),
                "method": str(method),
                "seed": int(getattr(args, "seed", 0)),
                "m": int(m),
                "n": int(n),
                "codesz": int(codesz),
                "nblocks": int(nblocks),
                "block_rows": int(block_rows),
            },
            "chunks": debug_chunks,
        }

        if debug_export:
            out_path = Path(str(debug_export)).expanduser()
            if out_path.suffix != ".pt":
                out_path.mkdir(parents=True, exist_ok=True)
                safe_module = str(module_name).replace("/", "_").replace(":", "_")
                out_path = out_path / f"debug_{safe_module}.pt"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(payload, str(out_path))
            print(f"WROTE {out_path}", flush=True)

        if debug_summary:
            def _flatten(key: str) -> torch.Tensor:
                parts: List[torch.Tensor] = []
                for rec in debug_chunks:
                    val: Any = rec
                    for tok in key.split("."):
                        if not isinstance(val, dict):
                            val = None
                            break
                        val = val.get(tok)
                    if isinstance(val, torch.Tensor):
                        parts.append(val.reshape(-1).to(dtype=torch.float32))
                if not parts:
                    return torch.empty(0, dtype=torch.float32)
                return torch.cat(parts, dim=0)

            changed = _flatten("changed_mask").to(dtype=torch.bool)
            gap = _flatten("block_stats.top2_gap")
            ent = _flatten("block_stats.entropy")
            cov = _flatten("block_stats.cov_trace")

            def _q(x: torch.Tensor) -> Tuple[float, float, float]:
                if x.numel() <= 0:
                    return float("nan"), float("nan"), float("nan")
                qs = torch.quantile(x, torch.tensor([0.1, 0.5, 0.9], dtype=torch.float32))
                return float(qs[0].item()), float(qs[1].item()), float(qs[2].item())

            total = int(changed.numel())
            changed_count = int(changed.sum().item()) if total else 0
            print("\n[MPQ debug summary]", flush=True)
            print(f"- blocks: {total}  changed: {changed_count}  frac: {changed_count / max(total, 1):.6f}", flush=True)
            if total and gap.numel() == total:
                g0, g1 = gap[~changed], gap[changed]
                e0, e1 = ent[~changed], ent[changed]
                c0, c1 = cov[~changed], cov[changed]
                print(f"- top2_gap p10/p50/p90 unchanged: {_q(g0)}  changed: {_q(g1)}", flush=True)
                print(f"- entropy  p10/p50/p90 unchanged: {_q(e0)}  changed: {_q(e1)}", flush=True)
                print(f"- cov_tr   p10/p50/p90 unchanged: {_q(c0)}  changed: {_q(c1)}", flush=True)

            hard = _flatten("hard_mask").to(dtype=torch.bool)
            if hard.numel() == total:
                hard_count = int(hard.sum().item())
                tp = int((hard & changed).sum().item())
                recall = tp / max(changed_count, 1)
                precision = tp / max(hard_count, 1)
                print(f"- hard_mask: {hard_count}  tp: {tp}  recall: {recall:.6f}  precision: {precision:.6f}", flush=True)

            ess = _flatten("survey.ess")
            if ess.numel() == total:
                ess_fin = ess[torch.isfinite(ess)]
                print(f"- survey ESS p10/p50/p90 (finite): {_q(ess_fin)}", flush=True)
            center_mass = _flatten("survey.center_mass")
            if center_mass.numel() == total:
                cm_fin = center_mass[torch.isfinite(center_mass)]
                print(f"- survey center_mass p10/p50/p90 (finite): {_q(cm_fin)}", flush=True)
            mean_shift = _flatten("survey.mean_shift")
            if mean_shift.numel() == total:
                ms_fin = mean_shift[torch.isfinite(mean_shift)]
                print(f"- survey mean_shift p10/p50/p90 (finite): {_q(ms_fin)}", flush=True)
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
    parser = argparse.ArgumentParser(description="Single-module QUIP# proxy benchmark (LDLQ vs RTN+MPQ).")
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
    parser.add_argument("--quip-tune-iters", type=int, default=-1, help="Override QUIP# tune sweeps (-1 keeps QUIP default).")

    parser.add_argument("--mpq-init", type=str, default="rtn", choices=["random", "rtn", "zero", "ldlq"])
    parser.add_argument(
        "--mpq-method",
        type=str,
        default="vamp",
        choices=["vamp", "vamp-topk-polish"],
    )
    parser.add_argument("--mpq-iters", type=int, default=16)
    parser.add_argument("--mpq-damping", type=float, default=0.5)
    parser.add_argument("--mpq-prior-tau", type=float, default=0.25)
    parser.add_argument(
        "--mpq-prior-tau-scale-mode",
        type=str,
        default="absmax",
        choices=["none", "absmax", "rms", "mean_abs"],
    )
    parser.add_argument("--mpq-gamma-start", type=float, default=1.0)
    parser.add_argument("--mpq-gamma-retry-factor", type=float, default=4.0)
    parser.add_argument("--mpq-gamma-retry-steps", type=int, default=3)
    parser.add_argument("--mpq-obs-precision", type=float, default=0.0)
    parser.add_argument("--mpq-em-obs-precision", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mpq-compile-prior", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mpq-em-obs-precision-damping", type=float, default=0.3)
    parser.add_argument("--mpq-em-obs-precision-min", type=float, default=1e-4)
    parser.add_argument("--mpq-em-obs-precision-max", type=float, default=1e8)
    parser.add_argument("--mpq-blocks-per-chunk", type=int, default=16)
    parser.add_argument("--mpq-block-rows", type=int, default=128)
    parser.add_argument("--mpq-polish-sweeps", type=int, default=3)
    parser.add_argument("--mpq-polish-metric", type=str, default="gram", choices=["gram", "isotropic", "diag"])
    parser.add_argument("--mpq-polish-max-blocks", type=int, default=0)
    parser.add_argument("--mpq-metric-rows-per-chunk", type=int, default=0)
    parser.add_argument("--mpq-topk-k", type=int, default=8)

    parser.add_argument(
        "--mpq-debug-export",
        type=str,
        default=None,
        help="If set, torch.save() a .pt debug bundle (path can be a .pt file or a directory).",
    )
    parser.add_argument("--mpq-debug-summary", action="store_true", help="Print a short debug summary after MPQ.")

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

    mpq_hatWr, mpq_Qidxs, mpq_stats = _run_mpq_refine_cached(
        Wr=Wr,
        Hr=Hr,
        hatWr0=hatWr0,
        Qidxs0=Qidxs0,
        cb=cb.to(device),
        args=quip_args,
        module_name=f"{int(args.layer)}_{args.suffix}",
    )

    out = {
        "base_model": str(args.base_model),
        "layer": int(args.layer),
        "suffix": str(args.suffix),
        "codebook": str(args.codebook),
        "device": str(device),
        "hessian_path": str(hessian_path),
        "ldlq_time_s": float(ldlq_time),
        "mpq": mpq_stats,
    }

    # Compute LDLQ no-tune error for ablation
    from mpq.adapters.quip_sharp import _exact_surrogate_chunk_means
    block_rows = min(Wr.shape[0], max(int(getattr(args, "mpq_block_rows", 32) or 32), 1))
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
    ldlq_mean = float(mpq_stats["exact_surrogate_ldlq_mean"])
    after_mean = float(mpq_stats["exact_surrogate_after_mean"])
    print(f"LDLQ no-tune proxy: {float(ldlq_notune_mean):.6g}")
    print(f"LDLQ mean proxy: {ldlq_mean:.6g}")
    print(f"MPQ mean proxy: {after_mean:.6g} (delta {after_mean-ldlq_mean:+.6g})")
    ratio = float(mpq_stats["solver_time_s"]) / max(ldlq_time, 1e-12)
    print(f"time ratio mpq/ldlq ~= {ratio:.2f}")


if __name__ == "__main__":
    main()
