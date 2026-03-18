from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch

from .linear import em_update_obs_precision_rankspace, linear_mmse_exact, vamp_rankspace_error
from .polish.block_cd import (
    BlockCDConfig,
    E8P12Provider,
    E8P12TopKProvider,
    UniformScalarProvider,
    UniformTopKProvider,
    block_coordinate_descent_rankspace,
)
from .priors.e8p12 import (
    prior_scale_from_center,
    vamp_vector_discrete_mmse_e8p12,
    vamp_vector_map_e8p12,
    vamp_vector_map_topk_e8p12,
)
from .priors.uniform import dequant_from_q, discrete_uniform_mmse, nearest_atom_idx, uniform_map_topk
from .utils import (
    alpha_valid_mask,
    log_damp_positive,
    precision_valid_mask,
    require_valid_alpha,
    require_valid_precision,
)


@dataclass(frozen=True)
class UniformVampConfig:
    iters: int = 10
    damping: float = 0.5
    final_decision: str = "map"  # "map" | "round_mean"
    init_from: str = "rtn"  # "rtn" | "float" | "zero"
    gamma_start: float = 1.0
    gamma_retry_factor: float = 4.0
    gamma_retry_steps: int = 3
    prior_tau: float = 0.2
    obs_precision: float = 0.0
    em_obs_precision: bool = True
    em_obs_precision_damping: float = 0.3
    em_obs_precision_min: float = 1e-4
    em_obs_precision_max: float = 1e8
    polish_sweeps: int = 3
    polish_mode: str = "block_cd"  # "none" | "block_cd" | "topk_block_cd"
    topk_k: int = 8


@dataclass(frozen=True)
class E8P12VampConfig:
    """Scalar-precision VAMP config for QUIP E8P12."""

    iters: int = 16
    damping: float = 0.5
    gamma_start: float = 1.0
    gamma_retry_factor: float = 4.0
    gamma_retry_steps: int = 3
    prior_tau: float = 0.25
    prior_tau_scale_mode: str = "absmax"
    compile_prior: bool = True
    obs_precision: float = 0.0
    em_obs_precision: bool = True
    em_obs_precision_damping: float = 0.3
    em_obs_precision_min: float = 1e-4
    em_obs_precision_max: float = 1e8
    blocks_per_chunk: int = 16
    polish_sweeps: int = 3
    polish_mode: str = "block_cd"  # "none" | "block_cd" | "topk_block_cd"
    polish_metric: str = "gram"  # "gram" | "diag"
    polish_metric_rows_per_chunk: int = 0
    polish_max_blocks: int = 0
    topk_k: int = 8


@torch.inference_mode()
def _rowwise_quant_mse(W: torch.Tensor, x_quant: torch.Tensor) -> torch.Tensor:
    return (W - x_quant).square().mean(dim=1, keepdim=True).clamp_min_(1e-12)


@torch.inference_mode()
def _global_quant_mse(W: torch.Tensor, x_quant: torch.Tensor) -> float:
    return float((W - x_quant).square().mean().clamp_min_(1e-12).item())


@torch.inference_mode()
def _e8p12_block_debug_stats(
    *,
    r1: torch.Tensor,  # (B, n)
    gamma1: torch.Tensor,  # (B, 1) or (B, n_blocks, 1|8)
    cb: object,
    blocks_per_chunk: int,
    prior_center_blocks: Optional[torch.Tensor],
    prior_scale_blocks: Optional[torch.Tensor],
    prior_tau: float,
    prior_tau_scale_mode: str,
) -> Dict[str, torch.Tensor]:
    """Return per-block diagnostics at a VAMP fixed point.

    Intended for post-hoc analysis (changed-by-polish, selector recall, etc.).
    All outputs are float32 tensors on the same device as `r1`.
    """
    B, n = r1.shape
    codesz = int(getattr(cb, "codesz", 8))
    if codesz != 8 or (n % codesz) != 0:
        raise ValueError(f"E8P12 debug expects n divisible by 8; got n={n}, codesz={codesz}.")
    nblocks = n // codesz

    post = vamp_vector_discrete_mmse_e8p12(
        r1_blocks=r1.reshape(B, nblocks, codesz),
        gamma1=gamma1,
        cb=cb,
        blocks_per_chunk=int(blocks_per_chunk),
        need_stats=False,
        need_var_diag=False,
        prior_center_blocks=prior_center_blocks,
        prior_scale_blocks=prior_scale_blocks,
        prior_tau=float(prior_tau),
        prior_tau_scale_mode=str(prior_tau_scale_mode),
        return_logZ_blocks=False,
        return_margin_blocks=True,
        return_entropy_blocks=True,
    )
    gap = post.get("margin_blocks")
    ent = post.get("entropy_blocks")
    if gap is None or ent is None:
        raise RuntimeError("E8P12 oracle did not return margin_blocks/entropy_blocks for debug.")

    var1 = post["var1"].to(dtype=torch.float32)
    cov_trace = (8.0 * var1).to(dtype=torch.float32)
    return {
        "top2_gap": gap.to(dtype=torch.float32),
        "entropy": ent.to(dtype=torch.float32),
        "cov_trace": cov_trace,
    }


@torch.inference_mode()
def _resolve_obs_precision(
    *,
    W: torch.Tensor,
    x_init: torch.Tensor,
    Vs: torch.Tensor,
    requested: float,
) -> float:
    requested = float(requested)
    if math.isfinite(requested) and requested > 0.0:
        return requested
    if Vs.numel() == 0:
        return 1.0 / _global_quant_mse(W, x_init)
    Y_u = W.matmul(Vs)
    Y_init_u = x_init.matmul(Vs)
    return 1.0 / float((Y_u - Y_init_u).square().mean().clamp_min_(1e-12).item())


@torch.inference_mode()
def _resolve_polish_mode(*, sweeps: int, mode: str) -> str:
    if int(sweeps) <= 0:
        return "none"
    mode_norm = str(mode or "block_cd").strip().lower()
    if mode_norm in {"none", "off"}:
        return "none"
    if mode_norm in {"block_cd", "cd", "full_cd"}:
        return "block_cd"
    if mode_norm in {"topk_block_cd", "topk_cd", "topk", "vamp_topk"}:
        return "topk_block_cd"
    raise ValueError(f"unsupported polish_mode={mode!r}; expected none|block_cd|topk_block_cd.")


@torch.inference_mode()
def _resolve_e8p12_polish_metric(metric: str) -> str:
    polish_metric = str(metric or "gram").strip().lower()
    if polish_metric in ("isotropic", "diag"):
        return "diag"
    if polish_metric == "gram":
        return "gram"
    raise ValueError(f"unsupported polish_metric={metric!r}; expected 'gram'|'diag'.")


@torch.inference_mode()
def build_uniform_vamp_topk_candidates(
    *,
    r1: torch.Tensor,
    gamma1: torch.Tensor,
    W: torch.Tensor,
    scale_full: torch.Tensor,
    zero_full: torch.Tensor,
    maxq: int,
    prior_tau: float,
    k: int,
    atoms: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    return uniform_map_topk(
        r1=r1,
        gamma1=gamma1,
        scale_full=scale_full,
        zero_full=zero_full,
        maxq=maxq,
        prior_center=W,
        prior_tau=float(prior_tau),
        k=max(int(k), 1),
        atoms=atoms,
    )


@torch.inference_mode()
def build_e8p12_vamp_topk_candidates(
    *,
    r1: torch.Tensor,
    gamma1: torch.Tensor,
    Wr: torch.Tensor,
    cb: object,
    blocks_per_chunk: int,
    prior_tau: float,
    prior_tau_scale_mode: str,
    k: int,
) -> Dict[str, torch.Tensor]:
    B, n = Wr.shape
    codesz = int(getattr(cb, "codesz", 8))
    if codesz != 8 or (n % codesz) != 0:
        raise ValueError(f"E8P12 top-k expects n divisible by 8; got n={n}, codesz={codesz}.")
    nblocks = n // codesz
    W_blocks = Wr.reshape(B, nblocks, codesz)
    prior_scale_blocks = None
    if float(prior_tau) > 0.0:
        prior_scale_blocks = prior_scale_from_center(W_blocks, prior_tau_scale_mode)
    return vamp_vector_map_topk_e8p12(
        r1_blocks=r1.reshape(B, nblocks, codesz),
        gamma1=gamma1,
        cb=cb,
        k=max(int(k), 1),
        blocks_per_chunk=max(int(blocks_per_chunk), 1),
        prior_center_blocks=W_blocks if float(prior_tau) > 0.0 else None,
        prior_scale_blocks=prior_scale_blocks,
        prior_tau=float(prior_tau),
        prior_tau_scale_mode=str(prior_tau_scale_mode),
    )


@torch.inference_mode()
def polish_uniform_rankspace(
    *,
    x_init: torch.Tensor,
    Y_u: torch.Tensor,
    Vs: torch.Tensor,
    scale_full: torch.Tensor,
    zero_full: torch.Tensor,
    maxq: int,
    sweeps: int,
    polish_mode: str,
    topk_values: Optional[torch.Tensor] = None,
    topk_q: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
    mode = _resolve_polish_mode(sweeps=int(sweeps), mode=str(polish_mode))
    x_work = x_init.to(dtype=torch.float32)
    if mode == "none" or Vs.numel() == 0:
        q_int = nearest_atom_idx(x=x_work, scale_full=scale_full, zero_full=zero_full, maxq=maxq).to(dtype=torch.int32)
        x_out = dequant_from_q(q_int=q_int, scale_full=scale_full, zero_full=zero_full)
        err = vamp_rankspace_error(Y_u=Y_u, x=x_out, Vs=Vs)
        return q_int, x_out, err, {"accepted_updates": 0.0}

    if mode == "topk_block_cd":
        if topk_values is None:
            raise ValueError("topk_block_cd requires topk_values for uniform polish.")
        provider = UniformTopKProvider(topk_values=topk_values, topk_q=topk_q)
    else:
        provider = UniformScalarProvider(scale=scale_full, zero=zero_full, maxq=maxq)

    x_polished, q_topk, polish_stats = block_coordinate_descent_rankspace(
        x=x_work,
        Y_u=Y_u,
        Vs=Vs,
        block_size=1,
        provider=provider,
        cfg=BlockCDConfig(sweeps=int(sweeps), metric="diag", accept_tol=0.0, order_mode="greedy_gain"),
    )
    if q_topk is not None:
        q_polished = q_topk.to(dtype=torch.int32)
    else:
        q_polished = nearest_atom_idx(x=x_polished, scale_full=scale_full, zero_full=zero_full, maxq=maxq).to(dtype=torch.int32)
    x_out = dequant_from_q(q_int=q_polished, scale_full=scale_full, zero_full=zero_full)
    err = vamp_rankspace_error(Y_u=Y_u, x=x_out, Vs=Vs)
    return q_polished, x_out, err, polish_stats


@torch.inference_mode()
def polish_e8p12_rankspace(
    *,
    x_init: torch.Tensor,
    idxs_init: torch.Tensor,
    Y_u: torch.Tensor,
    Vs: torch.Tensor,
    cb: object,
    sweeps: int,
    polish_mode: str,
    polish_metric: str,
    metric_rows_per_chunk: int = 0,
    polish_max_blocks: int = 0,
    topk_idxs: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
    mode = _resolve_polish_mode(sweeps=int(sweeps), mode=str(polish_mode))
    metric_norm = _resolve_e8p12_polish_metric(str(polish_metric))
    x_work = x_init.to(dtype=torch.float32)
    if mode == "none" or Vs.numel() == 0:
        err = vamp_rankspace_error(Y_u=Y_u, x=x_work, Vs=Vs)
        return x_work, idxs_init, err, {"accepted_updates": 0.0}

    if mode == "topk_block_cd":
        if topk_idxs is None:
            raise ValueError("topk_block_cd requires topk_idxs for E8P12 polish.")
        provider = E8P12TopKProvider(cb=cb, topk_idxs=topk_idxs)
    else:
        provider = E8P12Provider(cb=cb, metric_rows_per_chunk=int(metric_rows_per_chunk))

    codesz = int(getattr(cb, "codesz", 8))
    x_out, idxs_out, polish_stats = block_coordinate_descent_rankspace(
        x=x_work,
        Y_u=Y_u,
        Vs=Vs,
        block_size=codesz,
        provider=provider,
        cfg=BlockCDConfig(
            sweeps=int(sweeps),
            metric="gram" if metric_norm == "gram" else "diag",
            accept_tol=0.0,
            metric_rows_per_chunk=int(metric_rows_per_chunk),
            max_blocks_per_sweep=int(polish_max_blocks),
            order_mode="greedy_gain",
        ),
        idxs_init=idxs_init,
    )
    if idxs_out is None:
        raise RuntimeError("E8P12 polish must return block indices.")
    err = vamp_rankspace_error(Y_u=Y_u, x=x_out, Vs=Vs)
    return x_out, idxs_out, err, polish_stats


@torch.inference_mode()
def solve_uniform_rankspace_vamp(
    *,
    W: torch.Tensor,  # (B, n)
    x_rtn: torch.Tensor,  # (B, n)
    scale_full: torch.Tensor,  # (B, n)
    zero_full: torch.Tensor,  # (B, n)
    maxq: int,
    V: torch.Tensor,  # (n, R)
    Vh: torch.Tensor,  # (R, n)
    S: torch.Tensor,  # (R,)
    Vs: torch.Tensor,  # (n, R)
    cfg: UniformVampConfig,
    obs_precision_init: Optional[float] = None,
    log_stats: bool = True,
    debug_out: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Scalar uniform exact-full VAMP in rank space (MMSE x1), with optional polish.

    Returns:
      q_int: (B, n) int32
      x_final: (B, n) float32
      final_err: (B,) rank-space surrogate error
      rtn_err: (B,) rank-space surrogate error of RTN init
      stats: dict
    """
    W = W.to(dtype=torch.float32)
    x_rtn = x_rtn.to(dtype=torch.float32)
    scale_full = scale_full.to(dtype=torch.float32)
    zero_full = zero_full.to(dtype=torch.float32)
    maxq = int(maxq)

    iters = max(int(cfg.iters), 0)
    damping = float(cfg.damping)
    prior_tau = float(cfg.prior_tau)
    final_decision = str(cfg.final_decision or "map").strip().lower()
    init_from = str(cfg.init_from or "rtn").strip().lower()

    B, n = W.shape
    rank = int(S.numel())
    atoms = scale_full.unsqueeze(-1) * (
        torch.arange(maxq + 1, dtype=torch.float32, device=W.device).view(1, 1, -1) - zero_full.unsqueeze(-1)
    )

    if init_from == "zero":
        r1 = torch.zeros_like(W)
    elif init_from == "float":
        r1 = W.clone()
    elif init_from == "rtn":
        r1 = x_rtn.clone()
    else:
        raise ValueError(f"unknown init_from={init_from!r} (expected 'rtn', 'float', or 'zero').")

    gamma1 = float(cfg.gamma_start) * (0.25 / _rowwise_quant_mse(W, x_rtn))
    gamma1 = gamma1.to(dtype=torch.float32)
    require_valid_precision("gamma1_init", gamma1)

    Y_u = W.matmul(Vs)
    if obs_precision_init is not None and float(obs_precision_init) > 0.0:
        obs_precision_initial = float(obs_precision_init)
    else:
        obs_precision_initial = _resolve_obs_precision(W=W, x_init=x_rtn, Vs=Vs, requested=float(cfg.obs_precision))
    obs_precision_local = float(obs_precision_initial)
    em_obs_active = bool(cfg.em_obs_precision) and float(cfg.obs_precision) <= 0.0

    r2_prev: Optional[torch.Tensor] = None
    gamma2_prev: Optional[torch.Tensor] = None

    obs_precision_values = []
    alpha1_values = []
    alpha2_values = []
    entropy_values = []
    margin_values = []
    iter_logs = [] if debug_out is not None else None
    alpha1_retry_count = 0
    alpha2_clip_count = 0
    gamma2_invalid_count = 0

    alpha_eps = 1e-7
    gamma_retry_factor = max(float(cfg.gamma_retry_factor), 1.000001)
    gamma_retry_steps = max(int(cfg.gamma_retry_steps), 0)

    for it in range(iters):
        gamma1_work = gamma1
        post = None
        alpha1 = None
        x1 = None

        for retry_idx in range(gamma_retry_steps + 1):
            post_candidate = discrete_uniform_mmse(
                r1=r1,
                gamma1=gamma1_work,
                scale_full=scale_full,
                zero_full=zero_full,
                maxq=maxq,
                prior_center=W,
                prior_tau=prior_tau,
                need_stats=bool(log_stats),
                atoms=atoms,
            )
            alpha1_candidate = post_candidate["alpha1"]  # type: ignore[assignment]
            valid_alpha1 = alpha_valid_mask(alpha1_candidate, eps=alpha_eps)
            if bool(valid_alpha1.all().item()):
                post = post_candidate
                alpha1 = alpha1_candidate
                x1 = post_candidate["mean"]  # type: ignore[assignment]
                break

            if retry_idx >= gamma_retry_steps:
                if not torch.isfinite(alpha1_candidate).all():
                    require_valid_alpha("alpha1", alpha1_candidate, eps=alpha_eps)
                post = post_candidate
                alpha1 = torch.nan_to_num(alpha1_candidate, nan=0.5, posinf=1.0 - alpha_eps, neginf=alpha_eps).clamp(
                    min=alpha_eps, max=1.0 - alpha_eps
                )
                x1 = post_candidate["mean"]  # type: ignore[assignment]
                break

            alpha1_retry_count += int((~valid_alpha1).sum().item())
            gamma1_work = torch.where(alpha1_candidate <= alpha_eps, gamma1_work / gamma_retry_factor, gamma1_work)
            gamma1_work = torch.where(alpha1_candidate >= 1.0 - alpha_eps, gamma1_work * gamma_retry_factor, gamma1_work)
            gamma1_work = gamma1_work.clamp_min_(1e-12)

        assert post is not None and alpha1 is not None and x1 is not None

        eta1 = gamma1_work / alpha1
        gamma2_new = torch.nan_to_num(eta1 - gamma1_work, nan=1e-12, posinf=1e12, neginf=1e-12).clamp_min_(1e-12)
        if not bool((precision_valid_mask(gamma2_new).all().item())):
            gamma2_invalid_count += 1
            continue
        r2_new = (eta1 * x1 - gamma1_work * r1) / gamma2_new

        if r2_prev is None or gamma2_prev is None or damping >= 1.0:
            gamma2 = gamma2_new
            r2 = r2_new
        else:
            gamma2 = log_damp_positive(gamma2_prev, gamma2_new, damping)
            r2 = (1.0 - damping) * r2_prev + damping * r2_new
        if not bool((precision_valid_mask(gamma2).all().item())):
            gamma2_invalid_count += 1
            continue

        x2, alpha2_raw, residual_u = linear_mmse_exact(
            r2,
            gamma2,
            Y_u=Y_u,
            V=V,
            Vh=Vh,
            S=S,
            obs_precision=float(obs_precision_local),
        )
        valid_alpha2 = alpha_valid_mask(alpha2_raw, eps=alpha_eps)
        if not bool(valid_alpha2.all().item()):
            alpha2_clip_count += int((~valid_alpha2).sum().item())
        alpha2 = torch.nan_to_num(alpha2_raw, nan=0.5, posinf=1.0 - alpha_eps, neginf=alpha_eps).clamp(
            min=alpha_eps, max=1.0 - alpha_eps
        )

        if em_obs_active and rank > 0:
            obs_precision_local = em_update_obs_precision_rankspace(
                residual_u=residual_u,
                gamma2=gamma2,
                S=S,
                obs_precision=float(obs_precision_local),
                num_obs=rank,
                damping=float(cfg.em_obs_precision_damping),
                min_value=float(cfg.em_obs_precision_min),
                max_value=float(cfg.em_obs_precision_max),
            )

        eta2 = gamma2 / alpha2
        gamma1_new = torch.nan_to_num(eta2 - gamma2, nan=1e-12, posinf=1e12, neginf=1e-12).clamp_min_(1e-12)
        r1_new = (eta2 * x2 - gamma2 * r2) / gamma1_new
        if damping >= 1.0:
            gamma1 = gamma1_new
            r1 = r1_new
        else:
            gamma1 = log_damp_positive(gamma1, gamma1_new, damping)
            r1 = (1.0 - damping) * r1 + damping * r1_new

        r2_prev = r2
        gamma2_prev = gamma2

        obs_precision_values.append(float(obs_precision_local))
        alpha1_values.append(float(alpha1.mean().item()))
        alpha2_values.append(float(alpha2.mean().item()))
        if log_stats:
            entropy_values.append(float(post["entropy"]))  # type: ignore[arg-type]
            margin_values.append(float(post["margin"]))  # type: ignore[arg-type]
        if iter_logs is not None:
            gap = float(((x1 - x2).norm() / (x2.norm().clamp_min(1e-12))).item())
            iter_logs.append(
                {
                    "iter": int(it),
                    "alpha1": float(alpha1.mean().item()),
                    "alpha2": float(alpha2.mean().item()),
                    "gamma1": float(gamma1.mean().item()),
                    "gamma2": float(gamma2.mean().item()),
                    "obs_precision": float(obs_precision_local),
                    "gap": float(gap),
                }
            )

    final_post = discrete_uniform_mmse(
        r1=r1,
        gamma1=gamma1,
        scale_full=scale_full,
        zero_full=zero_full,
        maxq=maxq,
        prior_center=W,
        prior_tau=prior_tau,
        need_stats=False,
        atoms=atoms,
    )
    q_final, x_final = _uniform_hard_decision(
        x_mean=final_post["mean"],  # type: ignore[arg-type]
        x_map=final_post["map_value"],  # type: ignore[arg-type]
        scale_full=scale_full,
        zero_full=zero_full,
        maxq=maxq,
        final_decision=final_decision,
    )
    err_final = vamp_rankspace_error(Y_u=Y_u, x=x_final, Vs=Vs)

    polish_sweeps = max(int(cfg.polish_sweeps), 0)
    polish_mode = _resolve_polish_mode(sweeps=polish_sweeps, mode=str(getattr(cfg, "polish_mode", "block_cd")))
    topk_k = max(int(getattr(cfg, "topk_k", 8) or 8), 1)
    polish_gain = 0.0
    polish_updates = 0.0
    if polish_mode != "none" and Vs.numel() > 0:
        err_before = vamp_rankspace_error(Y_u=Y_u, x=x_final, Vs=Vs)
        topk = None
        if polish_mode == "topk_block_cd":
            topk = build_uniform_vamp_topk_candidates(
                r1=r1,
                gamma1=gamma1,
                W=W,
                scale_full=scale_full,
                zero_full=zero_full,
                maxq=maxq,
                prior_tau=prior_tau,
                k=topk_k,
                atoms=atoms,
            )
        q_polished, x_polished, err_after, polish_stats = polish_uniform_rankspace(
            x_init=x_final,
            Y_u=Y_u,
            Vs=Vs,
            scale_full=scale_full,
            zero_full=zero_full,
            maxq=maxq,
            sweeps=polish_sweeps,
            polish_mode=polish_mode,
            topk_values=None if topk is None else topk["topk_values"],
            topk_q=None if topk is None else topk["topk_q"],
        )
        improve = err_after + 1e-12 < err_before
        if bool(improve.any().item()):
            polish_gain = float((err_before - err_after).mean().item())
            q_final = torch.where(improve.unsqueeze(-1), q_polished, q_final)
            x_final = torch.where(improve.unsqueeze(-1), x_polished, x_final)
            err_final = torch.where(improve, err_after, err_final)
        polish_updates = float(polish_stats.get("accepted_updates", 0.0))

    if debug_out is not None:
        debug_out["solver"] = "uniform_exact_vamp"
        debug_out["q_pre_polish"] = q_final.detach().to(device="cpu")
        debug_out["r1"] = r1.detach().to(device="cpu")
        debug_out["gamma1"] = gamma1.detach().to(device="cpu")
        debug_out["W"] = W.detach().to(device="cpu")

    rtn_err = vamp_rankspace_error(Y_u=Y_u, x=x_rtn, Vs=Vs)

    stats: Dict[str, Any] = {
        "solver": "uniform_exact_vamp",
        "iters": int(iters),
        "rank": int(rank),
        "init_from": str(init_from),
        "damping": float(damping),
        "prior_tau": float(prior_tau),
        "gamma_start": float(cfg.gamma_start),
        "gamma_retry_factor": float(gamma_retry_factor),
        "gamma_retry_steps": int(gamma_retry_steps),
        "obs_precision_initial": float(obs_precision_initial),
        "obs_precision_final": float(obs_precision_local),
        "obs_precision_mean": float(sum(obs_precision_values) / len(obs_precision_values)) if obs_precision_values else float(obs_precision_local),
        "alpha1_retry_count": int(alpha1_retry_count),
        "alpha2_clip_count": int(alpha2_clip_count),
        "gamma2_invalid_count": int(gamma2_invalid_count),
        "alpha1_mean": float(sum(alpha1_values) / len(alpha1_values)) if alpha1_values else 0.0,
        "alpha2_mean": float(sum(alpha2_values) / len(alpha2_values)) if alpha2_values else 0.0,
        "mean_entropy": float(sum(entropy_values) / len(entropy_values)) if entropy_values else 0.0,
        "mean_margin": float(sum(margin_values) / len(margin_values)) if margin_values else 0.0,
        "final_decision": str(final_decision),
        "polish_sweeps": int(polish_sweeps),
        "polish_mode": str(polish_mode),
        "topk_k": int(topk_k),
        "polish_gain": float(polish_gain),
        "polish_updates": float(polish_updates),
        "rankspace_error_mean": float(err_final.mean().item()) if err_final.numel() else 0.0,
        "rtn_rankspace_error_mean": float(rtn_err.mean().item()) if rtn_err.numel() else 0.0,
    }
    return q_final.to(dtype=torch.int32), x_final, err_final, rtn_err, stats


@torch.inference_mode()
def _uniform_hard_decision(
    *,
    x_mean: torch.Tensor,
    x_map: torch.Tensor,
    scale_full: torch.Tensor,
    zero_full: torch.Tensor,
    maxq: int,
    final_decision: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    mode = str(final_decision or "map").strip().lower()
    if mode == "map":
        q_int = nearest_atom_idx(x=x_map, scale_full=scale_full, zero_full=zero_full, maxq=maxq).to(dtype=torch.int32)
        return q_int, dequant_from_q(q_int=q_int, scale_full=scale_full, zero_full=zero_full)
    if mode == "round_mean":
        q_int = nearest_atom_idx(x=x_mean, scale_full=scale_full, zero_full=zero_full, maxq=maxq).to(dtype=torch.int32)
        return q_int, dequant_from_q(q_int=q_int, scale_full=scale_full, zero_full=zero_full)
    raise ValueError(f"unsupported final_decision={final_decision!r}")


@torch.inference_mode()
def solve_e8p12_rankspace_vamp(
    *,
    Wr: torch.Tensor,  # (B, n)
    hatWr_init: torch.Tensor,  # (B, n)
    cb: object,
    V: torch.Tensor,  # (n, R)
    Vh: torch.Tensor,  # (R, n)
    S: torch.Tensor,  # (R,)
    Vs: torch.Tensor,  # (n, R)
    cfg: E8P12VampConfig,
    module_name: str = "",
    obs_precision_init: Optional[float] = None,
    log_stats: bool = True,
    debug_out: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Scalar-precision VAMP for QUIP E8P12 in rank space, with optional rank-space polish."""
    Wr = Wr.to(dtype=torch.float32)
    hatWr_init = hatWr_init.to(dtype=torch.float32)

    B, n = Wr.shape
    codesz = int(getattr(cb, "codesz", 8))
    if codesz != 8 or (n % codesz) != 0:
        raise ValueError(f"E8P12 solver expects n divisible by 8; got n={n}, codesz={codesz}.")
    nblocks = n // codesz
    rank = int(S.numel())

    iters = max(int(cfg.iters), 0)
    reported_iters = int(iters)
    damping = float(cfg.damping)
    prior_tau = float(cfg.prior_tau)
    prior_tau_scale_mode = str(cfg.prior_tau_scale_mode or "absmax").strip().lower()
    compile_prior = bool(getattr(cfg, "compile_prior", True))
    blocks_per_chunk = max(int(cfg.blocks_per_chunk), 1)

    W_blocks = Wr.reshape(B, nblocks, codesz)
    prior_scale_blocks = None
    if prior_tau > 0.0:
        prior_scale_blocks = prior_scale_from_center(W_blocks, prior_tau_scale_mode)

    Y_u = Wr.matmul(Vs)
    if float(cfg.obs_precision) > 0.0:
        obs_precision_start = float(cfg.obs_precision)
    elif obs_precision_init is not None and float(obs_precision_init) > 0.0:
        obs_precision_start = float(obs_precision_init)
    else:
        obs_precision_start = _resolve_obs_precision(W=Wr, x_init=hatWr_init, Vs=Vs, requested=float(cfg.obs_precision))
    obs_precision_local = float(obs_precision_start)
    em_obs_active = bool(cfg.em_obs_precision) and float(cfg.obs_precision) <= 0.0

    r1 = hatWr_init.clone()
    gamma1 = float(cfg.gamma_start) * (0.25 / _rowwise_quant_mse(Wr, hatWr_init))
    gamma1 = gamma1.to(dtype=torch.float32)
    require_valid_precision("gamma1_init", gamma1)

    r2_prev: Optional[torch.Tensor] = None
    gamma2_prev: Optional[torch.Tensor] = None

    obs_precision_values = []
    alpha1_values = []
    alpha2_values = []
    entropy_values = []
    margin_values = []
    iter_logs = [] if debug_out is not None else None
    alpha1_retry_count = 0
    alpha2_clip_count = 0
    gamma2_invalid_count = 0

    alpha_eps = 1e-7
    gamma_retry_factor = max(float(cfg.gamma_retry_factor), 1.000001)
    gamma_retry_steps = max(int(cfg.gamma_retry_steps), 0)

    for it in range(iters):
        gamma1_work = gamma1
        post = None
        alpha1 = None
        x1 = None

        for retry_idx in range(gamma_retry_steps + 1):
            post_candidate = vamp_vector_discrete_mmse_e8p12(
                r1_blocks=r1.reshape(B, nblocks, codesz),
                gamma1=gamma1_work,
                cb=cb,
                blocks_per_chunk=blocks_per_chunk,
                need_stats=bool(log_stats),
                need_var_diag=False,
                prior_center_blocks=W_blocks if prior_tau > 0.0 else None,
                prior_scale_blocks=prior_scale_blocks,
                prior_tau=prior_tau,
                prior_tau_scale_mode=prior_tau_scale_mode,
                compile_prior=compile_prior,
            )
            alpha1_candidate = post_candidate["alpha1"]  # type: ignore[assignment]
            valid_alpha1 = alpha_valid_mask(alpha1_candidate, eps=alpha_eps)
            if bool(valid_alpha1.all().item()):
                post = post_candidate
                alpha1 = alpha1_candidate
                x1 = post_candidate["x1_blocks"].reshape_as(r1)
                break

            if retry_idx >= gamma_retry_steps:
                if not torch.isfinite(alpha1_candidate).all():
                    require_valid_alpha("alpha1", alpha1_candidate, eps=alpha_eps)
                post = post_candidate
                alpha1 = torch.nan_to_num(alpha1_candidate, nan=0.5, posinf=1.0 - alpha_eps, neginf=alpha_eps).clamp(
                    min=alpha_eps, max=1.0 - alpha_eps
                )
                x1 = post_candidate["x1_blocks"].reshape_as(r1)
                break

            alpha1_retry_count += int((~valid_alpha1).sum().item())
            gamma1_work = torch.where(alpha1_candidate <= alpha_eps, gamma1_work / gamma_retry_factor, gamma1_work)
            gamma1_work = torch.where(alpha1_candidate >= 1.0 - alpha_eps, gamma1_work * gamma_retry_factor, gamma1_work)
            gamma1_work = gamma1_work.clamp_min_(1e-12)

        assert post is not None and alpha1 is not None and x1 is not None

        eta1 = gamma1_work / alpha1
        gamma2_new = torch.nan_to_num(eta1 - gamma1_work, nan=1e-12, posinf=1e12, neginf=1e-12).clamp_min_(1e-12)
        if not bool((precision_valid_mask(gamma2_new).all().item())):
            gamma2_invalid_count += 1
            continue
        r2_new = (eta1 * x1 - gamma1_work * r1) / gamma2_new

        if r2_prev is None or gamma2_prev is None or damping >= 1.0:
            gamma2 = gamma2_new
            r2 = r2_new
        else:
            gamma2 = log_damp_positive(gamma2_prev, gamma2_new, damping)
            r2 = (1.0 - damping) * r2_prev + damping * r2_new
        if not bool((precision_valid_mask(gamma2).all().item())):
            gamma2_invalid_count += 1
            continue

        x2, alpha2_raw, residual_u = linear_mmse_exact(
            r2,
            gamma2,
            Y_u=Y_u,
            V=V,
            Vh=Vh,
            S=S,
            obs_precision=float(obs_precision_local),
        )
        valid_alpha2 = alpha_valid_mask(alpha2_raw, eps=alpha_eps)
        if not bool(valid_alpha2.all().item()):
            alpha2_clip_count += int((~valid_alpha2).sum().item())
        alpha2 = torch.nan_to_num(alpha2_raw, nan=0.5, posinf=1.0 - alpha_eps, neginf=alpha_eps).clamp(
            min=alpha_eps, max=1.0 - alpha_eps
        )

        if em_obs_active and rank > 0:
            obs_precision_local = em_update_obs_precision_rankspace(
                residual_u=residual_u,
                gamma2=gamma2,
                S=S,
                obs_precision=float(obs_precision_local),
                num_obs=rank,
                damping=float(cfg.em_obs_precision_damping),
                min_value=float(cfg.em_obs_precision_min),
                max_value=float(cfg.em_obs_precision_max),
            )

        eta2 = gamma2 / alpha2
        gamma1_new = torch.nan_to_num(eta2 - gamma2, nan=1e-12, posinf=1e12, neginf=1e-12).clamp_min_(1e-12)
        r1_new = (eta2 * x2 - gamma2 * r2) / gamma1_new
        if damping >= 1.0:
            gamma1 = gamma1_new
            r1 = r1_new
        else:
            gamma1 = log_damp_positive(gamma1, gamma1_new, damping)
            r1 = (1.0 - damping) * r1 + damping * r1_new

        r2_prev = r2
        gamma2_prev = gamma2

        obs_precision_values.append(float(obs_precision_local))
        alpha1_values.append(float(alpha1.mean().item()))
        alpha2_values.append(float(alpha2.mean().item()))
        if log_stats:
            entropy_values.append(float(post["entropy"]))  # type: ignore[arg-type]
            margin_values.append(float(post["margin"]))  # type: ignore[arg-type]
        if iter_logs is not None:
            gap = float(((x1 - x2).norm() / (x2.norm().clamp_min(1e-12))).item())
            iter_logs.append(
                {
                    "iter": int(it),
                    "alpha1": float(alpha1.mean().item()),
                    "alpha2": float(alpha2.mean().item()),
                    "gamma1": float(gamma1.mean().item()),
                    "gamma2": float(gamma2.mean().item()),
                    "obs_precision": float(obs_precision_local),
                    "gap": float(gap),
                }
            )

    final_post = vamp_vector_map_e8p12(
        r1_blocks=r1.reshape(B, nblocks, codesz),
        gamma1=gamma1,
        cb=cb,
        blocks_per_chunk=blocks_per_chunk,
        need_stats=False,
        prior_center_blocks=W_blocks if prior_tau > 0.0 else None,
        prior_scale_blocks=prior_scale_blocks,
        prior_tau=prior_tau,
        prior_tau_scale_mode=prior_tau_scale_mode,
    )
    final_hatWr = final_post["map_vals"].reshape_as(r1)
    final_Qidxs = final_post["map_idxs"].to(dtype=getattr(cb, "idx_dtype", torch.int64))
    final_err = vamp_rankspace_error(Y_u=Y_u, x=final_hatWr, Vs=Vs)

    if debug_out is not None:
        dbg_stats = _e8p12_block_debug_stats(
            r1=r1,
            gamma1=gamma1,
            cb=cb,
            blocks_per_chunk=blocks_per_chunk,
            prior_center_blocks=W_blocks if prior_tau > 0.0 else None,
            prior_scale_blocks=prior_scale_blocks,
            prior_tau=prior_tau,
            prior_tau_scale_mode=prior_tau_scale_mode,
        )
        debug_out["solver"] = "e8p12_exact_vamp"
        debug_out["idxs_pre_polish"] = final_Qidxs.detach().to(device="cpu")
        debug_out["r1"] = r1.detach().to(device="cpu")
        debug_out["gamma1"] = gamma1.detach().to(device="cpu")
        debug_out["W"] = Wr.detach().to(device="cpu")
        debug_out["block_stats"] = {k: v.detach().to(device="cpu") for k, v in dbg_stats.items()}
        debug_out["iter_logs"] = iter_logs if iter_logs is not None else []

    polish_sweeps = max(int(cfg.polish_sweeps), 0)
    polish_mode = _resolve_polish_mode(sweeps=polish_sweeps, mode=str(getattr(cfg, "polish_mode", "block_cd")))
    polish_metric = _resolve_e8p12_polish_metric(str(cfg.polish_metric or "gram"))
    topk_k = max(int(getattr(cfg, "topk_k", 8) or 8), 1)

    polish_gain = 0.0
    polish_updates = 0.0
    if polish_mode != "none" and Vs.numel() > 0:
        err_before = vamp_rankspace_error(Y_u=Y_u, x=final_hatWr, Vs=Vs)
        topk = None
        if polish_mode == "topk_block_cd":
            topk = build_e8p12_vamp_topk_candidates(
                r1=r1,
                gamma1=gamma1,
                Wr=Wr,
                cb=cb,
                blocks_per_chunk=blocks_per_chunk,
                prior_tau=prior_tau,
                prior_tau_scale_mode=prior_tau_scale_mode,
                k=topk_k,
            )
        final_hatWr, final_Qidxs, err_after, polish_stats = polish_e8p12_rankspace(
            x_init=final_hatWr,
            idxs_init=final_Qidxs,
            Y_u=Y_u,
            Vs=Vs,
            cb=cb,
            sweeps=polish_sweeps,
            polish_mode=polish_mode,
            polish_metric=polish_metric,
            metric_rows_per_chunk=int(cfg.polish_metric_rows_per_chunk),
            polish_max_blocks=int(getattr(cfg, "polish_max_blocks", 0) or 0),
            topk_idxs=None if topk is None else topk["topk_idxs"].to(device=final_hatWr.device, dtype=torch.long),
        )
        polish_gain = float((err_before - err_after).mean().item())
        polish_updates = float(polish_stats.get("accepted_updates", 0.0))
        final_err = err_after
        if debug_out is not None and topk is not None:
            debug_out["topk_idxs"] = topk["topk_idxs"].detach().to(device="cpu")

    if debug_out is not None:
        idxs_pre = debug_out.get("idxs_pre_polish")
        idxs_post = final_Qidxs.detach().to(device="cpu")
        debug_out["idxs_post_polish"] = idxs_post
        if isinstance(idxs_pre, torch.Tensor) and idxs_pre.shape == idxs_post.shape:
            debug_out["changed_mask"] = (idxs_pre != idxs_post)

    stats: Dict[str, Any] = {
        "solver": "e8p12_exact_vamp",
        "module": str(module_name),
        "iters": int(reported_iters),
        "rank": int(rank),
        "nblocks": int(nblocks),
        "damping": float(damping),
        "prior_tau": float(prior_tau),
        "prior_tau_scale_mode": str(prior_tau_scale_mode),
        "compile_prior": bool(compile_prior),
        "gamma_start": float(cfg.gamma_start),
        "gamma_retry_factor": float(gamma_retry_factor),
        "gamma_retry_steps": int(gamma_retry_steps),
        "obs_precision_initial": float(obs_precision_start),
        "obs_precision_final": float(obs_precision_local),
        "obs_precision_mean": float(sum(obs_precision_values) / len(obs_precision_values)) if obs_precision_values else float(obs_precision_local),
        "alpha1_retry_count": int(alpha1_retry_count),
        "alpha2_clip_count": int(alpha2_clip_count),
        "gamma2_invalid_count": int(gamma2_invalid_count),
        "alpha1_mean": float(sum(alpha1_values) / len(alpha1_values)) if alpha1_values else 0.0,
        "alpha2_mean": float(sum(alpha2_values) / len(alpha2_values)) if alpha2_values else 0.0,
        "mean_entropy": float(sum(entropy_values) / len(entropy_values)) if entropy_values else 0.0,
        "mean_margin": float(sum(margin_values) / len(margin_values)) if margin_values else 0.0,
        "blocks_per_chunk": int(blocks_per_chunk),
        "polish_sweeps": int(polish_sweeps),
        "polish_mode": str(polish_mode),
        "polish_metric": str(polish_metric),
        "topk_k": int(topk_k),
        "polish_gain": float(polish_gain),
        "polish_updates": float(polish_updates),
        "rankspace_error_mean": float(final_err.mean().item()) if final_err.numel() else 0.0,
    }
    return final_hatWr, final_Qidxs, final_err, stats
