from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

from mpgq.priors.e8p12 import vamp_vector_map_e8p12
from mpgq.utils import log_damp_positive

from .denoisers.e8p_exact import rs_posterior_blocks
from .linear_module import SpectralLinearModule
from .utils.numerics import clip_alpha, clip_gamma


@dataclass(frozen=True)
class RsVampConfig:
    max_iters: int = 16
    damping: float = 0.5
    gamma_start: float = 1.0
    gamma_retry_factor: float = 4.0
    gamma_retry_steps: int = 3
    alpha_eps: float = 1e-6
    gamma_min: float = 1e-8
    gamma_max: float = 1e12
    blocks_per_chunk: int = 64
    tol: float = 1e-4
    prior_tau: float = 0.0
    prior_tau_scale_mode: str = "absmax"


@torch.inference_mode()
def _rowwise_mse(w: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return (w - x).square().mean(dim=1, keepdim=True).clamp_min_(1e-12)


@torch.inference_mode()
def run_rs_vamp_e8p12_rankspace(
    *,
    w: torch.Tensor,  # (B, n)
    r1_init: torch.Tensor,  # (B, n)
    cb: Any,
    V: torch.Tensor,  # (n, R)
    Vh: torch.Tensor,  # (R, n)
    S: torch.Tensor,  # (R,)
    lambda_eff: float,
    cfg: RsVampConfig,
    backend: str = "factorized",
    log_per_iter: bool = False,
) -> Dict[str, Any]:
    """RS-VAMP baseline (scalar precisions) for QUIP-style rank-space Hessians."""
    w = w.to(dtype=torch.float32)
    r1 = r1_init.to(dtype=torch.float32).clone()

    B, n = w.shape
    codesz = int(getattr(cb, "codesz", 8))
    if codesz != 8 or (n % codesz) != 0:
        raise ValueError(f"expected n divisible by 8; got n={n}, codesz={codesz}")
    nblocks = n // codesz

    W_blocks = w.reshape(B, nblocks, codesz)
    prior_tau = float(cfg.prior_tau)
    prior_tau_scale_mode = str(cfg.prior_tau_scale_mode or "absmax").strip().lower()
    prior_scale_blocks = None
    if prior_tau > 0.0:
        from mpgq.priors.e8p12 import prior_scale_from_center

        prior_scale_blocks = prior_scale_from_center(W_blocks, prior_tau_scale_mode)

    linear = SpectralLinearModule(V=V, Vh=Vh, S=S, w=w)
    trace_avg = float(linear.stats.trace_avg)
    lam = float(lambda_eff)
    if not (math.isfinite(lam) and lam > 0.0):
        raise ValueError(f"invalid lambda_eff={lambda_eff}")

    damping = float(cfg.damping)
    alpha_eps = float(cfg.alpha_eps)
    gamma_retry_factor = max(float(cfg.gamma_retry_factor), 1.000001)
    gamma_retry_steps = max(int(cfg.gamma_retry_steps), 0)

    # gamma1 initialization matched to existing MPGQ VAMP scaling.
    gamma1 = float(cfg.gamma_start) * (0.25 / _rowwise_mse(w, r1))
    gamma1 = clip_gamma(gamma1.to(dtype=torch.float32), min_value=float(cfg.gamma_min), max_value=float(cfg.gamma_max))

    r2_prev: Optional[torch.Tensor] = None
    gamma2_prev: Optional[torch.Tensor] = None

    logs: List[Dict[str, float]] = []
    x1 = r1
    x2 = r1
    alpha1 = torch.full((B, 1), 0.5, device=w.device, dtype=torch.float32)
    alpha2 = torch.full((B, 1), 0.5, device=w.device, dtype=torch.float32)

    for it in range(max(int(cfg.max_iters), 0)):
        gamma1_work = gamma1
        post = None
        cov_trace = None
        for retry_idx in range(gamma_retry_steps + 1):
            post = rs_posterior_blocks(
                center_blocks=r1.reshape(B, nblocks, codesz),
                gamma=gamma1_work,
                cb=cb,
                blocks_per_chunk=int(cfg.blocks_per_chunk),
                prior_center_blocks=W_blocks if prior_tau > 0.0 else None,
                prior_scale_blocks=prior_scale_blocks,
                prior_tau=prior_tau,
                prior_tau_scale_mode=prior_tau_scale_mode,
                return_var_diag=False,
                return_logZ=False,
                return_top2=False,
                backend=str(backend),
            )
            x1 = post["mean"].reshape(B, n)
            cov_trace = post["cov_trace"]  # (B, nblocks)
            var1_mean = (cov_trace / float(codesz)).mean(dim=1, keepdim=True)  # (B,1)
            alpha1_candidate = gamma1_work * var1_mean
            valid = torch.isfinite(alpha1_candidate) & (alpha1_candidate > alpha_eps) & (alpha1_candidate < 1.0 - alpha_eps)
            if bool(valid.all().item()):
                alpha1 = alpha1_candidate
                break
            if retry_idx >= gamma_retry_steps:
                alpha1 = clip_alpha(alpha1_candidate, eps=alpha_eps)
                break

            # Retry gamma to push alpha1 back into (0,1).
            gamma1_work = torch.where(alpha1_candidate <= alpha_eps, gamma1_work / gamma_retry_factor, gamma1_work)
            gamma1_work = torch.where(alpha1_candidate >= 1.0 - alpha_eps, gamma1_work * gamma_retry_factor, gamma1_work)
            gamma1_work = clip_gamma(gamma1_work, min_value=float(cfg.gamma_min), max_value=float(cfg.gamma_max))

        assert post is not None and cov_trace is not None

        eta1 = gamma1_work / alpha1
        gamma2_new = clip_gamma(eta1 - gamma1_work, min_value=float(cfg.gamma_min), max_value=float(cfg.gamma_max))
        r2_new = (eta1 * x1 - gamma1_work * r1) / gamma2_new

        if r2_prev is None or gamma2_prev is None or damping >= 1.0:
            r2 = r2_new
            gamma2 = gamma2_new
        else:
            r2 = (1.0 - damping) * r2_prev + damping * r2_new
            gamma2 = log_damp_positive(gamma2_prev, gamma2_new, damping)

        x2, alpha2_raw, _residual_u = linear.apply(r2=r2, gamma2=gamma2, lambda_eff=lam)
        alpha2 = clip_alpha(alpha2_raw, eps=alpha_eps)

        eta2 = gamma2 / alpha2
        gamma1_new = clip_gamma(eta2 - gamma2, min_value=float(cfg.gamma_min), max_value=float(cfg.gamma_max))
        r1_new = (eta2 * x2 - gamma2 * r2) / gamma1_new

        if damping >= 1.0:
            r1 = r1_new
            gamma1 = gamma1_new
        else:
            r1 = (1.0 - damping) * r1 + damping * r1_new
            gamma1 = log_damp_positive(gamma1, gamma1_new, damping)

        r2_prev = r2
        gamma2_prev = gamma2

        if log_per_iter:
            gap = float(((x1 - x2).norm() / (x2.norm().clamp_min(1e-12))).item())
            logs.append(
                {
                    "iter": float(it),
                    "lambda_eff": float(lam),
                    "trace_avg": float(trace_avg),
                    "alpha1": float(alpha1.mean().item()),
                    "alpha2": float(alpha2.mean().item()),
                    "gamma1": float(gamma1.mean().item()),
                    "gamma2": float(gamma2.mean().item()),
                    "gap": float(gap),
                }
            )
            if gap <= float(cfg.tol):
                break

    final = vamp_vector_map_e8p12(
        r1_blocks=r1.reshape(B, nblocks, codesz),
        gamma1=gamma1,
        cb=cb,
        blocks_per_chunk=int(cfg.blocks_per_chunk),
        need_stats=False,
        prior_center_blocks=W_blocks if prior_tau > 0.0 else None,
        prior_scale_blocks=prior_scale_blocks,
        prior_tau=prior_tau,
        prior_tau_scale_mode=prior_tau_scale_mode,
    )
    q_hard = final["map_vals"].reshape(B, n).to(dtype=torch.float32)
    q_idxs = final["map_idxs"].to(dtype=getattr(cb, "idx_dtype", torch.int64))

    return {
        "q_soft": x1.to(dtype=torch.float32),
        "q_hard": q_hard,
        "q_hard_idxs": q_idxs,
        "state": {
            "r1": r1,
            "gamma1": gamma1,
            "alpha1": alpha1,
            "alpha2": alpha2,
        },
        "logs": logs,
        "linear_stats": {
            "trace_avg": float(trace_avg),
            "rank": int(linear.stats.rank),
        },
    }
