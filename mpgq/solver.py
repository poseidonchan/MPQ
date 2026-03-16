from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch

from .linear import em_update_obs_precision_rankspace, linear_mmse_exact, vamp_rankspace_error
from .polish.block_cd import BlockCDConfig, E8P12Provider, UniformScalarProvider, block_coordinate_descent_rankspace
from .priors.e8p12 import prior_scale_from_center, vamp_vector_discrete_mmse_e8p12, vamp_vector_map_e8p12
from .priors.uniform import dequant_from_q, discrete_uniform_mmse, nearest_atom_idx
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
    obs_precision: float = 0.0
    em_obs_precision: bool = True
    em_obs_precision_damping: float = 0.3
    em_obs_precision_min: float = 1e-4
    em_obs_precision_max: float = 1e8
    blocks_per_chunk: int = 16
    polish_sweeps: int = 3
    polish_metric: str = "gram"  # "gram" | "diag"
    polish_metric_rows_per_chunk: int = 0
    polish_max_blocks: int = 0


@dataclass(frozen=True)
class E8P12LrsVaspConfig:
    """LRS-VASP (RS-VAMP + low-rank survey) config for QUIP E8P12.

    The solver internally uses the trace-normalized Hessian Hbar, so `lambda_eff`
    is interpreted against Hbar rather than the raw H. By default the provided
    `lambda_eff_grid` is treated as *relative* values and scaled by an automatic
    base precision derived from the init mismatch under Hbar.
    """

    method: str = "lrs-vasp"  # "rs-vamp" | "lrs-vasp"
    iters: int = 16
    rs_warmup_iters: int = 4
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

    lambda_eff_grid: tuple[float, ...] = (0.25, 1.0, 4.0)
    lambda_eff_scale_mode: str = "auto"  # "auto" | "none"

    # Survey settings (hard blocks only).
    enable_survey: bool = True
    survey_L: float = 2.0
    survey_Q: int = 9
    survey_v1_init: float = 0.0  # <=0 means "auto"
    survey_v1_min: float = 1e-6

    # Hard-block gating (top2 gap + variance).
    hard_gap_thresh: float = 0.5
    hard_var_mult: float = 2.0
    hard_max_frac: float = 0.25

    backend: str = "factorized"  # "factorized" | "bruteforce"

    # Optional polish (same rank-space block CD as scalar VAMP).
    polish_sweeps: int = 0
    polish_metric: str = "gram"  # "gram" | "diag"
    polish_metric_rows_per_chunk: int = 0
    polish_max_blocks: int = 0


@torch.inference_mode()
def _rowwise_quant_mse(W: torch.Tensor, x_quant: torch.Tensor) -> torch.Tensor:
    return (W - x_quant).square().mean(dim=1, keepdim=True).clamp_min_(1e-12)


@torch.inference_mode()
def _global_quant_mse(W: torch.Tensor, x_quant: torch.Tensor) -> float:
    return float((W - x_quant).square().mean().clamp_min_(1e-12).item())


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
def resolve_e8p12_lrs_lambda_eff_grid(
    *,
    Wr: torch.Tensor,  # (M,n)
    hatWr_init: torch.Tensor,  # (M,n)
    cb: object,
    S: torch.Tensor,  # (n,)
    Vs: torch.Tensor,  # (n,n)
    cfg: E8P12LrsVaspConfig,
    sample_rows: int = 32,
) -> tuple[tuple[float, ...], Dict[str, float]]:
    """Resolve an absolute `lambda_eff_grid` for LRS solvers.

    When `cfg.lambda_eff_scale_mode == "auto"`, the user-provided grid is treated
    as *relative* values and scaled by an estimated base precision derived from
    the init mismatch under Hbar and the RS denoiser divergence at init.

    This helper is intended to be called once per module (before row chunking)
    so that all chunks use a consistent lambda scale.
    """
    scale_mode = str(cfg.lambda_eff_scale_mode or "auto").strip().lower()
    raw_grid = tuple(float(x) for x in (cfg.lambda_eff_grid or (1.0,)))
    if not raw_grid:
        raw_grid = (1.0,)
    if scale_mode == "none":
        return raw_grid, {}
    if scale_mode != "auto":
        raise ValueError(f"unknown lambda_eff_scale_mode={scale_mode!r} (expected 'auto'|'none')")

    Wr = Wr.to(dtype=torch.float32)
    hatWr_init = hatWr_init.to(dtype=torch.float32)
    M, n = Wr.shape
    codesz = int(getattr(cb, "codesz", 8))
    if codesz != 8 or (n % codesz) != 0:
        raise ValueError(f"E8P12 solver expects n divisible by 8; got n={n}, codesz={codesz}.")
    nblocks = n // codesz

    # Trace-normalize Hbar: Vsbar = Vs / sqrt(tr(H)/n).
    eigs = S.to(dtype=torch.float32).square()
    trace_avg = float(eigs.mean().clamp_min_(1e-30).item()) if eigs.numel() else 1.0
    Vsbar = Vs.to(dtype=torch.float32) / float(trace_avg) ** 0.5

    # Estimate alpha1 at init on a small deterministic sample of rows.
    sample_rows = min(int(sample_rows), int(M))
    if sample_rows <= 0:
        sample_rows = int(M)
    rb = slice(0, sample_rows)
    w_s = Wr[rb]
    r1_s = hatWr_init[rb]

    # Use the same deterministic sample for the init mismatch estimate. This
    # avoids over-penalizing outlier rows when selecting a single module-wide
    # lambda scale.
    diff_u = (w_s - r1_s).matmul(Vsbar)
    mse_u = float(diff_u.square().mean().clamp_min_(1e-12).item())
    lambda_init = float(1.0 / mse_u)

    gamma1_s = float(cfg.gamma_start) * (0.25 / _rowwise_quant_mse(w_s, r1_s))

    prior_tau = float(cfg.prior_tau)
    prior_tau_scale_mode = str(cfg.prior_tau_scale_mode or "absmax").strip().lower()
    prior_center_blocks = w_s.reshape(sample_rows, nblocks, codesz) if prior_tau > 0.0 else None
    prior_scale_blocks = None
    if prior_tau > 0.0 and prior_center_blocks is not None:
        prior_scale_blocks = prior_scale_from_center(prior_center_blocks, prior_tau_scale_mode)

    post = vamp_vector_discrete_mmse_e8p12(
        r1_blocks=r1_s.reshape(sample_rows, nblocks, codesz),
        gamma1=gamma1_s,
        cb=cb,
        blocks_per_chunk=int(cfg.blocks_per_chunk),
        need_stats=False,
        need_var_diag=False,
        prior_center_blocks=prior_center_blocks,
        prior_scale_blocks=prior_scale_blocks,
        prior_tau=prior_tau,
        prior_tau_scale_mode=prior_tau_scale_mode,
    )
    alpha1_s = post["alpha1"]  # type: ignore[assignment]
    alpha_eps = float(cfg.alpha_eps)
    alpha1_mean = float(torch.nan_to_num(alpha1_s, nan=0.5).clamp(min=alpha_eps, max=1.0 - alpha_eps).mean().item())

    lambda_base = float(lambda_init / max(alpha1_mean, alpha_eps))
    abs_grid = tuple(float(lambda_base) * float(x) for x in raw_grid)
    stats = {
        "trace_avg": float(trace_avg),
        "init_mse_u": float(mse_u),
        "lambda_init": float(lambda_init),
        "alpha1_init_mean": float(alpha1_mean),
        "lambda_eff_base": float(lambda_base),
    }
    return abs_grid, stats


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
    alpha1_retry_count = 0
    alpha2_clip_count = 0
    gamma2_invalid_count = 0

    alpha_eps = 1e-7
    gamma_retry_factor = max(float(cfg.gamma_retry_factor), 1.000001)
    gamma_retry_steps = max(int(cfg.gamma_retry_steps), 0)

    for _ in range(iters):
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
    polish_gain = 0.0
    polish_updates = 0.0
    if polish_sweeps > 0 and Vs.numel() > 0:
        err_before = vamp_rankspace_error(Y_u=Y_u, x=x_final, Vs=Vs)
        x_polished, _, polish_stats = block_coordinate_descent_rankspace(
            x=x_final,
            Y_u=Y_u,
            Vs=Vs,
            block_size=1,
            provider=UniformScalarProvider(scale=scale_full, zero=zero_full, maxq=maxq),
            cfg=BlockCDConfig(sweeps=polish_sweeps, metric="diag", accept_tol=0.0, order_mode="greedy_gain"),
        )
        q_polished = nearest_atom_idx(x=x_polished, scale_full=scale_full, zero_full=zero_full, maxq=maxq).to(dtype=torch.int32)
        x_polished = dequant_from_q(q_int=q_polished, scale_full=scale_full, zero_full=zero_full)
        err_after = vamp_rankspace_error(Y_u=Y_u, x=x_polished, Vs=Vs)
        improve = err_after + 1e-12 < err_before
        if bool(improve.any().item()):
            polish_gain = float((err_before - err_after).mean().item())
            q_final = torch.where(improve.unsqueeze(-1), q_polished, q_final)
            x_final = torch.where(improve.unsqueeze(-1), x_polished, x_final)
            err_final = torch.where(improve, err_after, err_final)
        polish_updates = float(polish_stats.get("accepted_updates", 0.0))

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
    alpha1_retry_count = 0
    alpha2_clip_count = 0
    gamma2_invalid_count = 0

    alpha_eps = 1e-7
    gamma_retry_factor = max(float(cfg.gamma_retry_factor), 1.000001)
    gamma_retry_steps = max(int(cfg.gamma_retry_steps), 0)

    for _ in range(iters):
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

    polish_sweeps = max(int(cfg.polish_sweeps), 0)
    polish_metric = str(cfg.polish_metric or "gram").strip().lower()
    if polish_metric in ("isotropic", "diag"):
        polish_metric = "diag"
    elif polish_metric != "gram":
        raise ValueError(f"unsupported polish_metric={polish_metric!r}; expected 'gram'|'diag'.")

    polish_gain = 0.0
    polish_updates = 0.0
    if polish_sweeps > 0 and Vs.numel() > 0:
        err_before = vamp_rankspace_error(Y_u=Y_u, x=final_hatWr, Vs=Vs)
        final_hatWr, final_Qidxs, polish_stats = block_coordinate_descent_rankspace(
            x=final_hatWr,
            Y_u=Y_u,
            Vs=Vs,
            block_size=codesz,
            provider=E8P12Provider(cb=cb, metric_rows_per_chunk=int(cfg.polish_metric_rows_per_chunk)),
            cfg=BlockCDConfig(
                sweeps=polish_sweeps,
                metric="gram" if polish_metric == "gram" else "diag",
                accept_tol=0.0,
                metric_rows_per_chunk=int(cfg.polish_metric_rows_per_chunk),
                max_blocks_per_sweep=int(getattr(cfg, "polish_max_blocks", 0) or 0),
                order_mode="greedy_gain",
            ),
            idxs_init=final_Qidxs,
        )
        err_after = vamp_rankspace_error(Y_u=Y_u, x=final_hatWr, Vs=Vs)
        polish_gain = float((err_before - err_after).mean().item())
        polish_updates = float(polish_stats.get("accepted_updates", 0.0))
        final_err = err_after

    stats: Dict[str, Any] = {
        "solver": "e8p12_exact_vamp",
        "module": str(module_name),
        "iters": int(reported_iters),
        "rank": int(rank),
        "nblocks": int(nblocks),
        "damping": float(damping),
        "prior_tau": float(prior_tau),
        "prior_tau_scale_mode": str(prior_tau_scale_mode),
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
        "polish_metric": str(polish_metric),
        "polish_gain": float(polish_gain),
        "polish_updates": float(polish_updates),
        "rankspace_error_mean": float(final_err.mean().item()) if final_err.numel() else 0.0,
    }
    return final_hatWr, final_Qidxs, final_err, stats


@torch.inference_mode()
def solve_e8p12_rankspace_lrs_vasp(
    *,
    Wr: torch.Tensor,  # (B, n)
    hatWr_init: torch.Tensor,  # (B, n)
    cb: object,
    V: torch.Tensor,  # (n, R)
    Vh: torch.Tensor,  # (R, n)
    S: torch.Tensor,  # (R,)
    Vs: torch.Tensor,  # (n, R)
    cfg: E8P12LrsVaspConfig,
    module_name: str = "",
    log_per_iter: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Run RS-VAMP / LRS-VASP on trace-normalized Hbar and score on the raw surrogate."""
    from lrs_vasp_ptq.rs_vamp import RsVampConfig, run_rs_vamp_e8p12_rankspace
    from lrs_vasp_ptq.survey_vamp import LrsVaspConfig, run_lrs_vasp_e8p12_rankspace

    Wr = Wr.to(dtype=torch.float32)
    hatWr_init = hatWr_init.to(dtype=torch.float32)

    B, n = Wr.shape
    codesz = int(getattr(cb, "codesz", 8))
    if codesz != 8 or (n % codesz) != 0:
        raise ValueError(f"E8P12 solver expects n divisible by 8; got n={n}, codesz={codesz}.")

    # Diagnostics for Hbar init mismatch energy (not used by default scale).
    eigs = S.to(dtype=torch.float32).square()
    trace_avg = float(eigs.mean().clamp_min_(1e-30).item()) if eigs.numel() else 1.0
    Vsbar = Vs.to(dtype=torch.float32) / float(trace_avg) ** 0.5
    diff_u = (Wr - hatWr_init).matmul(Vsbar)
    mse_u = float(diff_u.square().mean().clamp_min_(1e-12).item())

    gamma1_init = float(cfg.gamma_start) * (0.25 / _rowwise_quant_mse(Wr, hatWr_init))
    gamma1_init_mean = float(gamma1_init.mean().item())
    lambda_base = float(gamma1_init_mean)
    lambda_base_mode = "gamma1_init_mean"
    gamma2_init_mean = 0.0
    alpha1_init_mean = 0.0

    scale_mode = str(cfg.lambda_eff_scale_mode or "auto").strip().lower()
    raw_grid = tuple(float(x) for x in (cfg.lambda_eff_grid or (1.0,)))
    if not raw_grid:
        raw_grid = (1.0,)
    if scale_mode == "auto":
        # More robust auto scaling: initialize lambda_eff from the *extrinsic*
        # linear-module precision implied by the RS denoiser at r1_init.
        #
        # With trace-normalized Hbar (mean eigenvalue ~= 1), the linear module's
        # denom is (lambda_eff * eig + gamma2). Setting lambda_eff on the same
        # order as gamma2 avoids alpha2 collapse and tends to be a better
        # starting scale than gamma1_init or Hbar mismatch alone.
        from lrs_vasp_ptq.denoisers.e8p_exact import rs_posterior_blocks

        W_blocks = Wr.reshape(B, n // codesz, codesz)
        prior_tau = float(cfg.prior_tau)
        prior_tau_scale_mode = str(cfg.prior_tau_scale_mode or "absmax").strip().lower()
        prior_scale_blocks = None
        if prior_tau > 0.0:
            prior_scale_blocks = prior_scale_from_center(W_blocks, prior_tau_scale_mode)

        post = rs_posterior_blocks(
            center_blocks=hatWr_init.reshape(B, n // codesz, codesz),
            gamma=gamma1_init,
            cb=cb,
            blocks_per_chunk=int(cfg.blocks_per_chunk),
            prior_center_blocks=W_blocks if prior_tau > 0.0 else None,
            prior_scale_blocks=prior_scale_blocks,
            prior_tau=prior_tau,
            prior_tau_scale_mode=prior_tau_scale_mode,
            return_var_diag=False,
            return_logZ=False,
            return_top2=False,
            backend=str(cfg.backend),
        )
        cov_trace = post["cov_trace"]  # (B,nblocks)
        alpha_eps = float(cfg.alpha_eps)
        alpha1_init = (gamma1_init * cov_trace).sum(dim=1, keepdim=True) / float(n)
        alpha1_init = torch.nan_to_num(alpha1_init, nan=0.5, posinf=1.0 - alpha_eps, neginf=alpha_eps).clamp(
            min=alpha_eps, max=1.0 - alpha_eps
        )
        eta1_init = gamma1_init / alpha1_init
        gamma2_init = torch.nan_to_num(
            eta1_init - gamma1_init,
            nan=float(cfg.gamma_min),
            posinf=float(cfg.gamma_max),
            neginf=float(cfg.gamma_min),
        ).clamp(min=float(cfg.gamma_min), max=float(cfg.gamma_max))
        alpha1_init_mean = float(alpha1_init.mean().item())
        gamma2_init_mean = float(gamma2_init.mean().item())

        # Scale 1/mse(init) by 1/alpha1_init to account for the VAMP extrinsic
        # normalization (r2, gamma2) implied by the denoiser at init.
        lambda_base = float((1.0 / float(mse_u)) / max(float(alpha1_init_mean), alpha_eps))
        lambda_base_mode = "mse_u_inv_over_alpha1_init"
        lambda_grid = tuple(float(lambda_base) * float(x) for x in raw_grid)
    elif scale_mode == "none":
        lambda_grid = raw_grid
    else:
        raise ValueError(f"unknown lambda_eff_scale_mode={scale_mode!r} (expected 'auto'|'none')")

    method = str(cfg.method or "lrs-vasp").strip().lower()
    if method not in {"rs-vamp", "lrs-vasp"}:
        raise ValueError(f"unknown method={method!r} (expected 'rs-vamp'|'lrs-vasp')")

    # Score candidates under the original (raw) rank-space surrogate.
    Y_u = Wr.matmul(Vs.to(dtype=torch.float32))

    best_hat = None
    best_idxs = None
    best_err = None
    best_err_mean = float("inf")
    best_stats: Dict[str, Any] = {}

    polish_sweeps = max(int(cfg.polish_sweeps), 0)
    polish_metric = str(cfg.polish_metric or "gram").strip().lower()
    if polish_metric in ("isotropic", "diag"):
        polish_metric = "diag"
    elif polish_metric != "gram":
        raise ValueError(f"unsupported polish_metric={polish_metric!r}; expected 'gram'|'diag'.")

    for lam in lambda_grid:
        if method == "rs-vamp" or not bool(cfg.enable_survey):
            out = run_rs_vamp_e8p12_rankspace(
                w=Wr,
                r1_init=hatWr_init,
                cb=cb,
                V=V,
                Vh=Vh,
                S=S,
                lambda_eff=float(lam),
                cfg=RsVampConfig(
                    max_iters=int(cfg.iters),
                    damping=float(cfg.damping),
                    gamma_start=float(cfg.gamma_start),
                    gamma_retry_factor=float(cfg.gamma_retry_factor),
                    gamma_retry_steps=int(cfg.gamma_retry_steps),
                    alpha_eps=float(cfg.alpha_eps),
                    gamma_min=float(cfg.gamma_min),
                    gamma_max=float(cfg.gamma_max),
                    blocks_per_chunk=int(cfg.blocks_per_chunk),
                    tol=float(cfg.tol),
                    prior_tau=float(cfg.prior_tau),
                    prior_tau_scale_mode=str(cfg.prior_tau_scale_mode),
                ),
                backend=str(cfg.backend),
                log_per_iter=bool(log_per_iter),
            )
            hat = out["q_hard"]
            idxs = out["q_hard_idxs"]
            logs = out.get("logs", [])
            survey_meta: Dict[str, Any] = {"solver": "rs-vamp"}
        else:
            out = run_lrs_vasp_e8p12_rankspace(
                w=Wr,
                r1_init=hatWr_init,
                cb=cb,
                V=V,
                Vh=Vh,
                S=S,
                lambda_eff=float(lam),
                cfg=LrsVaspConfig(
                    max_iters=int(cfg.iters),
                    rs_warmup_iters=int(cfg.rs_warmup_iters),
                    damping=float(cfg.damping),
                    gamma_start=float(cfg.gamma_start),
                    gamma_retry_factor=float(cfg.gamma_retry_factor),
                    gamma_retry_steps=int(cfg.gamma_retry_steps),
                    alpha_eps=float(cfg.alpha_eps),
                    gamma_min=float(cfg.gamma_min),
                    gamma_max=float(cfg.gamma_max),
                    blocks_per_chunk=int(cfg.blocks_per_chunk),
                    tol=float(cfg.tol),
                    prior_tau=float(cfg.prior_tau),
                    prior_tau_scale_mode=str(cfg.prior_tau_scale_mode),
                    enable_survey=bool(cfg.enable_survey),
                    survey_L=float(cfg.survey_L),
                    survey_Q=int(cfg.survey_Q),
                    survey_v1_init=float(cfg.survey_v1_init),
                    survey_v1_min=float(cfg.survey_v1_min),
                    hard_gap_thresh=float(cfg.hard_gap_thresh),
                    hard_var_mult=float(cfg.hard_var_mult),
                    hard_max_frac=float(cfg.hard_max_frac),
                ),
                backend=str(cfg.backend),
                log_per_iter=bool(log_per_iter),
            )
            hat = out["q_hard"]
            idxs = out["q_hard_idxs"]
            logs = out.get("logs", [])
            survey_meta = out.get("survey", {}) if isinstance(out.get("survey"), dict) else {"solver": "lrs-vasp"}

        err = vamp_rankspace_error(Y_u=Y_u, x=hat, Vs=Vs.to(dtype=torch.float32))
        err_mean = float(err.mean().item()) if err.numel() else 0.0

        polish_gain = 0.0
        polish_updates = 0.0
        if polish_sweeps > 0 and Vs.numel() > 0:
            err_before = err
            hat, idxs, polish_stats = block_coordinate_descent_rankspace(
                x=hat.to(dtype=torch.float32),
                Y_u=Y_u.to(dtype=torch.float32),
                Vs=Vs.to(dtype=torch.float32),
                block_size=codesz,
                provider=E8P12Provider(cb=cb, metric_rows_per_chunk=int(cfg.polish_metric_rows_per_chunk)),
                cfg=BlockCDConfig(
                    sweeps=polish_sweeps,
                    metric="gram" if polish_metric == "gram" else "diag",
                    accept_tol=0.0,
                    metric_rows_per_chunk=int(cfg.polish_metric_rows_per_chunk),
                    max_blocks_per_sweep=int(getattr(cfg, "polish_max_blocks", 0) or 0),
                    order_mode="greedy_gain",
                ),
                idxs_init=idxs,
            )
            err = vamp_rankspace_error(Y_u=Y_u, x=hat, Vs=Vs.to(dtype=torch.float32))
            err_mean = float(err.mean().item()) if err.numel() else 0.0
            polish_gain = float((err_before - err).mean().item()) if err.numel() else 0.0
            polish_updates = float(polish_stats.get("accepted_updates", 0.0))

        if err_mean < best_err_mean:
            best_err_mean = err_mean
            best_hat = hat
            best_idxs = idxs
            best_err = err
            best_stats = {
                "solver": "e8p12_lrs_vasp",
                "module": str(module_name),
                "method": str(method),
                "backend": str(cfg.backend),
                "iters": int(cfg.iters),
                "rs_warmup_iters": int(cfg.rs_warmup_iters),
                "damping": float(cfg.damping),
                "lambda_eff": float(lam),
                "lambda_eff_grid": [float(x) for x in lambda_grid],
                "lambda_eff_base": float(lambda_base),
                "lambda_eff_base_mode": str(lambda_base_mode),
                "lambda_eff_scale_mode": str(scale_mode),
                "trace_avg": float(trace_avg),
                "init_mse_u": float(mse_u),
                "gamma1_init_mean": float(gamma1_init_mean),
                "alpha1_init_mean": float(alpha1_init_mean),
                "gamma2_init_mean": float(gamma2_init_mean),
                "rankspace_error_mean": float(err_mean),
                "survey": survey_meta,
                "logs": logs,
                "polish_sweeps": int(polish_sweeps),
                "polish_metric": str(polish_metric),
                "polish_gain": float(polish_gain),
                "polish_updates": float(polish_updates),
            }

    if best_hat is None or best_idxs is None or best_err is None:
        raise RuntimeError("LRS-VASP failed to produce a candidate solution")
    return best_hat, best_idxs, best_err, best_stats
