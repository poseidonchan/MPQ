from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch

from .linear import em_update_obs_precision_rankspace, linear_mmse_exact, vamp_rankspace_error
from .polish.block_cd import BlockCDConfig, E8P12Provider, UniformScalarProvider, block_coordinate_descent_rankspace
from .priors.e8p12 import prior_scale_from_center, vamp_vector_discrete_mmse_e8p12
from .priors.uniform import dequant_from_q, discrete_uniform_mmse, nearest_atom_idx, select_x1_for_linear_step
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
    x1_mode: str = "mmse"  # "mmse" | "map"
    final_decision: str = "map"  # "map" | "round_mean" | "nearest_mean"
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
    best_iterate_export: bool = True
    adaptive_accept_reject: bool = True
    adaptive_retry_steps: int = 3
    adaptive_accept_tol: float = 1e-4
    adaptive_damping_min: float = 0.1
    adaptive_obs_precision_backoff: float = 0.5
    polish_sweeps: int = 3


@dataclass(frozen=True)
class E8P12VampConfig:
    iters: int = 4
    damping: float = 0.5
    x1_mode: str = "map"  # "mean" | "map" | "mix"
    map_mix: float = 0.5
    gamma_start: float = 1.0
    gamma_retry_factor: float = 4.0
    gamma_retry_steps: int = 3
    prior_tau: float = 0.0
    prior_tau_scale_mode: str = "absmax"
    obs_precision: float = 0.0
    obs_precision_end: Optional[float] = None
    em_obs_precision: bool = True
    em_obs_precision_damping: float = 0.3
    em_obs_precision_min: float = 1e-4
    em_obs_precision_max: float = 1e8
    best_iterate_export: bool = True
    adaptive_accept_reject: bool = True
    adaptive_retry_steps: int = 3
    adaptive_accept_tol: float = 1e-4
    adaptive_damping_min: float = 0.1
    adaptive_obs_precision_backoff: float = 0.5
    blocks_per_chunk: int = 16
    polish_sweeps: int = 3
    polish_metric: str = "gram"  # "gram" | "diag"
    polish_metric_rows_per_chunk: int = 0


@torch.inference_mode()
def _rowwise_quant_mse(W: torch.Tensor, x_quant: torch.Tensor) -> torch.Tensor:
    return (W - x_quant).square().mean(dim=1, keepdim=True).clamp_min_(1e-12)


@torch.inference_mode()
def _global_quant_mse(W: torch.Tensor, x_quant: torch.Tensor) -> float:
    return float((W - x_quant).square().mean().clamp_min_(1e-12).item())


def _positive_schedule(start: float, end: Optional[float], frac: float) -> float:
    start = float(start)
    if end is None:
        return start
    end = float(end)
    if not (math.isfinite(start) and start > 0.0 and math.isfinite(end) and end > 0.0):
        return start
    frac = float(max(0.0, min(1.0, frac)))
    if abs(end - start) <= 1e-12 or frac <= 0.0:
        return start
    if frac >= 1.0:
        return end
    return float(math.exp((1.0 - frac) * math.log(start) + frac * math.log(end)))


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
    log_stats: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Scalar uniform exact-full VAMP in rank space, with greedy polish.

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
    x1_mode = str(cfg.x1_mode or "mmse").strip().lower()
    final_decision = str(cfg.final_decision or "map").strip().lower()
    init_from = str(cfg.init_from or "rtn").strip().lower()

    B, n = W.shape
    rank = int(S.numel())
    atoms = scale_full.unsqueeze(-1) * (
        torch.arange(maxq + 1, dtype=torch.float32, device=W.device).view(1, 1, -1)
        - zero_full.unsqueeze(-1)
    )

    if init_from == "zero":
        r1 = torch.zeros_like(W)
    elif init_from == "float":
        r1 = W.clone()
    else:
        r1 = x_rtn.clone()

    gamma1 = float(cfg.gamma_start) * (0.25 / _rowwise_quant_mse(W, x_rtn))
    gamma1 = gamma1.to(dtype=torch.float32)
    require_valid_precision("gamma1_init", gamma1)

    Y_u = W.matmul(Vs)
    obs_precision_local = _resolve_obs_precision(W=W, x_init=x_rtn, Vs=Vs, requested=float(cfg.obs_precision))

    q_rtn_int = nearest_atom_idx(x=x_rtn, scale_full=scale_full, zero_full=zero_full, maxq=maxq).to(dtype=torch.int32)
    best_q_final = q_rtn_int.clone()
    best_x_final = x_rtn.clone()
    best_err = vamp_rankspace_error(Y_u=Y_u, x=x_rtn, Vs=Vs)
    current_err = best_err.clone()
    last_q_final = best_q_final.clone()
    last_x_final = best_x_final.clone()

    r2_prev: Optional[torch.Tensor] = None
    gamma2_prev: Optional[torch.Tensor] = None

    iter_residuals = [float("nan")] * iters if log_stats else []
    entropy_values = []
    margin_values = []
    alpha1_values = []
    alpha2_values = []
    obs_precision_values = []
    alpha1_retry_count = 0
    alpha2_clip_count = 0
    gamma2_invalid_count = 0
    step_reject_count = 0

    alpha_eps = 1e-7
    gamma_retry_factor = max(float(cfg.gamma_retry_factor), 1.000001)
    gamma_retry_steps = max(int(cfg.gamma_retry_steps), 0)
    em_obs_precision = bool(cfg.em_obs_precision)
    adaptive_accept_reject = bool(cfg.adaptive_accept_reject)
    adaptive_retry_steps = max(int(cfg.adaptive_retry_steps), 0) if adaptive_accept_reject else 0
    adaptive_accept_tol = float(cfg.adaptive_accept_tol)
    adaptive_damping_min = float(cfg.adaptive_damping_min)
    adaptive_obs_backoff = float(cfg.adaptive_obs_precision_backoff)

    for iter_idx in range(iters):
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
                x1 = select_x1_for_linear_step(post_candidate, x1_mode=x1_mode)
                break

            if retry_idx >= gamma_retry_steps:
                if not torch.isfinite(alpha1_candidate).all():
                    require_valid_alpha("alpha1", alpha1_candidate, eps=alpha_eps)
                post = post_candidate
                alpha1 = alpha1_candidate.clamp(min=alpha_eps, max=1.0 - alpha_eps)
                x1 = select_x1_for_linear_step(post_candidate, x1_mode=x1_mode)
                break

            alpha1_retry_count += int((~valid_alpha1).sum().item())
            gamma1_work = torch.where(alpha1_candidate <= alpha_eps, gamma1_work / gamma_retry_factor, gamma1_work)
            gamma1_work = torch.where(alpha1_candidate >= 1.0 - alpha_eps, gamma1_work * gamma_retry_factor, gamma1_work)
            gamma1_work = gamma1_work.clamp_min_(1e-12)

        assert post is not None and alpha1 is not None and x1 is not None
        eta1 = gamma1_work / alpha1
        gamma2_new = eta1 - gamma1_work
        if not bool((precision_valid_mask(gamma2_new).all().item())):
            gamma2_invalid_count += 1
            continue
        r2_new = (eta1 * x1 - gamma1_work * r1) / gamma2_new

        step_damping = damping
        step_obs_precision = float(obs_precision_local)
        accepted = False
        accepted_alpha2 = None
        accepted_residual_u = None
        accepted_obs_precision = obs_precision_local
        accepted_q_iter = None
        accepted_x_iter = None

        for retry_idx in range(adaptive_retry_steps + 1):
            if r2_prev is None or gamma2_prev is None or step_damping >= 1.0:
                gamma2 = gamma2_new
                r2 = r2_new
            else:
                gamma2 = log_damp_positive(gamma2_prev, gamma2_new, step_damping)
                r2 = (1.0 - step_damping) * r2_prev + step_damping * r2_new
            if not bool((precision_valid_mask(gamma2).all().item())):
                step_reject_count += 1
                if retry_idx >= adaptive_retry_steps:
                    break
                step_damping = max(step_damping * 0.5, adaptive_damping_min)
                step_obs_precision = max(step_obs_precision * adaptive_obs_backoff, float(cfg.em_obs_precision_min))
                continue

            x2, alpha2_raw, residual_u = linear_mmse_exact(
                r2,
                gamma2,
                Y_u=Y_u,
                V=V,
                Vh=Vh,
                S=S,
                obs_precision=step_obs_precision,
            )
            valid_alpha2 = alpha_valid_mask(alpha2_raw, eps=alpha_eps)
            alpha2 = torch.nan_to_num(alpha2_raw, nan=0.5, posinf=1.0 - alpha_eps, neginf=alpha_eps)
            alpha2 = alpha2.clamp(min=alpha_eps, max=1.0 - alpha_eps)

            obs_precision_candidate = float(step_obs_precision)
            if em_obs_precision and rank > 0:
                obs_precision_candidate = em_update_obs_precision_rankspace(
                    residual_u=residual_u,
                    gamma2=gamma2,
                    S=S,
                    obs_precision=step_obs_precision,
                    num_obs=rank,
                    damping=float(cfg.em_obs_precision_damping),
                    min_value=float(cfg.em_obs_precision_min),
                    max_value=float(cfg.em_obs_precision_max),
                )

            eta2 = gamma2 / alpha2
            gamma1_new = eta2 - gamma2
            gamma1_valid = bool(torch.isfinite(gamma1_new).all().item() and (gamma1_new > 1e-12).all().item())
            if not gamma1_valid:
                step_reject_count += 1
                if retry_idx >= adaptive_retry_steps:
                    break
                step_damping = max(step_damping * 0.5, adaptive_damping_min)
                step_obs_precision = max(step_obs_precision * adaptive_obs_backoff, float(cfg.em_obs_precision_min))
                continue

            r1_new = (eta2 * x2 - gamma2 * r2) / gamma1_new
            if step_damping >= 1.0:
                gamma1_candidate = gamma1_new
                r1_candidate = r1_new
            else:
                gamma1_candidate = log_damp_positive(gamma1, gamma1_new, step_damping)
                r1_candidate = (1.0 - step_damping) * r1 + step_damping * r1_new
            gamma1_valid = bool(torch.isfinite(gamma1_candidate).all().item() and (gamma1_candidate > 1e-12).all().item())

            q_iter = None
            x_iter = None
            err_valid = False
            allow_err = False
            iter_err = current_err
            if gamma1_valid:
                post_next = discrete_uniform_mmse(
                    r1=r1_candidate,
                    gamma1=gamma1_candidate,
                    scale_full=scale_full,
                    zero_full=zero_full,
                    maxq=maxq,
                    prior_center=W,
                    prior_tau=prior_tau,
                    need_stats=False,
                    atoms=atoms,
                )
                if final_decision in ("round_mean", "nearest_mean"):
                    q_iter = nearest_atom_idx(
                        x=post_next["mean"],  # type: ignore[arg-type]
                        scale_full=scale_full,
                        zero_full=zero_full,
                        maxq=maxq,
                    ).to(dtype=torch.int32)
                else:
                    q_iter = post_next["map_idx"].to(dtype=torch.int32)  # type: ignore[assignment]
                x_iter = dequant_from_q(q_int=q_iter, scale_full=scale_full, zero_full=zero_full)
                iter_err = vamp_rankspace_error(Y_u=Y_u, x=x_iter, Vs=Vs)
                err_valid = bool(torch.isfinite(iter_err).all().item())
                allow_err = True
                if adaptive_accept_reject:
                    allow_err = float(iter_err.mean().item()) <= float(current_err.mean().item()) * (1.0 + adaptive_accept_tol)

            candidate_good = bool(valid_alpha2.all().item()) and gamma1_valid and err_valid and allow_err
            if candidate_good and q_iter is not None and x_iter is not None:
                gamma1 = gamma1_candidate
                r1 = r1_candidate
                obs_precision_local = obs_precision_candidate
                r2_prev = r2
                gamma2_prev = gamma2
                current_err = iter_err
                accepted_alpha2 = alpha2
                accepted_residual_u = residual_u
                accepted_obs_precision = obs_precision_candidate
                accepted_q_iter = q_iter
                accepted_x_iter = x_iter
                accepted = True
                break

            if (not adaptive_accept_reject) or retry_idx >= adaptive_retry_steps:
                break

            step_reject_count += 1
            if not bool(valid_alpha2.all().item()):
                alpha2_clip_count += int((~valid_alpha2).sum().item())
            step_damping = max(step_damping * 0.5, adaptive_damping_min)
            step_obs_precision = max(step_obs_precision * adaptive_obs_backoff, float(cfg.em_obs_precision_min))

        if not accepted or accepted_q_iter is None or accepted_x_iter is None or accepted_alpha2 is None or accepted_residual_u is None:
            continue

        q_iter = accepted_q_iter
        x_iter = accepted_x_iter
        last_q_final = q_iter
        last_x_final = x_iter
        best_mask = (current_err < best_err).unsqueeze(-1)
        best_err = torch.minimum(best_err, current_err)
        best_q_final = torch.where(best_mask, q_iter, best_q_final)
        best_x_final = torch.where(best_mask, x_iter, best_x_final)

        obs_precision_values.append(float(accepted_obs_precision))
        if log_stats:
            iter_residuals[iter_idx] = float(accepted_residual_u.square().mean().item())
            entropy_values.append(float(post["entropy"]))  # type: ignore[arg-type]
            margin_values.append(float(post["margin"]))  # type: ignore[arg-type]
            alpha1_values.append(float(alpha1.mean().item()))
            alpha2_values.append(float(accepted_alpha2.mean().item()))

    if bool(cfg.best_iterate_export):
        q_final = best_q_final
        x_final = best_x_final
        err_final = best_err
    else:
        q_final = last_q_final
        x_final = last_x_final
        err_final = current_err

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
            cfg=BlockCDConfig(sweeps=polish_sweeps, metric="diag", accept_tol=0.0),
        )
        q_polished = nearest_atom_idx(x=x_polished, scale_full=scale_full, zero_full=zero_full, maxq=maxq).to(dtype=torch.int32)
        x_polished = dequant_from_q(q_int=q_polished, scale_full=scale_full, zero_full=zero_full)
        err_after = vamp_rankspace_error(Y_u=Y_u, x=x_polished, Vs=Vs)
        improve = err_after + 1e-12 < err_before
        if bool(improve.any().item()):
            polish_gain = float((err_before - torch.minimum(err_before, err_after)).mean().item())
            q_final = torch.where(improve.unsqueeze(-1), q_polished, q_final)
            x_final = torch.where(improve.unsqueeze(-1), x_polished, x_final)
            err_final = torch.where(improve, err_after, err_final)
        polish_updates = float(polish_stats.get("accepted_updates", 0.0))

    rtn_err = vamp_rankspace_error(Y_u=Y_u, x=x_rtn, Vs=Vs)

    stats: Dict[str, Any] = {
        "solver": "uniform_exact_vamp",
        "iters": int(iters),
        "rank": int(rank),
        "x1_mode": str(x1_mode),
        "final_decision": str(final_decision),
        "init_from": str(init_from),
        "damping": float(damping),
        "prior_tau": float(prior_tau),
        "gamma_start": float(cfg.gamma_start),
        "gamma_retry_factor": float(gamma_retry_factor),
        "gamma_retry_steps": int(gamma_retry_steps),
        "best_iterate_export": bool(cfg.best_iterate_export),
        "adaptive_accept_reject": bool(adaptive_accept_reject),
        "adaptive_retry_steps": int(adaptive_retry_steps),
        "adaptive_accept_tol": float(adaptive_accept_tol),
        "adaptive_damping_min": float(adaptive_damping_min),
        "adaptive_obs_precision_backoff": float(adaptive_obs_backoff),
        "obs_precision_initial": float(obs_precision_local) if not obs_precision_values else float(obs_precision_values[0]),
        "obs_precision_final": float(obs_precision_local),
        "obs_precision_mean": float(sum(obs_precision_values) / len(obs_precision_values)) if obs_precision_values else float(obs_precision_local),
        "iter_residuals": iter_residuals,
        "mean_entropy": float(sum(entropy_values) / len(entropy_values)) if entropy_values else 0.0,
        "mean_margin": float(sum(margin_values) / len(margin_values)) if margin_values else 0.0,
        "alpha1_mean": float(sum(alpha1_values) / len(alpha1_values)) if alpha1_values else 0.0,
        "alpha2_mean": float(sum(alpha2_values) / len(alpha2_values)) if alpha2_values else 0.0,
        "alpha1_retry_count": int(alpha1_retry_count),
        "alpha2_clip_count": int(alpha2_clip_count),
        "gamma2_invalid_count": int(gamma2_invalid_count),
        "step_reject_count": int(step_reject_count),
        "polish_sweeps": int(polish_sweeps),
        "polish_gain": float(polish_gain),
        "polish_updates": float(polish_updates),
        "rankspace_error_mean": float(err_final.mean().item()) if err_final.numel() else 0.0,
        "rtn_rankspace_error_mean": float(rtn_err.mean().item()) if rtn_err.numel() else 0.0,
    }
    return q_final.to(dtype=torch.int32), x_final, err_final, rtn_err, stats


@torch.inference_mode()
def solve_e8p12_rankspace_vamp(
    *,
    Wr: torch.Tensor,  # (B, n)
    hatWr_init: torch.Tensor,  # (B, n)
    Qidxs_init: torch.Tensor,  # (B, n_blocks)
    cb: object,
    V: torch.Tensor,  # (n, R)
    Vh: torch.Tensor,  # (R, n)
    S: torch.Tensor,  # (R,)
    Vs: torch.Tensor,  # (n, R)
    cfg: E8P12VampConfig,
    module_name: str = "",
    log_stats: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """E8P12 exact-full VAMP in rank space, with greedy gram polish."""
    Wr = Wr.to(dtype=torch.float32)
    hatWr_init = hatWr_init.to(dtype=torch.float32)
    Qidxs_init = Qidxs_init.to(dtype=getattr(cb, "idx_dtype", torch.int64))

    B, n = Wr.shape
    codesz = int(getattr(cb, "codesz", 8))
    if codesz != 8 or (n % codesz) != 0:
        raise ValueError(f"E8P12 solver expects n divisible by 8; got n={n}, codesz={codesz}.")
    nblocks = n // codesz
    rank = int(S.numel())

    iters = max(int(cfg.iters), 0)
    damping = float(cfg.damping)
    prior_tau = float(cfg.prior_tau)
    prior_tau_scale_mode = str(cfg.prior_tau_scale_mode or "absmax").strip().lower()
    x1_mode = str(cfg.x1_mode or "map").strip().lower()
    map_mix = float(cfg.map_mix)
    map_mix = max(0.0, min(1.0, map_mix))

    W_blocks = Wr.reshape(B, nblocks, codesz)
    prior_scale_blocks = None
    if prior_tau > 0.0:
        prior_scale_blocks = prior_scale_from_center(W_blocks, prior_tau_scale_mode)

    Y_u = Wr.matmul(Vs)
    obs_precision_start = _resolve_obs_precision(W=Wr, x_init=hatWr_init, Vs=Vs, requested=float(cfg.obs_precision))
    obs_precision_end = cfg.obs_precision_end if float(cfg.obs_precision) > 0.0 else None
    obs_precision_local = float(obs_precision_start)
    em_obs_active = bool(cfg.em_obs_precision) and float(cfg.obs_precision) <= 0.0

    r1 = hatWr_init.clone()
    gamma1 = float(cfg.gamma_start) * (0.25 / _rowwise_quant_mse(Wr, hatWr_init))
    gamma1 = gamma1.to(dtype=torch.float32)
    require_valid_precision("gamma1_init", gamma1)

    best_hatWr = hatWr_init.clone()
    best_Qidxs = Qidxs_init.clone()
    best_err = vamp_rankspace_error(Y_u=Y_u, x=hatWr_init, Vs=Vs)
    current_err = best_err.clone()
    last_hatWr = hatWr_init.clone()
    last_Qidxs = Qidxs_init.clone()

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
    step_reject_count = 0

    alpha_eps = 1e-7
    gamma_retry_factor = max(float(cfg.gamma_retry_factor), 1.000001)
    gamma_retry_steps = max(int(cfg.gamma_retry_steps), 0)
    adaptive_accept_reject = bool(cfg.adaptive_accept_reject)
    adaptive_retry_steps = max(int(cfg.adaptive_retry_steps), 0) if adaptive_accept_reject else 0
    adaptive_accept_tol = float(cfg.adaptive_accept_tol)
    adaptive_damping_min = float(cfg.adaptive_damping_min)
    adaptive_obs_backoff = float(cfg.adaptive_obs_precision_backoff)

    blocks_per_chunk = max(int(cfg.blocks_per_chunk), 1)

    for iter_idx in range(iters):
        frac = 1.0 if iters <= 1 else float(iter_idx) / float(iters - 1)
        step_obs_precision = (
            _positive_schedule(obs_precision_start, obs_precision_end, frac)
            if float(cfg.obs_precision) > 0.0
            else float(obs_precision_local)
        )

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
                if x1_mode == "map":
                    x1 = post_candidate["map_vals"].reshape_as(r1)  # type: ignore[assignment]
                elif x1_mode == "mix":
                    x_mean = post_candidate["x1_blocks"].reshape_as(r1)  # type: ignore[assignment]
                    x_map = post_candidate["map_vals"].reshape_as(r1)  # type: ignore[assignment]
                    x1 = (1.0 - map_mix) * x_mean + map_mix * x_map
                else:
                    x1 = post_candidate["x1_blocks"].reshape_as(r1)  # type: ignore[assignment]
                break

            if retry_idx >= gamma_retry_steps:
                if not torch.isfinite(alpha1_candidate).all():
                    require_valid_alpha("alpha1", alpha1_candidate, eps=alpha_eps)
                post = post_candidate
                alpha1 = torch.nan_to_num(alpha1_candidate, nan=0.5, posinf=1.0 - alpha_eps, neginf=alpha_eps).clamp(
                    min=alpha_eps, max=1.0 - alpha_eps
                )
                if x1_mode == "map":
                    x1 = post_candidate["map_vals"].reshape_as(r1)  # type: ignore[assignment]
                elif x1_mode == "mix":
                    x_mean = post_candidate["x1_blocks"].reshape_as(r1)  # type: ignore[assignment]
                    x_map = post_candidate["map_vals"].reshape_as(r1)  # type: ignore[assignment]
                    x1 = (1.0 - map_mix) * x_mean + map_mix * x_map
                else:
                    x1 = post_candidate["x1_blocks"].reshape_as(r1)  # type: ignore[assignment]
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

        step_damping = damping
        step_obs = float(step_obs_precision)
        accepted = False
        accepted_alpha2 = None
        accepted_obs = obs_precision_local
        accepted_hatWr = None
        accepted_Qidxs = None
        accepted_residual_u = None

        for retry_idx in range(adaptive_retry_steps + 1):
            if r2_prev is None or gamma2_prev is None or step_damping >= 1.0:
                gamma2_try = gamma2_new
                r2_try = r2_new
            else:
                gamma2_try = log_damp_positive(gamma2_prev, gamma2_new, step_damping)
                r2_try = (1.0 - step_damping) * r2_prev + step_damping * r2_new
            if not bool((precision_valid_mask(gamma2_try).all().item())):
                step_reject_count += 1
                if retry_idx >= adaptive_retry_steps:
                    break
                step_damping = max(step_damping * 0.5, adaptive_damping_min)
                step_obs = max(step_obs * adaptive_obs_backoff, float(cfg.em_obs_precision_min))
                continue

            x2, alpha2_raw, residual_u = linear_mmse_exact(
                r2_try,
                gamma2_try,
                Y_u=Y_u,
                V=V,
                Vh=Vh,
                S=S,
                obs_precision=step_obs,
            )
            valid_alpha2 = alpha_valid_mask(alpha2_raw, eps=alpha_eps)
            alpha2 = torch.nan_to_num(alpha2_raw, nan=0.5, posinf=1.0 - alpha_eps, neginf=alpha_eps).clamp(
                min=alpha_eps, max=1.0 - alpha_eps
            )

            obs_candidate = float(step_obs)
            if em_obs_active and rank > 0:
                obs_candidate = em_update_obs_precision_rankspace(
                    residual_u=residual_u,
                    gamma2=gamma2_try,
                    S=S,
                    obs_precision=step_obs,
                    num_obs=rank,
                    damping=float(cfg.em_obs_precision_damping),
                    min_value=float(cfg.em_obs_precision_min),
                    max_value=float(cfg.em_obs_precision_max),
                )

            eta2 = gamma2_try / alpha2
            gamma1_new = torch.nan_to_num(eta2 - gamma2_try, nan=1e-12, posinf=1e12, neginf=1e-12).clamp_min_(
                1e-12
            )
            r1_new = (eta2 * x2 - gamma2_try * r2_try) / gamma1_new
            if step_damping >= 1.0:
                gamma1_candidate = gamma1_new
                r1_candidate = r1_new
            else:
                gamma1_candidate = log_damp_positive(gamma1, gamma1_new, step_damping)
                r1_candidate = (1.0 - step_damping) * r1 + step_damping * r1_new
            gamma1_valid = bool(torch.isfinite(gamma1_candidate).all().item() and (gamma1_candidate > 1e-12).all().item())

            trial_err = current_err
            err_valid = False
            allow_err = False
            trial_hatWr = None
            trial_Qidxs = None
            if gamma1_valid:
                trial_post = vamp_vector_discrete_mmse_e8p12(
                    r1_blocks=r1_candidate.reshape(B, nblocks, codesz),
                    gamma1=gamma1_candidate,
                    cb=cb,
                    blocks_per_chunk=blocks_per_chunk,
                    need_stats=False,
                    need_var_diag=False,
                    prior_center_blocks=W_blocks if prior_tau > 0.0 else None,
                    prior_scale_blocks=prior_scale_blocks,
                    prior_tau=prior_tau,
                    prior_tau_scale_mode=prior_tau_scale_mode,
                )
                trial_hatWr = trial_post["map_vals"].reshape_as(r1)  # type: ignore[assignment]
                trial_Qidxs = trial_post["map_idxs"].to(dtype=getattr(cb, "idx_dtype", torch.int64))  # type: ignore[assignment]
                trial_err = vamp_rankspace_error(Y_u=Y_u, x=trial_hatWr, Vs=Vs)
                err_valid = bool(torch.isfinite(trial_err).all().item())
                allow_err = True
                if adaptive_accept_reject:
                    allow_err = float(trial_err.mean().item()) <= float(current_err.mean().item()) * (1.0 + adaptive_accept_tol)

            candidate_good = bool(valid_alpha2.all().item()) and gamma1_valid and err_valid and allow_err
            if candidate_good and trial_hatWr is not None and trial_Qidxs is not None:
                gamma1 = gamma1_candidate
                r1 = r1_candidate
                r2_prev = r2_try
                gamma2_prev = gamma2_try
                obs_precision_local = obs_candidate
                current_err = trial_err
                accepted = True
                accepted_alpha2 = alpha2
                accepted_obs = obs_candidate
                accepted_hatWr = trial_hatWr
                accepted_Qidxs = trial_Qidxs
                accepted_residual_u = residual_u
                break

            if (not adaptive_accept_reject) or retry_idx >= adaptive_retry_steps:
                break

            step_reject_count += 1
            if not bool(valid_alpha2.all().item()):
                alpha2_clip_count += int((~valid_alpha2).sum().item())
            step_damping = max(step_damping * 0.5, adaptive_damping_min)
            step_obs = max(step_obs * adaptive_obs_backoff, float(cfg.em_obs_precision_min))

        if not accepted or accepted_hatWr is None or accepted_Qidxs is None or accepted_alpha2 is None:
            continue

        last_hatWr = accepted_hatWr
        last_Qidxs = accepted_Qidxs
        if bool(cfg.best_iterate_export):
            improved = current_err < best_err
            if bool(improved.any().item()):
                mask = improved.unsqueeze(-1)
                best_hatWr = torch.where(mask, accepted_hatWr, best_hatWr)
                best_Qidxs = torch.where(improved.unsqueeze(-1), accepted_Qidxs, best_Qidxs)
                best_err = torch.minimum(best_err, current_err)

        obs_precision_values.append(float(accepted_obs))
        alpha1_values.append(float(alpha1.mean().item()))
        alpha2_values.append(float(accepted_alpha2.mean().item()))
        if log_stats:
            entropy_values.append(float(post["entropy"]))  # type: ignore[arg-type]
            margin_values.append(float(post["margin"]))  # type: ignore[arg-type]

    if bool(cfg.best_iterate_export):
        final_hatWr = best_hatWr
        final_Qidxs = best_Qidxs
        final_err = best_err
    else:
        final_hatWr = last_hatWr
        final_Qidxs = last_Qidxs
        final_err = current_err

    polish_sweeps = max(int(cfg.polish_sweeps), 0)
    polish_metric = str(cfg.polish_metric or "gram").strip().lower()
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
        "iters": int(iters),
        "rank": int(rank),
        "nblocks": int(nblocks),
        "x1_mode": str(x1_mode),
        "map_mix": float(map_mix),
        "damping": float(damping),
        "prior_tau": float(prior_tau),
        "prior_tau_scale_mode": str(prior_tau_scale_mode),
        "gamma_start": float(cfg.gamma_start),
        "gamma_retry_factor": float(gamma_retry_factor),
        "gamma_retry_steps": int(gamma_retry_steps),
        "best_iterate_export": bool(cfg.best_iterate_export),
        "adaptive_accept_reject": bool(adaptive_accept_reject),
        "adaptive_retry_steps": int(adaptive_retry_steps),
        "adaptive_accept_tol": float(adaptive_accept_tol),
        "adaptive_damping_min": float(adaptive_damping_min),
        "adaptive_obs_precision_backoff": float(adaptive_obs_backoff),
        "obs_precision_initial": float(obs_precision_start),
        "obs_precision_final": float(obs_precision_local),
        "obs_precision_mean": float(sum(obs_precision_values) / len(obs_precision_values)) if obs_precision_values else float(obs_precision_local),
        "alpha1_retry_count": int(alpha1_retry_count),
        "alpha2_clip_count": int(alpha2_clip_count),
        "gamma2_invalid_count": int(gamma2_invalid_count),
        "step_reject_count": int(step_reject_count),
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

