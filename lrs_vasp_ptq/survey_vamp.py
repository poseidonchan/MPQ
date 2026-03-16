from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

from mpgq.priors.e8p12 import vamp_vector_map_e8p12
from mpgq.utils import log_damp_positive

from .denoisers.e8p_exact import rs_posterior_blocks
from .hard_block_selector import HardBlockSelectorConfig, select_hard_blocks
from .linear_module import SpectralLinearModule
from .utils.numerics import clip_alpha, clip_gamma
from .utils.quadrature import normal_1d


@dataclass(frozen=True)
class LrsVaspConfig:
    max_iters: int = 16
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

    enable_survey: bool = True
    survey_L: float = 2.0
    survey_Q: int = 9
    survey_v1_init: float = 0.0  # <=0 means "auto"
    survey_v1_min: float = 1e-6

    hard_gap_thresh: float = 0.5
    hard_var_mult: float = 2.0
    hard_max_frac: float = 0.25


@torch.inference_mode()
def _rowwise_mse(w: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return (w - x).square().mean(dim=1, keepdim=True).clamp_min_(1e-12)


@torch.inference_mode()
def _stage_iters(total: int) -> List[int]:
    total = int(total)
    if total <= 0:
        return []
    # Spec suggests 2–3 iters per stage; use 3,3,rest when possible.
    s1 = min(3, total)
    s2 = min(3, max(0, total - s1))
    s3 = max(0, total - s1 - s2)
    out = [s1]
    if s2:
        out.append(s2)
    if s3:
        out.append(s3)
    return out


@torch.inference_mode()
def _apply_survey_on_hard_blocks(
    *,
    r1_blocks: torch.Tensor,  # (B, n_blocks, 8)
    gamma1: torch.Tensor,  # (B, 1)
    cb: Any,
    hard_mask: torch.Tensor,  # (B, n_blocks) bool
    L: float,
    Q: int,
    v1: float,
    blocks_per_chunk: int,
    backend: str,
    prior_center_blocks: Optional[torch.Tensor],
    prior_scale_blocks: Optional[torch.Tensor],
    prior_tau: float,
    prior_tau_scale_mode: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (mean_blocks, jac_trace_blocks) with survey overrides on hard blocks only."""
    B, n_blocks, codesz = r1_blocks.shape
    if codesz != 8:
        raise ValueError(f"expected codesz=8, got {codesz}")

    # Base RS for all blocks (cheap path).
    rs = rs_posterior_blocks(
        center_blocks=r1_blocks,
        gamma=gamma1,
        cb=cb,
        blocks_per_chunk=int(blocks_per_chunk),
        prior_center_blocks=prior_center_blocks,
        prior_scale_blocks=prior_scale_blocks,
        prior_tau=float(prior_tau),
        prior_tau_scale_mode=str(prior_tau_scale_mode),
        return_var_diag=False,
        return_logZ=False,
        return_top2=False,
        backend=str(backend),
    )
    mean_blocks = rs["mean"]  # (B, n_blocks, 8)
    cov_trace = rs["cov_trace"]  # (B, n_blocks)
    jac_trace_blocks = (gamma1 * cov_trace).to(dtype=torch.float32)  # (B, n_blocks)

    if not bool(hard_mask.any().item()) or float(v1) <= 0.0 or float(L) <= 0.0:
        return mean_blocks, jac_trace_blocks

    idx = hard_mask.nonzero(as_tuple=False)  # (Nh,2)
    Nh = int(idx.shape[0])
    if Nh <= 0:
        return mean_blocks, jac_trace_blocks

    rule = normal_1d(int(Q), var=float(v1), dtype=torch.float64)
    nodes = rule.nodes.to(device=r1_blocks.device, dtype=torch.float32)  # (Q,)
    quad_w = rule.weights.to(device=r1_blocks.device, dtype=torch.float32).clamp_min_(1e-30)  # (Q,)
    log_quad_w = torch.log(quad_w).view(1, -1)  # (1,Q)

    u1 = torch.full((8,), 1.0 / (8.0**0.5), device=r1_blocks.device, dtype=torch.float32)  # ones/sqrt(8)
    shift = nodes.view(-1, 1) * u1.view(1, -1)  # (Q,8)

    base = r1_blocks[idx[:, 0], idx[:, 1], :].to(dtype=torch.float32)  # (Nh,8)
    centers = base.unsqueeze(1) + shift.unsqueeze(0)  # (Nh,Q,8)

    gamma_items = gamma1[idx[:, 0], :].to(dtype=torch.float32)  # (Nh,1)
    gamma_flat = gamma_items.repeat_interleave(int(Q), dim=0)  # (Nh*Q,1)

    prior_center_flat = None
    prior_scale_flat = None
    if float(prior_tau) > 0.0 and prior_center_blocks is not None:
        pc = prior_center_blocks[idx[:, 0], idx[:, 1], :].to(dtype=torch.float32)  # (Nh,8)
        prior_center_flat = pc.repeat_interleave(int(Q), dim=0).reshape(Nh * int(Q), 1, 8)
        if prior_scale_blocks is not None:
            ps = prior_scale_blocks[idx[:, 0], idx[:, 1], :].to(dtype=torch.float32)  # (Nh,1)
            prior_scale_flat = ps.repeat_interleave(int(Q), dim=0).reshape(Nh * int(Q), 1, 1)

    post = rs_posterior_blocks(
        center_blocks=centers.reshape(Nh * int(Q), 1, 8),
        gamma=gamma_flat,
        cb=cb,
        blocks_per_chunk=int(blocks_per_chunk),
        prior_center_blocks=prior_center_flat,
        prior_scale_blocks=prior_scale_flat,
        prior_tau=float(prior_tau),
        prior_tau_scale_mode=str(prior_tau_scale_mode),
        return_var_diag=False,
        return_logZ=True,
        return_top2=False,
        backend=str(backend),
    )
    m_nodes = post["mean"].reshape(Nh, int(Q), 8)  # (Nh,Q,8)
    cov_trace_nodes = post["cov_trace"].reshape(Nh, int(Q))  # (Nh,Q)
    logZ_nodes = post["logZ"].reshape(Nh, int(Q))  # (Nh,Q)

    log_a = log_quad_w + float(L) * logZ_nodes  # (Nh,Q)
    log_pi = log_a - torch.logsumexp(log_a, dim=1, keepdim=True)
    pi = torch.exp(log_pi)  # (Nh,Q)

    m_mix = (pi.unsqueeze(-1) * m_nodes).sum(dim=1)  # (Nh,8)
    cov_trace_mix = (pi * cov_trace_nodes).sum(dim=1)  # (Nh,)

    a = m_nodes - m_mix.unsqueeze(1)  # (Nh,Q,8)
    b = m_nodes - base.unsqueeze(1) - shift.unsqueeze(0)  # (Nh,Q,8)
    dot = (a * b).sum(dim=-1)  # (Nh,Q)
    second_term = (pi * dot).sum(dim=1)  # (Nh,)

    gamma_scalar = gamma_items.squeeze(-1)  # (Nh,)
    jac_trace_mix = gamma_scalar * cov_trace_mix + float(L) * gamma_scalar * second_term  # (Nh,)

    mean_blocks[idx[:, 0], idx[:, 1], :] = m_mix
    jac_trace_blocks[idx[:, 0], idx[:, 1]] = jac_trace_mix.to(dtype=torch.float32)
    return mean_blocks, jac_trace_blocks


@torch.inference_mode()
def run_lrs_vasp_e8p12_rankspace(
    *,
    w: torch.Tensor,  # (B, n)
    r1_init: torch.Tensor,  # (B, n)
    cb: Any,
    V: torch.Tensor,
    Vh: torch.Tensor,
    S: torch.Tensor,
    lambda_eff: float,
    cfg: LrsVaspConfig,
    backend: str = "factorized",
    log_per_iter: bool = False,
) -> Dict[str, Any]:
    """LRS-VASP (RS-VAMP + low-rank survey on hard blocks) for QUIP rank space."""
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

    gamma1 = float(cfg.gamma_start) * (0.25 / _rowwise_mse(w, r1))
    gamma1 = clip_gamma(gamma1.to(dtype=torch.float32), min_value=float(cfg.gamma_min), max_value=float(cfg.gamma_max))

    r2_prev: Optional[torch.Tensor] = None
    gamma2_prev: Optional[torch.Tensor] = None

    logs: List[Dict[str, float]] = []
    x1 = r1
    x2 = r1
    alpha1 = torch.full((B, 1), 0.5, device=w.device, dtype=torch.float32)
    alpha2 = torch.full((B, 1), 0.5, device=w.device, dtype=torch.float32)

    # Phase 1: warmup RS-VAMP.
    warmup = min(max(int(cfg.rs_warmup_iters), 0), int(cfg.max_iters))
    for it in range(warmup):
        gamma1_work = gamma1
        jac_trace_blocks = None
        for retry_idx in range(gamma_retry_steps + 1):
            rs = rs_posterior_blocks(
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
            mean_blocks = rs["mean"]
            cov_trace = rs["cov_trace"]
            jac_trace_blocks = (gamma1_work * cov_trace).to(dtype=torch.float32)
            x1 = mean_blocks.reshape(B, n)
            alpha1_candidate = jac_trace_blocks.sum(dim=1, keepdim=True) / float(n)
            valid = torch.isfinite(alpha1_candidate) & (alpha1_candidate > alpha_eps) & (alpha1_candidate < 1.0 - alpha_eps)
            if bool(valid.all().item()):
                alpha1 = alpha1_candidate
                break
            if retry_idx >= gamma_retry_steps:
                alpha1 = clip_alpha(alpha1_candidate, eps=alpha_eps)
                break

            gamma1_work = torch.where(alpha1_candidate <= alpha_eps, gamma1_work / gamma_retry_factor, gamma1_work)
            gamma1_work = torch.where(alpha1_candidate >= 1.0 - alpha_eps, gamma1_work * gamma_retry_factor, gamma1_work)
            gamma1_work = clip_gamma(gamma1_work, min_value=float(cfg.gamma_min), max_value=float(cfg.gamma_max))

        assert jac_trace_blocks is not None
        eta1 = gamma1_work / alpha1
        gamma2_new = clip_gamma(eta1 - gamma1_work, min_value=float(cfg.gamma_min), max_value=float(cfg.gamma_max))
        r2_new = (eta1 * x1 - gamma1_work * r1) / gamma2_new

        if r2_prev is None or gamma2_prev is None or damping >= 1.0:
            r2 = r2_new
            gamma2 = gamma2_new
        else:
            r2 = (1.0 - damping) * r2_prev + damping * r2_new
            gamma2 = log_damp_positive(gamma2_prev, gamma2_new, damping)

        x2, alpha2_raw, _ = linear.apply(r2=r2, gamma2=gamma2, lambda_eff=lam)
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
                    "phase": 1.0,
                    "lambda_eff": float(lam),
                    "trace_avg": float(trace_avg),
                    "alpha1": float(alpha1.mean().item()),
                    "alpha2": float(alpha2.mean().item()),
                    "gamma1": float(gamma1.mean().item()),
                    "gamma2": float(gamma2.mean().item()),
                    "gap": float(gap),
                }
            )

    # Select hard blocks once (as in spec).
    hard_mask = torch.zeros((B, nblocks), device=w.device, dtype=torch.bool)
    v1_init = 0.0
    if bool(cfg.enable_survey):
        stats = rs_posterior_blocks(
            center_blocks=r1.reshape(B, nblocks, codesz),
            gamma=gamma1,
            cb=cb,
            blocks_per_chunk=int(cfg.blocks_per_chunk),
            prior_center_blocks=W_blocks if prior_tau > 0.0 else None,
            prior_scale_blocks=prior_scale_blocks,
            prior_tau=prior_tau,
            prior_tau_scale_mode=prior_tau_scale_mode,
            return_var_diag=False,
            return_logZ=False,
            return_top2=True,
            backend=str(backend),
        )
        cov_trace = stats["cov_trace"]
        var = (cov_trace / float(codesz)).to(dtype=torch.float32)
        gap = stats.get("top2_gap")
        hard_mask = select_hard_blocks(
            top2_gap=gap.to(dtype=torch.float32) if gap is not None else None,
            var=var,
            cfg=HardBlockSelectorConfig(
                gap_thresh=float(cfg.hard_gap_thresh),
                var_mult=float(cfg.hard_var_mult),
                max_frac=float(cfg.hard_max_frac),
            ),
        )
        if float(cfg.survey_v1_init) > 0.0:
            v1_init = float(cfg.survey_v1_init)
        else:
            if bool(hard_mask.any().item()):
                v1_init = float(var[hard_mask].median().clamp_min(0.0).item())
            else:
                v1_init = float(var.median().clamp_min(0.0).item()) if var.numel() else 0.0
        if bool(hard_mask.any().item()):
            v1_init = max(float(v1_init), float(cfg.survey_v1_min))
        else:
            v1_init = 0.0

    v1_schedule = [float(v1_init), 0.25 * float(v1_init), 0.0] if float(v1_init) > 0.0 else [0.0]
    remaining = max(int(cfg.max_iters) - warmup, 0)
    per_stage = _stage_iters(remaining)
    v1_schedule = v1_schedule[: len(per_stage)]

    phase2_iter = warmup
    for stage_idx, (v1, iters_stage) in enumerate(zip(v1_schedule, per_stage)):
        for _ in range(int(iters_stage)):
            gamma1_work = gamma1
            jac_trace_blocks = None
            for retry_idx in range(gamma_retry_steps + 1):
                mean_blocks, jac_trace_blocks = _apply_survey_on_hard_blocks(
                    r1_blocks=r1.reshape(B, nblocks, codesz),
                    gamma1=gamma1_work,
                    cb=cb,
                    hard_mask=hard_mask,
                    L=float(cfg.survey_L),
                    Q=int(cfg.survey_Q),
                    v1=float(v1),
                    blocks_per_chunk=int(cfg.blocks_per_chunk),
                    backend=str(backend),
                    prior_center_blocks=W_blocks if prior_tau > 0.0 else None,
                    prior_scale_blocks=prior_scale_blocks,
                    prior_tau=prior_tau,
                    prior_tau_scale_mode=prior_tau_scale_mode,
                )
                x1 = mean_blocks.reshape(B, n)
                alpha1_candidate = jac_trace_blocks.sum(dim=1, keepdim=True) / float(n)
                valid = torch.isfinite(alpha1_candidate) & (alpha1_candidate > alpha_eps) & (alpha1_candidate < 1.0 - alpha_eps)
                if bool(valid.all().item()):
                    alpha1 = alpha1_candidate
                    break
                if retry_idx >= gamma_retry_steps:
                    alpha1 = clip_alpha(alpha1_candidate, eps=alpha_eps)
                    break

                gamma1_work = torch.where(alpha1_candidate <= alpha_eps, gamma1_work / gamma_retry_factor, gamma1_work)
                gamma1_work = torch.where(alpha1_candidate >= 1.0 - alpha_eps, gamma1_work * gamma_retry_factor, gamma1_work)
                gamma1_work = clip_gamma(gamma1_work, min_value=float(cfg.gamma_min), max_value=float(cfg.gamma_max))

            assert jac_trace_blocks is not None
            eta1 = gamma1_work / alpha1
            gamma2_new = clip_gamma(eta1 - gamma1_work, min_value=float(cfg.gamma_min), max_value=float(cfg.gamma_max))
            r2_new = (eta1 * x1 - gamma1_work * r1) / gamma2_new

            if r2_prev is None or gamma2_prev is None or damping >= 1.0:
                r2 = r2_new
                gamma2 = gamma2_new
            else:
                r2 = (1.0 - damping) * r2_prev + damping * r2_new
                gamma2 = log_damp_positive(gamma2_prev, gamma2_new, damping)

            x2, alpha2_raw, _ = linear.apply(r2=r2, gamma2=gamma2, lambda_eff=lam)
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
                        "iter": float(phase2_iter),
                        "phase": 2.0,
                        "stage": float(stage_idx),
                        "v1": float(v1),
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
            phase2_iter += 1

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
        "survey": {
            "hard_frac": float(hard_mask.float().mean().item()) if hard_mask.numel() else 0.0,
            "v1_init": float(v1_init),
            "v1_schedule": [float(x) for x in v1_schedule],
        },
    }
