from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch

from ..brute_force import brute_force_posterior
from ..codebooks import FiniteE8PCodebook


@torch.inference_mode()
def rs_posterior_blocks(
    *,
    center_blocks: torch.Tensor,  # (B, n_blocks, 8)
    gamma: torch.Tensor,  # (B, 1) or broadcastable to (B,1)
    cb: Any,
    blocks_per_chunk: int = 64,
    prior_center_blocks: Optional[torch.Tensor] = None,
    prior_scale_blocks: Optional[torch.Tensor] = None,
    prior_tau: float = 0.0,
    prior_tau_scale_mode: str = "absmax",
    return_var_diag: bool = False,
    return_logZ: bool = False,
    return_top2: bool = False,
    backend: str = "factorized",
) -> Dict[str, torch.Tensor]:
    """RS posterior moments for finite E8P12 codebook.

    backend:
      - "factorized": uses mpgq.priors.e8p12 factorized exact backend (fast)
      - "bruteforce": enumerates the full codebook grid (reference)
    """
    center_blocks = torch.as_tensor(center_blocks)
    if center_blocks.ndim != 3 or center_blocks.shape[-1] != 8:
        raise ValueError(f"expected center_blocks shape (B,n_blocks,8), got {tuple(center_blocks.shape)}")
    gamma = torch.as_tensor(gamma, device=center_blocks.device, dtype=torch.float32)
    if gamma.ndim == 0:
        gamma = gamma.view(1, 1).expand(center_blocks.shape[0], 1)
    if gamma.ndim != 2 or gamma.shape[1] != 1:
        raise ValueError(f"expected gamma shape (B,1), got {tuple(gamma.shape)}")

    mode = str(backend).strip().lower()
    if mode == "factorized":
        from mpgq.priors.e8p12 import vamp_vector_discrete_mmse_e8p12

        need_stats = bool(return_top2)
        out = vamp_vector_discrete_mmse_e8p12(
            r1_blocks=center_blocks,
            gamma1=gamma,
            cb=cb,
            blocks_per_chunk=int(blocks_per_chunk),
            need_stats=need_stats,
            need_var_diag=bool(return_var_diag),
            prior_center_blocks=prior_center_blocks,
            prior_scale_blocks=prior_scale_blocks,
            prior_tau=float(prior_tau),
            prior_tau_scale_mode=str(prior_tau_scale_mode),
            return_logZ_blocks=bool(return_logZ),
            return_margin_blocks=bool(return_top2),
        )
        mean = out["x1_blocks"].to(dtype=torch.float32)  # type: ignore[assignment]
        var_trace_over8 = out["var1"].to(dtype=torch.float32)  # type: ignore[assignment]
        cov_trace = (8.0 * var_trace_over8).to(dtype=torch.float32)
        res: Dict[str, torch.Tensor] = {
            "mean": mean,
            "cov_trace": cov_trace,
            "map": out["map_vals"].to(dtype=torch.float32),  # type: ignore[assignment]
            "map_idxs": out["map_idxs"].to(dtype=torch.long),  # type: ignore[assignment]
        }
        if bool(return_var_diag):
            var_diag = out.get("var1_diag")
            if var_diag is None:
                raise RuntimeError("backend did not return var1_diag")
            res["var_diag"] = var_diag.to(dtype=torch.float32)  # type: ignore[assignment]
        if bool(return_logZ):
            logZ = out.get("logZ_blocks")
            if logZ is None:
                raise RuntimeError("backend did not return logZ_blocks")
            res["logZ"] = logZ.to(dtype=torch.float32)  # type: ignore[assignment]
        if bool(return_top2):
            gap = out.get("margin_blocks")
            if gap is None:
                raise RuntimeError("backend did not return margin_blocks")
            res["top2_gap"] = gap.to(dtype=torch.float32)  # type: ignore[assignment]
        return res

    if mode == "bruteforce":
        if float(prior_tau) > 0.0 and prior_center_blocks is not None:
            raise ValueError("bruteforce backend does not support prior_tau; set prior_tau=0 for reference checks")
        codebook = FiniteE8PCodebook.from_quip(cb, dedup_unique=True)
        flat = center_blocks.reshape(-1, 8)
        # Repeat per-block gamma (broadcast from per-row gamma).
        gamma_flat = gamma.repeat_interleave(center_blocks.shape[1], dim=0).reshape(-1, 1)
        post = brute_force_posterior(flat, gamma_flat, codebook, dedup_unique=True, return_cov=False, return_top2=bool(return_top2))
        mean = post["mean"].to(dtype=torch.float32).reshape_as(center_blocks)
        second = post["second_moment"].to(dtype=torch.float32).reshape_as(center_blocks)
        var_diag = (second - mean.square()).clamp_min_(0.0)
        cov_trace = var_diag.sum(dim=-1)
        # MAP / idx: brute_force_posterior returns map value but not idx; we recompute idx cheaply.
        grid = codebook.reconstructed(dtype=torch.float64, device=flat.device)  # (K,8)
        diff = grid.unsqueeze(0) - flat.to(dtype=torch.float64).unsqueeze(1)
        scores = -0.5 * gamma_flat.to(dtype=torch.float64) * diff.square().sum(dim=-1)
        map_idx = scores.argmax(dim=-1).to(dtype=torch.long)
        map_val = grid.index_select(0, map_idx).to(dtype=torch.float32).reshape_as(center_blocks)
        res = {
            "mean": mean,
            "cov_trace": cov_trace.to(dtype=torch.float32),
            "map": map_val,
            "map_idxs": map_idx.reshape(center_blocks.shape[:-1]),
        }
        if bool(return_var_diag):
            res["var_diag"] = var_diag.to(dtype=torch.float32)
        if bool(return_logZ):
            res["logZ"] = post["logZ"].to(dtype=torch.float32).reshape(center_blocks.shape[:-1])
        if bool(return_top2):
            res["top2_gap"] = post["top2_gap"].to(dtype=torch.float32).reshape(center_blocks.shape[:-1])
        return res

    raise ValueError(f"unknown backend={backend!r} (expected 'factorized'|'bruteforce')")
