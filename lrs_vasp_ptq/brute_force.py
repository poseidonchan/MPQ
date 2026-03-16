from __future__ import annotations

from typing import Any, Dict, Tuple

import torch


def _codebook_grid(codebook: Any, *, dtype: torch.dtype, device: torch.device, dedup_unique: bool) -> torch.Tensor:
    grid = getattr(codebook, "grid", None)
    if grid is None and hasattr(codebook, "reconstructed"):
        grid = codebook.reconstructed(dtype=dtype, device=device)  # type: ignore[assignment]
    if grid is None:
        raise ValueError("codebook must provide `.grid` or `.reconstructed()`")
    g = grid if isinstance(grid, torch.Tensor) else torch.as_tensor(grid)
    g = g.to(dtype=dtype, device=device)
    if g.ndim != 2 or g.shape[1] != 8:
        raise ValueError(f"expected codebook grid shape (K,8), got {tuple(g.shape)}")
    return torch.unique(g, dim=0) if bool(dedup_unique) else g


@torch.inference_mode()
def brute_force_posterior(
    center: torch.Tensor,
    gamma: torch.Tensor | float,
    codebook: Any,
    *,
    dedup_unique: bool = True,
    return_cov: bool = True,
    return_top2: bool = True,
) -> Dict[str, torch.Tensor]:
    """Exact posterior for a finite 8D codebook by enumerating all codewords.

    Posterior: p(c|r,gamma) ∝ exp(-gamma/2 * ||c-r||^2), c ∈ C.
    """
    center = torch.as_tensor(center)
    if center.shape[-1] != 8:
        raise ValueError(f"expected center[...,8], got {tuple(center.shape)}")

    device = center.device
    dtype = torch.float64
    grid = _codebook_grid(codebook, dtype=dtype, device=device, dedup_unique=dedup_unique)  # (K,8)

    flat = center.reshape(-1, 8).to(dtype=dtype)
    gamma_t = gamma if isinstance(gamma, torch.Tensor) else torch.tensor(float(gamma), device=device, dtype=dtype)
    gamma_flat = gamma_t.to(device=device, dtype=dtype).reshape(-1, 1)  # (B,1) broadcast
    if gamma_flat.shape[0] not in (1, flat.shape[0]):
        raise ValueError(f"gamma batch mismatch: center batch={flat.shape[0]}, gamma batch={gamma_flat.shape[0]}")
    if gamma_flat.shape[0] == 1 and flat.shape[0] > 1:
        gamma_flat = gamma_flat.expand(flat.shape[0], 1)

    diff = grid.unsqueeze(0) - flat.unsqueeze(1)  # (B,K,8)
    scores = -0.5 * gamma_flat * diff.square().sum(dim=-1)  # (B,K)
    logZ = torch.logsumexp(scores, dim=-1)  # (B,)
    logp = scores - logZ.unsqueeze(-1)
    p = torch.exp(logp)  # (B,K)

    mean = (p.unsqueeze(-1) * grid.unsqueeze(0)).sum(dim=1)  # (B,8)
    second = (p.unsqueeze(-1) * grid.square().unsqueeze(0)).sum(dim=1)  # (B,8)
    var = (second - mean.square()).clamp_min_(0.0)  # (B,8)

    if return_cov:
        w = (p.sqrt().unsqueeze(-1) * grid.unsqueeze(0))  # (B,K,8)
        eccT = torch.bmm(w.transpose(1, 2), w)  # (B,8,8)
        cov = eccT - mean.unsqueeze(-1) * mean.unsqueeze(-2)
    else:
        cov = torch.empty((flat.shape[0], 8, 8), device=device, dtype=dtype)

    map_idx = scores.argmax(dim=-1)  # (B,)
    map_val = grid.index_select(0, map_idx)  # (B,8)

    if return_top2:
        top2 = torch.topk(scores, k=min(2, scores.shape[-1]), dim=-1).values
        if top2.shape[-1] == 1:
            top2_gap = torch.full((flat.shape[0],), float("inf"), device=device, dtype=dtype)
        else:
            top2_gap = top2[:, 0] - top2[:, 1]
    else:
        top2_gap = torch.zeros((flat.shape[0],), device=device, dtype=dtype)

    out = {
        "logZ": logZ.reshape(center.shape[:-1]),
        "mean": mean.reshape(center.shape),
        "second_moment": second.reshape(center.shape),
        "cov": cov.reshape(center.shape[:-1] + (8, 8)),
        "var": var.reshape(center.shape),
        "map": map_val.reshape(center.shape),
        "top2_gap": top2_gap.reshape(center.shape[:-1]),
    }
    return out

