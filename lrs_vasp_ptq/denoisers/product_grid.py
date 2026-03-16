from __future__ import annotations

from typing import Dict

import torch

from ..codebooks import ProductGridCodebook


@torch.inference_mode()
def rs_posterior_product_grid(
    center: torch.Tensor,
    gamma: torch.Tensor | float,
    codebook: ProductGridCodebook,
    *,
    return_cov: bool = True,
    return_top2: bool = True,
) -> Dict[str, torch.Tensor]:
    """Exact RS denoiser for A^8 by coordinate factorization."""
    center = torch.as_tensor(center)
    if center.shape[-1] != int(codebook.codesz):
        raise ValueError(f"expected center[...,{codebook.codesz}], got {tuple(center.shape)}")

    atoms = codebook.atoms(dtype=torch.float64, device=center.device)  # (L,)
    r = center.to(dtype=torch.float64)
    g = gamma if isinstance(gamma, torch.Tensor) else torch.tensor(float(gamma), device=center.device, dtype=torch.float64)
    g = g.to(dtype=torch.float64, device=center.device)

    flat = r.reshape(-1, int(codebook.codesz))  # (B,8)
    g_flat = g.reshape(-1, 1)  # (Bg,1)
    if g_flat.shape[0] not in (1, flat.shape[0]):
        raise ValueError(f"gamma batch mismatch: center batch={flat.shape[0]}, gamma batch={g_flat.shape[0]}")
    if g_flat.shape[0] == 1 and flat.shape[0] > 1:
        g_flat = g_flat.expand(flat.shape[0], 1)

    # Scores per coordinate: (B,8,L)
    diff = atoms.view(1, 1, -1) - flat.unsqueeze(-1)
    scores = -0.5 * g_flat.unsqueeze(-1) * diff.square()
    logZ_i = torch.logsumexp(scores, dim=-1)  # (B,8)
    logp = scores - logZ_i.unsqueeze(-1)
    p = torch.exp(logp)

    mean = (p * atoms.view(1, 1, -1)).sum(dim=-1)  # (B,8)
    second = (p * atoms.square().view(1, 1, -1)).sum(dim=-1)  # (B,8)
    var = (second - mean.square()).clamp_min_(0.0)  # (B,8)

    # MAP is coordinate-wise MAP.
    map_idx = scores.argmax(dim=-1)  # (B,8)
    map_val = atoms.index_select(0, map_idx.reshape(-1)).reshape_as(mean)  # (B,8)

    # Full-grid logZ factorizes.
    logZ = logZ_i.sum(dim=-1)  # (B,)

    if return_top2:
        top2 = torch.topk(scores, k=min(2, scores.shape[-1]), dim=-1).values  # (B,8,k)
        if top2.shape[-1] == 1:
            top2_gap = torch.full((flat.shape[0],), float("inf"), device=center.device, dtype=torch.float64)
        else:
            gaps = top2[..., 0] - top2[..., 1]  # (B,8)
            top2_gap = gaps.min(dim=-1).values  # (B,)
    else:
        top2_gap = torch.zeros((flat.shape[0],), device=center.device, dtype=torch.float64)

    if return_cov:
        cov = torch.diag_embed(var)  # (B,8,8)
    else:
        cov = torch.empty((flat.shape[0], int(codebook.codesz), int(codebook.codesz)), device=center.device, dtype=torch.float64)

    out = {
        "logZ": logZ.reshape(center.shape[:-1]),
        "mean": mean.reshape_as(r),
        "second_moment": second.reshape_as(r),
        "cov": cov.reshape(center.shape[:-1] + (int(codebook.codesz), int(codebook.codesz))),
        "var": var.reshape_as(r),
        "map": map_val.reshape_as(r),
        "top2_gap": top2_gap.reshape(center.shape[:-1]),
    }
    return out

