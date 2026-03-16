from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class HardBlockSelectorConfig:
    gap_thresh: float = 0.5
    var_mult: float = 2.0
    max_frac: float = 0.25


@torch.inference_mode()
def select_hard_blocks(
    *,
    top2_gap: Optional[torch.Tensor],  # (B, n_blocks) or None
    var: torch.Tensor,  # (B, n_blocks) per-coordinate variance proxy (trace/8)
    cfg: HardBlockSelectorConfig,
) -> torch.Tensor:
    """Return boolean hard mask of shape (B, n_blocks)."""
    if var.ndim != 2:
        raise ValueError(f"expected var shape (B,n_blocks), got {tuple(var.shape)}")
    B, n_blocks = var.shape
    gap = top2_gap
    if gap is not None and gap.shape != (B, n_blocks):
        raise ValueError(f"expected top2_gap shape {(B, n_blocks)}, got {tuple(gap.shape)}")

    eps = 1e-12
    median_var = float(var.median().clamp_min(eps).item()) if var.numel() else 1.0
    hard_var = var > float(cfg.var_mult) * float(median_var)
    hard_gap = (gap < float(cfg.gap_thresh)) if gap is not None else torch.zeros_like(hard_var, dtype=torch.bool)
    hard = hard_var | hard_gap

    total = int(B * n_blocks)
    if total <= 0:
        return hard

    max_keep = int(float(cfg.max_frac) * float(total))
    max_keep = max(min(max_keep, total), 0)
    hard_count = int(hard.sum().item())
    if hard_count <= max_keep or max_keep <= 0:
        return hard

    # Keep the hardest by a combined score: large var and small gap.
    score = var / float(median_var)
    if gap is not None:
        score = score + (float(cfg.gap_thresh) - gap).clamp_min(0.0)
    score = score.reshape(-1)
    hard_flat = hard.reshape(-1)
    score = torch.where(hard_flat, score, torch.full_like(score, float("-inf")))

    topk = torch.topk(score, k=max_keep).indices
    keep = torch.zeros_like(hard_flat)
    keep[topk] = True
    return keep.reshape(B, n_blocks)

