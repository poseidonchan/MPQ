from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class ProductGridCodebook:
    """8D product-grid codebook C = A^8.

    This object stores the 1D levels A. The exact RS denoiser factorizes across
    coordinates and does not require enumerating the full A^8 grid.
    """

    levels: torch.Tensor  # (L,)
    scale: float = 1.0
    zero: float = 0.0
    codesz: int = 8

    def atoms(self, *, dtype: torch.dtype = torch.float64, device: Optional[torch.device] = None) -> torch.Tensor:
        lev = self.levels.to(dtype=dtype, device=device)
        return lev * float(self.scale) + float(self.zero)

    def enumerate_grid(self, *, max_entries: int = 1_000_000, dtype: torch.dtype = torch.float64) -> torch.Tensor:
        """Enumerate the full A^8 grid (for brute-force reference only)."""
        levels = self.atoms(dtype=dtype)
        L = int(levels.numel())
        total = int(L**int(self.codesz))
        if total > int(max_entries):
            raise ValueError(f"refusing to enumerate product grid of size {total} (> {max_entries})")
        coords = [levels] * int(self.codesz)
        return torch.cartesian_prod(*coords)


@dataclass(frozen=True)
class FiniteE8PCodebook:
    """Finite 8D codebook backed by an explicit grid of reconstructed vectors."""

    grid: torch.Tensor  # (K, 8)
    scale: float = 1.0
    zero: float = 0.0
    codesz: int = 8
    dedup_unique: bool = True

    def reconstructed(self, *, dtype: torch.dtype = torch.float64, device: Optional[torch.device] = None) -> torch.Tensor:
        g = self.grid.to(dtype=dtype, device=device)
        g = g * float(self.scale) + float(self.zero)
        if bool(self.dedup_unique):
            g = torch.unique(g, dim=0)
        return g

    @staticmethod
    def from_quip(cb: object, *, dedup_unique: bool = True) -> "FiniteE8PCodebook":
        grid = getattr(cb, "grid", None)
        if grid is None:
            raise ValueError("expected QUIP codebook with .grid")
        grid_t = grid if isinstance(grid, torch.Tensor) else torch.as_tensor(grid)
        if grid_t.ndim != 2 or grid_t.shape[1] != 8:
            raise ValueError(f"expected grid shape (K,8), got {tuple(grid_t.shape)}")
        return FiniteE8PCodebook(grid=grid_t, dedup_unique=bool(dedup_unique))


def trace_normalize_eigs(eigs: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Return (eigs_normalized, trace_avg) where trace_avg = tr(H)/n."""
    if eigs.ndim != 1:
        raise ValueError(f"expected eigs as 1D tensor, got {tuple(eigs.shape)}")
    n = int(eigs.numel())
    trace_avg = float(eigs.mean().clamp_min_(1e-30).item()) if n else 1.0
    if not math.isfinite(trace_avg) or trace_avg <= 0.0:
        raise ValueError(f"invalid trace_avg={trace_avg}")
    return eigs / float(trace_avg), float(trace_avg)

