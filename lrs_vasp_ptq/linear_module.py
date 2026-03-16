from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import torch

from mpgq.linear import linear_mmse_exact


@dataclass(frozen=True)
class LinearModuleStats:
    trace_avg: float
    rank: int


class SpectralLinearModule:
    """Rank-space linear module with Hessian trace-normalization.

    We assume H = V diag(eigs) V^T, with eigs = S^2 in the QUIP adapter.
    The solver operates on Hbar = H / (tr(H)/n) so that lambda_eff is more stable
    across layers/groups.
    """

    def __init__(self, *, V: torch.Tensor, Vh: torch.Tensor, S: torch.Tensor, w: torch.Tensor):
        if V.ndim != 2 or Vh.ndim != 2:
            raise ValueError(f"expected V,Vh 2D; got {tuple(V.shape)} {tuple(Vh.shape)}")
        if S.ndim != 1:
            raise ValueError(f"expected S 1D, got {tuple(S.shape)}")
        if w.ndim != 2:
            raise ValueError(f"expected w (B,n), got {tuple(w.shape)}")

        n = int(w.shape[1])
        rank = int(S.numel())
        if V.shape[0] != n or Vh.shape[1] != n:
            raise ValueError(f"V/Vh mismatch: n={n}, V={tuple(V.shape)}, Vh={tuple(Vh.shape)}")
        if V.shape[1] != rank or Vh.shape[0] != rank:
            raise ValueError(f"rank mismatch: S={tuple(S.shape)}, V={tuple(V.shape)}, Vh={tuple(Vh.shape)}")

        eigs = S.to(dtype=torch.float64).square()
        trace_avg = float(eigs.mean().clamp_min_(1e-30).item()) if eigs.numel() else 1.0
        if not (math.isfinite(trace_avg) and trace_avg > 0.0):
            raise ValueError(f"invalid trace_avg={trace_avg}")

        Sbar = (S.to(dtype=torch.float32) / float(trace_avg) ** 0.5).contiguous()
        Vsbar = (V.to(dtype=torch.float32) * Sbar.unsqueeze(0)).contiguous()
        Y_u = w.to(dtype=torch.float32).matmul(Vsbar)

        self._V = V.to(dtype=torch.float32)
        self._Vh = Vh.to(dtype=torch.float32)
        self._Sbar = Sbar
        self._Vsbar = Vsbar
        self._Y_u = Y_u
        self._stats = LinearModuleStats(trace_avg=float(trace_avg), rank=int(rank))

    @property
    def stats(self) -> LinearModuleStats:
        return self._stats

    @property
    def Vsbar(self) -> torch.Tensor:
        return self._Vsbar

    @torch.inference_mode()
    def apply(self, *, r2: torch.Tensor, gamma2: torch.Tensor, lambda_eff: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return linear_mmse_exact(
            r2.to(dtype=torch.float32),
            gamma2.to(dtype=torch.float32),
            Y_u=self._Y_u,
            V=self._V,
            Vh=self._Vh,
            S=self._Sbar,
            obs_precision=float(lambda_eff),
        )

