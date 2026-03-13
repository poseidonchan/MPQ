from __future__ import annotations

import math
from typing import Tuple

import torch

from .utils import log_damp_positive_scalar


@torch.inference_mode()
def linear_mmse_exact(
    r2: torch.Tensor,
    gamma2: torch.Tensor,
    *,
    Y_u: torch.Tensor,
    V: torch.Tensor,
    Vh: torch.Tensor,
    S: torch.Tensor,
    obs_precision: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact linear MMSE module in rank space.

    Shapes:
      r2: (B, n)
      gamma2: (B, 1)
      V: (n, R)
      Vh: (R, n)
      S: (R,)
      Y_u: (B, R)

    Returns:
      x2: (B, n)
      alpha2: (B, 1)
      residual_u: (B, R) residual after applying x2
    """
    n = int(r2.shape[1])
    rank = int(S.numel())
    if rank == 0:
        x2 = r2.clone()
        alpha2 = torch.ones((r2.shape[0], 1), dtype=torch.float32, device=r2.device)
        residual_u = Y_u.new_zeros((r2.shape[0], 0), dtype=torch.float32)
        return x2, alpha2, residual_u

    gamma2 = gamma2.to(dtype=torch.float32)
    s = S.unsqueeze(0)  # (1, R)
    denom = float(obs_precision) * s.square() + gamma2  # (B, R)

    r2_v = r2.matmul(V)  # (B, R)
    residual_before = Y_u - r2_v * s
    coeff = float(obs_precision) * s / denom
    delta = (residual_before * coeff).matmul(Vh)
    x2 = r2 + delta

    alpha2 = 1.0 - (float(rank) / float(n)) + (gamma2 / float(n)) * (1.0 / denom).sum(dim=1, keepdim=True)
    residual_after = residual_before * (gamma2 / denom)
    return x2, alpha2, residual_after


@torch.inference_mode()
def em_update_obs_precision_rankspace(
    *,
    residual_u: torch.Tensor,
    gamma2: torch.Tensor,
    S: torch.Tensor,
    obs_precision: float,
    num_obs: int,
    damping: float = 0.3,
    min_value: float = 1e-4,
    max_value: float = 1e8,
) -> float:
    if int(S.numel()) == 0 or int(num_obs) <= 0:
        return float(obs_precision)

    s2 = S.square().unsqueeze(0)
    trace_term = (s2 / (float(obs_precision) * s2 + gamma2)).sum(dim=1, keepdim=True)
    resid_term = residual_u.square().sum(dim=1, keepdim=True)
    denom = (resid_term + trace_term).mean().clamp_min_(1e-12)
    beta_new = float(num_obs / denom.item())
    beta_old = max(float(obs_precision), 1e-12)
    beta = log_damp_positive_scalar(beta_old, beta_new, damping=float(damping))
    beta = min(max(beta, float(min_value)), float(max_value))
    return float(beta)


@torch.inference_mode()
def vamp_rankspace_error(
    *,
    Y_u: torch.Tensor,
    x: torch.Tensor,
    Vs: torch.Tensor,
) -> torch.Tensor:
    if Vs.numel() == 0:
        return torch.zeros(x.shape[0], dtype=torch.float32, device=x.device)
    return (Y_u - x.matmul(Vs)).square().sum(dim=1)


def stabilize_cholesky(
    gram: torch.Tensor,
    *,
    jitter: float = 1e-6,
    max_tries: int = 6,
) -> torch.Tensor:
    """Return lower Cholesky factor of a (near) SPD gram matrix, with diagonal jitter."""
    if gram.ndim != 2 or gram.shape[0] != gram.shape[1]:
        raise ValueError(f"expected square gram, got {tuple(gram.shape)}")

    diag = torch.diag(gram)
    floor = float(diag.abs().max().item()) * float(jitter) + float(jitter)
    eye = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    out = gram
    for k in range(max(int(max_tries), 1)):
        try:
            return torch.linalg.cholesky(out)
        except RuntimeError:
            out = out + (2.0**k) * floor * eye
    # Final attempt: raise with original error message.
    return torch.linalg.cholesky(out)
