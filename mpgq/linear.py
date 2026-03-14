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


def _block_vamp_matvec(v: torch.Tensor, Vs: torch.Tensor, gamma2_inv: torch.Tensor, inv_beta: float) -> torch.Tensor:
    """Implicit matvec: M @ v = (1/β)v + Vsᵀ (γ₂⁻¹ ⊙ (Vs v)), v is (R,) or (B, R)."""
    w = v.matmul(Vs.T)  # (..., n)
    w = w * gamma2_inv  # element-wise scale
    return inv_beta * v + w.matmul(Vs)  # (..., R)


def _block_vamp_cg_solve(
    rhs: torch.Tensor,
    Vs: torch.Tensor,
    gamma2_inv: torch.Tensor,
    inv_beta: float,
    max_steps: int = 20,
    tol: float = 1e-6,
) -> torch.Tensor:
    """Batched CG solve: M @ x = rhs, where rhs is (B, R)."""
    x = torch.zeros_like(rhs)
    r = rhs.clone()
    p = r.clone()
    rs_old = (r * r).sum(dim=-1, keepdim=True)  # (B, 1)

    for _ in range(max_steps):
        Ap = _block_vamp_matvec(p, Vs, gamma2_inv, inv_beta)  # (B, R)
        pAp = (p * Ap).sum(dim=-1, keepdim=True)  # (B, 1)
        alpha = rs_old / pAp.clamp_min(1e-30)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = (r * r).sum(dim=-1, keepdim=True)
        if float(rs_new.max().item()) < tol * tol:
            break
        p = r + (rs_new / rs_old.clamp_min(1e-30)) * p
        rs_old = rs_new

    return x


@torch.inference_mode()
def linear_mmse_block_vamp(
    r2: torch.Tensor,
    gamma2_blocks: torch.Tensor,
    *,
    Y_u: torch.Tensor,
    Vs: torch.Tensor,
    S: torch.Tensor,
    obs_precision: float,
    codesz: int = 8,
    cg_steps: int = 20,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Block-VAMP linear MMSE with per-block precision, CG-accelerated.

    Uses conjugate gradient with implicit matvec to avoid O(R³) Cholesky.
    Alpha2 uses a per-block diagonal approximation for speed.

    Shapes:
      r2: (B, n)
      gamma2_blocks: (K,) -- per-block precision, shared across rows
      Vs: (n, R) -- V @ diag(S)
      S: (R,) -- singular values
      Y_u: (B, R)

    Returns:
      x2: (B, n)
      alpha2_blocks: (K,) -- per-block sensitivity
      residual_u: (B, R)
    """
    B, n = r2.shape
    R = Vs.shape[1]
    K = int(n // codesz)
    beta = float(obs_precision)

    if R == 0:
        x2 = r2.clone()
        alpha2_blocks = torch.ones(K, dtype=torch.float32, device=r2.device)
        residual_u = Y_u.new_zeros((B, 0), dtype=torch.float32)
        return x2, alpha2_blocks, residual_u

    gamma2_blocks = gamma2_blocks.to(dtype=torch.float32, device=r2.device)
    inv_beta = 1.0 / beta

    # Build per-element gamma2_inv: (n,)
    gamma2_inv = (1.0 / gamma2_blocks.repeat_interleave(codesz).clamp_min(1e-12))

    # G = Gamma2_inv @ Vs
    G = Vs * gamma2_inv.unsqueeze(1)  # (n, R)

    # residual = Y_u - r2 @ Vs
    residual = Y_u.to(dtype=torch.float32) - r2.to(dtype=torch.float32).matmul(Vs)  # (B, R)

    # Solve M @ Z = residual^T via CG, then x2 = r2 + G @ Z
    # Equivalently: solve M @ Z = residual (B separate R-dim systems)
    Z = _block_vamp_cg_solve(residual, Vs, gamma2_inv, inv_beta, max_steps=cg_steps)  # (B, R)

    # x2 = r2 + Z @ G^T = r2 + (Z @ Vs^T) * gamma2_inv ... expand
    # Actually: x2 = r2 + G @ M⁻¹ @ residual^T  →  x2[b] = r2[b] + G @ Z[b]
    # Z[b] = M⁻¹ residual[b], so G @ Z[b] = G @ Z[b] which is (n,R)@(R,) = (n,)
    x2 = r2.to(dtype=torch.float32) + Z.matmul(G.T)  # (B, n)

    # Per-block alpha2: diagonal approximation
    # For each block k with precision gamma2_k:
    #   α₂_k ≈ 1 - (1/(8·γ₂_k)) Σᵢ∈block_k Σᵣ s²ᵣ Vs[i,r]² / (γ₂_k/β + s²ᵣ ||Vs[:,r]||² / γ̄₂)
    # Simplified: use per-block scalar approximation
    #   α₂_k = 1 - (1/8) Σᵢ∈block_k Σᵣ β·Vs[i,r]² / (γ₂_k + β·Σⱼ Vs[j,r]²/γ₂_block(j))
    # For speed, use the diagonal of M⁻¹: diag(M⁻¹)_r ≈ 1/(1/β + Σᵢ Vs[i,r]²/γ₂_block(i))
    # Then α₂_k = 1 - (1/(8γ₂_k)) Σᵢ∈block_k Σᵣ Vs[i,r]² · diag(M⁻¹)_r
    Vs_sq = Vs.square()  # (n, R)
    M_diag = inv_beta + (Vs_sq * gamma2_inv.unsqueeze(1)).sum(dim=0)  # (R,)
    M_diag_inv = 1.0 / M_diag.clamp_min(1e-12)  # (R,)
    # trace_per_dim[i] ≈ Vs[i,:]² · M_diag_inv
    trace_per_dim = (Vs_sq * M_diag_inv.unsqueeze(0)).sum(dim=1)  # (n,)
    trace_per_block = trace_per_dim.view(K, codesz).sum(dim=1)  # (K,)
    alpha2_blocks = (1.0 - trace_per_block / (float(codesz) * gamma2_blocks.clamp_min(1e-12))).clamp(min=1e-7, max=1.0 - 1e-7)

    # Residual after update
    residual_after = Y_u.to(dtype=torch.float32) - x2.matmul(Vs)  # (B, R)

    return x2, alpha2_blocks, residual_after


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
