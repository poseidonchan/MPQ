from __future__ import annotations

import torch


@torch.inference_mode()
def clip_alpha(alpha: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor:
    eps = float(eps)
    if eps <= 0.0:
        raise ValueError(f"expected eps>0, got {eps}")
    alpha = torch.nan_to_num(alpha, nan=0.5, posinf=1.0 - eps, neginf=eps)
    return alpha.clamp(min=eps, max=1.0 - eps)


@torch.inference_mode()
def clip_gamma(gamma: torch.Tensor, *, min_value: float = 1e-8, max_value: float = 1e12) -> torch.Tensor:
    min_value = float(min_value)
    max_value = float(max_value)
    if not (0.0 < min_value < max_value):
        raise ValueError(f"expected 0 < min_value < max_value, got {min_value}, {max_value}")
    gamma = torch.nan_to_num(gamma, nan=min_value, posinf=max_value, neginf=min_value)
    return gamma.clamp(min=min_value, max=max_value)

