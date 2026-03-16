from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch


@dataclass(frozen=True)
class QuadratureRule:
    nodes: torch.Tensor  # (Q,) or (Q, k)
    weights: torch.Tensor  # (Q,)


def gauss_hermite_1d(Q: int, *, dtype: torch.dtype = torch.float64) -> QuadratureRule:
    """Return Gauss–Hermite nodes/weights for ∫ exp(-x^2) f(x) dx."""
    Q = int(Q)
    if Q <= 0:
        raise ValueError(f"expected Q>0, got {Q}")
    x, w = np.polynomial.hermite.hermgauss(Q)
    nodes = torch.as_tensor(x, dtype=dtype)
    weights = torch.as_tensor(w, dtype=dtype)
    return QuadratureRule(nodes=nodes, weights=weights)


def normal_1d(Q: int, *, var: float, dtype: torch.dtype = torch.float64) -> QuadratureRule:
    """Quadrature for z ~ N(0, var): ∫ N(z;0,var) f(z) dz."""
    var = float(var)
    if var < 0.0:
        raise ValueError(f"expected var>=0, got {var}")
    if var == 0.0:
        return QuadratureRule(nodes=torch.zeros((1,), dtype=dtype), weights=torch.ones((1,), dtype=dtype))

    gh = gauss_hermite_1d(Q, dtype=dtype)
    # For N(0,var): z = sqrt(2 var) x, weight = w / sqrt(pi)
    nodes = (np.sqrt(2.0 * var) * gh.nodes).to(dtype=dtype)
    weights = (gh.weights / np.sqrt(np.pi)).to(dtype=dtype)
    return QuadratureRule(nodes=nodes, weights=weights)


def normal_2d(Q: int, *, var: float, dtype: torch.dtype = torch.float64) -> QuadratureRule:
    """Product quadrature for z ~ N(0, var I_2)."""
    var = float(var)
    if var < 0.0:
        raise ValueError(f"expected var>=0, got {var}")
    if var == 0.0:
        return QuadratureRule(nodes=torch.zeros((1, 2), dtype=dtype), weights=torch.ones((1,), dtype=dtype))

    rule_1d = normal_1d(Q, var=var, dtype=dtype)
    x = rule_1d.nodes
    w = rule_1d.weights
    xx, yy = torch.meshgrid(x, x, indexing="ij")
    ww = torch.outer(w, w)
    nodes = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)
    weights = ww.reshape(-1)
    return QuadratureRule(nodes=nodes, weights=weights)

