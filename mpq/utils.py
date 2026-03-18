from __future__ import annotations

import hashlib
import math
from typing import Optional

import torch


def stable_int_hash(key: str, *, bytes_: int = 8) -> int:
    digest = hashlib.sha256(str(key).encode("utf-8")).digest()
    if bytes_ <= 0:
        raise ValueError(f"bytes_ must be > 0, got {bytes_}.")
    bytes_ = min(bytes_, len(digest))
    return int.from_bytes(digest[:bytes_], byteorder="little", signed=False)


def stable_seed(key: str, *, base_seed: int = 0) -> int:
    # Keep it within 31-bit to match typical torch manual_seed expectations.
    extra = stable_int_hash(key, bytes_=4)
    return int((int(base_seed) + int(extra)) % (2**31 - 1))


@torch.inference_mode()
def log_damp_positive(prev: torch.Tensor, new: torch.Tensor, damping: float) -> torch.Tensor:
    damping = float(damping)
    if damping >= 1.0:
        return new
    if damping <= 0.0:
        return prev
    prev = prev.clamp_min(1e-12)
    new = new.clamp_min(1e-12)
    return torch.exp((1.0 - damping) * torch.log(prev) + damping * torch.log(new))


@torch.inference_mode()
def log_damp_positive_scalar(prev: float, new: float, damping: float) -> float:
    damping = float(damping)
    if damping >= 1.0:
        return float(new)
    if damping <= 0.0:
        return float(prev)
    prev = max(float(prev), 1e-12)
    new = max(float(new), 1e-12)
    return float(math.exp((1.0 - damping) * math.log(prev) + damping * math.log(new)))


def clamp_prob(x: torch.Tensor, *, eps: float = 1e-7) -> torch.Tensor:
    return x.clamp(min=float(eps), max=1.0 - float(eps))


def alpha_valid_mask(alpha: torch.Tensor, *, eps: float = 1e-7) -> torch.Tensor:
    return torch.isfinite(alpha) & (alpha > eps) & (alpha < 1.0 - eps)


def precision_valid_mask(gamma: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
    return torch.isfinite(gamma) & (gamma > eps)


def require_valid_precision(name: str, gamma: torch.Tensor, *, eps: float = 1e-12) -> None:
    if not bool(precision_valid_mask(gamma, eps=eps).all().item()):
        raise RuntimeError(f"{name} invalid: expected finite and > {eps}.")


def require_valid_alpha(name: str, alpha: torch.Tensor, *, eps: float = 1e-7) -> None:
    if not bool(alpha_valid_mask(alpha, eps=eps).all().item()):
        raise RuntimeError(f"{name} invalid: expected all in ({eps}, 1-{eps}).")


def safe_device_index(device: torch.device) -> Optional[int]:
    if device.type != "cuda":
        return None
    return 0 if device.index is None else int(device.index)

