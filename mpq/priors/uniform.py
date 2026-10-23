from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import torch

try:
    from .triton_vamp import uniform_mmse_triton
except Exception:  # pragma: no cover - optional Triton dependency
    uniform_mmse_triton = None  # type: ignore[assignment]


_COMPILED_UNIFORM_MMSE_CORE: Optional[Callable[..., Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]] = None
_COMPILED_UNIFORM_MMSE_CORE_FAILED = False


def _disable_compiled_uniform_mmse_core() -> None:
    global _COMPILED_UNIFORM_MMSE_CORE, _COMPILED_UNIFORM_MMSE_CORE_FAILED
    _COMPILED_UNIFORM_MMSE_CORE = None
    _COMPILED_UNIFORM_MMSE_CORE_FAILED = True


def _get_compiled_uniform_mmse_core() -> Optional[Callable[..., Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]]:
    global _COMPILED_UNIFORM_MMSE_CORE, _COMPILED_UNIFORM_MMSE_CORE_FAILED
    if _COMPILED_UNIFORM_MMSE_CORE_FAILED:
        return None
    if _COMPILED_UNIFORM_MMSE_CORE is None:
        compile_fn = getattr(torch, "compile", None)
        if compile_fn is None:
            _COMPILED_UNIFORM_MMSE_CORE_FAILED = True
            return None
        try:
            _COMPILED_UNIFORM_MMSE_CORE = compile_fn(
                _uniform_mmse_core_compilable,
                mode="max-autotune-no-cudagraphs",
                fullgraph=False,
            )
        except Exception:
            _disable_compiled_uniform_mmse_core()
            return None
    return _COMPILED_UNIFORM_MMSE_CORE


@torch.inference_mode()
def candidate_tensor(maxq: int, *, device: torch.device) -> torch.Tensor:
    return torch.arange(int(maxq) + 1, dtype=torch.float32, device=device).view(1, 1, -1)


def _uniform_mmse_core_impl(
    *,
    r1: torch.Tensor,
    gamma1: torch.Tensor,
    scale_full: torch.Tensor,
    prior_center: torch.Tensor,
    atoms: torch.Tensor,
    prior_precision: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r1_f32 = r1.to(dtype=torch.float32)
    gamma1_f32 = gamma1.to(dtype=torch.float32)
    scale_f32 = scale_full.to(dtype=torch.float32).clamp_min_(1e-12)
    prior_center_f32 = prior_center.to(dtype=torch.float32)
    atoms_f32 = atoms.to(dtype=torch.float32)
    prior_precision_f32 = prior_precision.to(dtype=torch.float32)

    scores = -0.5 * gamma1_f32.unsqueeze(-1) * (atoms_f32 - r1_f32.unsqueeze(-1)).square()
    inv_tau2 = prior_precision_f32.view(1, 1, 1) / scale_f32.square().unsqueeze(-1)
    scores = scores - 0.5 * inv_tau2 * (atoms_f32 - prior_center_f32.unsqueeze(-1)).square()

    probs = torch.softmax(scores, dim=-1)
    mean = (probs * atoms_f32).sum(dim=-1)
    second = (probs * atoms_f32.square()).sum(dim=-1)
    var = (second - mean.square()).clamp_min_(0.0)

    map_idx_long = torch.argmax(scores, dim=-1)
    map_value = torch.gather(atoms_f32, dim=-1, index=map_idx_long.unsqueeze(-1)).squeeze(-1)
    alpha1 = gamma1_f32 * var.mean(dim=1, keepdim=True)
    return mean, var, map_value, map_idx_long, alpha1


def _uniform_mmse_core_compilable(
    *,
    r1: torch.Tensor,
    gamma1: torch.Tensor,
    scale_full: torch.Tensor,
    prior_center: torch.Tensor,
    atoms: torch.Tensor,
    prior_precision: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return _uniform_mmse_core_impl(
        r1=r1,
        gamma1=gamma1,
        scale_full=scale_full,
        prior_center=prior_center,
        atoms=atoms,
        prior_precision=prior_precision,
    )


@torch.inference_mode()
def discrete_uniform_mmse(
    *,
    r1: torch.Tensor,  # (B, n)
    gamma1: torch.Tensor,  # (B, 1)
    scale_full: torch.Tensor,  # (B, n)
    zero_full: torch.Tensor,  # (B, n)
    maxq: int,
    prior_center: torch.Tensor,  # (B, n)
    prior_tau: float,
    need_stats: bool,
    atoms: torch.Tensor | None = None,  # (B, n, L) optional precomputed atoms
    vamp_backend: str = "auto",
) -> Dict[str, torch.Tensor | float | str]:
    """Exact discrete MMSE over a uniform scalar quantizer grid per entry."""
    backend = str(vamp_backend or "auto").strip().lower()
    if backend not in {"eager", "compile", "triton", "auto"}:
        raise ValueError(
            f"unsupported vamp_backend={vamp_backend!r}; expected eager|compile|triton|auto."
        )
    if backend == "triton" and need_stats:
        raise ValueError("Triton uniform VAMP backend does not support posterior diagnostics.")

    if not need_stats and backend in {"triton", "auto"}:
        if r1.is_cuda and uniform_mmse_triton is not None:
            try:
                return uniform_mmse_triton(
                    r1=r1,
                    gamma1=gamma1,
                    scale_full=scale_full,
                    zero_full=zero_full,
                    maxq=int(maxq),
                    prior_center=prior_center,
                    prior_tau=float(prior_tau),
                )
            except Exception:
                if backend == "triton":
                    raise
        elif backend == "triton":
            raise RuntimeError("Triton uniform VAMP backend requested but CUDA/Triton is not available.")

    if atoms is None:
        candidates = candidate_tensor(maxq, device=r1.device)  # (1,1,L)
        atoms = scale_full.to(dtype=torch.float32).unsqueeze(-1) * (
            candidates - zero_full.to(dtype=torch.float32).unsqueeze(-1)
        )  # (B,n,L)
    gamma1 = gamma1.to(dtype=torch.float32)

    if not need_stats and r1.is_cuda and backend in {"compile", "auto"}:
        prior_precision = torch.tensor(
            0.0 if float(prior_tau) <= 0.0 else (1.0 / (float(prior_tau) * float(prior_tau))),
            dtype=torch.float32,
            device=r1.device,
        )
        compiled_fn = _get_compiled_uniform_mmse_core()
        if compiled_fn is not None:
            try:
                mean, var, map_value, map_idx_long, alpha1 = compiled_fn(
                    r1=r1,
                    gamma1=gamma1,
                    scale_full=scale_full,
                    prior_center=prior_center,
                    atoms=atoms,
                    prior_precision=prior_precision,
                )
                return {
                    "mean": mean.to(dtype=torch.float32),
                    "var": var.to(dtype=torch.float32),
                    "map_value": map_value.to(dtype=torch.float32),
                    "map_idx": map_idx_long.to(dtype=torch.long),
                    "alpha1": alpha1.to(dtype=torch.float32),
                    "entropy": 0.0,
                    "margin": 0.0,
                    "vamp_backend": "compile",
                }
            except Exception:
                _disable_compiled_uniform_mmse_core()

    scores = -0.5 * gamma1.unsqueeze(-1) * (atoms - r1.to(dtype=torch.float32).unsqueeze(-1)).square()
    if float(prior_tau) > 0.0:
        scaled_tau2 = (float(prior_tau) * scale_full.to(dtype=torch.float32).clamp_min(1e-12)).square().unsqueeze(-1)
        scores = scores - 0.5 * (atoms - prior_center.to(dtype=torch.float32).unsqueeze(-1)).square() / scaled_tau2

    log_probs = scores - torch.logsumexp(scores, dim=-1, keepdim=True)
    probs = torch.exp(log_probs)
    mean = (probs * atoms).sum(dim=-1)
    second = (probs * atoms.square()).sum(dim=-1)
    var = (second - mean.square()).clamp_min_(0.0)

    map_idx_long = torch.argmax(scores, dim=-1)
    map_value = torch.gather(atoms, dim=-1, index=map_idx_long.unsqueeze(-1)).squeeze(-1)

    alpha1 = gamma1 * var.mean(dim=1, keepdim=True)

    if need_stats:
        top2 = torch.topk(scores, k=min(2, scores.shape[-1]), dim=-1).values
        entropy = float((-(probs * log_probs).sum(dim=-1)).mean().item())
        if top2.shape[-1] == 1:
            margin = float("inf")
        else:
            margin = float((top2[..., 0] - top2[..., 1]).mean().item())
    else:
        entropy = 0.0
        margin = 0.0

    return {
        "mean": mean.to(dtype=torch.float32),
        "var": var.to(dtype=torch.float32),
        "map_value": map_value.to(dtype=torch.float32),
        "map_idx": map_idx_long.to(dtype=torch.long),
        "alpha1": alpha1.to(dtype=torch.float32),
        "entropy": float(entropy),
        "margin": float(margin),
        "vamp_backend": "eager",
    }


@torch.inference_mode()
def discrete_uniform_topk(
    *,
    r1: torch.Tensor,  # (B, n)
    gamma1: torch.Tensor,  # (B, 1)
    scale_full: torch.Tensor,  # (B, n)
    zero_full: torch.Tensor,  # (B, n)
    maxq: int,
    prior_center: torch.Tensor,  # (B, n)
    prior_tau: float,
    k: int,
    atoms: torch.Tensor | None = None,  # (B, n, L) optional precomputed atoms
) -> Dict[str, torch.Tensor]:
    """Return per-entry top-k atoms under the same exact scalar posterior used by VAMP."""
    if atoms is None:
        candidates = candidate_tensor(maxq, device=r1.device)  # (1,1,L)
        atoms = scale_full.to(dtype=torch.float32).unsqueeze(-1) * (
            candidates - zero_full.to(dtype=torch.float32).unsqueeze(-1)
        )  # (B,n,L)
    gamma1 = gamma1.to(dtype=torch.float32)

    scores = -0.5 * gamma1.unsqueeze(-1) * (atoms - r1.to(dtype=torch.float32).unsqueeze(-1)).square()
    if float(prior_tau) > 0.0:
        scaled_tau2 = (float(prior_tau) * scale_full.to(dtype=torch.float32).clamp_min(1e-12)).square().unsqueeze(-1)
        scores = scores - 0.5 * (atoms - prior_center.to(dtype=torch.float32).unsqueeze(-1)).square() / scaled_tau2

    k_eff = max(1, min(int(k), int(scores.shape[-1])))
    topk_scores, topk_q = torch.topk(scores, k=k_eff, dim=-1)
    topk_values = torch.gather(atoms, dim=-1, index=topk_q)
    return {
        "topk_q": topk_q.to(dtype=torch.long),
        "topk_values": topk_values.to(dtype=torch.float32),
        "topk_scores": topk_scores.to(dtype=torch.float32),
    }


@torch.inference_mode()
def uniform_map_topk(
    *,
    r1: torch.Tensor,  # (B, n)
    gamma1: torch.Tensor,  # (B, 1)
    scale_full: torch.Tensor,  # (B, n)
    zero_full: torch.Tensor,  # (B, n)
    maxq: int,
    prior_center: torch.Tensor,  # (B, n)
    prior_tau: float,
    k: int,
    atoms: torch.Tensor | None = None,  # (B, n, L) optional precomputed atoms
) -> Dict[str, torch.Tensor]:
    if atoms is None:
        candidates = candidate_tensor(maxq, device=r1.device)  # (1,1,L)
        atoms = scale_full.to(dtype=torch.float32).unsqueeze(-1) * (
            candidates - zero_full.to(dtype=torch.float32).unsqueeze(-1)
        )  # (B,n,L)
    gamma1 = gamma1.to(dtype=torch.float32)

    scores = -0.5 * gamma1.unsqueeze(-1) * (atoms - r1.to(dtype=torch.float32).unsqueeze(-1)).square()
    if float(prior_tau) > 0.0:
        scaled_tau2 = (float(prior_tau) * scale_full.to(dtype=torch.float32).clamp_min(1e-12)).square().unsqueeze(-1)
        scores = scores - 0.5 * (atoms - prior_center.to(dtype=torch.float32).unsqueeze(-1)).square() / scaled_tau2

    k_eff = min(max(int(k), 1), int(scores.shape[-1]))
    top_scores, top_q = torch.topk(scores, k=k_eff, dim=-1)
    top_values = torch.gather(atoms, dim=-1, index=top_q)
    return {
        "topk_q": top_q.to(dtype=torch.long),
        "topk_values": top_values.to(dtype=torch.float32),
        "topk_scores": top_scores.to(dtype=torch.float32),
    }


@torch.inference_mode()
def nearest_atom_idx(
    *,
    x: torch.Tensor,  # (B, n)
    scale_full: torch.Tensor,  # (B, n)
    zero_full: torch.Tensor,  # (B, n)
    maxq: int,
) -> torch.Tensor:
    scale = scale_full.to(dtype=torch.float32).clamp_min_(1e-12)
    zero = zero_full.to(dtype=torch.float32)
    q = torch.round(x.to(dtype=torch.float32) / scale + zero)
    q = torch.clamp(q, 0, int(maxq)).to(dtype=torch.int32)
    return q


@torch.inference_mode()
def dequant_from_q(
    *,
    q_int: torch.Tensor,  # (B, n) int32
    scale_full: torch.Tensor,
    zero_full: torch.Tensor,
) -> torch.Tensor:
    return scale_full.to(dtype=torch.float32) * (q_int.to(dtype=torch.float32) - zero_full.to(dtype=torch.float32))


def summarize_post_stats(stats: List[float]) -> float:
    if not stats:
        return 0.0
    return float(sum(stats) / len(stats))
