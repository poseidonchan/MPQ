from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch

from ..solver import E8P12VampConfig, solve_e8p12_rankspace_vamp
from ..utils import stable_seed


@dataclass(frozen=True)
class QuipAdapterConfig:
    block_rows: int = 32
    jitter_base: float = 1e-6
    eigen_floor_scale: float = 1e-6


@torch.inference_mode()
def _vamp_hessian_eigen(
    H: torch.Tensor,
    *,
    jitter_base: float = 1e-6,
    floor_scale: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, int, float]:
    """Eigendecomposition of Hessian H for the linear MMSE step.

    Returns (eigenvalues, eigvecs, rank, eigen_floor) where:
      eigenvalues: (n,)      floored eigenvalues (descending)
      eigvecs:     (n, n)    orthonormal columns
      rank:        nominal rank (= n after flooring)
      eigen_floor: scalar clamp floor used for stability
    """
    n = int(H.shape[0])
    H_f64 = H.to(dtype=torch.float64)
    H_f64 = 0.5 * (H_f64 + H_f64.T)
    lam, V = torch.linalg.eigh(H_f64)  # ascending order
    lam = lam.to(dtype=torch.float32)
    V = V.to(dtype=torch.float32)
    max_lam = float(lam.max().item()) if lam.numel() else 0.0
    diag_mean = float(torch.diagonal(H_f64).mean().abs().item()) if H_f64.numel() else 0.0
    eigen_floor = max(float(jitter_base), float(floor_scale) * max(diag_mean, max_lam, 1e-12))
    lam = lam.clamp_min_(eigen_floor)
    lam = lam.flip(0).contiguous()
    V = V.flip(1).contiguous()
    return lam, V, n, float(eigen_floor)


@torch.inference_mode()
def _exact_surrogate_rows(Wr: torch.Tensor, hatWr: torch.Tensor, Hr: torch.Tensor) -> torch.Tensor:
    delta = (Wr - hatWr).to(dtype=torch.float32)
    proj = delta.matmul(Hr.to(dtype=torch.float32))
    return (proj * delta).sum(dim=1)


@torch.inference_mode()
def _exact_surrogate_mean(
    Wr: torch.Tensor,
    hatWr: torch.Tensor,
    Hr: torch.Tensor,
    *,
    block_rows: int,
) -> float:
    m = int(Wr.shape[0])
    block_rows = _resolve_block_rows(m, block_rows)
    accum = 0.0
    count = 0
    for row_start in range(0, m, block_rows):
        row_end = min(row_start + block_rows, m)
        scores = _exact_surrogate_rows(Wr[row_start:row_end], hatWr[row_start:row_end], Hr)
        accum += float(scores.sum().item())
        count += int(scores.numel())
    return float(accum / max(count, 1))


@torch.inference_mode()
def _exact_surrogate_chunk_means(
    Wr: torch.Tensor,
    hatWr: torch.Tensor,
    Hr: torch.Tensor,
    *,
    block_rows: int,
) -> Tuple[float, list[float]]:
    """Return (overall_mean, per_chunk_means) over row chunks of size block_rows."""
    m = int(Wr.shape[0])
    block_rows = _resolve_block_rows(m, block_rows)
    total_sum = 0.0
    total_count = 0
    chunk_means: list[float] = []
    for row_start in range(0, m, block_rows):
        row_end = min(row_start + block_rows, m)
        scores = _exact_surrogate_rows(Wr[row_start:row_end], hatWr[row_start:row_end], Hr)
        chunk_means.append(float(scores.mean().item()) if scores.numel() else 0.0)
        total_sum += float(scores.sum().item())
        total_count += int(scores.numel())
    return float(total_sum / max(total_count, 1)), chunk_means


@torch.inference_mode()
def _resolve_block_rows(rows: int, requested: int) -> int:
    return min(int(rows), max(int(requested), 1))


@torch.inference_mode()
def _random_e8p12_init(
    *,
    cb: Any,
    rows: int,
    nblocks: int,
    seed_key: str,
    base_seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    codesz = int(getattr(cb, "codesz", 8))
    grid = cb.grid.to(device=device, dtype=torch.float32)
    codebook_size = int(grid.shape[0])
    gen = torch.Generator(device="cpu")
    gen.manual_seed(stable_seed(seed_key, base_seed=int(base_seed)))
    idxs = torch.randint(0, codebook_size, (int(rows), int(nblocks)), generator=gen, dtype=torch.int64)
    idxs = idxs.to(device=device, dtype=getattr(cb, "idx_dtype", torch.int64), non_blocking=False)
    vals = grid.index_select(0, idxs.reshape(-1)).view(int(rows), int(nblocks), int(codesz))
    return vals.to(dtype=dtype).reshape(int(rows), int(nblocks) * int(codesz)), idxs


@torch.inference_mode()
def mpgq_refine_codebook(
    Wr: torch.Tensor,
    Hr: torch.Tensor,
    hatWr0: torch.Tensor,
    Qidxs0: torch.Tensor,
    cb: Any,
    args: Any,
    *,
    module_name: str = "",
    adapter_cfg: QuipAdapterConfig = QuipAdapterConfig(),
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """QUIP# adapter: refine codebook decisions via MPGQ (VAMP + greedy polish)."""
    start = time.time()
    Wr = Wr.to(dtype=torch.float32)
    Hr = Hr.to(dtype=torch.float32)
    device = Wr.device
    m, n = Wr.shape
    codesz = int(getattr(cb, "codesz", 8))
    if codesz != 8 or (n % codesz) != 0:
        raise ValueError(f"mpgq QUIP adapter expects E8P12 codesz=8; got codesz={codesz}, n={n}.")
    nblocks = n // codesz

    init_mode = str(getattr(args, "mpgq_init", getattr(args, "wmpq_init", "random")) or "random").strip().lower()
    seed = int(getattr(args, "seed", getattr(args, "mpgq_seed", 0)) or 0)

    if init_mode == "random":
        hatWr_init, Qidxs_init = _random_e8p12_init(
            cb=cb,
            rows=m,
            nblocks=nblocks,
            seed_key=f"{module_name}:mpgq_init",
            base_seed=seed,
            device=device,
            dtype=Wr.dtype,
        )
    else:
        hatWr_init = hatWr0.to(dtype=torch.float32)
        Qidxs_init = Qidxs0.to(dtype=getattr(cb, "idx_dtype", torch.int64))

    eigenvalues, V, rank, eigen_floor = _vamp_hessian_eigen(
        Hr,
        jitter_base=float(adapter_cfg.jitter_base),
        floor_scale=float(adapter_cfg.eigen_floor_scale),
    )
    V = V.to(device=device, dtype=torch.float32)
    Vh = V.T.contiguous()
    eigenvalues = eigenvalues.to(device=device, dtype=torch.float32)
    S = torch.sqrt(eigenvalues).contiguous()
    Vs = V * S.unsqueeze(0)

    # Warm-start obs precision from init mismatch in rank space, unless fixed.
    requested_obs = float(getattr(args, "mpgq_obs_precision", getattr(args, "wmpq_obs_precision", 0.0)) or 0.0)
    if requested_obs > 0.0:
        obs_precision = requested_obs
    else:
        block_rows = _resolve_block_rows(m, int(getattr(args, "mpgq_block_rows", getattr(args, "wmpq_block_rows", adapter_cfg.block_rows)) or adapter_cfg.block_rows))
        sum_sq = 0.0
        count = 0
        for row_start in range(0, m, block_rows):
            row_end = min(row_start + block_rows, m)
            delta = Wr[row_start:row_end] - hatWr_init[row_start:row_end]
            diff_u = delta.matmul(Vs)
            sum_sq += float(diff_u.square().sum().item())
            count += int(diff_u.numel())
        mse = max(sum_sq / max(count, 1), 1e-12)
        obs_precision = 1.0 / float(mse)

    vamp_cfg = E8P12VampConfig(
        iters=int(getattr(args, "mpgq_iters", getattr(args, "wmpq_iters", 4)) or 4),
        damping=float(getattr(args, "mpgq_damping", getattr(args, "wmpq_damping", 0.5)) or 0.5),
        x1_mode=str(getattr(args, "mpgq_vamp_x1_mode", getattr(args, "wmpq_vamp_x1_mode", "map")) or "map"),
        map_mix=float(getattr(args, "mpgq_vamp_map_mix", getattr(args, "wmpq_vamp_map_mix", 0.5)) or 0.5),
        gamma_start=float(getattr(args, "mpgq_gamma_start", getattr(args, "wmpq_gamma_start", 1.0)) or 1.0),
        gamma_retry_factor=float(getattr(args, "mpgq_gamma_retry_factor", getattr(args, "wmpq_gamma_retry_factor", 4.0)) or 4.0),
        gamma_retry_steps=int(getattr(args, "mpgq_gamma_retry_steps", getattr(args, "wmpq_gamma_retry_steps", 3)) or 3),
        prior_tau=float(getattr(args, "mpgq_prior_tau", getattr(args, "wmpq_prior_tau", 0.0)) or 0.0),
        prior_tau_scale_mode=str(getattr(args, "mpgq_prior_tau_scale_mode", getattr(args, "wmpq_prior_tau_scale_mode", "absmax")) or "absmax"),
        obs_precision=float(requested_obs),
        obs_precision_end=(
            None
            if getattr(args, "mpgq_obs_precision_end", getattr(args, "wmpq_obs_precision_end", None)) is None
            else float(getattr(args, "mpgq_obs_precision_end", getattr(args, "wmpq_obs_precision_end", None)))
        ),
        em_obs_precision=bool(getattr(args, "mpgq_em_obs_precision", getattr(args, "wmpq_em_obs_precision", True))),
        em_obs_precision_damping=float(getattr(args, "mpgq_em_obs_precision_damping", getattr(args, "wmpq_em_obs_precision_damping", 0.3)) or 0.3),
        em_obs_precision_min=float(getattr(args, "mpgq_em_obs_precision_min", getattr(args, "wmpq_em_obs_precision_min", 1e-4)) or 1e-4),
        em_obs_precision_max=float(getattr(args, "mpgq_em_obs_precision_max", getattr(args, "wmpq_em_obs_precision_max", 1e8)) or 1e8),
        best_iterate_export=bool(getattr(args, "mpgq_best_iterate_export", getattr(args, "wmpq_best_iterate_export", True))),
        adaptive_accept_reject=bool(getattr(args, "mpgq_adaptive_accept_reject", getattr(args, "wmpq_adaptive_accept_reject", True))),
        adaptive_retry_steps=int(getattr(args, "mpgq_adaptive_retry_steps", getattr(args, "wmpq_adaptive_retry_steps", 3)) or 3),
        adaptive_accept_tol=float(getattr(args, "mpgq_adaptive_accept_tol", getattr(args, "wmpq_adaptive_accept_tol", 1e-4)) or 1e-4),
        adaptive_damping_min=float(getattr(args, "mpgq_adaptive_damping_min", getattr(args, "wmpq_adaptive_damping_min", 0.1)) or 0.1),
        adaptive_obs_precision_backoff=float(getattr(args, "mpgq_adaptive_obs_backoff", getattr(args, "wmpq_adaptive_obs_backoff", 0.5)) or 0.5),
        blocks_per_chunk=int(getattr(args, "mpgq_blocks_per_chunk", getattr(args, "wmpq_blocks_per_chunk", 16)) or 16),
        polish_sweeps=int(getattr(args, "mpgq_polish_sweeps", getattr(args, "wmpq_polish_sweeps", 3)) or 3),
        polish_metric=str(getattr(args, "mpgq_polish_metric", getattr(args, "wmpq_polish_metric", "gram")) or "gram"),
        polish_metric_rows_per_chunk=int(getattr(args, "mpgq_metric_rows_per_chunk", getattr(args, "wmpq_metric_rows_per_chunk", 0)) or 0),
        polish_max_blocks=int(getattr(args, "mpgq_polish_max_blocks", getattr(args, "wmpq_polish_max_blocks", 0)) or 0),
    )

    block_rows = _resolve_block_rows(
        m,
        int(getattr(args, "mpgq_block_rows", getattr(args, "wmpq_block_rows", adapter_cfg.block_rows)) or adapter_cfg.block_rows),
    )
    final_hatWr = hatWr_init.clone()
    final_Qidxs = Qidxs_init.clone()

    exact_ldlq, exact_ldlq_chunks = _exact_surrogate_chunk_means(
        Wr,
        hatWr0.to(dtype=torch.float32),
        Hr,
        block_rows=block_rows,
    )
    exact_before, exact_before_chunks = _exact_surrogate_chunk_means(Wr, hatWr_init, Hr, block_rows=block_rows)
    obs_precision_block = float(obs_precision)
    for row_start in range(0, m, block_rows):
        row_end = min(row_start + block_rows, m)
        rb = slice(row_start, row_end)
        hat_chunk, idx_chunk, _err_chunk, stats_chunk = solve_e8p12_rankspace_vamp(
            Wr=Wr[rb],
            hatWr_init=final_hatWr[rb],
            Qidxs_init=final_Qidxs[rb],
            cb=cb,
            V=V,
            Vh=Vh,
            S=S,
            Vs=Vs,
            cfg=vamp_cfg,
            module_name=module_name,
            obs_precision_init=obs_precision_block,
            log_stats=False,
        )
        final_hatWr[rb] = hat_chunk
        final_Qidxs[rb] = idx_chunk
        if bool(vamp_cfg.em_obs_precision) and float(vamp_cfg.obs_precision) <= 0.0:
            obs_precision_block = float(stats_chunk.get("obs_precision_final", obs_precision_block))

    exact_after, exact_after_chunks = _exact_surrogate_chunk_means(Wr, final_hatWr, Hr, block_rows=block_rows)

    # Surrogate gate helpers: compare per-row-chunk (block_rows) means vs LDLQ.
    chunk_eps = 1e-8
    violation_count = 0
    max_delta = 0.0
    max_ratio = 0.0
    for ldlq_val, mpgq_val in zip(exact_ldlq_chunks, exact_after_chunks):
        if mpgq_val > ldlq_val + chunk_eps:
            violation_count += 1
            max_delta = max(max_delta, float(mpgq_val - ldlq_val))
            if ldlq_val > 0:
                max_ratio = max(max_ratio, float(mpgq_val / ldlq_val))

    stats: Dict[str, Any] = {
        "method": "mpgq",
        "module": str(module_name),
        "init_mode": str(init_mode),
        "seed": int(seed),
        "m": int(m),
        "n": int(n),
        "rank": int(rank),
        "eigen_floor": float(eigen_floor),
        "obs_precision_initial": float(obs_precision),
        "obs_precision_final": float(obs_precision_block),
        "block_rows": int(block_rows),
        "exact_surrogate_ldlq_mean": float(exact_ldlq),
        "exact_surrogate_ldlq_chunk_means": exact_ldlq_chunks,
        "exact_surrogate_before_mean": float(exact_before),
        "exact_surrogate_before_chunk_means": exact_before_chunks,
        "exact_surrogate_after_mean": float(exact_after),
        "exact_surrogate_after_chunk_means": exact_after_chunks,
        "exact_surrogate_gain_mean": float(exact_before - exact_after),
        "surrogate_chunk_violation_count": int(violation_count),
        "surrogate_chunk_violation_max_delta": float(max_delta),
        "surrogate_chunk_violation_max_ratio": float(max_ratio),
        "solver_time_s": float(time.time() - start),
        "vamp": stats_chunk if "stats_chunk" in locals() else {},
    }
    return final_hatWr.to(dtype=hatWr0.dtype), final_Qidxs, stats
