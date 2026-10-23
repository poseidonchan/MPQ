from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import torch

from ..linear import stabilize_cholesky


_ALPHA_EPS = 1e-8


@dataclass
class OJBKQConfig:
    """OJBKQ discrete solver baseline (ported from our GPTQModel fork).

    This is not part of the MPQ solver; it is a baseline discrete solver that can be used
    as a GPTQ solver-swap using the Cholesky factor of H^{-1}.
    """

    enabled: bool = field(default=True)
    random_trials: int = field(default=4)
    alpha_mode: str = field(default="paper")  # "paper" | "fixed"
    alpha_override: Optional[float] = field(default=None)
    alpha_scale: float = field(default=1.0)
    sample_topk: Optional[int] = field(default=None)

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            self.enabled = bool(self.enabled)
        if not isinstance(self.random_trials, int) or self.random_trials < 0:
            raise ValueError("OJBKQConfig: `random_trials` must be an integer >= 0.")
        self.alpha_mode = str(self.alpha_mode).lower().strip()
        if self.alpha_mode not in {"paper", "fixed"}:
            raise ValueError("OJBKQConfig: `alpha_mode` must be either `paper` or `fixed`.")
        if self.alpha_override is not None and float(self.alpha_override) <= 0.0:
            raise ValueError("OJBKQConfig: `alpha_override` must be > 0 when provided.")
        if float(self.alpha_scale) <= 0.0:
            raise ValueError("OJBKQConfig: `alpha_scale` must be > 0.")
        if self.sample_topk is not None and (not isinstance(self.sample_topk, int) or self.sample_topk <= 0):
            raise ValueError("OJBKQConfig: `sample_topk` must be an integer > 0 when provided.")


def _resolve_group_size(total_cols: int, group_size: int) -> int:
    if group_size == -1 or group_size <= 0:
        return total_cols
    return min(group_size, total_cols)


def _paper_rho(num_trials: int, dim: int) -> float:
    if num_trials <= 1:
        return math.e

    log_k = math.log(float(num_trials))

    def score(rho: float) -> float:
        return (2.0 * float(dim) / rho) * (1.0 + math.log(rho)) - log_k

    lo = 1.0
    hi = 2.0
    while score(hi) > 0.0:
        hi *= 2.0
        if hi > 1e8:
            break

    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if score(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    return hi


def _resolve_alpha(*, r_diag: torch.Tensor, total_trials: int, config: OJBKQConfig) -> float:
    if config.alpha_override is not None:
        return float(config.alpha_override)
    if total_trials <= 1:
        return 1e12
    if config.alpha_mode == "fixed":
        return float(config.alpha_scale)
    rho = _paper_rho(total_trials, int(r_diag.numel()))
    min_diag_sq = float(torch.square(r_diag).amin().item())
    min_diag_sq = max(min_diag_sq, _ALPHA_EPS)
    return float(config.alpha_scale) * math.log(rho) / min_diag_sq


def _clamp_codes(codes: torch.Tensor, maxq: int) -> torch.Tensor:
    return torch.clamp(codes, 0.0, float(maxq))


def _quantize_with_params(*, weights: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor, maxq: int) -> Tuple[torch.Tensor, torch.Tensor]:
    q_int = _clamp_codes(torch.round(weights / scale) + zero, maxq)
    dequant = scale * (q_int - zero)
    return q_int, dequant


@torch.inference_mode()
def _e8p12_metric_sample(
    *,
    grid: torch.Tensor,
    target: torch.Tensor,
    chol: torch.Tensor,
    sample_topk: int,
    alpha: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    target_metric = target.to(dtype=torch.float32).matmul(chol.to(dtype=torch.float32))
    grid_metric = grid.to(device=target.device, dtype=torch.float32).matmul(chol.to(dtype=torch.float32))
    target_norm = target_metric.square().sum(dim=1, keepdim=True)
    grid_norm = grid_metric.square().sum(dim=1).unsqueeze(0)
    scores = (target_norm + grid_norm - 2.0 * target_metric.matmul(grid_metric.T)).clamp_min_(0.0)
    k = max(1, min(int(sample_topk), int(grid.shape[0])))
    top_scores, top_idxs = torch.topk(scores, k=k, dim=1, largest=False)
    logits = -float(alpha) * top_scores
    logits = torch.nan_to_num(
        logits - logits.amax(dim=1, keepdim=True),
        nan=0.0,
        posinf=0.0,
        neginf=-1e9,
    )
    sampled = torch.distributions.Categorical(logits=logits).sample()
    row_idx = torch.arange(int(target.shape[0]), device=target.device)
    idxs = top_idxs[row_idx, sampled]
    vals = grid.index_select(0, idxs)
    return vals.to(dtype=target.dtype), idxs


@torch.inference_mode()
def _e8p12_surrogate_rows(Wr: torch.Tensor, hatWr: torch.Tensor, Hr: torch.Tensor) -> torch.Tensor:
    delta = (Wr - hatWr).to(dtype=torch.float32)
    return (delta.matmul(Hr.to(dtype=torch.float32)) * delta).sum(dim=1)


def _refresh_group_params_for_paths(
    *,
    work_paths: torch.Tensor,
    group_start: int,
    group_end: int,
    quantizer,
    scale_paths: torch.Tensor,
    zero_paths: torch.Tensor,
    group_idx: int,
    share_reference_params: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    num_paths, rows, _ = work_paths.shape
    current_scale = torch.empty((num_paths, rows), dtype=torch.float32, device=work_paths.device)
    current_zero = torch.empty((num_paths, rows), dtype=torch.float32, device=work_paths.device)

    if share_reference_params:
        quantizer.find_params(work_paths[0, :, group_start:group_end], weight=True)
        scale = quantizer.scale.to(device=work_paths.device, dtype=torch.float32).reshape(rows)
        zero = quantizer.zero.to(device=work_paths.device, dtype=torch.float32).reshape(rows)
        current_scale.copy_(scale.unsqueeze(0))
        current_zero.copy_(zero.unsqueeze(0))
        scale_paths[:, :, group_idx] = scale.unsqueeze(0)
        zero_paths[:, :, group_idx] = zero.unsqueeze(0)
        return current_scale, current_zero

    for path_idx in range(num_paths):
        quantizer.find_params(work_paths[path_idx, :, group_start:group_end], weight=True)
        scale = quantizer.scale.to(device=work_paths.device, dtype=torch.float32).reshape(rows)
        zero = quantizer.zero.to(device=work_paths.device, dtype=torch.float32).reshape(rows)
        current_scale[path_idx] = scale
        current_zero[path_idx] = zero
        scale_paths[path_idx, :, group_idx] = scale
        zero_paths[path_idx, :, group_idx] = zero

    return current_scale, current_zero


@torch.inference_mode()
def run_ojbkq_gptq_quantization(
    *,
    W: torch.Tensor,
    Hinv: torch.Tensor,
    quantizer,
    qcfg,
    config: OJBKQConfig,
    nsamples: int,
    module_name: str = "",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float, Dict[str, object]]:
    """OJBKQ as a GPTQ solver-swap, operating on the GPTQ Hessian inverse factor `Hinv`.

    Returns:
      Q_dequant: (rows, cols) float32
      q_int: (rows, cols) int32 (quantized code indices)
      scale_grouped: (rows, n_groups) float32
      zero_grouped: (rows, n_groups) float32
      g_idx: (cols,) int32
      avg_loss: float (normalized by nsamples)
      stats: dict
    """
    if W.ndim != 2:
        raise ValueError("OJBKQ/GPTQ: `W` must be a 2D tensor.")
    if Hinv.ndim != 2 or Hinv.shape[0] != Hinv.shape[1] or Hinv.shape[0] != W.shape[1]:
        raise ValueError("OJBKQ/GPTQ: `Hinv` must be square with size equal to `W.shape[1]`.")
    if bool(getattr(qcfg, "act_group_aware", False)):
        raise ValueError("OJBKQ/GPTQ: solver-swap path currently requires `act_group_aware=False`.")

    start_time = time.perf_counter()
    device = W.device
    W = W.to(device=device, dtype=torch.float32, non_blocking=False).contiguous()
    Hinv = Hinv.to(device=device, dtype=torch.float32, non_blocking=False).contiguous()

    rows, cols = W.shape
    maxq = int(quantizer.maxq.item())
    random_trials = max(int(config.random_trials), 0)
    total_paths = 1 + random_trials
    group_size = _resolve_group_size(cols, int(getattr(qcfg, "group_size", -1)))
    num_groups = (cols + group_size - 1) // group_size
    sample_support = int(config.sample_topk) if config.sample_topk is not None else maxq + 1
    sample_support = max(1, min(sample_support, maxq + 1))
    levels = torch.arange(maxq + 1, device=device, dtype=torch.float32)

    inv_diag = torch.diagonal(Hinv).clamp_min(_ALPHA_EPS)
    r_diag = inv_diag.reciprocal()
    alpha = _resolve_alpha(r_diag=r_diag, total_trials=max(total_paths, 1), config=config)

    work_paths = W.unsqueeze(0).repeat(total_paths, 1, 1)
    qint_paths = torch.empty((total_paths, rows, cols), dtype=torch.float32, device=device)
    dequant_paths = torch.empty((total_paths, rows, cols), dtype=torch.float32, device=device)
    loss_paths = torch.zeros((total_paths, rows), dtype=torch.float32, device=device)
    scale_paths = torch.empty((total_paths, rows, num_groups), dtype=torch.float32, device=device)
    zero_paths = torch.empty((total_paths, rows, num_groups), dtype=torch.float32, device=device)

    current_scale = None
    current_zero = None
    blocksize = 128
    for block_start in range(0, cols, blocksize):
        block_end = min(block_start + blocksize, cols)
        block_count = block_end - block_start
        work_block = work_paths[:, :, block_start:block_end].clone()
        errors_block = torch.zeros(
            (total_paths, rows, block_count),
            dtype=torch.float32,
            device=device,
        )
        hinv_block = Hinv[block_start:block_end, block_start:block_end]

        for block_col in range(block_count):
            col_idx = block_start + block_col
            if col_idx % group_size == 0:
                group_idx = col_idx // group_size
                group_end = min(col_idx + group_size, cols)
                current_scale, current_zero = _refresh_group_params_for_paths(
                    work_paths=work_paths,
                    group_start=col_idx,
                    group_end=group_end,
                    quantizer=quantizer,
                    scale_paths=scale_paths,
                    zero_paths=zero_paths,
                    group_idx=group_idx,
                    share_reference_params=True,
                )

            if current_scale is None or current_zero is None:
                raise RuntimeError("OJBKQ/GPTQ: group parameters were not initialized.")

            d = float(hinv_block[block_col, block_col].item())
            w_current = work_block[:, :, block_col]

            # Keep path zero on the deterministic blocked Babai/GPTQ-style
            # loop. It is OJBKQ's greedy reference path, not a separately
            # computed native GPTQ solution.
            greedy_dequant = quantizer.quantize(w_current[0].unsqueeze(1)).flatten()
            greedy_qint = _clamp_codes(
                torch.round(greedy_dequant / current_scale[0]) + current_zero[0],
                maxq,
            )
            qint_paths[0, :, col_idx] = greedy_qint
            dequant_paths[0, :, col_idx] = greedy_dequant

            if total_paths > 1:
                current_random = w_current[1:]
                scale_random = current_scale[1:]
                zero_random = current_zero[1:]
                centers = current_random / scale_random + zero_random
                distances = (centers.unsqueeze(-1) - levels.view(1, 1, -1)).abs()
                _, support_idx = torch.topk(distances, k=sample_support, dim=-1, largest=False)
                support_levels = torch.gather(
                    levels.view(1, 1, -1).expand(total_paths - 1, rows, -1),
                    dim=-1,
                    index=support_idx,
                )

                # Paper Eq. (13) uses the transformed diagonal entry \bar{R}_{ii},
                # not its square, in the randomized rounding logits.
                local_rbar_diag = (scale_random / d).clamp_min(_ALPHA_EPS).unsqueeze(-1)
                logits = -alpha * local_rbar_diag * (centers.unsqueeze(-1) - support_levels).square()
                logits = torch.nan_to_num(
                    logits - logits.amax(dim=-1, keepdim=True),
                    nan=0.0,
                    posinf=0.0,
                    neginf=-1e9,
                )
                sampled_idx = torch.distributions.Categorical(logits=logits.reshape(-1, sample_support)).sample()
                sampled_idx = sampled_idx.reshape(total_paths - 1, rows, 1)
                sampled_levels = torch.gather(support_levels, dim=-1, index=sampled_idx).squeeze(-1)
                random_dequant = scale_random * (sampled_levels - zero_random)
                qint_paths[1:, :, col_idx] = sampled_levels
                dequant_paths[1:, :, col_idx] = random_dequant

            err = (w_current - dequant_paths[:, :, col_idx]) / d
            loss_paths += 0.5 * err.square()
            errors_block[:, :, block_col] = err

            # Match GPTQ's accumulation order for the greedy candidate.
            work_block[0, :, block_col:] -= err[0].unsqueeze(1).matmul(
                hinv_block[block_col, block_col:].unsqueeze(0)
            )
            if total_paths > 1:
                work_block[1:, :, block_col:] -= (
                    err[1:].unsqueeze(-1)
                    * hinv_block[block_col, block_col:].view(1, 1, -1)
                )

        if block_end < cols:
            trailing_hinv = Hinv[block_start:block_end, block_end:]
            work_paths[0, :, block_end:] -= errors_block[0].matmul(trailing_hinv)
            if total_paths > 1:
                work_paths[1:, :, block_end:] -= errors_block[1:].matmul(trailing_hinv)

    best_idx = torch.argmin(loss_paths, dim=0)
    row_ids = torch.arange(rows, device=device)
    qint_by_row = qint_paths.permute(1, 0, 2)
    dequant_by_row = dequant_paths.permute(1, 0, 2)
    scale_by_row = scale_paths.permute(1, 0, 2)
    zero_by_row = zero_paths.permute(1, 0, 2)
    loss_by_row = loss_paths.t()

    best_qint = qint_by_row[row_ids, best_idx].to(dtype=torch.int32)
    best_dequant = dequant_by_row[row_ids, best_idx]
    best_scale = scale_by_row[row_ids, best_idx]
    best_zero = zero_by_row[row_ids, best_idx]
    best_loss = loss_by_row[row_ids, best_idx]

    g_idx = torch.arange(cols, device=device, dtype=torch.int32) // group_size
    solver_time = time.perf_counter() - start_time
    avg_loss = float(best_loss.sum().item()) / float(max(int(nsamples), 1))
    non_greedy_ratio = float((best_idx > 0).float().mean().item())
    selected_path_mean = float(best_idx.to(dtype=torch.float32).mean().item())
    selected_path_max = int(best_idx.max().item()) if best_idx.numel() > 0 else 0

    stats: Dict[str, object] = {
        "solver": "ojbkq",
        "solver_scope": "gptq_random_k_solver",
        "module": module_name,
        "solver_time": float(solver_time),
        "random_trials": int(config.random_trials),
        "random_paths": int(random_trials),
        "total_candidates": int(total_paths),
        "greedy_candidates": 1,
        "alpha_mode": str(config.alpha_mode),
        "alpha": float(alpha),
        "sample_topk": int(sample_support),
        "non_greedy_row_ratio": non_greedy_ratio,
        "selected_path_mean": selected_path_mean,
        "selected_path_max": selected_path_max,
    }

    return best_dequant, best_qint, best_scale, best_zero, g_idx, avg_loss, stats


@torch.inference_mode()
def run_ojbkq_e8p12_hessian(
    *,
    Wr: torch.Tensor,
    Hr: torch.Tensor,
    hatWr0: torch.Tensor,
    Qidxs0: torch.Tensor,
    cb: Any,
    config: OJBKQConfig,
    module_name: str = "",
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, object]]:
    start_time = time.perf_counter()
    Wr_f32 = Wr.to(dtype=torch.float32)
    Hr_f32 = Hr.to(device=Wr_f32.device, dtype=torch.float32)
    Hr_f32 = 0.5 * (Hr_f32 + Hr_f32.T)
    hat0 = hatWr0.to(device=Wr_f32.device, dtype=torch.float32)
    rows, cols = Wr_f32.shape
    codesz = int(getattr(cb, "codesz", 8))
    if codesz <= 0 or cols % codesz != 0:
        raise ValueError(f"OJBKQ/E8P12 expects cols divisible by code size; got cols={cols}, codesz={codesz}.")

    nblocks = cols // codesz
    random_trials = max(int(config.random_trials), 0)
    total_paths = 1 + random_trials
    grid = cb.grid.to(device=Wr_f32.device, dtype=torch.float32)
    sample_support = int(config.sample_topk) if config.sample_topk is not None else min(16, int(grid.shape[0]))
    sample_support = max(1, min(sample_support, int(grid.shape[0])))
    diag = torch.diagonal(Hr_f32).abs().clamp_min(_ALPHA_EPS)
    alpha = _resolve_alpha(r_diag=diag, total_trials=max(total_paths, 1), config=config)

    path_hats = torch.empty((total_paths, rows, cols), dtype=torch.float32, device=Wr_f32.device)
    path_idxs = torch.empty((total_paths, rows, nblocks), dtype=getattr(cb, "idx_dtype", torch.int64), device=Wr_f32.device)
    path_hats[0] = hat0
    path_idxs[0] = Qidxs0.to(device=Wr_f32.device, dtype=path_idxs.dtype)

    if random_trials > 0:
        block_meta = []
        for block_idx in range(nblocks):
            start = block_idx * codesz
            end = start + codesz
            gram = Hr_f32[start:end, start:end].contiguous()
            chol = stabilize_cholesky(gram)
            gram_inv = torch.cholesky_inverse(chol).contiguous()
            block_meta.append((start, end, Hr_f32[:, start:end].contiguous(), chol, gram_inv))

        for path_idx in range(1, total_paths):
            hat = Wr_f32.clone()
            delta = Wr_f32 - hat
            idxs = torch.empty((rows, nblocks), dtype=path_idxs.dtype, device=Wr_f32.device)
            for block_idx, (start, end, H_cols, chol, gram_inv) in enumerate(block_meta):
                target = hat[:, start:end] + delta.matmul(H_cols).matmul(gram_inv)
                vals, sampled_idxs = _e8p12_metric_sample(
                    grid=grid,
                    target=target,
                    chol=chol,
                    sample_topk=sample_support,
                    alpha=alpha,
                )
                vals = vals.to(dtype=torch.float32)
                hat[:, start:end] = vals
                idxs[:, block_idx] = sampled_idxs.to(dtype=idxs.dtype)
                delta[:, start:end] = Wr_f32[:, start:end] - vals
            path_hats[path_idx] = hat
            path_idxs[path_idx] = idxs

    losses = torch.empty((total_paths, rows), dtype=torch.float32, device=Wr_f32.device)
    for path_idx in range(total_paths):
        losses[path_idx] = _e8p12_surrogate_rows(Wr_f32, path_hats[path_idx], Hr_f32)

    best_idx = torch.argmin(losses, dim=0)
    row_idx = torch.arange(rows, device=Wr_f32.device)
    best_hat = path_hats.permute(1, 0, 2)[row_idx, best_idx]
    best_Qidxs = path_idxs.permute(1, 0, 2)[row_idx, best_idx]
    non_greedy_ratio = float((best_idx > 0).to(dtype=torch.float32).mean().item()) if best_idx.numel() else 0.0

    stats: Dict[str, object] = {
        "solver": "ojbkq",
        "solver_scope": "e8p12_random_k_solver",
        "module": str(module_name),
        "solver_time": float(time.perf_counter() - start_time),
        "random_trials": int(config.random_trials),
        "random_paths": int(random_trials),
        "total_candidates": int(total_paths),
        "greedy_candidates": 1,
        "alpha_mode": str(config.alpha_mode),
        "alpha": float(alpha),
        "sample_topk": int(sample_support),
        "non_greedy_row_ratio": non_greedy_ratio,
        "selected_path_mean": float(best_idx.to(dtype=torch.float32).mean().item()) if best_idx.numel() else 0.0,
        "selected_path_max": int(best_idx.max().item()) if best_idx.numel() else 0,
    }
    return best_hat, best_Qidxs, stats


__all__ = ["OJBKQConfig", "run_ojbkq_e8p12_hessian", "run_ojbkq_gptq_quantization"]
