from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Protocol, Tuple

import torch

from ..linear import stabilize_cholesky


class CandidateProvider(Protocol):
    @torch.inference_mode()
    def propose(
        self,
        *,
        block_idx: int,
        start: int,
        end: int,
        x_block: torch.Tensor,  # (B, bs)
        target: torch.Tensor,  # (B, bs)
        metric: str,  # "diag" | "gram"
        gram_chol: Optional[torch.Tensor],  # (bs, bs) if metric=="gram"
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Return (cand_vals, cand_idxs) for this block."""


@dataclass(frozen=True)
class BlockCDConfig:
    sweeps: int = 3
    metric: str = "diag"  # "diag" | "gram"
    accept_tol: float = 0.0
    metric_rows_per_chunk: int = 0  # for large codebooks (optional)
    max_blocks_per_sweep: int = 0  # if >0, only polish top-K blocks by static importance


@torch.inference_mode()
def _resolve_blocks(n: int, block_size: int) -> Tuple[Tuple[int, int], ...]:
    if block_size <= 0:
        raise ValueError(f"block_size must be > 0, got {block_size}.")
    if n % block_size != 0:
        raise ValueError(f"n must be divisible by block_size; got n={n}, block_size={block_size}.")
    blocks = []
    for start in range(0, n, block_size):
        blocks.append((start, start + block_size))
    return tuple(blocks)


@torch.inference_mode()
def block_coordinate_descent_rankspace(
    *,
    x: torch.Tensor,  # (B, n)
    Y_u: torch.Tensor,  # (B, R)
    Vs: torch.Tensor,  # (n, R)
    block_size: int,
    provider: CandidateProvider,
    cfg: BlockCDConfig,
    idxs_init: Optional[torch.Tensor] = None,  # (B, n_blocks) for codebook-style decisions
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, float]]:
    """Generic rank-space block-coordinate polish.

    Objective: minimize ||Y_u - x Vs||^2 subject to discrete constraints supplied by `provider`.

    Returns:
      x_new: (B, n)
      idxs:  optional discrete decision tensor, provider-defined
      stats: summary dict
    """
    sweeps = max(int(cfg.sweeps), 0)
    if sweeps <= 0 or x.numel() == 0 or Vs.numel() == 0:
        return x, None, {"accepted_updates": 0.0}

    metric = str(cfg.metric or "diag").strip().lower()
    if metric not in ("diag", "gram", "isotropic"):
        raise ValueError(f"unsupported metric={metric!r}; expected 'diag'|'gram'.")
    if metric == "isotropic":
        metric = "diag"

    B, n = x.shape
    blocks = _resolve_blocks(n, int(block_size))
    x_work = x.clone().to(dtype=torch.float32)
    Vs_f32 = Vs.to(dtype=torch.float32)
    Y_u_f32 = Y_u.to(dtype=torch.float32)

    residual_u = Y_u_f32 - x_work.matmul(Vs_f32)  # (B, R)
    residual_norm = residual_u.square().sum(dim=1)  # (B,)

    # Precompute per-block metadata.
    block_meta = []
    importance = torch.empty(len(blocks), dtype=torch.float32, device=x_work.device)
    for bi, (start, end) in enumerate(blocks):
        Vs_blk = Vs_f32[start:end]  # (bs, R)
        gram_raw = Vs_blk.matmul(Vs_blk.T).contiguous()  # (bs, bs)
        if metric == "gram":
            gram_chol = stabilize_cholesky(gram_raw)
        else:
            gram_chol = None
        block_meta.append((start, end, Vs_blk, gram_raw, gram_chol))
        importance[bi] = float(torch.diag(gram_raw).sum().item())
    order = torch.argsort(importance, descending=True)
    order_list = order.tolist()
    max_blocks = int(getattr(cfg, "max_blocks_per_sweep", 0) or 0)
    if max_blocks > 0:
        order_list = order_list[:max_blocks]

    accept_tol = max(float(cfg.accept_tol), 0.0)
    accepted_updates = 0
    idxs_out: Optional[torch.Tensor] = idxs_init.clone() if idxs_init is not None else None

    for _ in range(sweeps):
        for bi in order_list:
            start, end, Vs_blk, gram_raw, gram_chol = block_meta[bi]
            x_blk = x_work[:, start:end]  # (B, bs)
            rhs = residual_u.matmul(Vs_blk.T)  # (B, bs)

            if metric == "gram":
                assert gram_chol is not None
                delta_star = torch.cholesky_solve(rhs.T.contiguous(), gram_chol).T
                target = x_blk + delta_star
            else:
                # Diagonal metric fallback: use per-dim steps  rhs / diag(gram)
                diag = torch.diag(gram_raw).clamp_min_(1e-12).unsqueeze(0)  # (1, bs)
                target = x_blk + rhs / diag

            cand_vals, cand_idxs = provider.propose(
                block_idx=bi,
                start=start,
                end=end,
                x_block=x_blk,
                target=target,
                metric=metric,
                gram_chol=gram_chol,
            )
            cand_vals = cand_vals.to(dtype=torch.float32)
            delta = cand_vals - x_blk

            cross = (rhs * delta).sum(dim=1)
            delta_gram = delta.matmul(gram_raw)
            delta_norm = (delta_gram * delta).sum(dim=1).clamp_min_(0.0)
            cand_err = (residual_norm + delta_norm - 2.0 * cross).clamp_min_(0.0)

            accept = cand_err + 1e-12 < residual_norm * (1.0 + accept_tol)
            if not bool(accept.any().item()):
                continue

            x_work[accept, start:end] = cand_vals[accept]
            residual_u[accept] = residual_u[accept] - delta[accept].matmul(Vs_blk)
            residual_norm[accept] = cand_err[accept]
            accepted_updates += int(accept.sum().item())

            if cand_idxs is not None:
                if idxs_out is None:
                    idxs_out = cand_idxs.new_empty((B, len(blocks)), dtype=cand_idxs.dtype, device=cand_idxs.device)
                idxs_out[accept, bi] = cand_idxs[accept]

    return x_work.to(dtype=x.dtype), idxs_out, {"accepted_updates": float(accepted_updates)}


@dataclass(frozen=True)
class UniformScalarProvider:
    scale: torch.Tensor  # (B, n)
    zero: torch.Tensor  # (B, n)
    maxq: int

    @torch.inference_mode()
    def propose(
        self,
        *,
        block_idx: int,
        start: int,
        end: int,
        x_block: torch.Tensor,
        target: torch.Tensor,
        metric: str,
        gram_chol: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if end != start + 1:
            raise ValueError("UniformScalarProvider expects block_size=1.")
        scale = self.scale[:, start:end].to(dtype=torch.float32).clamp_min_(1e-12)
        zero = self.zero[:, start:end].to(dtype=torch.float32)
        q = torch.round(target.to(dtype=torch.float32) / scale + zero)
        q = torch.clamp(q, 0, int(self.maxq)).to(dtype=torch.int32)
        xq = scale * (q.to(dtype=torch.float32) - zero)
        return xq.to(dtype=x_block.dtype), None


@dataclass
class E8P12Provider:
    cb: object
    metric_rows_per_chunk: int = 0
    _cache: Optional[Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = None

    def __post_init__(self) -> None:
        self._cache = {}

    @torch.inference_mode()
    def _metric_map_assign(
        self,
        *,
        target: torch.Tensor,  # (B, 8)
        chol: torch.Tensor,  # (8,8)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        grid = self.cb.grid.to(dtype=torch.float32, device=target.device)
        cache_key = int(chol.data_ptr())
        metric_grid = None
        metric_grid_norm = None
        if self._cache is not None and cache_key in self._cache:
            metric_grid, metric_grid_norm = self._cache[cache_key]
            if metric_grid.device != target.device:
                metric_grid = metric_grid.to(device=target.device)
                metric_grid_norm = metric_grid_norm.to(device=target.device)
        if metric_grid is None or metric_grid_norm is None:
            chol_f32 = chol.to(dtype=torch.float32)
            metric_grid = grid.matmul(chol_f32)
            metric_grid_norm = metric_grid.square().sum(dim=-1)
            if self._cache is not None:
                self._cache[cache_key] = (metric_grid, metric_grid_norm)

        target_metric = target.to(dtype=torch.float32).matmul(chol.to(dtype=torch.float32))
        rows = int(target_metric.shape[0])
        rows_per_chunk = rows
        if self.metric_rows_per_chunk and int(self.metric_rows_per_chunk) > 0:
            rows_per_chunk = max(1, min(rows, int(self.metric_rows_per_chunk)))

        map_idx = torch.empty(rows, dtype=torch.long, device=target_metric.device)
        for start in range(0, rows, rows_per_chunk):
            end = min(start + rows_per_chunk, rows)
            chunk = target_metric[start:end]
            chunk_norm = chunk.square().sum(dim=-1, keepdim=True)
            scores = chunk_norm + metric_grid_norm.unsqueeze(0) - 2.0 * chunk.matmul(metric_grid.T)
            map_idx[start:end] = torch.argmin(scores, dim=1)
        map_val = grid.index_select(0, map_idx).to(dtype=target.dtype)
        return map_val, map_idx.to(dtype=getattr(self.cb, "idx_dtype", torch.int64))

    @torch.inference_mode()
    def propose(
        self,
        *,
        block_idx: int,
        start: int,
        end: int,
        x_block: torch.Tensor,
        target: torch.Tensor,
        metric: str,
        gram_chol: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if (end - start) != int(getattr(self.cb, "codesz", 8)):
            raise ValueError("E8P12Provider expects block_size=codesz.")
        if metric == "gram":
            if gram_chol is None:
                raise ValueError("metric='gram' requires gram_chol.")
            return self._metric_map_assign(target=target, chol=gram_chol)

        # isotropic/diag: rely on codebook rounding
        if not hasattr(self.cb, "quantize"):
            raise ValueError("cb.quantize is required for isotropic polish.")
        cand_vals, cand_idxs = self.cb.quantize(target, resid_scale_override=-1)
        return cand_vals.to(dtype=x_block.dtype), cand_idxs
