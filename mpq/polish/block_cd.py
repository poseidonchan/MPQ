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


def _cd_accept_step_impl(
    x_blk: torch.Tensor,
    cand_vals: torch.Tensor,
    rhs: torch.Tensor,
    gram_raw: torch.Tensor,
    residual_norm: torch.Tensor,
    accept_tol: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    delta = cand_vals - x_blk
    cross = (rhs * delta).sum(dim=1)
    delta_gram = delta.matmul(gram_raw)
    delta_norm = (delta_gram * delta).sum(dim=1).clamp_min_(0.0)
    cand_err = (residual_norm + delta_norm - 2.0 * cross).clamp_min_(0.0)
    accept = cand_err + 1e-12 < residual_norm * (1.0 + accept_tol)
    return delta, cand_err, accept


_COMPILED_CD_ACCEPT = None
_COMPILED_CD_ACCEPT_FAILED = False
_COMPILED_TOPK_METRIC_SELECT = None
_COMPILED_TOPK_METRIC_SELECT_FAILED = False


def _get_cd_accept_step(*, device: torch.device):
    global _COMPILED_CD_ACCEPT, _COMPILED_CD_ACCEPT_FAILED
    if device.type != "cuda" or not hasattr(torch, "compile"):
        return _cd_accept_step_impl
    if _COMPILED_CD_ACCEPT_FAILED:
        return _cd_accept_step_impl
    if _COMPILED_CD_ACCEPT is None:
        try:
            _COMPILED_CD_ACCEPT = torch.compile(
                _cd_accept_step_impl,
                mode="max-autotune-no-cudagraphs",
                fullgraph=False,
            )
        except Exception:
            _COMPILED_CD_ACCEPT_FAILED = True
            return _cd_accept_step_impl
    return _COMPILED_CD_ACCEPT


def _topk_metric_select_impl(
    cand_metric: torch.Tensor,
    cand_metric_norm: torch.Tensor,
    target_metric: torch.Tensor,
) -> torch.Tensor:
    target_norm = target_metric.square().sum(dim=1, keepdim=True)
    cross = (cand_metric * target_metric.unsqueeze(1)).sum(dim=-1)
    dist = (target_norm + cand_metric_norm - 2.0 * cross).clamp_min_(0.0)
    return torch.argmin(dist, dim=1)


def _get_topk_metric_select(*, device: torch.device):
    global _COMPILED_TOPK_METRIC_SELECT, _COMPILED_TOPK_METRIC_SELECT_FAILED
    if device.type != "cuda" or not hasattr(torch, "compile"):
        return _topk_metric_select_impl
    if _COMPILED_TOPK_METRIC_SELECT_FAILED:
        return _topk_metric_select_impl
    if _COMPILED_TOPK_METRIC_SELECT is None:
        try:
            _COMPILED_TOPK_METRIC_SELECT = torch.compile(
                _topk_metric_select_impl,
                mode="max-autotune-no-cudagraphs",
                fullgraph=False,
            )
        except Exception:
            _COMPILED_TOPK_METRIC_SELECT_FAILED = True
            return _topk_metric_select_impl
    return _COMPILED_TOPK_METRIC_SELECT


def _topk_gram_step_impl(
    residual_u: torch.Tensor,
    x_blk: torch.Tensor,
    Vs_blk_t: torch.Tensor,
    gram_inv: torch.Tensor,
    chol: torch.Tensor,
    cand_metric: torch.Tensor,
    cand_metric_norm: torch.Tensor,
    cand_vals_f32: torch.Tensor,
    gram_raw: torch.Tensor,
    Vs_blk: torch.Tensor,
    residual_norm: torch.Tensor,
    accept_tol: float,
    row_idx: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rhs = residual_u.matmul(Vs_blk_t)
    target = x_blk + rhs.matmul(gram_inv)
    target_metric = target.matmul(chol)
    target_norm = target_metric.square().sum(dim=1, keepdim=True)
    cross_m = (cand_metric * target_metric.unsqueeze(1)).sum(dim=-1)
    dist = (target_norm + cand_metric_norm - 2.0 * cross_m).clamp_min_(0.0)
    best = torch.argmin(dist, dim=1)
    out_vals = cand_vals_f32[row_idx, best]
    delta = out_vals - x_blk
    cross_r = (rhs * delta).sum(dim=1)
    delta_gram = delta.matmul(gram_raw)
    delta_norm = (delta_gram * delta).sum(dim=1).clamp_min_(0.0)
    cand_err = (residual_norm + delta_norm - 2.0 * cross_r).clamp_min_(0.0)
    accept = cand_err + 1e-12 < residual_norm * (1.0 + accept_tol)
    mask = accept.unsqueeze(1)
    new_x = torch.where(mask, out_vals, x_blk)
    delta_masked = delta * mask
    residual_update = delta_masked.matmul(Vs_blk)
    new_norm = torch.where(accept, cand_err, residual_norm)
    accepted = accept.sum()
    return new_x, best, accept, residual_update, new_norm, accepted


_COMPILED_TOPK_GRAM_STEP = None
_COMPILED_TOPK_GRAM_STEP_FAILED = False


def _get_topk_gram_step(*, device: torch.device):
    global _COMPILED_TOPK_GRAM_STEP, _COMPILED_TOPK_GRAM_STEP_FAILED
    if device.type != "cuda" or not hasattr(torch, "compile"):
        return _topk_gram_step_impl
    if _COMPILED_TOPK_GRAM_STEP_FAILED:
        return _topk_gram_step_impl
    if _COMPILED_TOPK_GRAM_STEP is None:
        try:
            _COMPILED_TOPK_GRAM_STEP = torch.compile(
                _topk_gram_step_impl,
                mode="max-autotune-no-cudagraphs",
                fullgraph=False,
            )
        except Exception:
            _COMPILED_TOPK_GRAM_STEP_FAILED = True
            return _topk_gram_step_impl
    return _COMPILED_TOPK_GRAM_STEP


def _cd_gram_step_impl(
    residual_u: torch.Tensor,
    x_blk: torch.Tensor,
    Vs_blk_t: torch.Tensor,
    gram_inv: torch.Tensor,
    chol: torch.Tensor,
    grid: torch.Tensor,
    gram_raw: torch.Tensor,
    Vs_blk: torch.Tensor,
    residual_norm: torch.Tensor,
    accept_tol: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rhs = residual_u.matmul(Vs_blk_t)
    target = x_blk + rhs.matmul(gram_inv)
    target_metric = target.matmul(chol)
    metric_grid = grid.matmul(chol)
    target_norm = target_metric.square().sum(dim=1, keepdim=True)
    metric_grid_norm = metric_grid.square().sum(dim=-1).unsqueeze(0)
    cross_m = target_metric.matmul(metric_grid.T)
    scores = target_norm + metric_grid_norm - 2.0 * cross_m
    best = torch.argmin(scores, dim=1)
    out_vals = grid.index_select(0, best)
    delta = out_vals - x_blk
    cross_r = (rhs * delta).sum(dim=1)
    delta_gram = delta.matmul(gram_raw)
    delta_norm = (delta_gram * delta).sum(dim=1).clamp_min_(0.0)
    cand_err = (residual_norm + delta_norm - 2.0 * cross_r).clamp_min_(0.0)
    accept = cand_err + 1e-12 < residual_norm * (1.0 + accept_tol)
    mask = accept.unsqueeze(1)
    new_x = torch.where(mask, out_vals, x_blk)
    delta_masked = delta * mask
    residual_update = delta_masked.matmul(Vs_blk)
    new_norm = torch.where(accept, cand_err, residual_norm)
    accepted = accept.sum()
    return new_x, best, accept, residual_update, new_norm, accepted


_COMPILED_CD_GRAM_STEP = None
_COMPILED_CD_GRAM_STEP_FAILED = False


def _get_cd_gram_step(*, device: torch.device):
    global _COMPILED_CD_GRAM_STEP, _COMPILED_CD_GRAM_STEP_FAILED
    if device.type != "cuda" or not hasattr(torch, "compile"):
        return _cd_gram_step_impl
    if _COMPILED_CD_GRAM_STEP_FAILED:
        return _cd_gram_step_impl
    if _COMPILED_CD_GRAM_STEP is None:
        try:
            _COMPILED_CD_GRAM_STEP = torch.compile(
                _cd_gram_step_impl,
                mode="max-autotune-no-cudagraphs",
                fullgraph=False,
            )
        except Exception:
            _COMPILED_CD_GRAM_STEP_FAILED = True
            return _cd_gram_step_impl
    return _COMPILED_CD_GRAM_STEP


@torch.inference_mode()
def _batched_greedy_gain(
    *,
    residual_u: torch.Tensor,  # (B, R)
    Vs_full_t: torch.Tensor,  # (R, n)
    block_size: int,
    metric: str,
    gram_inv_stack: Optional[torch.Tensor],  # (nblocks, bs, bs) if metric=="gram"
    diag_inv: Optional[torch.Tensor],  # (nblocks, bs) if metric!="gram"
) -> torch.Tensor:
    rhs_all = residual_u.matmul(Vs_full_t)  # (B, n)
    rhs_blocks = rhs_all.reshape(int(rhs_all.shape[0]), -1, int(block_size))  # (B, nblocks, bs)
    if metric == "gram":
        if gram_inv_stack is None:
            raise ValueError("batched greedy gain requires gram_inv_stack for metric='gram'.")
        rhs_perm = rhs_blocks.permute(1, 0, 2).contiguous()  # (nblocks, B, bs)
        delta_star_perm = torch.bmm(rhs_perm, gram_inv_stack)  # (nblocks, B, bs)
        return (rhs_perm * delta_star_perm).sum(dim=2).mean(dim=1)
    if diag_inv is None:
        raise ValueError("batched greedy gain requires diag_inv for metric='diag'.")
    return (rhs_blocks.square() * diag_inv.unsqueeze(0)).sum(dim=2).mean(dim=0)


@dataclass(frozen=True)
class BlockCDConfig:
    sweeps: int = 3
    metric: str = "diag"  # "diag" | "gram"
    accept_tol: float = 0.0
    metric_rows_per_chunk: int = 0  # for large codebooks (optional)
    max_blocks_per_sweep: int = 0  # if >0, only polish top-K blocks by static importance
    order_mode: str = "greedy_gain"  # "static" | "greedy_gain"


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
    order_mode = str(getattr(cfg, "order_mode", "greedy_gain") or "greedy_gain").strip().lower()
    if order_mode not in ("static", "greedy_gain"):
        raise ValueError(f"unsupported order_mode={order_mode!r}; expected 'static'|'greedy_gain'.")

    B, n = x.shape
    blocks = _resolve_blocks(n, int(block_size))
    x_work = x.clone().to(dtype=torch.float32)
    Vs_f32 = Vs.to(dtype=torch.float32)
    Vs_full_t = Vs_f32.T.contiguous()
    Y_u_f32 = Y_u.to(dtype=torch.float32)

    residual_u = Y_u_f32 - x_work.matmul(Vs_f32)  # (B, R)
    residual_norm = residual_u.square().sum(dim=1)  # (B,)

    # Precompute per-block metadata.
    block_meta = []
    importance = torch.empty(len(blocks), dtype=torch.float32, device=x_work.device)
    gram_inv_list = []
    diag_inv_list = []
    for bi, (start, end) in enumerate(blocks):
        Vs_blk = Vs_f32[start:end]  # (bs, R)
        Vs_blk_t = Vs_blk.T.contiguous()  # (R, bs)
        gram_raw = Vs_blk.matmul(Vs_blk.T).contiguous()  # (bs, bs)
        if metric == "gram":
            gram_chol = stabilize_cholesky(gram_raw)
            gram_inv = torch.cholesky_inverse(gram_chol).contiguous()
            gram_inv_list.append(gram_inv)
            diag_inv = None
        else:
            gram_chol = None
            gram_inv = None
            diag_inv = torch.diag(gram_raw).clamp_min_(1e-12).reciprocal_().contiguous()
            diag_inv_list.append(diag_inv)
        block_meta.append((start, end, Vs_blk_t, Vs_blk.contiguous(), gram_raw, gram_chol, diag_inv, gram_inv))
        importance[bi] = float(torch.diag(gram_raw).sum().item())
    static_order_list = torch.argsort(importance, descending=True).tolist()
    max_blocks = int(getattr(cfg, "max_blocks_per_sweep", 0) or 0)
    gram_inv_stack = torch.stack(gram_inv_list, dim=0).contiguous() if gram_inv_list else None
    diag_inv_stack = torch.stack(diag_inv_list, dim=0).contiguous() if diag_inv_list else None

    accept_tol = max(float(cfg.accept_tol), 0.0)
    accepted_count = torch.zeros((), dtype=torch.long, device=x_work.device)
    idxs_out: Optional[torch.Tensor] = idxs_init.clone() if idxs_init is not None else None

    # Detect compiled block-step paths.
    _use_compiled_topk = isinstance(provider, E8P12TopKProvider) and metric == "gram"
    _use_compiled_cd = (
        isinstance(provider, E8P12Provider)
        and not isinstance(provider, E8P12TopKProvider)
        and metric == "gram"
        and not getattr(provider, "metric_rows_per_chunk", 0)
    )

    if _use_compiled_topk:
        _step_fn = _get_topk_gram_step(device=x_work.device)
        _tv = provider._topk_vals_on(x_work.device)
        _ti = provider._topk_idxs_device
        _topk_cv: list = []
        _topk_cm: list = []
        _topk_cmn: list = []
        _topk_chol: list = []
        for _bi in range(len(blocks)):
            _c = block_meta[_bi][5].to(dtype=torch.float32)
            _v = _tv[:, _bi, :, :].to(dtype=torch.float32).contiguous()
            _m = _v.matmul(_c)
            _topk_cv.append(_v)
            _topk_cm.append(_m)
            _topk_cmn.append(_m.square().sum(dim=-1))
            _topk_chol.append(_c)
        _row_idx = torch.arange(B, device=x_work.device)
    elif _use_compiled_cd:
        _step_fn = _get_cd_gram_step(device=x_work.device)
        _grid = provider._grid_on(x_work.device)
        _cd_chols: list = []
        for _bi in range(len(blocks)):
            _cd_chols.append(block_meta[_bi][5].to(dtype=torch.float32))
    else:
        accept_step = _get_cd_accept_step(device=x_work.device)

    for _ in range(sweeps):
        if order_mode == "greedy_gain":
            gain = _batched_greedy_gain(
                residual_u=residual_u,
                Vs_full_t=Vs_full_t,
                block_size=int(block_size),
                metric=metric,
                gram_inv_stack=gram_inv_stack,
                diag_inv=diag_inv_stack,
            )
            order_list = torch.argsort(gain, descending=True).tolist()
        else:
            order_list = list(static_order_list)
        if max_blocks > 0:
            order_list = order_list[:max_blocks]

        if _use_compiled_topk:
            for bi in order_list:
                start, end, Vs_blk_t, Vs_blk, gram_raw, gram_chol, diag_inv, gram_inv = block_meta[bi]
                x_blk = x_work[:, start:end]
                new_x, best, accept, res_update, new_norm, acc = _step_fn(
                    residual_u, x_blk, Vs_blk_t, gram_inv, _topk_chol[bi],
                    _topk_cm[bi], _topk_cmn[bi], _topk_cv[bi],
                    gram_raw, Vs_blk, residual_norm, accept_tol, _row_idx,
                )
                x_work[:, start:end] = new_x
                residual_u.sub_(res_update)
                residual_norm = new_norm
                accepted_count += acc
                if idxs_out is not None:
                    out_idxs = _ti[:, bi, :][_row_idx, best]
                    idxs_out[:, bi] = torch.where(accept, out_idxs, idxs_out[:, bi])

        elif _use_compiled_cd:
            for bi in order_list:
                start, end, Vs_blk_t, Vs_blk, gram_raw, gram_chol, diag_inv, gram_inv = block_meta[bi]
                x_blk = x_work[:, start:end]
                new_x, best, accept, res_update, new_norm, acc = _step_fn(
                    residual_u, x_blk, Vs_blk_t, gram_inv, _cd_chols[bi],
                    _grid, gram_raw, Vs_blk, residual_norm, accept_tol,
                )
                x_work[:, start:end] = new_x
                residual_u.sub_(res_update)
                residual_norm = new_norm
                accepted_count += acc
                if idxs_out is not None:
                    idxs_out[:, bi] = torch.where(accept, best.to(dtype=idxs_out.dtype), idxs_out[:, bi])

        else:
            for bi in order_list:
                start, end, Vs_blk_t, Vs_blk, gram_raw, gram_chol, diag_inv, gram_inv = block_meta[bi]
                x_blk = x_work[:, start:end]

                rhs = residual_u.matmul(Vs_blk_t)
                if metric == "gram":
                    target = x_blk + rhs.matmul(gram_inv)
                else:
                    assert diag_inv is not None
                    target = x_blk + rhs * diag_inv.unsqueeze(0)

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
                delta, cand_err, accept = accept_step(
                    x_blk,
                    cand_vals,
                    rhs,
                    gram_raw,
                    residual_norm,
                    accept_tol,
                )

                mask = accept.unsqueeze(1)
                x_work[:, start:end] = torch.where(mask, cand_vals, x_blk)
                delta.mul_(mask)
                residual_u.sub_(delta.matmul(Vs_blk))
                residual_norm = torch.where(accept, cand_err, residual_norm)
                accepted_count += accept.sum()

                if cand_idxs is not None:
                    if idxs_out is None:
                        idxs_out = cand_idxs.new_empty((B, len(blocks)), dtype=cand_idxs.dtype, device=cand_idxs.device)
                    idxs_out[:, bi] = torch.where(accept, cand_idxs, idxs_out[:, bi])

    return x_work.to(dtype=x.dtype), idxs_out, {"accepted_updates": float(accepted_count.item())}


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


@dataclass(frozen=True)
class UniformTopKProvider:
    topk_values: torch.Tensor  # (B, n, k)
    topk_q: Optional[torch.Tensor] = None  # (B, n, k)

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
            raise ValueError("UniformTopKProvider expects block_size=1.")
        cand_vals = self.topk_values[:, start:end, :].to(device=x_block.device, dtype=torch.float32).transpose(1, 2)
        dist = (cand_vals.squeeze(-1) - target.to(dtype=torch.float32)).square()
        best = torch.argmin(dist, dim=1)
        row_idx = torch.arange(int(x_block.shape[0]), device=x_block.device)
        out_vals = cand_vals[row_idx, best].to(dtype=x_block.dtype)
        if self.topk_q is None:
            return out_vals, None
        cand_q = self.topk_q[:, start:end, :].to(device=x_block.device, dtype=torch.long).transpose(1, 2)
        out_q = cand_q[row_idx, best].squeeze(-1).to(dtype=torch.int32)
        return out_vals, out_q


@dataclass
class E8P12Provider:
    cb: object
    metric_rows_per_chunk: int = 0
    _cache: Optional[Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = None
    _grid: Optional[torch.Tensor] = None

    def __post_init__(self) -> None:
        self._cache = {}

    def _grid_on(self, device: torch.device) -> torch.Tensor:
        if self._grid is None or self._grid.device != device:
            self._grid = self.cb.grid.to(device=device, dtype=torch.float32)
        return self._grid

    @torch.inference_mode()
    def _metric_map_assign(
        self,
        *,
        target: torch.Tensor,  # (B, 8)
        chol: torch.Tensor,  # (8,8)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        grid = self._grid_on(target.device)
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


@dataclass
class E8P12TopKProvider:
    cb: object
    topk_idxs: torch.Tensor  # (rows, nblocks, k)
    _grid: Optional[torch.Tensor] = None
    _topk_idxs_device: Optional[torch.Tensor] = None
    _topk_vals: Optional[torch.Tensor] = None
    _metric_cache: Optional[Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]]] = None
    _row_idx_cache: Optional[Dict[Tuple[torch.device, int], torch.Tensor]] = None

    def __post_init__(self) -> None:
        self._metric_cache = {}
        self._row_idx_cache = {}

    def _grid_on(self, device: torch.device) -> torch.Tensor:
        if self._grid is None or self._grid.device != device:
            self._grid = self.cb.grid.to(device=device, dtype=torch.float32)
        return self._grid

    def _topk_vals_on(self, device: torch.device) -> torch.Tensor:
        if self._topk_idxs_device is None or self._topk_idxs_device.device != device:
            self._topk_idxs_device = self.topk_idxs.to(device=device, dtype=getattr(self.cb, "idx_dtype", torch.int64))
            if self._metric_cache is not None:
                self._metric_cache.clear()
        if self._topk_vals is None or self._topk_vals.device != device:
            width = int(getattr(self.cb, "codesz", 8))
            grid = self._grid_on(device)
            idxs = self._topk_idxs_device
            self._topk_vals = grid.index_select(0, idxs.reshape(-1).to(dtype=torch.long)).view(*idxs.shape, width)
            if self._metric_cache is not None:
                self._metric_cache.clear()
        return self._topk_vals

    def _row_idx_on(self, device: torch.device, rows: int) -> torch.Tensor:
        key = (device, int(rows))
        if self._row_idx_cache is None:
            self._row_idx_cache = {}
        row_idx = self._row_idx_cache.get(key)
        if row_idx is None:
            row_idx = torch.arange(int(rows), device=device)
            self._row_idx_cache[key] = row_idx
        return row_idx

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
        rows = int(x_block.shape[0])
        width = int(end - start)
        codesz = int(getattr(self.cb, "codesz", 8))
        if width != codesz:
            raise ValueError(f"E8P12TopKProvider expects block_size={codesz}, got {width}.")

        cand_vals = self._topk_vals_on(x_block.device)[:, int(block_idx), :, :]
        cand_idxs = self._topk_idxs_device[:, int(block_idx), :]

        if metric == "gram":
            if gram_chol is None:
                raise ValueError("metric='gram' requires gram_chol.")
            chol = gram_chol.to(device=x_block.device, dtype=torch.float32)
            target_metric = target.to(dtype=torch.float32).matmul(chol)
            cache_key = (int(block_idx), int(chol.data_ptr()))
            cached = None
            if self._metric_cache is not None:
                cached = self._metric_cache.get(cache_key)
            cand_metric = None
            cand_metric_norm = None
            if cached is not None:
                cand_metric, cand_metric_norm = cached
            if (
                cand_metric is None
                or cand_metric_norm is None
                or cand_metric.device != x_block.device
                or cand_metric_norm.device != x_block.device
            ):
                cand_metric = cand_vals.to(dtype=torch.float32).matmul(chol)
                cand_metric_norm = cand_metric.square().sum(dim=-1)
                if self._metric_cache is not None:
                    self._metric_cache[cache_key] = (cand_metric, cand_metric_norm)
            best = _get_topk_metric_select(device=x_block.device)(cand_metric, cand_metric_norm, target_metric)
        else:
            dist = (cand_vals.to(dtype=torch.float32) - target.to(dtype=torch.float32).unsqueeze(1)).square().sum(dim=-1)
            best = torch.argmin(dist, dim=1)
        row_idx = self._row_idx_on(x_block.device, rows)
        out_vals = cand_vals[row_idx, best].to(dtype=x_block.dtype)
        out_idxs = cand_idxs[row_idx, best]
        return out_vals, out_idxs
