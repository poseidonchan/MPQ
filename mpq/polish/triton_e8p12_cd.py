"""Persistent E8P12 block CD sweep kernel (triton).

Mirrors GPTQ's scalar persistent kernel in `triton_scalar_cd.py`, but each
"column step" is replaced with an 8-dim block step whose inner codebook
search scans the full E8P12 grid (K=65536 entries, 8-d) via tensor cores.

Kernel layout
-------------
  program_id(0) -> chunk of BLOCK_B rows, iterates over all n_blocks in order
  Inner per-block:
    1. load x_blk (BLOCK_B, 8), rhs_blk (BLOCK_B, 8)
    2. target = x_blk + rhs_blk @ gram_inv[bi]          (8x8 small matmul)
    3. target_metric = target @ chol[bi]                 (8x8)
    4. codebook search: for k tile in [0, K):
         cross = target_metric @ metric_grid[bi, k].T    (tl.dot, B x 16 @ 16 x K_TILE)
         scores = ||target_metric||^2 + metric_grid_norm[bi, k] - 2*cross
         track running min and argmin
    5. delta = grid[best] - x_blk
    6. cross_r = (rhs_blk * delta).sum(1)
    7. delta_gram = delta @ gram_raw[bi]                  (8x8)
    8. delta_norm = (delta_gram * delta).sum(1).clamp_min(0)
    9. cand_err = residual_norm + delta_norm - 2 cross_r; clamp_min(0)
    10. accept = cand_err + eps < residual_norm * (1+tol)
    11. store x_blk <- (accept? grid[best] : x_blk), idxs, residual_norm
    12. residual update (rhs_all -= delta @ gram[bi, :]) over all blocks:
        iterate chunks of BLOCK_N cols; gram_ptr[8*bi:8*bi+8, col_start:col_start+BLOCK_N]
        is a (8, BLOCK_N) tile.

Codebook padding
----------------
tl.dot requires contraction K>=16. E8P12 codesz=8, so we ask the caller to
pad `metric_grid` to width 16 (last 8 columns zero) and pad `target_metric`
similarly; dot product is unchanged.

The precomputed per-block `metric_grid_stack` lives in GMEM as
    (n_blocks, K, 16)  f32
which is 3500 * 65536 * 16 * 4 ≈ 15 GB for Llama-70B down_proj. That is far too
much. So we do NOT precompute; instead, we keep the raw global codebook
`grid (K, 8)` and compute `metric_grid_bi = grid @ chol[bi]` ON-THE-FLY per
block (tl.dot of (K, 8) @ (8, 8) at block-entry). cost: 65536*8*8*1 fp = 4M
flops per block, amortized across BLOCK_B rows.

Actually for 3584 blocks this is still 14G flops per sweep just for transforms.
Better: move chol INTO the target side: target'= chol @ target, then score is
||grid[k] - target'_new||^2 in a different metric. Doesn't help since we still
need a per-block whitening.

Simplest and still-fast: keep grid unwhitened; compute scores as
    score[k] = (target - grid[k])^T H (target - grid[k])
             = target^T H target + grid[k]^T H grid[k] - 2 target^T H grid[k]
the "grid Hessian form" grid[k]^T H grid[k] is a 65536-vector per block;
computing it is K * (8 H matvec + dot) ≈ 65536 * 128 = 8M flops per block;
we'd need it per block step → expensive.

DECISION: stream chol-transformed grid. Per-block we re-whiten codebook with
one (K, 8) @ (8, 8) matmul once at block entry. Keep metric_grid resident
in shared + registers for the block's full search phase. With BLOCK_K=128
and K=65536, we do K/BLOCK_K = 512 matmul dots per block; all fit in shared.
"""
from __future__ import annotations

from typing import Tuple

import torch

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except Exception:  # pragma: no cover
    triton = None
    tl = None
    _TRITON_AVAILABLE = False


def has_triton_e8p12_cd() -> bool:
    return bool(_TRITON_AVAILABLE)


if _TRITON_AVAILABLE:

    @triton.jit
    def _e8p12_cd_codebook_search_kernel(
        target_metric_ptr,      # (B, 16) f32 padded (caller must zero cols 8..15)
        metric_grid_ptr,        # (K, 16) f32 padded
        metric_grid_norm_ptr,   # (K,)    f32
        best_score_ptr,         # (B,)    f32 out
        best_idx_ptr,           # (B,)    i32 out
        n_rows,
        n_codes,
        BLOCK_B: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid = tl.program_id(0)
        row_offsets = pid * BLOCK_B + tl.arange(0, BLOCK_B)
        row_mask = row_offsets < n_rows

        col_offs = tl.arange(0, 16)
        tgt_ptrs = target_metric_ptr + row_offsets[:, None] * 16 + col_offs[None, :]
        tgt = tl.load(tgt_ptrs, mask=row_mask[:, None], other=0.0).to(tl.float32)
        tgt_norm = tl.sum(tgt * tgt, axis=1)

        best_score = tl.full([BLOCK_B], float("inf"), dtype=tl.float32)
        best_idx = tl.zeros([BLOCK_B], dtype=tl.int32)

        k_start = 0
        while k_start < n_codes:
            k_offs = k_start + tl.arange(0, BLOCK_K)
            k_mask = k_offs < n_codes
            mg_ptrs = metric_grid_ptr + k_offs[:, None] * 16 + col_offs[None, :]
            mg = tl.load(mg_ptrs, mask=k_mask[:, None], other=0.0).to(tl.float32)
            mgn = tl.load(metric_grid_norm_ptr + k_offs, mask=k_mask, other=float("inf")).to(tl.float32)

            # cross = tgt @ mg.T  on tensor cores in IEEE fp32 (no TF32 loss)
            cross = tl.dot(tgt, tl.trans(mg), input_precision="ieee")
            scores = tgt_norm[:, None] + mgn[None, :] - 2.0 * cross
            scores = tl.where(k_mask[None, :], scores, float("inf"))

            tile_min = tl.min(scores, axis=1)
            tile_argmin = tl.argmin(scores, axis=1).to(tl.int32) + k_start

            update = tile_min < best_score
            best_score = tl.where(update, tile_min, best_score)
            best_idx = tl.where(update, tile_argmin, best_idx)

            k_start += BLOCK_K

        tl.store(best_score_ptr + row_offsets, best_score, mask=row_mask)
        tl.store(best_idx_ptr + row_offsets, best_idx, mask=row_mask)


def e8p12_codebook_search_triton(
    *,
    target_metric: torch.Tensor,        # (B, 8) f32
    metric_grid: torch.Tensor,          # (K, 8) f32 (= grid @ chol)
    metric_grid_norm: torch.Tensor,     # (K,)   f32
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Find the best E8P12 codebook index per row under whitened metric.

    score[b, k] = ||target_metric[b] - metric_grid[k]||^2
                = ||target_metric[b]||^2 + ||metric_grid[k]||^2 - 2<.,.>
    """
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton not available.")
    if target_metric.device.type != "cuda":
        raise ValueError("target_metric must be on CUDA.")
    assert target_metric.dim() == 2 and target_metric.shape[1] == 8
    assert metric_grid.dim() == 2 and metric_grid.shape[1] == 8
    assert metric_grid_norm.dim() == 1 and metric_grid_norm.shape[0] == metric_grid.shape[0]

    B = int(target_metric.shape[0])
    K = int(metric_grid.shape[0])

    target_padded = torch.zeros(B, 16, device=target_metric.device, dtype=torch.float32)
    target_padded[:, :8] = target_metric
    metric_padded = torch.zeros(K, 16, device=metric_grid.device, dtype=torch.float32)
    metric_padded[:, :8] = metric_grid
    target_padded = target_padded.contiguous()
    metric_padded = metric_padded.contiguous()

    best_score = torch.empty(B, device=target_metric.device, dtype=torch.float32)
    best_idx = torch.empty(B, device=target_metric.device, dtype=torch.int32)

    BLOCK_B = 32 if B >= 32 else 16
    BLOCK_K = 128
    grid = (triton.cdiv(B, BLOCK_B),)
    _e8p12_cd_codebook_search_kernel[grid](
        target_padded,
        metric_padded,
        metric_grid_norm,
        best_score,
        best_idx,
        B,
        K,
        BLOCK_B=BLOCK_B,
        BLOCK_K=BLOCK_K,
        num_warps=4,
    )
    return best_score, best_idx.long()
