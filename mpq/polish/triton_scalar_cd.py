from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from ..linear import hessian_objective_mean

try:
    import triton
    import triton.language as tl
    from triton.language.extra import cuda as tl_cuda_extra

    _TRITON_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    triton = None
    tl = None
    tl_cuda_extra = None
    _TRITON_AVAILABLE = False


def has_triton_scalar_cd() -> bool:
    return bool(_TRITON_AVAILABLE)


if _TRITON_AVAILABLE:

    @triton.jit
    def _scalar_cd_persistent_sweep_kernel(
        rhs_ptr,
        rhs_row_stride,
        x_ptr,
        x_row_stride,
        scale_ptr,
        scale_row_stride,
        zero_ptr,
        zero_row_stride,
        residual_ptr,
        accepted_ptr,
        order_ptr,
        gram_ptr,
        gram_row_stride,
        diag_ptr,
        n_rows,
        n_cols,
        maxq,
        accept_tol,
        BLOCK_B: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid = tl.program_id(0)
        row_offsets = pid * BLOCK_B + tl.arange(0, BLOCK_B)
        row_mask = row_offsets < n_rows

        residual_norm = tl.load(residual_ptr + row_offsets, mask=row_mask, other=0.0).to(tl.float32)
        accepted_local = tl.zeros([BLOCK_B], dtype=tl.int32)

        ord_idx = 0
        while ord_idx < n_cols:
            col = tl.load(order_ptr + ord_idx).to(tl.int64)

            x_ptrs = x_ptr + row_offsets * x_row_stride + col
            rhs_ptrs = rhs_ptr + row_offsets * rhs_row_stride + col
            scale_ptrs = scale_ptr + row_offsets * scale_row_stride + col
            zero_ptrs = zero_ptr + row_offsets * zero_row_stride + col

            x_col = tl.load(x_ptrs, mask=row_mask, other=0.0).to(tl.float32)
            rhs_col = tl.load(rhs_ptrs, mask=row_mask, other=0.0).to(tl.float32)
            scale_col = tl.maximum(tl.load(scale_ptrs, mask=row_mask, other=1.0).to(tl.float32), 1e-12)
            zero_col = tl.load(zero_ptrs, mask=row_mask, other=0.0).to(tl.float32)
            diag_col = tl.load(diag_ptr + col).to(tl.float32)

            target = x_col + rhs_col / diag_col
            qf = tl_cuda_extra.libdevice.rint(target / scale_col + zero_col)
            qf = tl.maximum(tl.minimum(qf, tl.full([BLOCK_B], maxq, dtype=tl.float32)), 0.0)
            q = qf.to(tl.int32)
            xq = scale_col * (q.to(tl.float32) - zero_col)
            delta = xq - x_col
            cand_err = tl.maximum(residual_norm + delta * delta * diag_col - 2.0 * rhs_col * delta, 0.0)
            accept = (cand_err + 1e-12) < residual_norm * (1.0 + accept_tol)

            tl.store(x_ptrs, tl.where(accept, xq, x_col), mask=row_mask)
            residual_norm = tl.where(accept, cand_err, residual_norm)
            delta_masked = tl.where(accept, delta, 0.0)
            accepted_local += accept.to(tl.int32)

            chunk_start = 0
            while chunk_start < n_cols:
                col_offsets = chunk_start + tl.arange(0, BLOCK_N)
                col_mask = col_offsets < n_cols
                h_ptrs = gram_ptr + col * gram_row_stride + col_offsets
                h_row = tl.load(h_ptrs, mask=col_mask, other=0.0).to(tl.float32)
                rhs_chunk_ptrs = rhs_ptr + row_offsets[:, None] * rhs_row_stride + col_offsets[None, :]
                rhs_chunk = tl.load(
                    rhs_chunk_ptrs,
                    mask=row_mask[:, None] & col_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
                rhs_chunk = rhs_chunk - delta_masked[:, None] * h_row[None, :]
                tl.store(
                    rhs_chunk_ptrs,
                    rhs_chunk,
                    mask=row_mask[:, None] & col_mask[None, :],
                )
                chunk_start += BLOCK_N

            ord_idx += 1

        tl.store(residual_ptr + row_offsets, residual_norm, mask=row_mask)
        tl.store(accepted_ptr + row_offsets, accepted_local, mask=row_mask)


def _launch_persistent_scalar_cd_sweep(
    *,
    rhs_all: torch.Tensor,
    x_work: torch.Tensor,
    scale_f32: torch.Tensor,
    zero_f32: torch.Tensor,
    residual_norm: torch.Tensor,
    order: torch.Tensor,
    gram_f32: torch.Tensor,
    diag: torch.Tensor,
    maxq: int,
    accept_tol: float,
) -> int:
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton persistent scalar CD requested but Triton is not available.")
    if rhs_all.device.type != "cuda":
        raise ValueError("Triton persistent scalar CD requires CUDA tensors.")
    if not rhs_all.is_contiguous():
        raise ValueError("rhs_all must be contiguous for Triton persistent scalar CD.")

    rows, cols = rhs_all.shape
    accepted = torch.zeros(rows, device=rhs_all.device, dtype=torch.int32)
    block_b = 32 if rows >= 64 else 16
    block_n = 128 if cols <= 2048 else 256
    grid = (triton.cdiv(rows, block_b),)
    _scalar_cd_persistent_sweep_kernel[grid](
        rhs_all,
        rhs_all.stride(0),
        x_work,
        x_work.stride(0),
        scale_f32,
        scale_f32.stride(0),
        zero_f32,
        zero_f32.stride(0),
        residual_norm,
        accepted,
        order,
        gram_f32,
        gram_f32.stride(0),
        diag,
        rows,
        cols,
        int(maxq),
        float(accept_tol),
        BLOCK_B=block_b,
        BLOCK_N=block_n,
        num_warps=4 if block_b <= 16 else 8,
    )
    return int(accepted.sum().item())


@torch.inference_mode()
def uniform_scalar_diag_greedy_cd(
    *,
    x: torch.Tensor,
    Y_u: torch.Tensor,
    Vs: torch.Tensor,
    gram_full: torch.Tensor,
    scale: torch.Tensor,
    zero: torch.Tensor,
    maxq: int,
    sweeps: int,
    accept_tol: float = 0.0,
    update_backend: str = "torch",
    raw_target: Optional[torch.Tensor] = None,
    raw_gram: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, Dict[str, Any]]:
    """Exact scalar diagonal greedy CD with an optional Triton persistent sweep."""

    backend = str(update_backend or "torch").strip().lower()
    if backend not in {"torch", "triton"}:
        raise ValueError(f"unsupported update_backend={update_backend!r}; expected 'torch'|'triton'.")
    if backend == "triton" and not _TRITON_AVAILABLE:
        raise RuntimeError("Triton scalar CD backend requested but Triton is not available.")

    x_work = x.to(dtype=torch.float32).clone().contiguous()
    Y_u_f32 = Y_u.to(dtype=torch.float32)
    Vs_f32 = Vs.to(dtype=torch.float32)
    gram_f32 = gram_full.to(device=x_work.device, dtype=torch.float32).contiguous()
    scale_f32 = scale.to(device=x_work.device, dtype=torch.float32).contiguous()
    zero_f32 = zero.to(device=x_work.device, dtype=torch.float32).contiguous()

    residual_u = Y_u_f32 - x_work.matmul(Vs_f32)
    rhs_all = residual_u.matmul(Vs_f32.T.contiguous()).contiguous()
    residual_norm = residual_u.square().sum(dim=1).contiguous()
    diag = Vs_f32.square().sum(dim=1).clamp_min_(1e-12).contiguous()
    diag_inv = diag.reciprocal()
    q_initial = torch.round(
        x_work / scale_f32.clamp_min(1e-12) + zero_f32
    ).clamp_(0, int(maxq)).to(dtype=torch.int32)

    accepted_count = 0
    zero_delta = torch.zeros(int(x_work.shape[0]), device=x_work.device, dtype=torch.float32)
    if (raw_target is None) != (raw_gram is None):
        raise ValueError("raw_target and raw_gram must be provided together.")

    def trace_item(
        *,
        budget: int,
        accepted_updates_sweep: int,
        accepted_updates_cumulative: int,
        assignment_changes_from_cd0: int,
    ) -> dict[str, float | int]:
        item: dict[str, float | int] = {
            "budget": int(budget),
            "stabilized_objective_mean": float(residual_norm.mean().item()),
            "accepted_updates_sweep": int(accepted_updates_sweep),
            "accepted_updates_cumulative": int(accepted_updates_cumulative),
            "assignment_changes_from_cd0": int(assignment_changes_from_cd0),
            "assignments_total": int(q_initial.numel()),
        }
        if raw_target is not None and raw_gram is not None:
            item["raw_objective_mean"] = hessian_objective_mean(
                target=raw_target,
                x=x_work,
                gram=raw_gram,
            )
        return item

    trace: list[dict[str, float | int]] = [
        trace_item(
            budget=0,
            accepted_updates_sweep=0,
            accepted_updates_cumulative=0,
            assignment_changes_from_cd0=0,
        )
    ]

    for sweep in range(max(int(sweeps), 0)):
        gain = (rhs_all.square() * diag_inv.unsqueeze(0)).sum(dim=0) / float(max(int(rhs_all.shape[0]), 1))
        order = torch.argsort(gain, descending=True)
        accepted_before = int(accepted_count)
        if backend == "triton":
            accepted_count += _launch_persistent_scalar_cd_sweep(
                rhs_all=rhs_all,
                x_work=x_work,
                scale_f32=scale_f32,
                zero_f32=zero_f32,
                residual_norm=residual_norm,
                order=order.to(device=x_work.device, dtype=torch.int32).contiguous(),
                gram_f32=gram_f32,
                diag=diag,
                maxq=int(maxq),
                accept_tol=float(accept_tol),
            )
        else:
            for col in order.tolist():
                rhs = rhs_all[:, col]
                x_col = x_work[:, col]
                scale_col = scale_f32[:, col].clamp_min(1e-12)
                zero_col = zero_f32[:, col]
                diag_col = diag[col]
                target = x_col + rhs * diag_inv[col]
                q = torch.round(target / scale_col + zero_col)
                q = torch.clamp(q, 0, int(maxq)).to(dtype=torch.int32)
                xq = scale_col * (q.to(dtype=torch.float32) - zero_col)
                delta = xq - x_col
                cand_err = (
                    residual_norm
                    + delta.square() * diag_col
                    - 2.0 * rhs * delta
                ).clamp_min_(0.0)
                accept = (
                    cand_err + 1e-12
                    < residual_norm * (1.0 + float(accept_tol))
                )
                if not bool(accept.any().item()):
                    continue

                x_work[:, col] = torch.where(accept, xq, x_col)
                residual_norm = torch.where(accept, cand_err, residual_norm)
                delta_masked = torch.where(accept, delta, zero_delta).contiguous()
                accepted_count += int(accept.sum().item())
                rhs_all.sub_(
                    delta_masked.unsqueeze(1) * gram_f32[col].unsqueeze(0)
                )

        q_current = torch.round(
            x_work / scale_f32.clamp_min(1e-12) + zero_f32
        ).clamp_(0, int(maxq)).to(dtype=torch.int32)
        trace.append(
            trace_item(
                budget=int(sweep + 1),
                accepted_updates_sweep=int(accepted_count - accepted_before),
                accepted_updates_cumulative=int(accepted_count),
                assignment_changes_from_cd0=int(
                    (q_current != q_initial).to(dtype=torch.int64).sum().item()
                ),
            )
        )

    return x_work.to(dtype=x.dtype), {
        "accepted_updates": float(accepted_count),
        "cd_trace": trace,
    }
