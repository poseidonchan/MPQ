from __future__ import annotations

from typing import Dict

import torch

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    triton = None
    tl = None
    _TRITON_AVAILABLE = False


def has_triton_vamp() -> bool:
    return bool(_TRITON_AVAILABLE)


if _TRITON_AVAILABLE:

    @triton.jit
    def _uniform_mmse_kernel(
        r1_ptr,
        gamma_ptr,
        scale_ptr,
        zero_ptr,
        center_ptr,
        mean_ptr,
        var_ptr,
        map_value_ptr,
        map_idx_ptr,
        n_rows,
        n_cols,
        r1_row_stride,
        gamma_row_stride,
        scale_row_stride,
        zero_row_stride,
        center_row_stride,
        out_row_stride,
        maxq,
        prior_precision,
        BLOCK_B: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_n = tl.program_id(1)
        rows = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
        cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = (rows[:, None] < n_rows) & (cols[None, :] < n_cols)

        r1 = tl.load(r1_ptr + rows[:, None] * r1_row_stride + cols[None, :], mask=mask, other=0.0).to(tl.float32)
        gamma = tl.load(gamma_ptr + rows * gamma_row_stride, mask=rows < n_rows, other=0.0).to(tl.float32)
        scale = tl.load(scale_ptr + rows[:, None] * scale_row_stride + cols[None, :], mask=mask, other=1.0).to(tl.float32)
        zero = tl.load(zero_ptr + rows[:, None] * zero_row_stride + cols[None, :], mask=mask, other=0.0).to(tl.float32)
        center = tl.load(center_ptr + rows[:, None] * center_row_stride + cols[None, :], mask=mask, other=0.0).to(tl.float32)
        scale = tl.maximum(scale, 1e-12)

        max_score = tl.full((BLOCK_B, BLOCK_N), -float("inf"), dtype=tl.float32)
        sum_w = tl.zeros((BLOCK_B, BLOCK_N), dtype=tl.float32)
        sum_x = tl.zeros((BLOCK_B, BLOCK_N), dtype=tl.float32)
        sum_x2 = tl.zeros((BLOCK_B, BLOCK_N), dtype=tl.float32)
        best_score = tl.full((BLOCK_B, BLOCK_N), -float("inf"), dtype=tl.float32)
        best_value = tl.zeros((BLOCK_B, BLOCK_N), dtype=tl.float32)
        best_idx = tl.zeros((BLOCK_B, BLOCK_N), dtype=tl.int64)

        q = 0
        while q <= maxq:
            qf = q + tl.zeros((BLOCK_B, BLOCK_N), dtype=tl.float32)
            atom = scale * (qf - zero)
            diff = atom - r1
            score = -0.5 * gamma[:, None] * diff * diff
            prior_diff = atom - center
            score = score - 0.5 * prior_precision * prior_diff * prior_diff / (scale * scale)
            score = tl.where(mask, score, -float("inf"))

            new_max = tl.maximum(max_score, score)
            old_w = tl.exp(max_score - new_max)
            new_w = tl.exp(score - new_max)
            sum_w = sum_w * old_w + new_w
            sum_x = sum_x * old_w + new_w * atom
            sum_x2 = sum_x2 * old_w + new_w * atom * atom
            max_score = new_max

            better = score > best_score
            best_score = tl.where(better, score, best_score)
            best_value = tl.where(better, atom, best_value)
            best_idx = tl.where(better, q, best_idx)
            q += 1

        denom = tl.maximum(sum_w, 1e-30)
        mean = sum_x / denom
        second = sum_x2 / denom
        var = tl.maximum(second - mean * mean, 0.0)

        out_ptrs = rows[:, None] * out_row_stride + cols[None, :]
        tl.store(mean_ptr + out_ptrs, mean, mask=mask)
        tl.store(var_ptr + out_ptrs, var, mask=mask)
        tl.store(map_value_ptr + out_ptrs, best_value, mask=mask)
        tl.store(map_idx_ptr + out_ptrs, best_idx, mask=mask)

    @triton.jit
    def _tanh_approx(x):
        return 2.0 * tl.sigmoid(2.0 * x) - 1.0

    @triton.jit
    def _e8p12_component(
        r1_ptr,
        gamma_ptr,
        base_abs_ptr,
        base_sqnorm_ptr,
        base_parity_ptr,
        idx_lut_ptr,
        item_offsets,
        item_mask,
        branch,
        base,
        offset,
        BLOCK_M: tl.constexpr,
    ):
        gamma = tl.load(gamma_ptr + item_offsets, mask=item_mask, other=0.0).to(tl.float32)

        r0 = tl.load(r1_ptr + item_offsets * 8 + 0, mask=item_mask, other=0.0).to(tl.float32) - offset
        r1 = tl.load(r1_ptr + item_offsets * 8 + 1, mask=item_mask, other=0.0).to(tl.float32) - offset
        r2 = tl.load(r1_ptr + item_offsets * 8 + 2, mask=item_mask, other=0.0).to(tl.float32) - offset
        r3 = tl.load(r1_ptr + item_offsets * 8 + 3, mask=item_mask, other=0.0).to(tl.float32) - offset
        r4 = tl.load(r1_ptr + item_offsets * 8 + 4, mask=item_mask, other=0.0).to(tl.float32) - offset
        r5 = tl.load(r1_ptr + item_offsets * 8 + 5, mask=item_mask, other=0.0).to(tl.float32) - offset
        r6 = tl.load(r1_ptr + item_offsets * 8 + 6, mask=item_mask, other=0.0).to(tl.float32) - offset
        r7 = tl.load(r1_ptr + item_offsets * 8 + 7, mask=item_mask, other=0.0).to(tl.float32) - offset

        b0 = tl.load(base_abs_ptr + base * 8 + 0).to(tl.float32)
        b1 = tl.load(base_abs_ptr + base * 8 + 1).to(tl.float32)
        b2 = tl.load(base_abs_ptr + base * 8 + 2).to(tl.float32)
        b3 = tl.load(base_abs_ptr + base * 8 + 3).to(tl.float32)
        b4 = tl.load(base_abs_ptr + base * 8 + 4).to(tl.float32)
        b5 = tl.load(base_abs_ptr + base * 8 + 5).to(tl.float32)
        b6 = tl.load(base_abs_ptr + base * 8 + 6).to(tl.float32)
        b7 = tl.load(base_abs_ptr + base * 8 + 7).to(tl.float32)
        base_sq = tl.load(base_sqnorm_ptr + base).to(tl.float32)
        base_parity = tl.load(base_parity_ptr + base).to(tl.int64)

        a0 = gamma * r0 * b0
        a1 = gamma * r1 * b1
        a2 = gamma * r2 * b2
        a3 = gamma * r3 * b3
        a4 = gamma * r4 * b4
        a5 = gamma * r5 * b5
        a6 = gamma * r6 * b6
        a7 = gamma * r7 * b7

        ab0 = tl.abs(a0)
        ab1 = tl.abs(a1)
        ab2 = tl.abs(a2)
        ab3 = tl.abs(a3)
        ab4 = tl.abs(a4)
        ab5 = tl.abs(a5)
        ab6 = tl.abs(a6)
        ab7 = tl.abs(a7)
        sum_abs = ab0 + ab1 + ab2 + ab3 + ab4 + ab5 + ab6 + ab7

        t0 = tl.maximum(tl.minimum(_tanh_approx(a0), 1.0 - 1e-7), -1.0 + 1e-7)
        t1 = tl.maximum(tl.minimum(_tanh_approx(a1), 1.0 - 1e-7), -1.0 + 1e-7)
        t2 = tl.maximum(tl.minimum(_tanh_approx(a2), 1.0 - 1e-7), -1.0 + 1e-7)
        t3 = tl.maximum(tl.minimum(_tanh_approx(a3), 1.0 - 1e-7), -1.0 + 1e-7)
        t4 = tl.maximum(tl.minimum(_tanh_approx(a4), 1.0 - 1e-7), -1.0 + 1e-7)
        t5 = tl.maximum(tl.minimum(_tanh_approx(a5), 1.0 - 1e-7), -1.0 + 1e-7)
        t6 = tl.maximum(tl.minimum(_tanh_approx(a6), 1.0 - 1e-7), -1.0 + 1e-7)
        t7 = tl.maximum(tl.minimum(_tanh_approx(a7), 1.0 - 1e-7), -1.0 + 1e-7)
        prod = t0 * t1 * t2 * t3 * t4 * t5 * t6 * t7

        logc = (
            ab0 + tl.log(1.0 + tl.exp(-2.0 * ab0))
            + ab1 + tl.log(1.0 + tl.exp(-2.0 * ab1))
            + ab2 + tl.log(1.0 + tl.exp(-2.0 * ab2))
            + ab3 + tl.log(1.0 + tl.exp(-2.0 * ab3))
            + ab4 + tl.log(1.0 + tl.exp(-2.0 * ab4))
            + ab5 + tl.log(1.0 + tl.exp(-2.0 * ab5))
            + ab6 + tl.log(1.0 + tl.exp(-2.0 * ab6))
            + ab7 + tl.log(1.0 + tl.exp(-2.0 * ab7))
        )
        log_half = 0.6931471805599453
        logz_even = logc + tl.log(tl.maximum(1.0 + prod, 1e-12)) - log_half
        logz_odd = logc + tl.log(tl.maximum(1.0 - prod, 1e-12)) - log_half
        parity_even = base_parity == 0
        logz_sign = tl.where(parity_even, logz_even, logz_odd)

        denom_even = tl.maximum(1.0 + prod, 1e-12)
        denom_odd = tl.maximum(1.0 - prod, 1e-12)
        e0 = t1 * t2 * t3 * t4 * t5 * t6 * t7
        e1 = t0 * t2 * t3 * t4 * t5 * t6 * t7
        e2 = t0 * t1 * t3 * t4 * t5 * t6 * t7
        e3 = t0 * t1 * t2 * t4 * t5 * t6 * t7
        e4 = t0 * t1 * t2 * t3 * t5 * t6 * t7
        e5 = t0 * t1 * t2 * t3 * t4 * t6 * t7
        e6 = t0 * t1 * t2 * t3 * t4 * t5 * t7
        e7 = t0 * t1 * t2 * t3 * t4 * t5 * t6

        sm0 = tl.where(parity_even, (t0 + e0) / denom_even, (t0 - e0) / denom_odd)
        sm1 = tl.where(parity_even, (t1 + e1) / denom_even, (t1 - e1) / denom_odd)
        sm2 = tl.where(parity_even, (t2 + e2) / denom_even, (t2 - e2) / denom_odd)
        sm3 = tl.where(parity_even, (t3 + e3) / denom_even, (t3 - e3) / denom_odd)
        sm4 = tl.where(parity_even, (t4 + e4) / denom_even, (t4 - e4) / denom_odd)
        sm5 = tl.where(parity_even, (t5 + e5) / denom_even, (t5 - e5) / denom_odd)
        sm6 = tl.where(parity_even, (t6 + e6) / denom_even, (t6 - e6) / denom_odd)
        sm7 = tl.where(parity_even, (t7 + e7) / denom_even, (t7 - e7) / denom_odd)

        mu_core0 = sm0 * b0
        mu_core1 = sm1 * b1
        mu_core2 = sm2 * b2
        mu_core3 = sm3 * b3
        mu_core4 = sm4 * b4
        mu_core5 = sm5 * b5
        mu_core6 = sm6 * b6
        mu_core7 = sm7 * b7
        mu0 = mu_core0 + offset
        mu1 = mu_core1 + offset
        mu2 = mu_core2 + offset
        mu3 = mu_core3 + offset
        mu4 = mu_core4 + offset
        mu5 = mu_core5 + offset
        mu6 = mu_core6 + offset
        mu7 = mu_core7 + offset

        y_norm = r0 * r0 + r1 * r1 + r2 * r2 + r3 * r3 + r4 * r4 + r5 * r5 + r6 * r6 + r7 * r7
        const = -0.5 * gamma * (y_norm + base_sq)
        log_component = const + logz_sign
        second_component = base_sq + 8.0 * offset * offset + 2.0 * offset * (
            mu_core0 + mu_core1 + mu_core2 + mu_core3 + mu_core4 + mu_core5 + mu_core6 + mu_core7
        )

        sign_bits_sum = (
            (a0 < 0.0).to(tl.int64)
            + (a1 < 0.0).to(tl.int64)
            + (a2 < 0.0).to(tl.int64)
            + (a3 < 0.0).to(tl.int64)
            + (a4 < 0.0).to(tl.int64)
            + (a5 < 0.0).to(tl.int64)
            + (a6 < 0.0).to(tl.int64)
            + (a7 < 0.0).to(tl.int64)
        )
        best_parity = sign_bits_sum & 1
        mismatch = best_parity != base_parity
        sign_id = (
            (a0 < 0.0).to(tl.int64)
            + 2 * (a1 < 0.0).to(tl.int64)
            + 4 * (a2 < 0.0).to(tl.int64)
            + 8 * (a3 < 0.0).to(tl.int64)
            + 16 * (a4 < 0.0).to(tl.int64)
            + 32 * (a5 < 0.0).to(tl.int64)
            + 64 * (a6 < 0.0).to(tl.int64)
            + 128 * (a7 < 0.0).to(tl.int64)
        )
        min_abs = ab0
        min_bit = tl.full((BLOCK_M,), 1, dtype=tl.int64)
        take = ab1 < min_abs
        min_abs = tl.where(take, ab1, min_abs)
        min_bit = tl.where(take, 2, min_bit)
        take = ab2 < min_abs
        min_abs = tl.where(take, ab2, min_abs)
        min_bit = tl.where(take, 4, min_bit)
        take = ab3 < min_abs
        min_abs = tl.where(take, ab3, min_abs)
        min_bit = tl.where(take, 8, min_bit)
        take = ab4 < min_abs
        min_abs = tl.where(take, ab4, min_abs)
        min_bit = tl.where(take, 16, min_bit)
        take = ab5 < min_abs
        min_abs = tl.where(take, ab5, min_abs)
        min_bit = tl.where(take, 32, min_bit)
        take = ab6 < min_abs
        min_abs = tl.where(take, ab6, min_abs)
        min_bit = tl.where(take, 64, min_bit)
        take = ab7 < min_abs
        min_abs = tl.where(take, ab7, min_abs)
        min_bit = tl.where(take, 128, min_bit)
        sign_id = tl.where(mismatch, sign_id ^ min_bit, sign_id)
        best_linear = sum_abs - 2.0 * mismatch.to(tl.float32) * min_abs
        best_logscore = const + best_linear
        map_idx = tl.load(
            idx_lut_ptr + branch * 65536 + base * 256 + sign_id,
            mask=item_mask,
            other=0,
        ).to(tl.int64)

        return log_component, best_logscore, map_idx, mu0, mu1, mu2, mu3, mu4, mu5, mu6, mu7, second_component

    @triton.jit
    def _e8p12_mmse_kernel(
        r1_ptr,
        gamma_ptr,
        grid_ptr,
        base_abs_ptr,
        base_sqnorm_ptr,
        base_parity_ptr,
        idx_lut_ptr,
        branch_offsets_ptr,
        x1_ptr,
        var_ptr,
        map_idx_ptr,
        map_vals_ptr,
        n_items,
        BLOCK_M: tl.constexpr,
    ):
        pid = tl.program_id(0)
        item_offsets = pid * BLOCK_M + tl.arange(0, BLOCK_M)
        item_mask = item_offsets < n_items

        max_log = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
        best_score = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
        best_idx = tl.zeros((BLOCK_M,), dtype=tl.int64)

        branch = 0
        while branch < 2:
            offset = tl.load(branch_offsets_ptr + branch).to(tl.float32)
            base = 0
            while base < 256:
                log_component, score, idx, _m0, _m1, _m2, _m3, _m4, _m5, _m6, _m7, _second = _e8p12_component(
                    r1_ptr,
                    gamma_ptr,
                    base_abs_ptr,
                    base_sqnorm_ptr,
                    base_parity_ptr,
                    idx_lut_ptr,
                    item_offsets,
                    item_mask,
                    branch,
                    base,
                    offset,
                    BLOCK_M,
                )
                log_component = tl.where(item_mask, log_component, -float("inf"))
                score = tl.where(item_mask, score, -float("inf"))
                max_log = tl.maximum(max_log, log_component)
                better = score > best_score
                best_score = tl.where(better, score, best_score)
                best_idx = tl.where(better, idx, best_idx)
                base += 1
            branch += 1

        sum_w = tl.zeros((BLOCK_M,), dtype=tl.float32)
        mu0_acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
        mu1_acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
        mu2_acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
        mu3_acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
        mu4_acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
        mu5_acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
        mu6_acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
        mu7_acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
        second_acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

        branch = 0
        while branch < 2:
            offset = tl.load(branch_offsets_ptr + branch).to(tl.float32)
            base = 0
            while base < 256:
                log_component, _score, _idx, m0, m1, m2, m3, m4, m5, m6, m7, second = _e8p12_component(
                    r1_ptr,
                    gamma_ptr,
                    base_abs_ptr,
                    base_sqnorm_ptr,
                    base_parity_ptr,
                    idx_lut_ptr,
                    item_offsets,
                    item_mask,
                    branch,
                    base,
                    offset,
                    BLOCK_M,
                )
                w = tl.exp(log_component - max_log)
                w = tl.where(item_mask, w, 0.0)
                sum_w += w
                mu0_acc += w * m0
                mu1_acc += w * m1
                mu2_acc += w * m2
                mu3_acc += w * m3
                mu4_acc += w * m4
                mu5_acc += w * m5
                mu6_acc += w * m6
                mu7_acc += w * m7
                second_acc += w * second
                base += 1
            branch += 1

        denom = tl.maximum(sum_w, 1e-30)
        mu0_acc /= denom
        mu1_acc /= denom
        mu2_acc /= denom
        mu3_acc /= denom
        mu4_acc /= denom
        mu5_acc /= denom
        mu6_acc /= denom
        mu7_acc /= denom
        second_acc /= denom

        map0 = tl.load(grid_ptr + best_idx * 8 + 0, mask=item_mask, other=0.0).to(tl.float32)
        map1 = tl.load(grid_ptr + best_idx * 8 + 1, mask=item_mask, other=0.0).to(tl.float32)
        map2 = tl.load(grid_ptr + best_idx * 8 + 2, mask=item_mask, other=0.0).to(tl.float32)
        map3 = tl.load(grid_ptr + best_idx * 8 + 3, mask=item_mask, other=0.0).to(tl.float32)
        map4 = tl.load(grid_ptr + best_idx * 8 + 4, mask=item_mask, other=0.0).to(tl.float32)
        map5 = tl.load(grid_ptr + best_idx * 8 + 5, mask=item_mask, other=0.0).to(tl.float32)
        map6 = tl.load(grid_ptr + best_idx * 8 + 6, mask=item_mask, other=0.0).to(tl.float32)
        map7 = tl.load(grid_ptr + best_idx * 8 + 7, mask=item_mask, other=0.0).to(tl.float32)

        gamma = tl.load(gamma_ptr + item_offsets, mask=item_mask, other=0.0).to(tl.float32)
        snap = gamma >= 1000.0
        out0 = tl.where(snap, map0, mu0_acc)
        out1 = tl.where(snap, map1, mu1_acc)
        out2 = tl.where(snap, map2, mu2_acc)
        out3 = tl.where(snap, map3, mu3_acc)
        out4 = tl.where(snap, map4, mu4_acc)
        out5 = tl.where(snap, map5, mu5_acc)
        out6 = tl.where(snap, map6, mu6_acc)
        out7 = tl.where(snap, map7, mu7_acc)

        mu_norm = (
            out0 * out0 + out1 * out1 + out2 * out2 + out3 * out3
            + out4 * out4 + out5 * out5 + out6 * out6 + out7 * out7
        )
        block_var = tl.maximum((second_acc - mu_norm) / 8.0, 0.0)
        block_var = tl.where(snap, 0.0, block_var)

        tl.store(x1_ptr + item_offsets * 8 + 0, out0, mask=item_mask)
        tl.store(x1_ptr + item_offsets * 8 + 1, out1, mask=item_mask)
        tl.store(x1_ptr + item_offsets * 8 + 2, out2, mask=item_mask)
        tl.store(x1_ptr + item_offsets * 8 + 3, out3, mask=item_mask)
        tl.store(x1_ptr + item_offsets * 8 + 4, out4, mask=item_mask)
        tl.store(x1_ptr + item_offsets * 8 + 5, out5, mask=item_mask)
        tl.store(x1_ptr + item_offsets * 8 + 6, out6, mask=item_mask)
        tl.store(x1_ptr + item_offsets * 8 + 7, out7, mask=item_mask)
        tl.store(var_ptr + item_offsets, block_var, mask=item_mask)
        tl.store(map_idx_ptr + item_offsets, best_idx, mask=item_mask)
        tl.store(map_vals_ptr + item_offsets * 8 + 0, map0, mask=item_mask)
        tl.store(map_vals_ptr + item_offsets * 8 + 1, map1, mask=item_mask)
        tl.store(map_vals_ptr + item_offsets * 8 + 2, map2, mask=item_mask)
        tl.store(map_vals_ptr + item_offsets * 8 + 3, map3, mask=item_mask)
        tl.store(map_vals_ptr + item_offsets * 8 + 4, map4, mask=item_mask)
        tl.store(map_vals_ptr + item_offsets * 8 + 5, map5, mask=item_mask)
        tl.store(map_vals_ptr + item_offsets * 8 + 6, map6, mask=item_mask)
        tl.store(map_vals_ptr + item_offsets * 8 + 7, map7, mask=item_mask)


@torch.inference_mode()
def uniform_mmse_triton(
    *,
    r1: torch.Tensor,
    gamma1: torch.Tensor,
    scale_full: torch.Tensor,
    zero_full: torch.Tensor,
    maxq: int,
    prior_center: torch.Tensor,
    prior_tau: float,
) -> Dict[str, torch.Tensor | float | str]:
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton uniform VAMP backend requested but Triton is not available.")
    if r1.device.type != "cuda":
        raise ValueError("Triton uniform VAMP backend requires CUDA tensors.")

    r1_f32 = r1.to(dtype=torch.float32).contiguous()
    gamma_f32 = gamma1.to(device=r1.device, dtype=torch.float32).contiguous()
    scale_f32 = scale_full.to(device=r1.device, dtype=torch.float32).contiguous()
    zero_f32 = zero_full.to(device=r1.device, dtype=torch.float32).contiguous()
    center_f32 = prior_center.to(device=r1.device, dtype=torch.float32).contiguous()

    if gamma_f32.ndim != 2 or int(gamma_f32.shape[1]) != 1:
        raise ValueError(f"expected gamma1 with shape (B,1), got {tuple(gamma1.shape)}")
    if r1_f32.shape != scale_f32.shape or r1_f32.shape != zero_f32.shape or r1_f32.shape != center_f32.shape:
        raise ValueError("r1, scale_full, zero_full, and prior_center must have matching shapes.")

    rows, cols = r1_f32.shape
    mean = torch.empty_like(r1_f32, dtype=torch.float32)
    var = torch.empty_like(r1_f32, dtype=torch.float32)
    map_value = torch.empty_like(r1_f32, dtype=torch.float32)
    map_idx = torch.empty((rows, cols), dtype=torch.long, device=r1.device)
    prior_precision = 0.0 if float(prior_tau) <= 0.0 else 1.0 / (float(prior_tau) * float(prior_tau))

    block_b = 8
    block_n = 128
    grid = (triton.cdiv(rows, block_b), triton.cdiv(cols, block_n))
    _uniform_mmse_kernel[grid](
        r1_f32,
        gamma_f32,
        scale_f32,
        zero_f32,
        center_f32,
        mean,
        var,
        map_value,
        map_idx,
        int(rows),
        int(cols),
        r1_f32.stride(0),
        gamma_f32.stride(0),
        scale_f32.stride(0),
        zero_f32.stride(0),
        center_f32.stride(0),
        mean.stride(0),
        int(maxq),
        float(prior_precision),
        BLOCK_B=block_b,
        BLOCK_N=block_n,
        num_warps=4,
    )
    alpha1 = gamma_f32 * var.mean(dim=1, keepdim=True)
    return {
        "mean": mean,
        "var": var,
        "map_value": map_value,
        "map_idx": map_idx,
        "alpha1": alpha1.to(dtype=torch.float32),
        "entropy": 0.0,
        "margin": 0.0,
        "vamp_backend": "triton",
    }


@torch.inference_mode()
def e8p12_factorized_mmse_triton(
    *,
    r1_work: torch.Tensor,
    gamma_work: torch.Tensor,
    gamma1: torch.Tensor,
    grid_values: torch.Tensor,
    cache: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor | float | str]:
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton E8P12 VAMP backend requested but Triton is not available.")
    if r1_work.device.type != "cuda":
        raise ValueError("Triton E8P12 VAMP backend requires CUDA tensors.")

    batch, n_blocks, codesz = r1_work.shape
    if int(codesz) != 8:
        raise ValueError(f"E8P12 Triton VAMP expects codesz=8, got {codesz}.")

    r1_flat = r1_work.to(dtype=torch.float32).contiguous().view(batch * n_blocks, codesz)
    gamma_flat = gamma_work.to(device=r1_work.device, dtype=torch.float32).contiguous().view(batch * n_blocks)
    grid_f32 = grid_values.to(device=r1_work.device, dtype=torch.float32).contiguous()
    base_abs = cache["base_abs"].to(device=r1_work.device, dtype=torch.float32).contiguous()
    base_sqnorm = cache["base_sqnorm"].to(device=r1_work.device, dtype=torch.float32).contiguous()
    base_parity = cache["base_parity"].to(device=r1_work.device, dtype=torch.long).contiguous()
    idx_lut = cache["idx_lut"].to(device=r1_work.device, dtype=torch.long).contiguous()
    branch_offsets = cache["branch_offsets"].to(device=r1_work.device, dtype=torch.float32).contiguous()

    total = int(batch * n_blocks)
    x1_flat = torch.empty((total, codesz), device=r1_work.device, dtype=torch.float32)
    var_flat = torch.empty((total,), device=r1_work.device, dtype=torch.float32)
    map_idxs_flat = torch.empty((total,), device=r1_work.device, dtype=torch.long)
    map_vals_flat = torch.empty((total, codesz), device=r1_work.device, dtype=torch.float32)

    block_m = 8
    launch_grid = (triton.cdiv(total, block_m),)
    _e8p12_mmse_kernel[launch_grid](
        r1_flat,
        gamma_flat,
        grid_f32,
        base_abs,
        base_sqnorm,
        base_parity,
        idx_lut,
        branch_offsets,
        x1_flat,
        var_flat,
        map_idxs_flat,
        map_vals_flat,
        int(total),
        BLOCK_M=block_m,
        num_warps=4,
    )

    var1 = var_flat.view(batch, n_blocks)
    alpha1 = gamma1.to(device=r1_work.device, dtype=torch.float32) * var1.mean(dim=1, keepdim=True)
    return {
        "x1_blocks": x1_flat.view(batch, n_blocks, codesz),
        "var1": var1,
        "map_idxs": map_idxs_flat.view(batch, n_blocks),
        "map_vals": map_vals_flat.view(batch, n_blocks, codesz),
        "alpha1": alpha1.to(dtype=torch.float32),
        "entropy": 0.0,
        "margin": 0.0,
        "prior_backend": "triton",
    }
