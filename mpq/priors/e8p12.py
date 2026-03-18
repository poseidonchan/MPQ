from __future__ import annotations

import math
from typing import Callable, Dict, Optional

import torch


_COMPILED_BRANCH_FACTORIZED_STATS: Optional[Callable[..., Dict[str, torch.Tensor]]] = None
_COMPILED_BRANCH_FACTORIZED_STATS_FAILED = False


def _disable_compiled_branch_factorized_stats() -> None:
    global _COMPILED_BRANCH_FACTORIZED_STATS, _COMPILED_BRANCH_FACTORIZED_STATS_FAILED
    _COMPILED_BRANCH_FACTORIZED_STATS = None
    _COMPILED_BRANCH_FACTORIZED_STATS_FAILED = True


def _get_compiled_branch_factorized_stats() -> Optional[Callable[..., Dict[str, torch.Tensor]]]:
    global _COMPILED_BRANCH_FACTORIZED_STATS, _COMPILED_BRANCH_FACTORIZED_STATS_FAILED
    if _COMPILED_BRANCH_FACTORIZED_STATS_FAILED:
        return None
    if _COMPILED_BRANCH_FACTORIZED_STATS is None:
        compile_fn = getattr(torch, "compile", None)
        if compile_fn is None:
            _COMPILED_BRANCH_FACTORIZED_STATS_FAILED = True
            return None
        try:
            _COMPILED_BRANCH_FACTORIZED_STATS = compile_fn(
                _branch_factorized_stats_compilable,
                mode="max-autotune-no-cudagraphs",
                fullgraph=False,
            )
        except Exception:
            _disable_compiled_branch_factorized_stats()
            return None
    return _COMPILED_BRANCH_FACTORIZED_STATS


@torch.inference_mode()
def _e8p12_sign_parity(sign_ids: torch.Tensor, *, bit_count: int = 8) -> torch.Tensor:
    bit_values = (1 << torch.arange(bit_count, device=sign_ids.device, dtype=torch.long)).view(1, bit_count)
    bits = ((sign_ids.to(dtype=torch.long).unsqueeze(-1) & bit_values) != 0).to(dtype=torch.long)
    return (bits.sum(dim=-1) & 1).to(dtype=torch.long)


@torch.inference_mode()
def prior_scale_from_center(prior_center_blocks: Optional[torch.Tensor], mode: str) -> Optional[torch.Tensor]:
    if prior_center_blocks is None:
        return None
    mode = str(mode or "absmax").strip().lower()
    if mode in ("none", "unit", "legacy"):
        shape = prior_center_blocks.shape[:-1] + (1,)
        return torch.ones(shape, dtype=torch.float32, device=prior_center_blocks.device)
    if mode in ("absmax", "exact_full", "exact_full_absmax"):
        scale = prior_center_blocks.abs().amax(dim=-1, keepdim=True)
    elif mode in ("rms", "std"):
        scale = prior_center_blocks.square().mean(dim=-1, keepdim=True).sqrt()
    elif mode in ("mean_abs", "l1"):
        scale = prior_center_blocks.abs().mean(dim=-1, keepdim=True)
    else:
        raise ValueError(f"unsupported prior_tau_scale_mode={mode!r}; supported: none|absmax|rms|mean_abs")
    return scale.to(dtype=torch.float32).clamp_min_(1e-12)


@torch.inference_mode()
def resolve_factorized_blocks_per_chunk(batch_rows: int, n_blocks: int, requested: int, device: torch.device) -> int:
    requested = max(int(requested or 1), 1)
    if n_blocks <= requested:
        return int(n_blocks)
    if device.type != "cuda":
        return int(min(n_blocks, max(requested, 64)))
    index = int(device.index) if device.index is not None else int(torch.cuda.current_device())
    total_bytes = int(torch.cuda.get_device_properties(index).total_memory)

    bytes_per_item_base = 4 * 48
    budget = min(max(int(total_bytes // 8), 256 * 1024**2), 4 * 1024**3)
    denom = max(int(batch_rows) * 256 * bytes_per_item_base, 1)
    auto = max(1, int(budget // denom))
    return int(min(n_blocks, max(requested, auto)))


@torch.inference_mode()
def get_e8p12_factor_cache(cb: object, *, device: Optional[torch.device] = None) -> Dict[str, torch.Tensor]:
    target_device = cb.grid.device if device is None else device
    cache = getattr(cb, "_mpq_factor_cache", None)
    if cache is not None and cache.get("device") == target_device:
        return cache

    if cb.__class__.__name__ != "E8P12_codebook":
        raise ValueError(f"factorized E8P12 cache requires E8P12_codebook, got {cb.__class__.__name__}.")

    grid = cb.grid.to(device=target_device, dtype=torch.float32)
    codesz = int(cb.codesz)
    abs_count = int(cb.grid_packed_abs.shape[0])
    if abs_count != 256 or codesz != 8:
        raise ValueError(f"expected 256 abs patterns with codesz=8, got abs_count={abs_count}, codesz={codesz}.")

    bit_values = (1 << torch.arange(codesz, device=target_device, dtype=torch.long)).contiguous()
    valid_even_sign = (_e8p12_sign_parity(torch.arange(1 << codesz, device=target_device, dtype=torch.long)) == 0)

    seed_idxs = (torch.arange(abs_count, device=target_device, dtype=torch.long) << 8).contiguous()
    base_template = (grid.index_select(0, seed_idxs) - 0.25).contiguous()
    base_sign_mask = (base_template < 0).to(dtype=torch.long).contiguous()
    base_parity = (base_sign_mask.sum(dim=-1) & 1).to(dtype=torch.long).contiguous()
    base_abs = base_template.abs().contiguous()
    base_sqnorm = base_abs.square().sum(dim=-1).contiguous()
    base_abs_sq = base_abs.square().contiguous()

    full_idxs = torch.arange(grid.shape[0], device=target_device, dtype=torch.long)
    abs_idx = (full_idxs >> 8).contiguous()
    raw_sign = (full_idxs & 255).contiguous()
    branch = _e8p12_sign_parity(raw_sign)
    branch_offsets = torch.tensor([0.25, -0.25], dtype=torch.float32, device=target_device)
    offset = branch_offsets.index_select(0, branch).unsqueeze(-1)
    residual = grid - offset
    total_sign_bits = (residual < 0).to(dtype=torch.long)
    sign_id_total = (total_sign_bits * bit_values.view(1, codesz)).sum(dim=-1)

    idx_lut = torch.full((2, abs_count, 1 << codesz), -1, dtype=torch.long, device=target_device)
    idx_lut[branch, abs_idx, sign_id_total] = full_idxs

    expected_abs = base_abs.index_select(0, abs_idx)
    if not torch.allclose(residual.abs(), expected_abs, atol=1e-6, rtol=1e-6):
        raise RuntimeError("failed to reconstruct E8P12 abs base patterns from current codebook grid.")

    sign_parity = _e8p12_sign_parity(torch.arange(1 << codesz, device=target_device, dtype=torch.long))
    expected_valid = sign_parity.view(1, 1, -1) == base_parity.view(1, abs_count, 1)
    if not torch.all(idx_lut[expected_valid.expand(2, -1, -1)] >= 0):
        raise RuntimeError("missing valid sign entries in E8P12 factor cache.")
    if not torch.all(idx_lut[~expected_valid.expand(2, -1, -1)] < 0):
        raise RuntimeError("unexpected invalid sign entries in E8P12 factor cache.")

    cache = {
        "device": target_device,
        "codesz": torch.tensor(codesz, device=target_device),
        "abs_count": torch.tensor(abs_count, device=target_device),
        "base_ids": torch.arange(abs_count, device=target_device, dtype=torch.long).view(1, abs_count),
        "base_sign_mask": base_sign_mask,
        "base_parity": base_parity,
        "base_parity_even": (base_parity == 0).view(1, abs_count),
        "base_abs": base_abs,
        "base_abs_view": base_abs.view(1, abs_count, codesz),
        "base_abs_sq": base_abs_sq,
        "base_abs_sq_t": base_abs_sq.T.contiguous(),
        "base_sqnorm": base_sqnorm,
        "base_sqnorm_view": base_sqnorm.view(1, abs_count),
        "idx_lut": idx_lut,
        "valid_even_sign": valid_even_sign,
        "bit_values": bit_values,
        "bit_values_view": bit_values.view(1, 1, codesz),
        "branch_offsets": branch_offsets,
    }
    setattr(cb, "_mpq_factor_cache", cache)
    return cache


def _branch_factorized_stats_impl(
    *,
    y_branch: torch.Tensor,  # (N, 8)
    gamma_flat: torch.Tensor,  # (N, 1) or (N, 8)
    offset: float,
    branch_idx: int,
    cache: Dict[str, torch.Tensor],
    need_stats: bool,
    validate_map_idx: bool,
) -> Dict[str, torch.Tensor]:
    base_abs = cache["base_abs"]
    base_abs_view = cache["base_abs_view"]
    base_abs_sq_t = cache["base_abs_sq_t"]
    base_sqnorm_view = cache["base_sqnorm_view"]
    base_parity = cache["base_parity"]
    parity_mask = cache["base_parity_even"]
    bit_values = cache["bit_values"]
    bit_values_view = cache["bit_values_view"]
    base_ids = cache["base_ids"]
    idx_lut_branch = cache["idx_lut"][branch_idx]

    n_items = int(y_branch.shape[0])
    n_bases = int(base_abs.shape[0])
    codesz = int(base_abs.shape[1])
    device = y_branch.device
    dtype = y_branch.dtype

    if gamma_flat.ndim != 2:
        raise ValueError(f"expected gamma_flat with shape (N,1) or (N,8), got {tuple(gamma_flat.shape)}")

    if gamma_flat.shape[1] == 1:
        gamma_eff = gamma_flat.view(n_items, 1, 1)
    else:
        gamma_eff = gamma_flat.view(n_items, 1, codesz)

    a = gamma_eff * y_branch.view(n_items, 1, codesz) * base_abs_view
    if gamma_flat.shape[1] == 1:
        y_norm = y_branch.square().sum(dim=-1, keepdim=True)
        const = -0.5 * gamma_flat[:, :1] * (y_norm + base_sqnorm_view)
    else:
        gamma_vec = gamma_flat
        y_norm = (gamma_vec * y_branch.square()).sum(dim=-1, keepdim=True)
        base_norm = gamma_vec.matmul(base_abs_sq_t)
        const = -0.5 * (y_norm + base_norm)

    logC = torch.logaddexp(a, -a).sum(dim=-1)
    t = torch.tanh(a)
    t_eps = 1e-7 if t.dtype == torch.float32 else 1e-12
    t_safe = t.clamp(min=-1.0 + t_eps, max=1.0 - t_eps)
    log_abs_prod_t = torch.log(t_safe.abs().clamp_min(t_eps)).sum(dim=-1)
    sign_prod_neg = ((t_safe < 0).sum(dim=-1) & 1).bool()
    prod_abs_t = torch.exp(log_abs_prod_t)
    prod_t = torch.where(sign_prod_neg, -prod_abs_t, prod_abs_t)
    one = torch.ones_like(prod_t)

    def _log1mexp(x: torch.Tensor) -> torch.Tensor:
        cutoff = -math.log(2.0)
        return torch.where(
            x < cutoff,
            torch.log1p(-torch.exp(x)),
            torch.log(-torch.expm1(x)),
        )

    log_one_plus_prod = torch.where(
        sign_prod_neg,
        _log1mexp(log_abs_prod_t),
        torch.logaddexp(torch.zeros_like(log_abs_prod_t), log_abs_prod_t),
    )
    log_one_minus_prod = torch.where(
        sign_prod_neg,
        torch.logaddexp(torch.zeros_like(log_abs_prod_t), log_abs_prod_t),
        _log1mexp(log_abs_prod_t),
    )
    logZ_even = logC + log_one_plus_prod - math.log(2.0)
    logZ_odd = logC + log_one_minus_prod - math.log(2.0)
    logZ_sign = torch.where(parity_mask, logZ_even, logZ_odd)

    prefix = []
    acc = one
    for coord in range(codesz):
        prefix.append(acc)
        acc = acc * t_safe[:, :, coord]

    sign_mean = torch.empty((n_items, n_bases, codesz), dtype=dtype, device=device)
    suffix = one
    denom_even = one + prod_t
    denom_odd = one - prod_t
    denom_eps = 1e-12
    denom_even_safe = torch.where(denom_even.abs() > denom_eps, denom_even, torch.ones_like(denom_even))
    denom_odd_safe = torch.where(denom_odd.abs() > denom_eps, denom_odd, torch.ones_like(denom_odd))
    for coord in reversed(range(codesz)):
        t_coord = t_safe[:, :, coord]
        excl_prod = prefix[coord] * suffix
        even_mean = (t_coord + excl_prod) / denom_even_safe
        odd_mean = (t_coord - excl_prod) / denom_odd_safe
        sign_mean[:, :, coord] = torch.where(parity_mask, even_mean, odd_mean)
        suffix = suffix * t_coord
    sign_mean = torch.nan_to_num(sign_mean, nan=0.0, posinf=0.0, neginf=0.0)

    offset_f = float(offset)
    offset_sq = offset_f * offset_f
    mu_core = sign_mean * base_abs_view
    mu_component = mu_core + offset_f
    second_component = (
        base_sqnorm_view
        + codesz * offset_sq
        + 2.0 * offset_f * mu_core.sum(dim=-1)
    )
    log_component = const + logZ_sign

    abs_a = a.abs()
    sum_abs = abs_a.sum(dim=-1)
    best_sign_bits = (a < 0)
    best_parity = (best_sign_bits.to(dtype=torch.long).sum(dim=-1) & 1)
    min_pos = abs_a.argmin(dim=-1)
    min_abs = abs_a.gather(-1, min_pos.unsqueeze(-1)).squeeze(-1)
    mismatch = best_parity != base_parity.view(1, n_bases)
    best_sign_id = (best_sign_bits.to(dtype=torch.long) * bit_values_view).sum(dim=-1)
    min_bit = bit_values[min_pos]
    best_sign_id = torch.where(mismatch, torch.bitwise_xor(best_sign_id, min_bit), best_sign_id)
    best_linear = sum_abs - 2.0 * mismatch.to(dtype=dtype) * min_abs
    best_logscore = const + best_linear

    map_idx = idx_lut_branch[base_ids.expand(n_items, -1), best_sign_id]
    if validate_map_idx and not torch.all(map_idx >= 0):
        raise RuntimeError("failed to map factorized E8P12 MAP sign patterns back to full codebook indices.")

    out = {
        "log_component": log_component,
        "mu_component": mu_component,
        "second_component": second_component,
        "best_logscore": best_logscore,
        "best_idx": map_idx,
    }
    if need_stats:
        second_abs = abs_a.clone()
        second_abs.scatter_(-1, min_pos.unsqueeze(-1), float("inf"))
        second_min_abs = second_abs.min(dim=-1).values
        second_linear = torch.where(
            mismatch,
            sum_abs - 2.0 * second_min_abs,
            sum_abs - 2.0 * (min_abs + second_min_abs),
        )
        out["second_logscore"] = const + second_linear
        sign_entropy = torch.where(
            torch.isfinite(logZ_sign),
            (logZ_sign - (a * sign_mean).sum(dim=-1)).clamp_min_(0.0),
            torch.zeros_like(logZ_sign),
        )
        out["sign_entropy"] = sign_entropy
    return out


def _branch_factorized_stats_compilable(
    *,
    y_branch: torch.Tensor,  # (N, 8)
    gamma_flat: torch.Tensor,  # (N, 1) or (N, 8)
    offset: float,
    branch_idx: int,
    cache: Dict[str, torch.Tensor],
    need_stats: bool,
) -> Dict[str, torch.Tensor]:
    return _branch_factorized_stats_impl(
        y_branch=y_branch,
        gamma_flat=gamma_flat,
        offset=offset,
        branch_idx=branch_idx,
        cache=cache,
        need_stats=need_stats,
        validate_map_idx=False,
    )


@torch.inference_mode()
def _branch_factorized_stats(
    *,
    y_branch: torch.Tensor,  # (N, 8)
    gamma_flat: torch.Tensor,  # (N, 1) or (N, 8)
    offset: float,
    branch_idx: int,
    cache: Dict[str, torch.Tensor],
    need_stats: bool,
    use_compiled: bool = False,
) -> Dict[str, torch.Tensor]:
    if bool(use_compiled) and y_branch.is_cuda:
        compiled_fn = _get_compiled_branch_factorized_stats()
        if compiled_fn is not None:
            try:
                return compiled_fn(
                    y_branch=y_branch,
                    gamma_flat=gamma_flat,
                    offset=offset,
                    branch_idx=branch_idx,
                    cache=cache,
                    need_stats=need_stats,
                )
            except Exception:
                _disable_compiled_branch_factorized_stats()
    return _branch_factorized_stats_impl(
        y_branch=y_branch,
        gamma_flat=gamma_flat,
        offset=offset,
        branch_idx=branch_idx,
        cache=cache,
        need_stats=need_stats,
        validate_map_idx=True,
    )


@torch.inference_mode()
def _branch_factorized_map(
    *,
    y_branch: torch.Tensor,  # (N, 8)
    gamma_flat: torch.Tensor,  # (N, 1) or (N, 8)
    branch_idx: int,
    cache: Dict[str, torch.Tensor],
    need_stats: bool,
) -> Dict[str, torch.Tensor]:
    base_abs = cache["base_abs"]
    base_abs_view = cache["base_abs_view"]
    base_abs_sq_t = cache["base_abs_sq_t"]
    base_sqnorm_view = cache["base_sqnorm_view"]
    base_parity = cache["base_parity"]
    bit_values = cache["bit_values"]
    bit_values_view = cache["bit_values_view"]
    base_ids = cache["base_ids"]
    idx_lut_branch = cache["idx_lut"][branch_idx]

    n_items = int(y_branch.shape[0])
    n_bases = int(base_abs.shape[0])
    codesz = int(base_abs.shape[1])
    device = y_branch.device
    dtype = y_branch.dtype

    if gamma_flat.ndim != 2:
        raise ValueError(f"expected gamma_flat with shape (N,1) or (N,8), got {tuple(gamma_flat.shape)}")

    if gamma_flat.shape[1] == 1:
        gamma_eff = gamma_flat.view(n_items, 1, 1)
    else:
        gamma_eff = gamma_flat.view(n_items, 1, codesz)

    a = gamma_eff * y_branch.view(n_items, 1, codesz) * base_abs_view
    if gamma_flat.shape[1] == 1:
        y_norm = y_branch.square().sum(dim=-1, keepdim=True)
        const = -0.5 * gamma_flat[:, :1] * (y_norm + base_sqnorm_view)
    else:
        gamma_vec = gamma_flat
        y_norm = (gamma_vec * y_branch.square()).sum(dim=-1, keepdim=True)
        base_norm = gamma_vec.matmul(base_abs_sq_t)
        const = -0.5 * (y_norm + base_norm)

    abs_a = a.abs()
    sum_abs = abs_a.sum(dim=-1)
    best_sign_bits = (a < 0)
    best_parity = (best_sign_bits.to(dtype=torch.long).sum(dim=-1) & 1)
    min_pos = abs_a.argmin(dim=-1)
    min_abs = abs_a.gather(-1, min_pos.unsqueeze(-1)).squeeze(-1)
    mismatch = best_parity != base_parity.view(1, n_bases)
    best_sign_id = (best_sign_bits.to(dtype=torch.long) * bit_values_view).sum(dim=-1)
    min_bit = bit_values[min_pos]
    best_sign_id = torch.where(mismatch, torch.bitwise_xor(best_sign_id, min_bit), best_sign_id)
    best_linear = sum_abs - 2.0 * mismatch.to(dtype=dtype) * min_abs
    best_logscore = const + best_linear

    map_idx = idx_lut_branch[base_ids.expand(n_items, -1), best_sign_id]
    if not torch.all(map_idx >= 0):
        raise RuntimeError("failed to map factorized E8P12 MAP sign patterns back to full codebook indices.")

    out = {
        "best_logscore": best_logscore,
        "best_idx": map_idx,
    }
    if need_stats:
        second_abs = abs_a.clone()
        second_abs.scatter_(-1, min_pos.unsqueeze(-1), float("inf"))
        second_min = second_abs.min(dim=-1)
        second_min_abs = second_min.values
        second_pos = second_min.indices
        second_linear = torch.where(
            mismatch,
            sum_abs - 2.0 * second_min_abs,
            sum_abs - 2.0 * (min_abs + second_min_abs),
        )
        out["second_logscore"] = const + second_linear
        # Second-best MAP index under the parity constraint.
        #
        # Let s0 be the unconstrained best sign assignment (per coordinate). Each bit flip i
        # reduces the linear score by 2|a_i|. The parity constraint means:
        #   - if s0 parity mismatches base parity, the best valid sign flips the smallest |a_i|
        #     bit; the second-best flips the second-smallest |a_i| bit.
        #   - if s0 parity already matches, the second-best valid sign flips the two smallest
        #     |a_i| bits (to keep parity).
        orig_sign_id = (best_sign_bits.to(dtype=torch.long) * bit_values_view).sum(dim=-1)
        min_bit = bit_values[min_pos]
        second_bit = bit_values[second_pos]
        flip2 = torch.bitwise_xor(min_bit, second_bit)
        second_sign_id = torch.where(mismatch, torch.bitwise_xor(orig_sign_id, second_bit), torch.bitwise_xor(orig_sign_id, flip2))
        second_idx = idx_lut_branch[base_ids, second_sign_id]
        if not torch.all(second_idx >= 0):
            raise RuntimeError("failed to map factorized E8P12 2nd MAP sign patterns back to full codebook indices.")
        out["second_idx"] = second_idx
    return out


@torch.inference_mode()
def vamp_vector_map_topk_e8p12(
    *,
    r1_blocks: torch.Tensor,  # (B, n_blocks, 8)
    gamma1: torch.Tensor,  # (B, 1) or (B, n_blocks, 1) or (B, n_blocks, 8)
    cb: object,
    k: int,
    blocks_per_chunk: int = 64,
    prior_center_blocks: Optional[torch.Tensor] = None,
    prior_scale_blocks: Optional[torch.Tensor] = None,
    prior_tau: float = 0.0,
    prior_tau_scale_mode: str = "absmax",
) -> Dict[str, torch.Tensor]:
    """Return per-block top-k MAP candidates (indices + scores) under the factorized E8P12 model.

    Candidate set is restricted to the {best, 2nd-best} parity-valid sign assignments for each
    (branch, abs-base) pattern. This yields 2 * 256 * 2 = 1024 candidates per block.
    """
    batch, n_blocks, codesz = r1_blocks.shape
    if int(codesz) != 8:
        raise ValueError(f"E8P12 top-k expects codesz=8, got {codesz}.")
    k = int(k)
    if k <= 0:
        raise ValueError(f"expected k>0, got k={k}")

    r1_work = r1_blocks
    gamma_blocks = None
    if float(prior_tau) > 0.0 and prior_center_blocks is not None:
        scale_blocks = prior_scale_blocks
        if scale_blocks is None:
            scale_blocks = prior_scale_from_center(prior_center_blocks, prior_tau_scale_mode)
        assert scale_blocks is not None
        scale_blocks = scale_blocks.to(dtype=torch.float32, device=r1_blocks.device).clamp_min_(1e-12)
        tau2 = (max(float(prior_tau), 1e-12) * scale_blocks).square()
        prior_prec = 1.0 / tau2
        if gamma1.ndim == 2:
            gamma_base = gamma1.unsqueeze(1).expand(batch, n_blocks, 1).to(dtype=torch.float32)
        elif gamma1.ndim == 3:
            gamma_base = gamma1.to(dtype=torch.float32)
            if gamma_base.shape[-1] not in (1, codesz):
                raise ValueError(f"expected gamma1 last dim in {{1,{codesz}}}, got {tuple(gamma1.shape)}")
        else:
            raise ValueError(f"expected gamma1 with ndim in {{2,3}}, got {tuple(gamma1.shape)}")
        gamma_blocks = gamma_base + prior_prec
        r1_work = (
            gamma_base * r1_blocks.to(dtype=torch.float32)
            + prior_prec * prior_center_blocks.to(dtype=torch.float32)
        ) / gamma_blocks
        r1_work = r1_work.to(dtype=r1_blocks.dtype)

    blocks_per_chunk = resolve_factorized_blocks_per_chunk(batch, n_blocks, blocks_per_chunk, r1_blocks.device)
    cache = get_e8p12_factor_cache(cb, device=r1_blocks.device)
    branch_offsets = cache["branch_offsets"]

    # 1024 candidates per block.
    cand_count = 2 * int(cache["abs_count"].item()) * 2
    k_eff = min(int(k), int(cand_count))

    topk_idxs = torch.empty(batch, n_blocks, k_eff, dtype=torch.long, device=r1_blocks.device)
    topk_scores = torch.empty(batch, n_blocks, k_eff, dtype=torch.float32, device=r1_blocks.device)

    for bstart in range(0, n_blocks, blocks_per_chunk):
        bend = min(bstart + blocks_per_chunk, n_blocks)
        chunk_size = bend - bstart
        r1_chunk = r1_work[:, bstart:bend, :]
        r1_flat = r1_chunk.reshape(batch * chunk_size, codesz).to(dtype=torch.float32)

        if gamma_blocks is None:
            if gamma1.ndim == 2:
                gamma_flat = (
                    gamma1.unsqueeze(1)
                    .expand(batch, chunk_size, 1)
                    .reshape(batch * chunk_size, 1)
                    .to(dtype=torch.float32)
                )
            else:
                gamma_src = gamma1[:, bstart:bend, :]
                if gamma_src.shape[-1] not in (1, codesz):
                    raise ValueError(f"expected gamma1 last dim in {{1,{codesz}}}, got {tuple(gamma1.shape)}")
                gamma_flat = gamma_src.reshape(batch * chunk_size, gamma_src.shape[-1]).to(dtype=torch.float32)
        else:
            gamma_src = gamma_blocks[:, bstart:bend, :]
            gamma_flat = gamma_src.reshape(batch * chunk_size, gamma_src.shape[-1]).to(dtype=torch.float32)

        branch_outputs = []
        for branch_idx, offset in enumerate(branch_offsets.tolist()):
            branch_outputs.append(
                _branch_factorized_map(
                    y_branch=r1_flat - float(offset),
                    gamma_flat=gamma_flat,
                    branch_idx=int(branch_idx),
                    cache=cache,
                    need_stats=True,
                )
            )

        best_logscore = torch.stack([out["best_logscore"] for out in branch_outputs], dim=1)  # (N,2,256)
        second_logscore = torch.stack([out["second_logscore"] for out in branch_outputs], dim=1)  # (N,2,256)
        best_idx = torch.stack([out["best_idx"] for out in branch_outputs], dim=1)  # (N,2,256)
        second_idx = torch.stack([out["second_idx"] for out in branch_outputs], dim=1)  # (N,2,256)

        scores = torch.stack([best_logscore, second_logscore], dim=-1).reshape(batch * chunk_size, -1)
        idxs = torch.stack([best_idx, second_idx], dim=-1).reshape(batch * chunk_size, -1)
        top_scores, top_pos = torch.topk(scores, k=k_eff, dim=1)
        top_idxs = idxs.gather(1, top_pos)

        topk_idxs[:, bstart:bend, :] = top_idxs.view(batch, chunk_size, k_eff)
        topk_scores[:, bstart:bend, :] = top_scores.view(batch, chunk_size, k_eff).to(dtype=torch.float32)

    return {"topk_idxs": topk_idxs, "topk_scores": topk_scores}


@torch.inference_mode()
def vamp_vector_map_e8p12(
    *,
    r1_blocks: torch.Tensor,  # (B, n_blocks, 8)
    gamma1: torch.Tensor,  # (B, 1) or (B, n_blocks, 1) or (B, n_blocks, 8)
    cb: object,
    blocks_per_chunk: int = 64,
    need_stats: bool = False,
    prior_center_blocks: Optional[torch.Tensor] = None,
    prior_scale_blocks: Optional[torch.Tensor] = None,
    prior_tau: float = 0.0,
    prior_tau_scale_mode: str = "absmax",
) -> Dict[str, torch.Tensor | float]:
    batch, n_blocks, codesz = r1_blocks.shape
    if int(codesz) != 8:
        raise ValueError(f"E8P12 MAP expects codesz=8, got {codesz}.")

    r1_work = r1_blocks
    gamma_blocks = None
    if float(prior_tau) > 0.0 and prior_center_blocks is not None:
        scale_blocks = prior_scale_blocks
        if scale_blocks is None:
            scale_blocks = prior_scale_from_center(prior_center_blocks, prior_tau_scale_mode)
        assert scale_blocks is not None
        scale_blocks = scale_blocks.to(dtype=torch.float32, device=r1_blocks.device).clamp_min_(1e-12)
        tau2 = (max(float(prior_tau), 1e-12) * scale_blocks).square()
        prior_prec = 1.0 / tau2
        if gamma1.ndim == 2:
            gamma_base = gamma1.unsqueeze(1).expand(batch, n_blocks, 1).to(dtype=torch.float32)
        elif gamma1.ndim == 3:
            gamma_base = gamma1.to(dtype=torch.float32)
            if gamma_base.shape[-1] not in (1, codesz):
                raise ValueError(f"expected gamma1 last dim in {{1,{codesz}}}, got {tuple(gamma1.shape)}")
        else:
            raise ValueError(f"expected gamma1 with ndim in {{2,3}}, got {tuple(gamma1.shape)}")
        gamma_blocks = gamma_base + prior_prec
        r1_work = (
            gamma_base * r1_blocks.to(dtype=torch.float32)
            + prior_prec * prior_center_blocks.to(dtype=torch.float32)
        ) / gamma_blocks
        r1_work = r1_work.to(dtype=r1_blocks.dtype)

    blocks_per_chunk = resolve_factorized_blocks_per_chunk(batch, n_blocks, blocks_per_chunk, r1_blocks.device)
    cache = get_e8p12_factor_cache(cb, device=r1_blocks.device)
    branch_offsets = cache["branch_offsets"]

    map_idxs = torch.empty(batch, n_blocks, dtype=torch.long, device=r1_blocks.device)
    map_vals = torch.empty_like(r1_blocks)
    margin_values = []

    for bstart in range(0, n_blocks, blocks_per_chunk):
        bend = min(bstart + blocks_per_chunk, n_blocks)
        chunk_size = bend - bstart
        r1_chunk = r1_work[:, bstart:bend, :]
        r1_flat = r1_chunk.reshape(batch * chunk_size, codesz).to(dtype=torch.float32)

        if gamma_blocks is None:
            if gamma1.ndim == 2:
                gamma_flat = (
                    gamma1.unsqueeze(1)
                    .expand(batch, chunk_size, 1)
                    .reshape(batch * chunk_size, 1)
                    .to(dtype=torch.float32)
                )
            else:
                gamma_src = gamma1[:, bstart:bend, :]
                if gamma_src.shape[-1] not in (1, codesz):
                    raise ValueError(f"expected gamma1 last dim in {{1,{codesz}}}, got {tuple(gamma1.shape)}")
                gamma_flat = gamma_src.reshape(batch * chunk_size, gamma_src.shape[-1]).to(dtype=torch.float32)
        else:
            gamma_src = gamma_blocks[:, bstart:bend, :]
            gamma_flat = gamma_src.reshape(batch * chunk_size, gamma_src.shape[-1]).to(dtype=torch.float32)

        branch_outputs = []
        for branch_idx, offset in enumerate(branch_offsets.tolist()):
            branch_outputs.append(
                _branch_factorized_map(
                    y_branch=r1_flat - float(offset),
                    gamma_flat=gamma_flat,
                    branch_idx=int(branch_idx),
                    cache=cache,
                    need_stats=need_stats,
                )
            )

        best_logscore = torch.stack([out["best_logscore"] for out in branch_outputs], dim=1)
        best_flat = best_logscore.reshape(batch * chunk_size, -1)
        best_pos = best_flat.argmax(dim=-1)
        best_branch = torch.div(best_pos, int(cache["abs_count"].item()), rounding_mode="floor")
        best_abs = best_pos % int(cache["abs_count"].item())

        map_idx_per_branch = torch.stack([out["best_idx"] for out in branch_outputs], dim=1)
        map_idx = map_idx_per_branch[torch.arange(batch * chunk_size, device=r1_blocks.device), best_branch, best_abs]
        map_val = cb.grid.index_select(0, map_idx).to(dtype=r1_blocks.dtype)

        map_idxs[:, bstart:bend] = map_idx.view(batch, chunk_size)
        map_vals[:, bstart:bend, :] = map_val.view(batch, chunk_size, codesz)

        if need_stats:
            second_logscore = torch.stack([out["second_logscore"] for out in branch_outputs], dim=1)
            candidate_scores = torch.cat(
                [best_logscore.reshape(batch * chunk_size, -1), second_logscore.reshape(batch * chunk_size, -1)],
                dim=1,
            )
            top2 = torch.topk(candidate_scores, k=2, dim=1).values
            margin_values.append(float((top2[:, 0] - top2[:, 1]).mean().item()))

    out: Dict[str, torch.Tensor | float] = {
        "map_idxs": map_idxs,
        "map_vals": map_vals,
    }
    if need_stats:
        out["margin"] = float(sum(margin_values) / len(margin_values)) if margin_values else 0.0
    return out


@torch.inference_mode()
def gram_metric_map_blocks(
    *,
    r1_blocks: torch.Tensor,  # (B, n_blocks, 8)
    gram_chols: torch.Tensor,  # (n_blocks, 8, 8) lower-triangular Cholesky
    cb: object,
    rows_per_chunk: int = 0,
    prior_center_blocks: Optional[torch.Tensor] = None,
    prior_scale_blocks: Optional[torch.Tensor] = None,
    prior_tau: float = 0.0,
    gamma1: Optional[torch.Tensor] = None,  # (B, 1) -- scalar precision for prior folding
) -> Dict[str, torch.Tensor]:
    """MAP-only assignment under per-block Gram metric (no posterior statistics)."""
    out = gram_metric_denoiser_blocks(
        r1_blocks=r1_blocks,
        gamma1_scalar=gamma1 if gamma1 is not None else torch.ones(r1_blocks.shape[0], 1, device=r1_blocks.device),
        gram_chols=gram_chols,
        cb=cb,
        rows_per_chunk=rows_per_chunk,
        prior_center_blocks=prior_center_blocks,
        prior_scale_blocks=prior_scale_blocks,
        prior_tau=prior_tau,
        need_stats=False,
    )
    return {"map_idxs": out["map_idxs"], "map_vals": out["map_vals"]}


@torch.inference_mode()
def gram_metric_denoiser_blocks(
    *,
    r1_blocks: torch.Tensor,  # (B, n_blocks, 8)
    gamma1_scalar: torch.Tensor,  # (B, 1) -- scalar VAMP precision
    gram_chols: torch.Tensor,  # (n_blocks, 8, 8) lower-triangular Cholesky of G_k
    cb: object,
    rows_per_chunk: int = 0,
    prior_center_blocks: Optional[torch.Tensor] = None,
    prior_scale_blocks: Optional[torch.Tensor] = None,
    prior_tau: float = 0.0,
    need_stats: bool = False,
) -> Dict[str, torch.Tensor | float]:
    """Full discrete MMSE denoiser under per-block Gram metric.

    Posterior for block k:
      p(c | r1_k, gamma1, G_k) ∝ exp(-0.5 * gamma1 * (c - r1_k)^T G_k (c - r1_k))

    This is equivalent to an isotropic channel in the whitened space z = L_k^T x.

    Returns:
      x1_blocks: (B, n_blocks, 8) -- posterior mean in original space
      var1:      (B, n_blocks)     -- per-block mean Gram-metric variance / codesz
      map_idxs:  (B, n_blocks)     -- MAP codeword index
      map_vals:  (B, n_blocks, 8)  -- MAP codeword values
      alpha1:    (B, 1)            -- gamma1 * mean(var1), Gram-calibrated
    """
    batch, n_blocks, codesz = r1_blocks.shape
    device = r1_blocks.device
    grid = cb.grid.to(dtype=torch.float32, device=device)  # (C, 8)
    C = grid.shape[0]

    x1_blocks = torch.empty(batch, n_blocks, codesz, dtype=torch.float32, device=device)
    var1 = torch.empty(batch, n_blocks, dtype=torch.float32, device=device)
    map_idxs = torch.empty(batch, n_blocks, dtype=torch.long, device=device)
    map_vals = torch.empty(batch, n_blocks, codesz, dtype=r1_blocks.dtype, device=device)
    entropy_acc = 0.0
    margin_acc = 0.0
    stat_count = 0

    # Prior folding: combine channel message with prior into effective target & precision.
    r1_work = r1_blocks.to(dtype=torch.float32)
    gamma1_eff = gamma1_scalar.to(dtype=torch.float32)  # (B, 1)
    # For the Gram-metric denoiser, the prior is isotropic (per-element) while the
    # channel is Gram-weighted. The combined posterior is:
    #   score = -0.5 * gamma1 * (c-r1)^T G_k (c-r1) - 0.5 * prior_prec * ||c-center||^2
    # We handle this as a combined score in the brute-force loop below.
    has_prior = float(prior_tau) > 0.0 and prior_center_blocks is not None
    prior_prec_blocks: Optional[torch.Tensor] = None
    if has_prior:
        assert prior_center_blocks is not None
        scale_blocks = prior_scale_blocks
        if scale_blocks is None:
            scale_blocks = prior_scale_from_center(prior_center_blocks, "absmax")
        scale_blocks = scale_blocks.to(dtype=torch.float32, device=device).clamp_min_(1e-12)
        tau2 = (max(float(prior_tau), 1e-12) * scale_blocks).square()
        prior_prec_blocks = (1.0 / tau2)  # (B, n_blocks, 1)

    rows_per_chunk = max(int(rows_per_chunk), 0)
    if rows_per_chunk <= 0:
        rows_per_chunk = batch

    high_gamma_snap = 1e3

    for bi in range(n_blocks):
        chol = gram_chols[bi].to(dtype=torch.float32)  # (8, 8)
        # Pre-transform grid to whitened space for this block.
        grid_white = grid.matmul(chol)  # (C, 8)
        grid_white_norm = grid_white.square().sum(dim=-1)  # (C,)

        for rstart in range(0, batch, rows_per_chunk):
            rend = min(rstart + rows_per_chunk, batch)
            chunk_B = rend - rstart
            r1_chunk = r1_work[rstart:rend, bi, :]  # (chunk_B, 8)
            gamma1_chunk = gamma1_eff[rstart:rend, :]  # (chunk_B, 1)

            # Gram-metric distances: (c - r1)^T G (c - r1) = ||L^T c - L^T r1||^2
            r1_white = r1_chunk.matmul(chol)  # (chunk_B, 8)
            r1_white_norm = r1_white.square().sum(dim=-1, keepdim=True)  # (chunk_B, 1)
            # sq_dist[i,j] = ||grid_white[j] - r1_white[i]||^2
            sq_dist = r1_white_norm + grid_white_norm.unsqueeze(0) - 2.0 * r1_white.matmul(grid_white.T)
            # (chunk_B, C)

            # Log-scores: -0.5 * gamma1 * sq_dist
            log_scores = -0.5 * gamma1_chunk * sq_dist  # (chunk_B, C)

            # Add prior term if present: -0.5 * prior_prec * ||c - center||^2
            if has_prior and prior_prec_blocks is not None:
                assert prior_center_blocks is not None
                center = prior_center_blocks[rstart:rend, bi, :]  # (chunk_B, 8)
                pp = prior_prec_blocks[rstart:rend, bi, :]  # (chunk_B, 1)
                # ||c - center||^2 for all C codewords
                center_norm = center.square().sum(dim=-1, keepdim=True)  # (chunk_B, 1)
                grid_norm_iso = grid.square().sum(dim=-1).unsqueeze(0)  # (1, C)
                prior_sq = center_norm + grid_norm_iso - 2.0 * center.matmul(grid.T)  # (chunk_B, C)
                log_scores = log_scores - 0.5 * pp * prior_sq

            # MAP
            map_idx = log_scores.argmax(dim=-1)  # (chunk_B,)
            map_val = grid.index_select(0, map_idx)  # (chunk_B, 8)

            # Posterior probabilities
            probs = torch.softmax(log_scores, dim=-1)  # (chunk_B, C)

            # Posterior mean: E[c] = sum_c p(c) * c
            mu = probs.matmul(grid)  # (chunk_B, 8)

            # Gram-metric variance:
            # V_G = E[(c-mu)^T G (c-mu)] / codesz
            #     = (E[||L^T c||^2] - ||L^T mu||^2) / codesz
            mu_white = mu.matmul(chol)  # (chunk_B, 8)
            mu_white_norm = mu_white.square().sum(dim=-1)  # (chunk_B,)
            E_white_norm = (probs * grid_white_norm.unsqueeze(0)).sum(dim=-1)  # (chunk_B,)
            block_var = ((E_white_norm - mu_white_norm) / float(codesz)).clamp_min_(0.0)  # (chunk_B,)

            # High-gamma snap: when posterior is near-deterministic, use MAP directly
            high_mask = gamma1_chunk.squeeze(-1) >= high_gamma_snap
            if bool(high_mask.any().item()):
                mu[high_mask] = map_val[high_mask]
                block_var[high_mask] = 0.0

            x1_blocks[rstart:rend, bi, :] = mu
            var1[rstart:rend, bi] = block_var
            map_idxs[rstart:rend, bi] = map_idx
            map_vals[rstart:rend, bi, :] = map_val.to(dtype=r1_blocks.dtype)

            if need_stats:
                log_probs = log_scores - torch.logsumexp(log_scores, dim=-1, keepdim=True)
                ent = -(probs * log_probs).sum(dim=-1).mean().item()
                top2 = torch.topk(log_scores, k=min(2, C), dim=-1).values
                mrg = (top2[:, 0] - top2[:, 1]).mean().item() if top2.shape[-1] >= 2 else float("inf")
                entropy_acc += float(ent)
                margin_acc += float(mrg)
                stat_count += 1

    # Alpha1: gamma1 * mean(var1) -- properly calibrated to Gram metric
    alpha1 = gamma1_eff * var1.mean(dim=1, keepdim=True)  # (B, 1)

    out: Dict[str, torch.Tensor | float] = {
        "x1_blocks": x1_blocks,
        "var1": var1,
        "var1_diag": None,
        "map_idxs": map_idxs,
        "map_vals": map_vals,
        "alpha1": alpha1,
    }
    if need_stats:
        out["entropy"] = float(entropy_acc / max(stat_count, 1))
        out["margin"] = float(margin_acc / max(stat_count, 1))
    else:
        out["entropy"] = 0.0
        out["margin"] = 0.0
    return out


@torch.inference_mode()
def vamp_vector_discrete_mmse_e8p12(
    *,
    r1_blocks: torch.Tensor,  # (B, n_blocks, 8)
    gamma1: torch.Tensor,  # (B, 1) or (B, n_blocks, 1) or (B, n_blocks, 8)
    cb: object,
    blocks_per_chunk: int = 64,
    need_stats: bool = False,
    need_var_diag: bool = False,
    prior_center_blocks: Optional[torch.Tensor] = None,
    prior_scale_blocks: Optional[torch.Tensor] = None,
    prior_tau: float = 0.0,
    prior_tau_scale_mode: str = "absmax",
    compile_prior: bool = True,
    return_logZ_blocks: bool = False,
    return_margin_blocks: bool = False,
    return_entropy_blocks: bool = False,
) -> Dict[str, torch.Tensor | float]:
    return_logZ_blocks = bool(return_logZ_blocks)
    return_margin_blocks = bool(return_margin_blocks)
    return_entropy_blocks = bool(return_entropy_blocks)
    need_stats = bool(need_stats) or return_margin_blocks or return_entropy_blocks

    batch, n_blocks, codesz = r1_blocks.shape
    if int(codesz) != 8:
        raise ValueError(f"E8P12 posterior expects codesz=8, got {codesz}.")

    r1_work = r1_blocks
    gamma_blocks = None
    if float(prior_tau) > 0.0 and prior_center_blocks is not None:
        scale_blocks = prior_scale_blocks
        if scale_blocks is None:
            scale_blocks = prior_scale_from_center(prior_center_blocks, prior_tau_scale_mode)
        assert scale_blocks is not None
        scale_blocks = scale_blocks.to(dtype=torch.float32, device=r1_blocks.device).clamp_min_(1e-12)
        tau2 = (max(float(prior_tau), 1e-12) * scale_blocks).square()
        prior_prec = 1.0 / tau2
        if gamma1.ndim == 2:
            gamma_base = gamma1.unsqueeze(1).expand(batch, n_blocks, 1).to(dtype=torch.float32)
        elif gamma1.ndim == 3:
            gamma_base = gamma1.to(dtype=torch.float32)
            if gamma_base.shape[-1] not in (1, codesz):
                raise ValueError(f"expected gamma1 last dim in {{1,{codesz}}}, got {tuple(gamma1.shape)}")
        else:
            raise ValueError(f"expected gamma1 with ndim in {{2,3}}, got {tuple(gamma1.shape)}")
        gamma_blocks = gamma_base + prior_prec
        r1_work = (
            gamma_base * r1_blocks.to(dtype=torch.float32)
            + prior_prec * prior_center_blocks.to(dtype=torch.float32)
        ) / gamma_blocks
        r1_work = r1_work.to(dtype=r1_blocks.dtype)

    blocks_per_chunk = resolve_factorized_blocks_per_chunk(batch, n_blocks, blocks_per_chunk, r1_blocks.device)
    cache = get_e8p12_factor_cache(cb, device=r1_blocks.device)
    branch_offsets = cache["branch_offsets"]
    base_abs_sq = cache["base_abs_sq"]
    use_compiled_prior = bool(compile_prior) and r1_blocks.is_cuda

    x1_blocks = torch.empty_like(r1_blocks)
    var1 = torch.empty(batch, n_blocks, dtype=torch.float32, device=r1_blocks.device)
    var1_diag = (
        torch.empty(batch, n_blocks, codesz, dtype=torch.float32, device=r1_blocks.device)
        if bool(need_var_diag)
        else None
    )
    map_idxs = torch.empty(batch, n_blocks, dtype=torch.long, device=r1_blocks.device)
    map_vals = torch.empty_like(r1_blocks)
    entropy_values = []
    margin_values = []
    logZ_blocks = torch.empty(batch, n_blocks, dtype=torch.float32, device=r1_blocks.device) if return_logZ_blocks else None
    margin_blocks = torch.empty(batch, n_blocks, dtype=torch.float32, device=r1_blocks.device) if return_margin_blocks else None
    entropy_blocks = torch.empty(batch, n_blocks, dtype=torch.float32, device=r1_blocks.device) if return_entropy_blocks else None

    high_gamma_snap = 1e3

    for bstart in range(0, n_blocks, blocks_per_chunk):
        bend = min(bstart + blocks_per_chunk, n_blocks)
        chunk_size = bend - bstart
        r1_chunk = r1_work[:, bstart:bend, :]
        r1_flat = r1_chunk.reshape(batch * chunk_size, codesz).to(dtype=torch.float32)

        if gamma_blocks is None:
            if gamma1.ndim == 2:
                gamma_flat = (
                    gamma1.unsqueeze(1)
                    .expand(batch, chunk_size, 1)
                    .reshape(batch * chunk_size, 1)
                    .to(dtype=torch.float32)
                )
            else:
                gamma_src = gamma1[:, bstart:bend, :]
                if gamma_src.shape[-1] not in (1, codesz):
                    raise ValueError(f"expected gamma1 last dim in {{1,{codesz}}}, got {tuple(gamma1.shape)}")
                gamma_flat = gamma_src.reshape(batch * chunk_size, gamma_src.shape[-1]).to(dtype=torch.float32)
        else:
            gamma_src = gamma_blocks[:, bstart:bend, :]
            gamma_flat = gamma_src.reshape(batch * chunk_size, gamma_src.shape[-1]).to(dtype=torch.float32)

        branch_outputs = []
        for branch_idx, offset in enumerate(branch_offsets.tolist()):
            y_branch = r1_flat - float(offset)
            branch_outputs.append(
                _branch_factorized_stats(
                    y_branch=y_branch,
                    gamma_flat=gamma_flat,
                    offset=float(offset),
                    branch_idx=int(branch_idx),
                    cache=cache,
                    need_stats=need_stats,
                    use_compiled=use_compiled_prior,
                )
            )

        log_component = torch.stack([out["log_component"] for out in branch_outputs], dim=1)
        logZ_flat = torch.logsumexp(log_component.reshape(batch * chunk_size, -1), dim=-1)  # (batch*chunk_size,)
        log_weights = log_component - logZ_flat.view(batch * chunk_size, 1, 1)
        weights = torch.exp(log_weights)
        if logZ_blocks is not None:
            logZ_blocks[:, bstart:bend] = logZ_flat.view(batch, chunk_size).to(dtype=torch.float32)

        mu = torch.zeros((batch * chunk_size, codesz), dtype=torch.float32, device=r1_blocks.device)
        second_moment = torch.zeros((batch * chunk_size,), dtype=torch.float32, device=r1_blocks.device)
        var_diag = None

        if bool(need_var_diag):
            second_moment_diag = torch.zeros((batch * chunk_size, codesz), dtype=torch.float32, device=r1_blocks.device)
            for branch_idx, offset in enumerate(branch_offsets.tolist()):
                w_branch = weights[:, branch_idx, :]
                mu_comp = branch_outputs[branch_idx]["mu_component"]
                mu_branch = (w_branch.unsqueeze(-1) * mu_comp).sum(dim=1)
                mu += mu_branch
                second_moment += (w_branch * branch_outputs[branch_idx]["second_component"]).sum(dim=1)
                base_abs_sq_weighted = w_branch.matmul(base_abs_sq)
                w_sum = w_branch.sum(dim=1, keepdim=True)
                second_moment_diag += base_abs_sq_weighted - float(offset) ** 2 * w_sum + 2.0 * float(offset) * mu_branch
            var_diag = (second_moment_diag - mu.square()).clamp_min_(0.0)
            block_var = var_diag.mean(dim=-1)
        else:
            for branch_idx, _offset in enumerate(branch_offsets.tolist()):
                w_branch = weights[:, branch_idx, :]
                mu_comp = branch_outputs[branch_idx]["mu_component"]
                mu += (w_branch.unsqueeze(-1) * mu_comp).sum(dim=1)
                second_moment += (w_branch * branch_outputs[branch_idx]["second_component"]).sum(dim=1)
            block_var = ((second_moment - mu.square().sum(dim=-1)) / float(codesz)).clamp_min_(0.0)

        best_logscore = torch.stack([out["best_logscore"] for out in branch_outputs], dim=1)
        best_flat = best_logscore.reshape(batch * chunk_size, -1)
        best_pos = best_flat.argmax(dim=-1)
        best_branch = torch.div(best_pos, int(cache["abs_count"].item()), rounding_mode="floor")
        best_abs = best_pos % int(cache["abs_count"].item())
        map_idx_per_branch = torch.stack([out["best_idx"] for out in branch_outputs], dim=1)
        map_idx = map_idx_per_branch[torch.arange(batch * chunk_size, device=r1_blocks.device), best_branch, best_abs]
        map_val_flat = cb.grid.index_select(0, map_idx).to(dtype=torch.float32)

        if gamma_flat.shape[1] == 1:
            high_gamma_mask = gamma_flat.squeeze(-1) >= high_gamma_snap
        else:
            high_gamma_mask = gamma_flat.min(dim=-1).values >= high_gamma_snap
        if bool(high_gamma_mask.any().item()):
            mu[high_gamma_mask] = map_val_flat[high_gamma_mask]
            block_var[high_gamma_mask] = 0.0
            if var_diag is not None:
                var_diag[high_gamma_mask] = 0.0

        map_val = map_val_flat.view(batch, chunk_size, codesz).to(dtype=r1_blocks.dtype)
        x1_blocks[:, bstart:bend, :] = mu.view(batch, chunk_size, codesz).to(dtype=r1_blocks.dtype)
        var1[:, bstart:bend] = block_var.view(batch, chunk_size)
        if var1_diag is not None and var_diag is not None:
            var1_diag[:, bstart:bend, :] = var_diag.view(batch, chunk_size, codesz)
        map_idxs[:, bstart:bend] = map_idx.view(batch, chunk_size)
        map_vals[:, bstart:bend, :] = map_val

        if need_stats:
            sign_entropy = torch.stack([out["sign_entropy"] for out in branch_outputs], dim=1)
            weight_log_term = torch.where(
                torch.isfinite(log_weights), weights * log_weights, torch.zeros_like(log_weights)
            )
            entropy_flat = (-(weight_log_term).sum(dim=(1, 2)) + (weights * sign_entropy).sum(dim=(1, 2)))
            if entropy_blocks is not None:
                entropy_blocks[:, bstart:bend] = entropy_flat.view(batch, chunk_size).to(dtype=torch.float32)
            entropy_values.append(float(entropy_flat.mean().item()))

            second_logscore = torch.stack([out["second_logscore"] for out in branch_outputs], dim=1)
            top_candidates = torch.stack([best_logscore, second_logscore], dim=-1).reshape(batch * chunk_size, -1)
            top2 = torch.topk(top_candidates, k=min(2, top_candidates.shape[-1]), dim=-1).values
            if top2.shape[-1] == 1:
                margin_flat = torch.full((batch * chunk_size,), float("inf"), device=r1_blocks.device, dtype=torch.float32)
            else:
                margin_flat = (top2[:, 0] - top2[:, 1]).to(dtype=torch.float32)
            if margin_blocks is not None:
                margin_blocks[:, bstart:bend] = margin_flat.view(batch, chunk_size)
            margin_values.append(float(margin_flat.mean().item()))

    if gamma1.ndim == 2:
        alpha1 = gamma1.to(dtype=torch.float32) * var1.mean(dim=1, keepdim=True)
    else:
        if gamma1.shape[-1] == 1:
            alpha1 = (gamma1.squeeze(-1).to(dtype=torch.float32) * var1).mean(dim=1, keepdim=True)
        elif var1_diag is not None:
            alpha1 = (gamma1.to(dtype=torch.float32) * var1_diag).mean(dim=(1, 2)).unsqueeze(-1)
        else:
            alpha1 = (gamma1.to(dtype=torch.float32).mean(dim=-1) * var1).mean(dim=1, keepdim=True)

    return {
        "x1_blocks": x1_blocks,
        "var1": var1,
        "var1_diag": var1_diag,
        "map_idxs": map_idxs,
        "map_vals": map_vals,
        "alpha1": alpha1,
        "entropy": float(sum(entropy_values) / len(entropy_values)) if entropy_values else 0.0,
        "margin": float(sum(margin_values) / len(margin_values)) if margin_values else 0.0,
        **({"logZ_blocks": logZ_blocks} if logZ_blocks is not None else {}),
        **({"margin_blocks": margin_blocks} if margin_blocks is not None else {}),
        **({"entropy_blocks": entropy_blocks} if entropy_blocks is not None else {}),
    }
