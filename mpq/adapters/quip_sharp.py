from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch

from ..baselines.cd_solvers import (
    run_cdquant_e8p12_rankspace,
    run_quantease_e8p12_hessian,
)
from ..baselines.ojbkq import OJBKQConfig, run_ojbkq_e8p12_hessian
from ..combo import QuantCombo, make_combo
from ..solver import (
    E8P12VampConfig,
    build_e8p12_vamp_topk_candidates,
    polish_e8p12_rankspace,
    solve_e8p12_rankspace_vamp,
)
from ..utils import stable_seed


@dataclass(frozen=True)
class QuipAdapterConfig:
    block_rows: int = 128
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
    if not torch.isfinite(H_f64).all():
        H_f64 = torch.nan_to_num(H_f64, nan=0.0, posinf=0.0, neginf=0.0)
    try:
        lam, V = torch.linalg.eigh(H_f64)  # ascending order
    except Exception:
        lam_cpu, V_cpu = torch.linalg.eigh(H_f64.detach().cpu())
        lam = lam_cpu.to(device=H_f64.device)
        V = V_cpu.to(device=H_f64.device)
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


def _parse_cd_snapshot_budgets(args: Any, *, max_sweeps: int) -> set[int]:
    raw = getattr(args, "mpq_snapshot_budgets", "")
    if raw is None:
        return set()
    if isinstance(raw, str):
        values = [item.strip() for item in raw.split(",") if item.strip()]
    elif isinstance(raw, (list, tuple, set)):
        values = list(raw)
    else:
        values = [raw]
    budgets = {int(value) for value in values}
    invalid = sorted(budget for budget in budgets if budget < 0 or budget > int(max_sweeps))
    if invalid:
        raise ValueError(
            f"CD snapshot budgets {invalid} fall outside [0, {int(max_sweeps)}]."
        )
    return budgets


def _extract_layer_index(module_name: str) -> Optional[int]:
    match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", str(module_name))
    if match is None:
        return None
    try:
        return int(match.group(1))
    except (TypeError, ValueError):
        return None


def _summarize_e8p12_mechanism(
    *,
    module_name: str,
    rows: int,
    cols: int,
    nblocks: int,
    gamma_sum: float,
    gamma_count: int,
    gamma_min: float,
    gamma_max: float,
    changed_count: int,
    changed_total: int,
) -> Dict[str, Any]:
    total_count = int(changed_total) if int(changed_total) > 0 else int(rows) * int(nblocks)
    changed = int(changed_count)
    gamma_mean = float(gamma_sum / max(int(gamma_count), 1))
    return {
        "regime": "2bit_e8p12",
        "cleanup_unit": "e8p12_block",
        "module": str(module_name),
        "layer_index": _extract_layer_index(str(module_name)),
        "bits": 2,
        "rows": int(rows),
        "cols": int(cols),
        "nblocks": int(nblocks),
        "stage1_gamma1_mean": gamma_mean,
        "stage1_gamma1_min": float(gamma_min) if math.isfinite(gamma_min) else gamma_mean,
        "stage1_gamma1_max": float(gamma_max) if math.isfinite(gamma_max) else gamma_mean,
        "stage1_gamma1_count": int(gamma_count),
        "cleanup_changed_count": int(changed),
        "cleanup_total_count": int(total_count),
        "cleanup_changed_fraction": float(changed / max(total_count, 1)),
        "cleanup_agreement_fraction": float(1.0 - (changed / max(total_count, 1))),
    }


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
def _rtn_e8p12_init(
    *,
    Wr: torch.Tensor,
    cb: Any,
    resid_scale_override: float | None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    m, n = Wr.shape
    codesz = int(getattr(cb, "codesz", 8))
    if codesz != 8 or (n % codesz) != 0:
        raise ValueError(f"mpq RTN init expects E8P12 codesz=8; got codesz={codesz}, n={n}.")
    nblocks = n // codesz
    blocks = Wr.reshape(int(m) * int(nblocks), int(codesz))
    kwargs: Dict[str, Any] = {}
    if resid_scale_override is not None:
        kwargs["resid_scale_override"] = float(resid_scale_override)
    total_blocks = int(blocks.shape[0])
    blocks_per_chunk = 8192

    vals_out = torch.empty(total_blocks, int(codesz), device=Wr.device, dtype=Wr.dtype)
    idxs_out = torch.empty(total_blocks, device=Wr.device, dtype=getattr(cb, "idx_dtype", torch.int64))
    for start in range(0, total_blocks, blocks_per_chunk):
        end = min(start + blocks_per_chunk, total_blocks)
        vals, idxs = cb.quantize(blocks[start:end], return_idx=True, **kwargs)
        vals_out[start:end] = vals.to(dtype=Wr.dtype)
        idxs_out[start:end] = idxs.to(dtype=idxs_out.dtype)

    idxs_out = idxs_out.view(int(m), int(nblocks))
    vals_out = vals_out.view(int(m), int(nblocks), int(codesz)).reshape(int(m), int(n))
    return vals_out, idxs_out


@torch.inference_mode()
def _zero_e8p12_init(
    *,
    Wr: torch.Tensor,
    cb: Any,
    resid_scale_override: float | None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    m, n = Wr.shape
    codesz = int(getattr(cb, "codesz", 8))
    if codesz != 8 or (n % codesz) != 0:
        raise ValueError(f"mpq zero init expects E8P12 codesz=8; got codesz={codesz}, n={n}.")
    nblocks = n // codesz
    zero_blocks = Wr.new_zeros((int(m) * int(nblocks), int(codesz)))
    kwargs: Dict[str, Any] = {}
    if resid_scale_override is not None:
        kwargs["resid_scale_override"] = float(resid_scale_override)
    _, idxs = cb.quantize(zero_blocks, return_idx=True, **kwargs)
    idxs = idxs.to(device=Wr.device, dtype=getattr(cb, "idx_dtype", torch.int64)).view(int(m), int(nblocks))
    return Wr.new_zeros((int(m), int(n))), idxs


@torch.inference_mode()
def quip_tune_e8p12(
    *,
    Wr: torch.Tensor,
    Hr: torch.Tensor,
    hatWr: torch.Tensor,
    cb: Any,
    iters: int,
    resid_scale_override: float | None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run QUIP#'s block-coordinate tune sweeps starting from a given hatWr.

    Mirrors the update used in QUIP#'s LDLQ tune loop:

        WXWX_k = hat_k + (Wr - hat) @ Hr[:,k] @ inv(Hr[k,k])
        hat_k  = Q(WXWX_k)

    Returns (hatWr_tuned, Qidxs_tuned).
    """
    iters = int(iters)
    if iters <= 0:
        n = int(Wr.shape[1])
        codesz = int(getattr(cb, "codesz", 8))
        if codesz != 8 or (n % codesz) != 0:
            raise ValueError(f"quip_tune_e8p12 expects codesz=8 with n divisible by 8; got n={n}, codesz={codesz}.")
        nblocks = n // codesz
        # Best-effort indices for callers that need them.
        blocks = hatWr.reshape(int(Wr.shape[0]) * int(nblocks), int(codesz))
        _, idxs = cb.quantize(blocks, return_idx=True, resid_scale_override=resid_scale_override)
        idxs = idxs.view(int(Wr.shape[0]), int(nblocks)).to(dtype=getattr(cb, "idx_dtype", torch.int64))
        return hatWr, idxs

    Wr = Wr.to(dtype=torch.float32)
    Hr = Hr.to(dtype=torch.float32)
    hat = hatWr.to(dtype=torch.float32).clone()
    device = Wr.device
    m, n = Wr.shape
    codesz = int(getattr(cb, "codesz", 8))
    if codesz != 8 or (n % codesz) != 0:
        raise ValueError(f"quip_tune_e8p12 expects codesz=8 with n divisible by 8; got n={n}, codesz={codesz}.")
    nblocks = n // codesz

    inv_blocks = torch.empty(nblocks, codesz, codesz, dtype=torch.float32, device=device)
    for k in range(nblocks):
        s = codesz * k
        e = s + codesz
        inv_blocks[k] = torch.linalg.inv(Hr[s:e, s:e])

    Qidxs = torch.empty(m, nblocks, dtype=getattr(cb, "idx_dtype", torch.int64), device=device)
    quant_kwargs: Dict[str, Any] = {}
    if resid_scale_override is not None:
        quant_kwargs["resid_scale_override"] = float(resid_scale_override)

    for ie in range(iters):
        delta = Wr - hat
        last = (ie == (iters - 1))
        for k in reversed(range(nblocks)):
            s = codesz * k
            e = s + codesz
            WXWX = hat[:, s:e] + delta.matmul(Hr[:, s:e]).matmul(inv_blocks[k])

            # Update delta in-place so subsequent blocks see the new decision.
            delta[:, s:e] = delta[:, s:e] + hat[:, s:e]
            if not last:
                hat[:, s:e] = cb.quantize(WXWX, return_idx=False, **quant_kwargs).to(dtype=torch.float32)
            else:
                q, qidx = cb.quantize(WXWX, return_idx=True, **quant_kwargs)
                hat[:, s:e] = q.to(dtype=torch.float32)
                Qidxs[:, k] = qidx.to(dtype=Qidxs.dtype)
            delta[:, s:e] = delta[:, s:e] - hat[:, s:e]

    return hat, Qidxs


def _resolve_quip_combo(args: Any) -> Tuple[Optional[QuantCombo], Optional[str]]:
    quant_solver = getattr(args, "quant_solver", None)
    if quant_solver is None:
        quant_solver = getattr(args, "mpq_solver", None)
    if quant_solver is not None:
        combo = make_combo(
            backend=str(getattr(args, "quant_backend", "quip")),
            codebook=str(getattr(args, "quant_codebook", "e8p12")),
            preprocess=str(getattr(args, "quant_preprocess", "none")),
            rotation=str(getattr(args, "quant_rotation", getattr(args, "incoh_mode", "had"))),
            solver=str(quant_solver),
            polish_mode=str(getattr(args, "quant_polish_mode", getattr(args, "mpq_polish_mode", "none"))),
            topk_source=str(getattr(args, "quant_topk_source", "aux_vamp")),
        )
        return combo, None

    legacy_method = str(getattr(args, "mpq_method", "vamp") or "vamp").strip().lower()
    polish_mode = "topk_block_cd" if legacy_method == "vamp-topk-polish" else "block_cd"
    if int(getattr(args, "mpq_polish_sweeps", 3) or 0) <= 0:
        polish_mode = "none"
    combo = make_combo(
        backend="quip",
        codebook="e8p12",
        preprocess="none",
        rotation=str(getattr(args, "incoh_mode", "had")),
        solver="vamp",
        polish_mode=polish_mode,
    )
    return combo, None


def _build_vamp_cfg(args: Any, *, polish_mode: str) -> E8P12VampConfig:
    return E8P12VampConfig(
        iters=int(getattr(args, "mpq_iters", 16)),
        damping=float(getattr(args, "mpq_damping", 0.5)),
        gamma_start=float(getattr(args, "mpq_gamma_start", 1.0)),
        gamma_retry_factor=float(getattr(args, "mpq_gamma_retry_factor", 4.0)),
        gamma_retry_steps=int(getattr(args, "mpq_gamma_retry_steps", 3)),
        prior_tau=float(getattr(args, "mpq_prior_tau", 0.25)),
        prior_tau_scale_mode=str(getattr(args, "mpq_prior_tau_scale_mode", "absmax")),
        compile_prior=bool(getattr(args, "mpq_compile_prior", True)),
        obs_precision=float(getattr(args, "mpq_obs_precision", 0.0)),
        em_obs_precision=bool(getattr(args, "mpq_em_obs_precision", True)),
        em_obs_precision_damping=float(getattr(args, "mpq_em_obs_precision_damping", 0.3)),
        em_obs_precision_min=float(getattr(args, "mpq_em_obs_precision_min", 1e-4)),
        em_obs_precision_max=float(getattr(args, "mpq_em_obs_precision_max", 1e8)),
        blocks_per_chunk=int(getattr(args, "mpq_blocks_per_chunk", 16)),
        polish_sweeps=int(getattr(args, "mpq_polish_sweeps", 3)),
        polish_mode=str(polish_mode),
        polish_metric=str(getattr(args, "mpq_polish_metric", "gram")),
        polish_metric_rows_per_chunk=int(getattr(args, "mpq_metric_rows_per_chunk", 0)),
        polish_max_blocks=int(getattr(args, "mpq_polish_max_blocks", 0)),
        topk_k=int(getattr(args, "mpq_topk_k", getattr(args, "quant_topk_k", 16)) or 16),
        best_iterate=bool(getattr(args, "mpq_best_iterate", False)),
    )


@torch.inference_mode()
def mpq_refine_codebook(
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
    """QUIP# adapter: refine codebook decisions via MPQ (VAMP + greedy polish)."""
    start = time.time()
    Wr = Wr.to(dtype=torch.float32)
    Hr = Hr.to(dtype=torch.float32)
    device = Wr.device
    m, n = Wr.shape
    codesz = int(getattr(cb, "codesz", 8))
    if codesz != 8 or (n % codesz) != 0:
        raise ValueError(f"mpq QUIP adapter expects E8P12 codesz=8; got codesz={codesz}, n={n}.")
    nblocks = n // codesz

    init_mode = str(getattr(args, "mpq_init", "rtn") or "rtn").strip().lower()
    seed = int(getattr(args, "seed", getattr(args, "mpq_seed", 0)) or 0)

    resid_scale_override = getattr(args, "resid_scale_override", None)
    if resid_scale_override is not None:
        resid_scale_override = float(resid_scale_override)

    if init_mode == "random":
        hatWr_init, Qidxs_init = _random_e8p12_init(
            cb=cb,
            rows=m,
            nblocks=nblocks,
            seed_key=f"{module_name}:mpq_init",
            base_seed=seed,
            device=device,
            dtype=Wr.dtype,
        )
    elif init_mode == "rtn":
        hatWr_init, Qidxs_init = _rtn_e8p12_init(Wr=Wr, cb=cb, resid_scale_override=resid_scale_override)
    elif init_mode in {"float", "paper", "unquantized"}:
        hatWr_init = Wr.to(dtype=torch.float32).clone()
        _, Qidxs_init = _rtn_e8p12_init(Wr=Wr, cb=cb, resid_scale_override=resid_scale_override)
    elif init_mode == "zero":
        hatWr_init, Qidxs_init = _zero_e8p12_init(Wr=Wr, cb=cb, resid_scale_override=resid_scale_override)
    elif init_mode == "ldlq":
        hatWr_init = hatWr0.to(dtype=torch.float32)
        Qidxs_init = Qidxs0.to(dtype=getattr(cb, "idx_dtype", torch.int64))
    else:
        raise ValueError(f"unknown mpq_init={init_mode!r} (expected 'float', 'random', 'rtn', 'zero', or 'ldlq').")

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
    requested_obs = float(getattr(args, "mpq_obs_precision", 0.0) or 0.0)
    if requested_obs > 0.0:
        obs_precision = requested_obs
    else:
        block_rows = _resolve_block_rows(m, int(getattr(args, "mpq_block_rows", adapter_cfg.block_rows) or adapter_cfg.block_rows))
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

    combo, legacy_method = _resolve_quip_combo(args)
    if combo is not None:
        if combo.solver == "ldlq_tune10":
            raise ValueError("QUIP adapter does not handle solver=ldlq_tune10; route it through the native QUIP backend path.")
        vamp_cfg = _build_vamp_cfg(args, polish_mode=combo.polish_mode)
    else:
        vamp_cfg = _build_vamp_cfg(args, polish_mode="block_cd")

    block_rows = _resolve_block_rows(
        m,
        int(getattr(args, "mpq_block_rows", adapter_cfg.block_rows) or adapter_cfg.block_rows),
    )
    requested_snapshot_budgets = _parse_cd_snapshot_budgets(
        args,
        max_sweeps=int(getattr(args, "mpq_polish_sweeps", 0) or 0),
    )
    cd_snapshot_chunks: Dict[int, list[torch.Tensor]] = {
        budget: [] for budget in sorted(requested_snapshot_budgets)
    }
    cd_trace_accumulator: Dict[int, Dict[str, float]] = {}
    exact_ldlq, exact_ldlq_chunks = _exact_surrogate_chunk_means(
        Wr,
        hatWr0.to(dtype=torch.float32),
        Hr,
        block_rows=block_rows,
    )
    if combo is not None and combo.solver == "ldlq" and init_mode == "random":
        final_hatWr = hatWr_init.clone()
        final_Qidxs = Qidxs_init.clone()
        exact_before, exact_before_chunks = _exact_surrogate_chunk_means(
            Wr,
            final_hatWr,
            Hr,
            block_rows=block_rows,
        )
    elif combo is not None and combo.solver in {"ldlq", "cdquant", "ojbkq"}:
        final_hatWr = hatWr0.to(dtype=torch.float32).clone()
        final_Qidxs = Qidxs0.to(dtype=getattr(cb, "idx_dtype", torch.int64)).clone()
        exact_before, exact_before_chunks = exact_ldlq, list(exact_ldlq_chunks)
    elif combo is not None and combo.solver == "quantease":
        qe_init_mode = str(getattr(args, "mpq_init", "float") or "float").strip().lower()
        if qe_init_mode == "random":
            final_hatWr, final_Qidxs = _random_e8p12_init(
                cb=cb,
                rows=m,
                nblocks=nblocks,
                seed_key=f"{module_name}:quantease_init",
                base_seed=seed,
                device=device,
                dtype=Wr.dtype,
            )
            qe_init_label = "random_e8p12"
        elif qe_init_mode == "rtn":
            final_hatWr, final_Qidxs = _rtn_e8p12_init(Wr=Wr, cb=cb, resid_scale_override=resid_scale_override)
            qe_init_label = "rtn"
        elif qe_init_mode == "ldlq":
            final_hatWr = hatWr0.to(dtype=torch.float32).clone()
            final_Qidxs = Qidxs0.to(dtype=getattr(cb, "idx_dtype", torch.int64)).clone()
            qe_init_label = "ldlq"
        elif qe_init_mode in {"float", "paper", "unquantized"}:
            final_hatWr = Wr.to(dtype=torch.float32).clone()
            _, final_Qidxs = _rtn_e8p12_init(Wr=Wr, cb=cb, resid_scale_override=resid_scale_override)
            qe_init_label = "float"
        else:
            raise ValueError(f"unknown QuantEase mpq_init={qe_init_mode!r} (expected 'float', 'random', 'rtn', or 'ldlq').")
        exact_before, exact_before_chunks = _exact_surrogate_chunk_means(Wr, final_hatWr, Hr, block_rows=block_rows)
    else:
        final_hatWr = hatWr_init.clone()
        final_Qidxs = Qidxs_init.clone()
        exact_before, exact_before_chunks = _exact_surrogate_chunk_means(Wr, hatWr_init, Hr, block_rows=block_rows)
    obs_precision_block = float(obs_precision)
    stats_chunk: Dict[str, Any] = {}
    gamma_sum = 0.0
    gamma_count = 0
    gamma_min = float("inf")
    gamma_max = float("-inf")
    changed_count = 0
    changed_total = 0
    for row_start in range(0, m, block_rows):
        row_end = min(row_start + block_rows, m)
        rb = slice(row_start, row_end)
        chunk_snapshots: Dict[int, torch.Tensor] = {}
        chunk_trace: list[Dict[str, Any]] = []
        if combo is not None and combo.solver == "vamp":
            debug_chunk: Dict[str, Any] = {
                "summary_only": True,
                "cd_snapshot_budgets": sorted(requested_snapshot_budgets),
            }
            hat_chunk, idx_chunk, _err_chunk, stats_chunk = solve_e8p12_rankspace_vamp(
                Wr=Wr[rb],
                hatWr_init=final_hatWr[rb],
                cb=cb,
                V=V,
                Vh=Vh,
                S=S,
                Vs=Vs,
                cfg=vamp_cfg,
                module_name=module_name,
                obs_precision_init=obs_precision_block,
                log_stats=False,
                debug_out=debug_chunk,
            )
            final_hatWr[rb] = hat_chunk
            final_Qidxs[rb] = idx_chunk
            chunk_snapshots = debug_chunk.get("cd_snapshots", {})
            chunk_trace = list(
                stats_chunk.get("polish_trace", {}).get("sweep_trace", [])
            )
            if bool(vamp_cfg.em_obs_precision) and float(vamp_cfg.obs_precision) <= 0.0:
                obs_precision_block = float(stats_chunk.get("obs_precision_final", obs_precision_block))
            if "gamma1_sum" in debug_chunk:
                gamma_sum += float(debug_chunk.get("gamma1_sum", 0.0) or 0.0)
                gamma_count += int(debug_chunk.get("gamma1_count", 0) or 0)
                gamma_min = min(gamma_min, float(debug_chunk.get("gamma1_min", float("inf"))))
                gamma_max = max(gamma_max, float(debug_chunk.get("gamma1_max", float("-inf"))))
            else:
                gamma_chunk = debug_chunk.get("gamma1")
                if isinstance(gamma_chunk, torch.Tensor) and gamma_chunk.numel() > 0:
                    gamma_chunk = gamma_chunk.to(dtype=torch.float64)
                    gamma_sum += float(gamma_chunk.sum().item())
                    gamma_count += int(gamma_chunk.numel())
                    gamma_min = min(gamma_min, float(gamma_chunk.min().item()))
                    gamma_max = max(gamma_max, float(gamma_chunk.max().item()))
            if "changed_count" in debug_chunk:
                changed_count += int(debug_chunk.get("changed_count", 0) or 0)
                changed_total += int(debug_chunk.get("changed_total", 0) or 0)
            else:
                changed_mask = debug_chunk.get("changed_mask")
                if isinstance(changed_mask, torch.Tensor):
                    changed_count += int(changed_mask.to(dtype=torch.int64).sum().item())
                    changed_total += int(changed_mask.numel())
        elif combo is not None and combo.solver == "ldlq":
            solver_label = "e8p12_random_hard" if init_mode == "random" else "e8p12_ldlq"
            if combo.polish_mode == "none":
                hat_chunk = final_hatWr[rb]
                idx_chunk = final_Qidxs[rb]
                stats_chunk = {
                    "solver": solver_label,
                    "initialization": init_mode,
                    "polish_mode": "none",
                    "accepted_updates": 0.0,
                }
            elif combo.polish_mode == "block_cd":
                hat_chunk, idx_chunk, _err_chunk, polish_stats = polish_e8p12_rankspace(
                    x_init=final_hatWr[rb],
                    idxs_init=final_Qidxs[rb],
                    Y_u=Wr[rb].matmul(Vs),
                    Vs=Vs,
                    cb=cb,
                    sweeps=int(getattr(args, "mpq_polish_sweeps", 3)),
                    polish_mode="block_cd",
                    polish_metric=str(getattr(args, "mpq_polish_metric", "gram")),
                    metric_rows_per_chunk=int(getattr(args, "mpq_metric_rows_per_chunk", 0)),
                    polish_max_blocks=int(getattr(args, "mpq_polish_max_blocks", 0)),
                    snapshot_budgets=requested_snapshot_budgets,
                    snapshots_out=chunk_snapshots,
                )
                stats_chunk = {
                    "solver": solver_label,
                    "initialization": init_mode,
                    "polish_mode": "block_cd",
                    "polish_updates": float(polish_stats.get("accepted_updates", 0.0)),
                    "polish_trace": polish_stats,
                }
                chunk_trace = list(polish_stats.get("sweep_trace", []))
            else:
                aux_cfg = _build_vamp_cfg(args, polish_mode="none")
                aux_debug: Dict[str, Any] = {"capture_r1": True}
                _aux_hat, _aux_idx, _aux_err, aux_stats = solve_e8p12_rankspace_vamp(
                    Wr=Wr[rb],
                    hatWr_init=final_hatWr[rb],
                    cb=cb,
                    V=V,
                    Vh=Vh,
                    S=S,
                    Vs=Vs,
                    cfg=aux_cfg,
                    module_name=module_name,
                    obs_precision_init=obs_precision_block,
                    log_stats=False,
                    debug_out=aux_debug,
                )
                topk = build_e8p12_vamp_topk_candidates(
                    r1=aux_debug["r1"].to(device=Wr.device, dtype=torch.float32),
                    gamma1=aux_debug["gamma1"].to(device=Wr.device, dtype=torch.float32),
                    Wr=Wr[rb],
                    cb=cb,
                    blocks_per_chunk=int(getattr(args, "mpq_blocks_per_chunk", 16)),
                    prior_tau=float(getattr(args, "mpq_prior_tau", 0.25)),
                    prior_tau_scale_mode=str(getattr(args, "mpq_prior_tau_scale_mode", "absmax")),
                    k=int(getattr(args, "mpq_topk_k", getattr(args, "quant_topk_k", 16)) or 16),
                )
                hat_chunk, idx_chunk, _err_chunk, polish_stats = polish_e8p12_rankspace(
                    x_init=final_hatWr[rb],
                    idxs_init=final_Qidxs[rb],
                    Y_u=Wr[rb].matmul(Vs),
                    Vs=Vs,
                    cb=cb,
                    sweeps=int(getattr(args, "mpq_polish_sweeps", 3)),
                    polish_mode="topk_block_cd",
                    polish_metric=str(getattr(args, "mpq_polish_metric", "gram")),
                    metric_rows_per_chunk=int(getattr(args, "mpq_metric_rows_per_chunk", 0)),
                    polish_max_blocks=int(getattr(args, "mpq_polish_max_blocks", 0)),
                    topk_idxs=topk["topk_idxs"].to(device=Wr.device, dtype=torch.long),
                )
                if bool(aux_cfg.em_obs_precision) and float(aux_cfg.obs_precision) <= 0.0:
                    obs_precision_block = float(aux_stats.get("obs_precision_final", obs_precision_block))
                stats_chunk = {
                    "solver": solver_label,
                    "initialization": init_mode,
                    "polish_mode": "topk_block_cd",
                    "topk_k": int(topk["topk_idxs"].shape[-1]),
                    "polish_updates": float(polish_stats.get("accepted_updates", 0.0)),
                }
            final_hatWr[rb] = hat_chunk
            final_Qidxs[rb] = idx_chunk
        elif combo is not None and combo.solver in {"cdquant", "quantease", "ojbkq"}:
            Y_u_chunk = Wr[rb].matmul(Vs)
            if combo.solver == "cdquant":
                hat_chunk, idx_chunk, _err_chunk, stats_chunk = run_cdquant_e8p12_rankspace(
                    hatWr_init=final_hatWr[rb],
                    Qidxs_init=final_Qidxs[rb],
                    Y_u=Y_u_chunk,
                    Vs=Vs,
                    cb=cb,
                    sweeps=max(int(getattr(args, "mpq_polish_sweeps", 3)), 1),
                    polish_metric=str(getattr(args, "mpq_polish_metric", "gram")),
                    metric_rows_per_chunk=int(getattr(args, "mpq_metric_rows_per_chunk", 0)),
                    polish_max_blocks=int(getattr(args, "mpq_polish_max_blocks", 0)),
                )
            elif combo.solver == "quantease":
                hat_chunk, idx_chunk, _err_chunk, stats_chunk = run_quantease_e8p12_hessian(
                    Wr=Wr[rb],
                    Hr=Hr,
                    cb=cb,
                    Y_u=Y_u_chunk,
                    Vs=Vs,
                    sweeps=max(int(getattr(args, "mpq_iters", 16)), 1),
                    metric_rows_per_chunk=int(getattr(args, "mpq_metric_rows_per_chunk", 0)),
                    hatWr_init=final_hatWr[rb],
                    Qidxs_init=final_Qidxs[rb],
                    init_label=qe_init_label,
                )
            else:
                ojbkq_cfg = OJBKQConfig(
                    enabled=True,
                    random_trials=int(getattr(args, "ojbkq_random_trials", 4)),
                    alpha_mode=str(getattr(args, "ojbkq_alpha_mode", "paper")),
                    alpha_override=getattr(args, "ojbkq_alpha_override", None),
                    alpha_scale=float(getattr(args, "ojbkq_alpha_scale", 1.0)),
                    sample_topk=getattr(args, "ojbkq_sample_topk", None),
                )
                hat_chunk, idx_chunk, stats_chunk = run_ojbkq_e8p12_hessian(
                    Wr=Wr[rb],
                    Hr=Hr,
                    hatWr0=final_hatWr[rb],
                    Qidxs0=final_Qidxs[rb],
                    cb=cb,
                    config=ojbkq_cfg,
                    module_name=module_name,
                )
                if combo.polish_mode != "none" and int(getattr(args, "mpq_polish_sweeps", 0)) > 0:
                    hat_chunk, idx_chunk, _err_chunk, polish_stats = polish_e8p12_rankspace(
                        x_init=hat_chunk,
                        idxs_init=idx_chunk,
                        Y_u=Y_u_chunk,
                        Vs=Vs,
                        cb=cb,
                        sweeps=int(getattr(args, "mpq_polish_sweeps", 3)),
                        polish_mode=str(combo.polish_mode),
                        polish_metric=str(getattr(args, "mpq_polish_metric", "gram")),
                        metric_rows_per_chunk=int(getattr(args, "mpq_metric_rows_per_chunk", 0)),
                        polish_max_blocks=int(getattr(args, "mpq_polish_max_blocks", 0)),
                        snapshot_budgets=requested_snapshot_budgets,
                        snapshots_out=chunk_snapshots,
                    )
                    stats_chunk = {
                        **stats_chunk,
                        "post_ojbkq_polish_mode": str(combo.polish_mode),
                        "post_ojbkq_polish_sweeps": int(getattr(args, "mpq_polish_sweeps", 3)),
                        "post_ojbkq_polish_updates": float(
                            polish_stats.get("accepted_updates", 0.0)
                        ),
                        "post_ojbkq_polish_trace": polish_stats,
                    }
                    chunk_trace = list(polish_stats.get("sweep_trace", []))
            changed_count += int((idx_chunk != final_Qidxs[rb]).to(dtype=torch.int64).sum().item())
            changed_total += int(idx_chunk.numel())
            final_hatWr[rb] = hat_chunk
            final_Qidxs[rb] = idx_chunk
        else:
            raise RuntimeError("failed to resolve QUIP solver combination")

        if requested_snapshot_budgets:
            missing_budgets = sorted(requested_snapshot_budgets - set(chunk_snapshots))
            if missing_budgets:
                raise RuntimeError(
                    f"missing CD snapshots {missing_budgets} for {module_name} rows "
                    f"[{row_start}, {row_end})"
                )
            for budget in sorted(requested_snapshot_budgets):
                cd_snapshot_chunks[budget].append(chunk_snapshots[budget])
        chunk_rows = int(row_end - row_start)
        for trace_item in chunk_trace:
            budget = int(trace_item["budget"])
            accumulator = cd_trace_accumulator.setdefault(
                budget,
                {
                    "weighted_objective": 0.0,
                    "rows": 0.0,
                    "accepted_updates_sweep": 0.0,
                    "accepted_updates_cumulative": 0.0,
                    "assignment_changes_from_cd0": 0.0,
                    "assignments_total": 0.0,
                },
            )
            accumulator["weighted_objective"] += (
                float(trace_item["stabilized_objective_mean"]) * chunk_rows
            )
            accumulator["rows"] += chunk_rows
            for key in (
                "accepted_updates_sweep",
                "accepted_updates_cumulative",
                "assignment_changes_from_cd0",
                "assignments_total",
            ):
                accumulator[key] += float(trace_item.get(key, 0))

    exact_after, exact_after_chunks = _exact_surrogate_chunk_means(Wr, final_hatWr, Hr, block_rows=block_rows)

    # Surrogate gate helpers: compare per-row-chunk (block_rows) means vs LDLQ.
    chunk_eps = 1e-8
    violation_count = 0
    max_delta = 0.0
    max_ratio = 0.0
    for ldlq_val, mpq_val in zip(exact_ldlq_chunks, exact_after_chunks):
        if mpq_val > ldlq_val + chunk_eps:
            violation_count += 1
            max_delta = max(max_delta, float(mpq_val - ldlq_val))
            if ldlq_val > 0:
                max_ratio = max(max_ratio, float(mpq_val / ldlq_val))

    stats: Dict[str, Any] = {
        "method": "mpq",
        "backend": "quip",
        "solver": str(combo.solver) if combo is not None else str(legacy_method or "vamp"),
        "polish_mode": str(combo.polish_mode) if combo is not None else "block_cd",
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
        "solver_stats": stats_chunk if "stats_chunk" in locals() else {},
        "cd_trace": [
            {
                "budget": int(budget),
                "stabilized_objective_mean": float(
                    values["weighted_objective"] / max(values["rows"], 1.0)
                ),
                "accepted_updates_sweep": int(values["accepted_updates_sweep"]),
                "accepted_updates_cumulative": int(values["accepted_updates_cumulative"]),
                "assignment_changes_from_cd0": int(values["assignment_changes_from_cd0"]),
                "assignments_total": int(values["assignments_total"]),
            }
            for budget, values in sorted(cd_trace_accumulator.items())
        ],
        "mechanism": _summarize_e8p12_mechanism(
            module_name=str(module_name),
            rows=int(m),
            cols=int(n),
            nblocks=int(nblocks),
            gamma_sum=float(gamma_sum),
            gamma_count=int(gamma_count),
            gamma_min=float(gamma_min),
            gamma_max=float(gamma_max),
            changed_count=int(changed_count),
            changed_total=int(changed_total),
        ),
    }
    if requested_snapshot_budgets:
        stats["cd_snapshot_budgets"] = sorted(requested_snapshot_budgets)
        stats["_cd_snapshot_idxs"] = {
            int(budget): torch.cat(chunks, dim=0)
            for budget, chunks in cd_snapshot_chunks.items()
        }
    return final_hatWr.to(dtype=hatWr0.dtype), final_Qidxs, stats
