from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

from ..rotation import apply_right_hadamard, apply_right_hadamard_inverse, sample_hadamard_signs
from ..linear import vamp_rankspace_error
from ..solver import (
    UniformVampConfig,
    build_uniform_vamp_topk_candidates,
    polish_uniform_rankspace,
    solve_uniform_rankspace_vamp,
)


@dataclass(frozen=True)
class GPTQAdapterConfig:
    hessian_scale: float = 0.5  # GPTQModel upstream stores H = 2/m X^T X; divide by 2 to match X^T X / m.
    block_rows: int = 32
    jitter_base: float = 1e-6
    eigen_floor_scale: float = 1e-6
    rotation_seed: int = 0


@torch.inference_mode()
def _group_metadata(cols: int, group_size: int) -> Tuple[List[Tuple[int, int]], torch.Tensor]:
    slices: List[Tuple[int, int]] = []
    widths: List[int] = []
    for start in range(0, cols, group_size):
        end = min(start + group_size, cols)
        slices.append((start, end))
        widths.append(end - start)
    return slices, torch.tensor(widths, dtype=torch.long)


@torch.inference_mode()
def _expand_group_params(grouped: torch.Tensor, widths: torch.Tensor) -> torch.Tensor:
    if grouped.numel() == 0 or widths.numel() == 0:
        return grouped.new_zeros((grouped.shape[0], 0))
    widths = widths.to(device=grouped.device)
    return torch.repeat_interleave(grouped, widths, dim=1)


@torch.inference_mode()
def _expand_group_params_from_gidx(grouped: torch.Tensor, g_idx: torch.Tensor) -> torch.Tensor:
    if grouped.numel() == 0 or g_idx.numel() == 0:
        return grouped.new_zeros((grouped.shape[0], int(g_idx.numel())))
    return grouped.index_select(1, g_idx.to(device=grouped.device, dtype=torch.long))


@torch.inference_mode()
def _vamp_hessian_eigen(
    H: torch.Tensor,
    *,
    jitter_base: float = 1e-6,
    floor_scale: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Eigendecomposition of Hessian H for the linear MMSE step.

    Returns (eigenvalues, eigvecs, eigen_floor) where eigenvalues are floored and descending.
    """
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
    return lam, V, float(eigen_floor)


@torch.inference_mode()
def _resolve_block_rows(rows: int, cols: int, maxq: int, requested: int, device: torch.device) -> int:
    block_rows = min(int(rows), max(int(requested), 1))
    if device.type != "cuda" or not hasattr(torch.cuda, "mem_get_info"):
        return block_rows

    # Query memory for the requested device via a device context. This avoids
    # version-specific `mem_get_info` call signatures.
    with torch.cuda.device(device):
        free_bytes, _ = torch.cuda.mem_get_info()

    bytes_per_row = 4 * (8 * int(cols) + 4 * int(cols) * (int(maxq) + 1))
    budget = min(4 * 1024**3, int(free_bytes) // 2)
    auto = max(8, min(int(rows), int(budget // max(bytes_per_row, 1))))
    return min(int(rows), max(block_rows, int(auto)))


@torch.inference_mode()
def _uniform_vamp_cfg_from_qcfg(qcfg: Any) -> UniformVampConfig:
    mpgq_cfg = getattr(qcfg, "mpgq", None)
    return UniformVampConfig(
        iters=int(getattr(mpgq_cfg, "iters", 10)),
        damping=float(getattr(mpgq_cfg, "damping", 0.5)),
        final_decision=str(getattr(mpgq_cfg, "final_decision", "map") or "map"),
        init_from="rtn",
        gamma_start=float(getattr(mpgq_cfg, "gamma_start", 1.0)),
        gamma_retry_factor=float(getattr(mpgq_cfg, "gamma_retry_factor", 4.0)),
        gamma_retry_steps=int(getattr(mpgq_cfg, "gamma_retry_steps", 3)),
        prior_tau=float(getattr(mpgq_cfg, "prior_tau", 0.2)),
        obs_precision=float(getattr(mpgq_cfg, "obs_precision", 0.0)),
        em_obs_precision=bool(getattr(mpgq_cfg, "em_obs_precision", True)),
        em_obs_precision_damping=float(getattr(mpgq_cfg, "em_obs_precision_damping", 0.3)),
        em_obs_precision_min=float(getattr(mpgq_cfg, "em_obs_precision_min", 1e-4)),
        em_obs_precision_max=float(getattr(mpgq_cfg, "em_obs_precision_max", 1e8)),
        polish_sweeps=int(getattr(mpgq_cfg, "polish_sweeps", 3)),
        polish_mode=str(getattr(mpgq_cfg, "polish_mode", "block_cd") or "block_cd"),
        topk_k=int(getattr(mpgq_cfg, "topk_k", 8) or 8),
    )


@torch.inference_mode()
def _adapter_cfg_from_qcfg(qcfg: Any) -> GPTQAdapterConfig:
    mpgq_cfg = getattr(qcfg, "mpgq", None)
    return GPTQAdapterConfig(
        hessian_scale=float(getattr(mpgq_cfg, "hessian_scale", 0.5)),
        block_rows=int(getattr(mpgq_cfg, "block_rows", 32)),
        jitter_base=float(getattr(mpgq_cfg, "eigen_jitter", 1e-6)),
        eigen_floor_scale=float(getattr(mpgq_cfg, "eigen_floor_scale", 1e-6)),
        rotation_seed=int(getattr(mpgq_cfg, "rotation_seed", 0)),
    )


@torch.inference_mode()
def run_mpgq_quantization(
    *,
    W: torch.Tensor,
    H: torch.Tensor,
    quantizer: Any,
    qcfg: Any,
    module_name: str,
    vamp_cfg: UniformVampConfig,
    adapter_cfg: GPTQAdapterConfig = GPTQAdapterConfig(),
) -> Tuple[
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    float,
    Dict[str, Any],
]:
    """Run MPGQ on a GPTQModel layer given weights and accumulated Hessian.

    Returns:
      Q: (rows, cols) dense float32 weights (research path)
      q_int: (rows, cols) int32 quantized indices (None if internal rotation used)
      scale_groups: (rows, n_groups) float32 (None if internal rotation used)
      zero_groups: (rows, n_groups) float32 (None if internal rotation used)
      g_idx: (cols,) int32 group index per column (None if internal rotation used)
      quant_time_s: float
      stats: dict
    """
    start = time.time()
    W = W.to(dtype=torch.float32)
    H = H.to(dtype=torch.float32)
    device = W.device
    rows, cols = W.shape

    bits = int(getattr(qcfg, "bits"))
    maxq = int((2**bits) - 1)
    group_size = int(getattr(qcfg, "group_size", cols))
    if group_size == -1:
        group_size = cols

    rotation = str(getattr(qcfg, "rotation", None) or "none").strip().lower()
    if rotation not in ("none", "hadamard", "random", ""):
        raise ValueError(f"mpgq only supports rotation in {{none, hadamard, random}}; got rotation={rotation!r}.")
    use_hadamard = rotation == "hadamard"
    if bool(getattr(qcfg, "desc_act", False)):
        raise ValueError("mpgq exact-full path does not support desc_act=True.")

    H_work = float(adapter_cfg.hessian_scale) * H
    eigenvalues, V, eigen_floor = _vamp_hessian_eigen(
        H_work,
        jitter_base=float(adapter_cfg.jitter_base),
        floor_scale=float(adapter_cfg.eigen_floor_scale),
    )
    V = V.to(device=device, dtype=torch.float32).contiguous()
    eigenvalues = eigenvalues.to(device=device, dtype=torch.float32).contiguous()
    Vh = V.T.contiguous()
    S = torch.sqrt(eigenvalues).contiguous()

    if use_hadamard:
        key = f"{module_name}:full"
        signs = sample_hadamard_signs(width=cols, key=key, seed=int(adapter_cfg.rotation_seed), device=device)
        W_work = apply_right_hadamard(W, signs=signs)
        V_work = apply_right_hadamard(V.T.contiguous(), signs=signs).T.contiguous()
        V = V_work
        Vh = V.T.contiguous()
    else:
        signs = None
        W_work = W

    Vs = V * S.unsqueeze(0)

    group_slices, widths = _group_metadata(cols, group_size)
    # Collect fixed quantizer group params (scale/zero) over ALL rows, once.
    scale_chunks: List[torch.Tensor] = []
    zero_chunks: List[torch.Tensor] = []
    for start_col, end_col in group_slices:
        Wg = W_work[:, start_col:end_col].contiguous()
        quantizer.find_params(Wg, weight=True)
        scale_chunks.append(quantizer.scale.to(dtype=torch.float32).clone())
        zero_chunks.append(quantizer.zero.to(dtype=torch.float32).clone())
    scale_groups = torch.cat(scale_chunks, dim=1) if scale_chunks else W_work.new_zeros((rows, 0))
    zero_groups = torch.cat(zero_chunks, dim=1) if zero_chunks else W_work.new_zeros((rows, 0))

    # Warm-start obs precision from RTN error in rank space, unless explicitly overridden.
    block_rows = _resolve_block_rows(
        rows=rows,
        cols=cols,
        maxq=maxq,
        requested=int(adapter_cfg.block_rows),
        device=device,
    )

    if float(getattr(vamp_cfg, "obs_precision", 0.0)) > 0.0:
        obs_precision = float(getattr(vamp_cfg, "obs_precision"))
    else:
        sum_sq = 0.0
        count = 0
        for row_start in range(0, rows, block_rows):
            row_end = min(row_start + block_rows, rows)
            rb = slice(row_start, row_end)
            scale_full = _expand_group_params(scale_groups[rb], widths)
            zero_full = _expand_group_params(zero_groups[rb], widths)
            q_rtn = torch.clamp(torch.round(W_work[rb] / scale_full) + zero_full, 0, maxq).to(dtype=torch.float32)
            x_rtn = scale_full * (q_rtn - zero_full)
            delta = (W_work[rb] - x_rtn).to(dtype=torch.float32)
            if Vs.numel():
                diff_u = delta.matmul(Vs)
                sum_sq += float(diff_u.square().sum().item())
                count += int(diff_u.numel())
            else:
                sum_sq += float(delta.square().sum().item())
                count += int(delta.numel())
        mse = max(sum_sq / max(count, 1), 1e-12)
        obs_precision = 1.0 / float(mse)

    Q_work = torch.empty_like(W_work, dtype=torch.float32)
    q_int_work = torch.empty((rows, cols), dtype=torch.int32, device=device) if not use_hadamard else None
    layer_err_sum = 0.0
    rtn_err_sum = 0.0
    obs_precision_block = float(obs_precision)

    block_stats: List[Dict[str, Any]] = []
    for row_start in range(0, rows, block_rows):
        row_end = min(row_start + block_rows, rows)
        rb = slice(row_start, row_end)
        scale_full = _expand_group_params(scale_groups[rb], widths)
        zero_full = _expand_group_params(zero_groups[rb], widths)
        q_rtn = torch.clamp(torch.round(W_work[rb] / scale_full) + zero_full, 0, maxq).to(dtype=torch.float32)
        x_rtn = scale_full * (q_rtn - zero_full)

        q_int, x_final, final_err, rtn_err, stats = solve_uniform_rankspace_vamp(
            W=W_work[rb],
            x_rtn=x_rtn,
            scale_full=scale_full,
            zero_full=zero_full,
            maxq=maxq,
            V=V,
            Vh=Vh,
            S=S,
            Vs=Vs,
            cfg=vamp_cfg,
            obs_precision_init=obs_precision_block,
            log_stats=False,
        )
        Q_work[rb] = x_final
        if q_int_work is not None:
            q_int_work[rb] = q_int
        layer_err_sum += float(final_err.sum().item())
        rtn_err_sum += float(rtn_err.sum().item())
        if bool(getattr(vamp_cfg, "em_obs_precision", True)):
            obs_precision_block = float(stats.get("obs_precision_final", obs_precision_block))
        block_stats.append(stats)

    g_idx = torch.arange(cols, device=device, dtype=torch.int32) // int(group_size)

    if use_hadamard and signs is not None:
        Q = apply_right_hadamard_inverse(Q_work, signs=signs)
        q_int_out = None
        scale_out = None
        zero_out = None
        g_idx_out = None
    else:
        Q = Q_work
        q_int_out = q_int_work
        scale_out = scale_groups
        zero_out = zero_groups
        g_idx_out = g_idx

    stats: Dict[str, Any] = {
        "method": "mpgq",
        "solver": str(getattr(getattr(qcfg, "mpgq", None), "solver", "vamp") or "vamp"),
        "polish_mode": str(getattr(getattr(qcfg, "mpgq", None), "polish_mode", getattr(vamp_cfg, "polish_mode", "block_cd"))),
        "module": str(module_name),
        "bits": int(bits),
        "group_size": int(group_size),
        "rotation": "hadamard" if use_hadamard else "none",
        "rows": int(rows),
        "cols": int(cols),
        "rank": int(S.numel()),
        "eigen_floor": float(eigen_floor),
        "block_rows": int(block_rows),
        "obs_precision_initial": float(obs_precision),
        "obs_precision_final": float(obs_precision_block),
        "layer_error_sum": float(layer_err_sum),
        "rtn_error_sum": float(rtn_err_sum),
        "block_stats": block_stats,
        "quant_time_s": float(time.time() - start),
    }
    return Q, q_int_out, scale_out, zero_out, g_idx_out, float(time.time() - start), stats


@torch.inference_mode()
def postprocess_gptq_quantization(
    *,
    W: torch.Tensor,
    H: torch.Tensor,
    Q_init: torch.Tensor,
    scale_groups: torch.Tensor,
    zero_groups: torch.Tensor,
    g_idx: torch.Tensor,
    qcfg: Any,
    module_name: str,
    adapter_cfg: GPTQAdapterConfig = GPTQAdapterConfig(),
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    float,
    Dict[str, Any],
]:
    start = time.time()
    W = W.to(dtype=torch.float32)
    H = H.to(dtype=torch.float32)
    Q_init = Q_init.to(dtype=torch.float32)
    scale_groups = scale_groups.to(dtype=torch.float32)
    zero_groups = zero_groups.to(dtype=torch.float32)
    g_idx = g_idx.to(dtype=torch.int32, device=W.device)

    bits = int(getattr(qcfg, "bits"))
    maxq = int((2**bits) - 1)
    rows, cols = W.shape
    block_rows = _resolve_block_rows(
        rows=rows,
        cols=cols,
        maxq=maxq,
        requested=int(adapter_cfg.block_rows),
        device=W.device,
    )

    scale_full = _expand_group_params_from_gidx(scale_groups.to(device=W.device), g_idx)
    zero_full = _expand_group_params_from_gidx(zero_groups.to(device=W.device), g_idx)
    q_init = torch.round(Q_init / scale_full.clamp_min(1e-12) + zero_full).clamp_(0, maxq).to(dtype=torch.int32)

    H_work = float(adapter_cfg.hessian_scale) * H
    eigenvalues, V, eigen_floor = _vamp_hessian_eigen(
        H_work,
        jitter_base=float(adapter_cfg.jitter_base),
        floor_scale=float(adapter_cfg.eigen_floor_scale),
    )
    V = V.to(device=W.device, dtype=torch.float32).contiguous()
    eigenvalues = eigenvalues.to(device=W.device, dtype=torch.float32).contiguous()
    Vh = V.T.contiguous()
    S = torch.sqrt(eigenvalues).contiguous()
    Vs = V * S.unsqueeze(0)
    Y_u = W.matmul(Vs)

    mpgq_cfg = getattr(qcfg, "mpgq", None)
    polish_mode = str(getattr(mpgq_cfg, "polish_mode", "none") or "none").strip().lower()
    polish_sweeps = int(getattr(mpgq_cfg, "polish_sweeps", 0))
    vamp_cfg = _uniform_vamp_cfg_from_qcfg(qcfg)

    err_before = vamp_rankspace_error(Y_u=Y_u, x=Q_init, Vs=Vs)

    topk = None
    aux_stats: Dict[str, Any] = {}
    if polish_mode == "topk_block_cd":
        aux_debug: Dict[str, Any] = {}
        aux_cfg = UniformVampConfig(
            iters=vamp_cfg.iters,
            damping=vamp_cfg.damping,
            final_decision=vamp_cfg.final_decision,
            init_from="rtn",
            gamma_start=vamp_cfg.gamma_start,
            gamma_retry_factor=vamp_cfg.gamma_retry_factor,
            gamma_retry_steps=vamp_cfg.gamma_retry_steps,
            prior_tau=vamp_cfg.prior_tau,
            obs_precision=vamp_cfg.obs_precision,
            em_obs_precision=vamp_cfg.em_obs_precision,
            em_obs_precision_damping=vamp_cfg.em_obs_precision_damping,
            em_obs_precision_min=vamp_cfg.em_obs_precision_min,
            em_obs_precision_max=vamp_cfg.em_obs_precision_max,
            polish_sweeps=0,
            polish_mode="none",
            topk_k=vamp_cfg.topk_k,
        )
        _aux_q, _aux_x, _aux_err, _rtn_err, aux_stats = solve_uniform_rankspace_vamp(
            W=W,
            x_rtn=Q_init,
            scale_full=scale_full,
            zero_full=zero_full,
            maxq=maxq,
            V=V,
            Vh=Vh,
            S=S,
            Vs=Vs,
            cfg=aux_cfg,
            log_stats=False,
            debug_out=aux_debug,
        )
        topk = build_uniform_vamp_topk_candidates(
            r1=aux_debug["r1"].to(device=W.device, dtype=torch.float32),
            gamma1=aux_debug["gamma1"].to(device=W.device, dtype=torch.float32),
            W=W,
            scale_full=scale_full,
            zero_full=zero_full,
            maxq=maxq,
            prior_tau=float(vamp_cfg.prior_tau),
            k=int(vamp_cfg.topk_k),
        )

    q_polished, x_polished, err_after, polish_stats = polish_uniform_rankspace(
        x_init=Q_init,
        Y_u=Y_u,
        Vs=Vs,
        scale_full=scale_full,
        zero_full=zero_full,
        maxq=maxq,
        sweeps=polish_sweeps,
        polish_mode=polish_mode,
        topk_values=None if topk is None else topk["topk_values"],
        topk_q=None if topk is None else topk["topk_q"],
    )
    improve = err_after + 1e-12 < err_before
    if bool(improve.any().item()):
        Q_final = torch.where(improve.unsqueeze(-1), x_polished, Q_init)
        q_final = torch.where(improve.unsqueeze(-1), q_polished, q_init)
    else:
        Q_final = Q_init
        q_final = q_init

    stats: Dict[str, Any] = {
        "method": "mpgq",
        "solver": "gptq",
        "polish_mode": str(polish_mode),
        "module": str(module_name),
        "bits": int(bits),
        "group_size": int(getattr(qcfg, "group_size", cols)),
        "rows": int(rows),
        "cols": int(cols),
        "rank": int(S.numel()),
        "eigen_floor": float(eigen_floor),
        "block_rows": int(block_rows),
        "polish_sweeps": int(polish_sweeps),
        "topk_k": int(getattr(mpgq_cfg, "topk_k", 8) or 8),
        "rankspace_error_mean_before": float(err_before.mean().item()) if err_before.numel() else 0.0,
        "rankspace_error_mean_after": float(err_after.mean().item()) if err_after.numel() else 0.0,
        "polish_gain": float((err_before - err_after).mean().item()) if err_after.numel() else 0.0,
        "polish_updates": float(polish_stats.get("accepted_updates", 0.0)),
        "aux_vamp": aux_stats if topk is not None else {},
        "quant_time_s": float(time.time() - start),
    }
    return Q_final, q_final, scale_groups, zero_groups, g_idx, float(time.time() - start), stats
