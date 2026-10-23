from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch

from ..linear import stabilize_cholesky, vamp_rankspace_error
from ..polish.block_cd import E8P12Provider
from ..priors.uniform import dequant_from_q, nearest_atom_idx
from ..solver import polish_e8p12_rankspace, polish_uniform_rankspace


_EPS = 1e-12


@torch.inference_mode()
def run_gptq_fixed_uniform_hessian(
    *,
    W: torch.Tensor,
    H: torch.Tensor,
    scale_full: torch.Tensor,
    zero_full: torch.Tensor,
    maxq: int,
    damp_percent: float = 0.01,
) -> Tuple[torch.Tensor, torch.Tensor]:
    W_f32 = W.to(dtype=torch.float32).clone()
    H_f32 = H.to(device=W_f32.device, dtype=torch.float32)
    H_f32 = 0.5 * (H_f32 + H_f32.T)
    diag = torch.diagonal(H_f32)
    damp = float(damp_percent) * float(diag.mean().abs().item()) if diag.numel() else 0.0
    H_damped = H_f32.clone()
    H_damped.diagonal().add_(max(damp, _EPS))
    dead = torch.diagonal(H_damped) <= _EPS
    if bool(dead.any().item()):
        H_damped[dead, dead] = 1.0
        W_f32[:, dead] = 0.0
    chol = stabilize_cholesky(H_damped)
    Hinv = torch.cholesky_inverse(chol)
    Hinv = stabilize_cholesky(Hinv).T.contiguous()

    rows, cols = W_f32.shape
    q_out = torch.empty((rows, cols), dtype=torch.int32, device=W_f32.device)
    x_out = torch.empty_like(W_f32)
    work = W_f32.clone()
    for col_idx in range(cols):
        d = float(Hinv[col_idx, col_idx].item())
        q_col, x_col = _uniform_quantize(
            x=work[:, col_idx: col_idx + 1],
            scale=scale_full[:, col_idx: col_idx + 1],
            zero=zero_full[:, col_idx: col_idx + 1],
            maxq=maxq,
        )
        q_out[:, col_idx] = q_col.squeeze(1)
        x_out[:, col_idx] = x_col.squeeze(1)
        err = (work[:, col_idx] - x_out[:, col_idx]) / max(abs(d), _EPS)
        if col_idx < cols - 1:
            work[:, col_idx:] -= err.unsqueeze(1) * Hinv[col_idx, col_idx:].unsqueeze(0)
    return q_out, x_out


@torch.inference_mode()
def _uniform_quantize(
    *,
    x: torch.Tensor,
    scale: torch.Tensor,
    zero: torch.Tensor,
    maxq: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    scale_f32 = scale.to(device=x.device, dtype=torch.float32).clamp_min(_EPS)
    zero_f32 = zero.to(device=x.device, dtype=torch.float32)
    q = torch.round(x.to(dtype=torch.float32) / scale_f32 + zero_f32)
    q = torch.clamp(q, 0, int(maxq)).to(dtype=torch.int32)
    xq = dequant_from_q(q_int=q, scale_full=scale_f32, zero_full=zero_f32)
    return q, xq


@torch.inference_mode()
def run_cdquant_uniform_rankspace(
    *,
    x_rtn: torch.Tensor,
    Y_u: torch.Tensor,
    Vs: torch.Tensor,
    scale_full: torch.Tensor,
    zero_full: torch.Tensor,
    maxq: int,
    sweeps: int,
    compile_block_cd: bool = True,
    block_cd_backend: Optional[str] = None,
    gram_full: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
    q_int, x_out, err, stats = polish_uniform_rankspace(
        x_init=x_rtn,
        Y_u=Y_u,
        Vs=Vs,
        scale_full=scale_full,
        zero_full=zero_full,
        maxq=int(maxq),
        sweeps=max(int(sweeps), 0),
        polish_mode="block_cd",
        compile_block_cd=bool(compile_block_cd),
        block_cd_backend=block_cd_backend,
        gram_full=gram_full,
        order_mode="greedy_gain",
    )
    out_stats = dict(stats)
    out_stats["solver"] = "cdquant"
    out_stats["order_mode"] = "greedy_gain"
    return q_int, x_out, err, out_stats


@torch.inference_mode()
def run_quantease_uniform_hessian(
    *,
    W: torch.Tensor,
    H: torch.Tensor,
    Y_u: torch.Tensor,
    Vs: torch.Tensor,
    scale_full: torch.Tensor,
    zero_full: torch.Tensor,
    maxq: int,
    sweeps: int,
    x_init: Optional[torch.Tensor] = None,
    init_label: str = "",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
    W_f32 = W.to(dtype=torch.float32)
    H_f32 = H.to(device=W_f32.device, dtype=torch.float32)
    H_f32 = 0.5 * (H_f32 + H_f32.T)
    sweeps = max(int(sweeps), 0)
    if sweeps <= 0 or W_f32.numel() == 0:
        q_int, x_out = _uniform_quantize(x=W_f32, scale=scale_full, zero=zero_full, maxq=maxq)
        err = vamp_rankspace_error(Y_u=Y_u, x=x_out, Vs=Vs)
        return q_int, x_out, err, {"solver": "quantease", "accepted_updates": 0.0, "sweeps": 0.0}

    diag = torch.diagonal(H_f32).to(dtype=torch.float32)
    valid = torch.isfinite(diag) & (diag.abs() > _EPS)
    diag_safe = torch.where(valid, diag, torch.ones_like(diag))
    norm_h = H_f32 / diag_safe.view(1, -1)
    norm_h[:, ~valid] = 0.0
    P = W_f32.matmul(norm_h)
    norm_h.fill_diagonal_(0.0)
    norm_h_t = norm_h.T.contiguous()

    W_hat = W_f32.clone() if x_init is None else x_init.to(device=W_f32.device, dtype=torch.float32).clone()
    init_name = str(init_label or ("float" if x_init is None else "provided"))
    changed = 0
    for sweep_idx in range(sweeps):
        quantize_step = (sweep_idx % 3 != 1) or (sweep_idx == sweeps - 1) or (sweep_idx > 25)
        residual = P - W_hat.matmul(norm_h)
        for col_idx in range(int(W_hat.shape[1])):
            old_col = W_hat[:, col_idx].clone()
            if bool(valid[col_idx].item()):
                target = residual[:, col_idx]
                force_quantize = False
            else:
                target = old_col
                force_quantize = True
            if quantize_step or force_quantize:
                _, new_col = _uniform_quantize(
                    x=target.unsqueeze(1),
                    scale=scale_full[:, col_idx: col_idx + 1],
                    zero=zero_full[:, col_idx: col_idx + 1],
                    maxq=maxq,
                )
                new_col = new_col.squeeze(1)
            else:
                new_col = target.to(dtype=torch.float32)
            W_hat[:, col_idx] = new_col
            delta = old_col - new_col
            if col_idx < int(W_hat.shape[1]) - 1:
                residual[:, col_idx + 1:] += delta.unsqueeze(1) * norm_h[col_idx, col_idx + 1:].unsqueeze(0)
            changed += int((old_col != new_col).to(dtype=torch.int64).sum().item())

    q_int = nearest_atom_idx(x=W_hat, scale_full=scale_full, zero_full=zero_full, maxq=int(maxq)).to(dtype=torch.int32)
    x_out = dequant_from_q(q_int=q_int, scale_full=scale_full, zero_full=zero_full)
    err = vamp_rankspace_error(Y_u=Y_u, x=x_out, Vs=Vs)
    return q_int, x_out, err, {
        "solver": "quantease",
        "accepted_updates": float(changed),
        "sweeps": float(sweeps),
        "order_mode": "cyclic",
        "float_skip_schedule": "official_mod3",
        "init": init_name,
    }


@torch.inference_mode()
def run_cdquant_e8p12_rankspace(
    *,
    hatWr_init: torch.Tensor,
    Qidxs_init: torch.Tensor,
    Y_u: torch.Tensor,
    Vs: torch.Tensor,
    cb: Any,
    sweeps: int,
    polish_metric: str = "gram",
    metric_rows_per_chunk: int = 0,
    polish_max_blocks: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
    hat, idxs, err, stats = polish_e8p12_rankspace(
        x_init=hatWr_init,
        idxs_init=Qidxs_init,
        Y_u=Y_u,
        Vs=Vs,
        cb=cb,
        sweeps=max(int(sweeps), 0),
        polish_mode="block_cd",
        polish_metric=str(polish_metric or "gram"),
        metric_rows_per_chunk=int(metric_rows_per_chunk),
        polish_max_blocks=int(polish_max_blocks),
        order_mode="greedy_gain",
    )
    out_stats = dict(stats)
    out_stats["solver"] = "cdquant"
    out_stats["order_mode"] = "greedy_gain"
    return hat, idxs, err, out_stats


@torch.inference_mode()
def run_quantease_e8p12_hessian(
    *,
    Wr: torch.Tensor,
    Hr: torch.Tensor,
    cb: Any,
    Y_u: torch.Tensor,
    Vs: torch.Tensor,
    sweeps: int,
    metric_rows_per_chunk: int = 0,
    hatWr_init: Optional[torch.Tensor] = None,
    Qidxs_init: Optional[torch.Tensor] = None,
    init_label: str = "",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
    Wr_f32 = Wr.to(dtype=torch.float32)
    Hr_f32 = Hr.to(device=Wr_f32.device, dtype=torch.float32)
    Hr_f32 = 0.5 * (Hr_f32 + Hr_f32.T)
    rows, cols = Wr_f32.shape
    codesz = int(getattr(cb, "codesz", 8))
    if codesz <= 0 or cols % codesz != 0:
        raise ValueError(f"QuantEase/E8P12 expects cols divisible by code size; got cols={cols}, codesz={codesz}.")
    nblocks = cols // codesz
    provider = E8P12Provider(cb=cb, metric_rows_per_chunk=int(metric_rows_per_chunk))
    idx_dtype = getattr(cb, "idx_dtype", torch.int64)

    sweeps = max(int(sweeps), 0)
    hat = Wr_f32.clone() if hatWr_init is None else hatWr_init.to(device=Wr_f32.device, dtype=torch.float32).clone()
    init_name = str(init_label or ("float" if hatWr_init is None else "provided"))
    if Qidxs_init is None:
        idxs = torch.empty((rows, nblocks), dtype=idx_dtype, device=Wr_f32.device)
    else:
        idxs = Qidxs_init.to(device=Wr_f32.device, dtype=idx_dtype).clone()
    changed = 0
    if sweeps <= 0 or Wr_f32.numel() == 0:
        for block_idx in range(nblocks):
            start = block_idx * codesz
            end = start + codesz
            gram = Hr_f32[start:end, start:end].contiguous()
            chol = stabilize_cholesky(gram)
            vals, block_idxs = provider.propose(
                block_idx=block_idx,
                start=start,
                end=end,
                x_block=hat[:, start:end],
                target=hat[:, start:end],
                metric="gram",
                gram_chol=chol,
            )
            hat[:, start:end] = vals.to(dtype=torch.float32)
            idxs[:, block_idx] = block_idxs.to(dtype=idx_dtype)
        err = vamp_rankspace_error(Y_u=Y_u, x=hat, Vs=Vs)
        return hat, idxs, err, {"solver": "quantease", "accepted_updates": 0.0, "sweeps": 0.0}

    block_meta = []
    for block_idx in range(nblocks):
        start = block_idx * codesz
        end = start + codesz
        gram = Hr_f32[start:end, start:end].contiguous()
        chol = stabilize_cholesky(gram)
        gram_inv = torch.cholesky_inverse(chol).contiguous()
        block_meta.append((start, end, Hr_f32[:, start:end].contiguous(), chol, gram_inv))

    delta = Wr_f32 - hat
    for sweep_idx in range(sweeps):
        quantize_step = (sweep_idx % 3 != 1) or (sweep_idx == sweeps - 1) or (sweep_idx > 25)
        for block_idx, (start, end, H_cols, chol, gram_inv) in enumerate(block_meta):
            old_block = hat[:, start:end].clone()
            rhs = delta.matmul(H_cols)
            target = old_block + rhs.matmul(gram_inv)
            if quantize_step:
                vals, block_idxs = provider.propose(
                    block_idx=block_idx,
                    start=start,
                    end=end,
                    x_block=old_block,
                    target=target,
                    metric="gram",
                    gram_chol=chol,
                )
                vals = vals.to(dtype=torch.float32)
                idxs[:, block_idx] = block_idxs.to(dtype=idx_dtype)
            else:
                vals = target.to(dtype=torch.float32)
            hat[:, start:end] = vals
            delta[:, start:end] = Wr_f32[:, start:end] - vals
            changed += int((old_block != vals).any(dim=1).to(dtype=torch.int64).sum().item())

    err = vamp_rankspace_error(Y_u=Y_u, x=hat, Vs=Vs)
    return hat, idxs, err, {
        "solver": "quantease",
        "accepted_updates": float(changed),
        "sweeps": float(sweeps),
        "order_mode": "cyclic",
        "float_skip_schedule": "official_mod3",
        "init": init_name,
    }
