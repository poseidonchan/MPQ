from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from ..baselines.cd_solvers import (
    run_cdquant_uniform_rankspace,
    run_quantease_uniform_hessian,
)
from ..data import tensor_sha256
from ..rotation import (
    apply_right_block_hadamard,
    apply_right_block_hadamard_inverse,
    resolve_block_hadamard_plan,
    sample_rademacher_signs,
)
from ..linear import stabilize_cholesky, vamp_rankspace_error
from ..solver import (
    UniformVampConfig,
    build_uniform_vamp_topk_candidates,
    polish_uniform_rankspace,
    solve_uniform_rankspace_vamp,
)
from ..utils import stable_seed


@dataclass(frozen=True)
class GPTQAdapterConfig:
    hessian_scale: float = 0.5  # GPTQModel upstream stores H = 2/m X^T X; divide by 2 to match X^T X / m.
    block_rows: int = 128
    jitter_base: float = 1e-6
    eigen_floor_scale: float = 1e-6
    rotation_seed: int = 0


@torch.inference_mode()
def _resolve_internal_rotation(
    *,
    requested: str,
    cols: int,
    module_name: str,
    seed: int,
    device: torch.device,
) -> Dict[str, Any]:
    requested_norm = str(requested or "none").strip().lower()
    if requested_norm in {"", "none"}:
        return {
            "requested": "none",
            "effective": "none",
            "mode": "none",
            "signs": None,
            "block_dim": 1,
            "fallback_reason": "",
            "internal": False,
        }
    if requested_norm == "model_random":
        return {
            "requested": "model_random",
            "effective": "model_random",
            "mode": "none",
            "signs": None,
            "block_dim": 1,
            "fallback_reason": "",
            "internal": False,
        }
    if requested_norm == "model_hadamard":
        return {
            "requested": "model_hadamard",
            "effective": "model_hadamard",
            "mode": "none",
            "signs": None,
            "block_dim": 1,
            "fallback_reason": "",
            "internal": False,
        }
    if requested_norm != "internal_hadamard":
        raise ValueError(
            "mpq only supports rotation in {none, model_hadamard, model_random, internal_hadamard}; "
            f"got rotation={requested!r}."
        )

    mode, block_dim = resolve_block_hadamard_plan(int(cols))
    if mode == "none":
        return {
            "requested": "internal_hadamard",
            "effective": "none",
            "mode": "none",
            "signs": None,
            "block_dim": 1,
            "fallback_reason": f"unsupported_block_hadamard_width:{int(cols)}",
            "internal": False,
        }

    signs = sample_rademacher_signs(
        width=int(cols),
        key=f"{module_name}:full",
        seed=int(seed),
        device=device,
    )
    return {
        "requested": "internal_hadamard",
        "effective": "internal_block_hadamard",
        "mode": "block",
        "signs": signs,
        "block_dim": int(block_dim),
        "fallback_reason": "",
        "internal": True,
    }


def _paired_matrix_cache(
    *,
    W: torch.Tensor,
    H: torch.Tensor,
    qcfg: Any,
    module_name: str,
    basis: str,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    mpq_cfg = getattr(qcfg, "mpq", None)
    cache_dir_value = getattr(mpq_cfg, "paired_matrix_cache_dir", None)
    W_f32 = W.to(dtype=torch.float32).contiguous()
    H_f32 = H.to(dtype=torch.float32).contiguous()
    w_sha256 = tensor_sha256(W_f32)
    if not cache_dir_value:
        fingerprint = {
            "protocol": "paired_w_h_sha256_v1",
            "w_sha256": w_sha256,
            "h_sha256": tensor_sha256(H_f32),
            "w_shape": [int(dim) for dim in W_f32.shape],
            "h_shape": [int(dim) for dim in H_f32.shape],
            "basis": str(basis),
            "cache": "disabled",
        }
        return W_f32, H_f32, fingerprint

    cache_dir = Path(str(cache_dir_value))
    manifests_dir = cache_dir / "manifests"
    blobs_dir = cache_dir / "hessian_blobs"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    blobs_dir.mkdir(parents=True, exist_ok=True)
    module_digest = hashlib.sha256(str(module_name).encode("utf-8")).hexdigest()
    manifest_path = manifests_dir / f"{module_digest}.json"
    lock_path = manifests_dir / f"{module_digest}.lock"

    require_consumer = str(
        os.environ.get("MPQ_PAIRED_CACHE_REQUIRE_CONSUMER", "")
    ).strip().lower() in {"1", "true", "yes", "on"}
    if require_consumer:
        deadline = time.monotonic() + float(
            os.environ.get("MPQ_PAIRED_CACHE_WAIT_SECONDS", "7200")
        )
        while not manifest_path.exists():
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"timed out waiting for paired matrix producer: {manifest_path}"
                )
            time.sleep(0.25)

    with lock_path.open("w", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if str(manifest.get("module")) != str(module_name):
                raise RuntimeError(f"paired matrix cache module mismatch at {manifest_path}")
            if str(manifest.get("w_sha256")) != w_sha256:
                raise RuntimeError(
                    f"paired W mismatch for {module_name}: "
                    f"{w_sha256} != {manifest.get('w_sha256')}"
                )
            if list(manifest.get("w_shape", [])) != [int(dim) for dim in W_f32.shape]:
                raise RuntimeError(f"paired W shape mismatch for {module_name}")
            blob_path = blobs_dir / f"{manifest['h_sha256']}.pt"
            if not blob_path.exists():
                raise RuntimeError(f"missing paired Hessian blob: {blob_path}")
            cached_hessian = torch.load(blob_path, map_location="cpu", weights_only=True)
            if tensor_sha256(cached_hessian) != str(manifest["h_sha256"]):
                raise RuntimeError(f"paired Hessian blob hash mismatch: {blob_path}")
            if list(cached_hessian.shape) != list(H_f32.shape):
                raise RuntimeError(f"paired Hessian shape mismatch for {module_name}")
            H_f32 = cached_hessian.to(device=H_f32.device, dtype=torch.float32)
            cache_state = "consumer"
        elif require_consumer:
            raise RuntimeError(
                f"paired matrix manifest disappeared before consumer read: {manifest_path}"
            )
        else:
            h_sha256 = tensor_sha256(H_f32)
            blob_path = blobs_dir / f"{h_sha256}.pt"
            if not blob_path.exists():
                temporary_blob = blobs_dir / f".{h_sha256}.{os.getpid()}.tmp"
                torch.save(H_f32.detach().cpu(), temporary_blob)
                temporary_blob.replace(blob_path)
            manifest = {
                "protocol": "paired_w_h_sha256_v1",
                "module": str(module_name),
                "w_sha256": w_sha256,
                "h_sha256": h_sha256,
                "w_shape": [int(dim) for dim in W_f32.shape],
                "h_shape": [int(dim) for dim in H_f32.shape],
                "basis": str(basis),
            }
            temporary_manifest = manifest_path.with_suffix(".json.tmp")
            temporary_manifest.write_text(
                json.dumps(manifest, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            temporary_manifest.replace(manifest_path)
            cache_state = "producer"

    fingerprint = dict(manifest)
    fingerprint["cache"] = cache_state
    fingerprint["manifest_path"] = str(manifest_path)
    return W_f32, H_f32, fingerprint


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
def _random_uniform_init(
    *,
    scale_full: torch.Tensor,
    zero_full: torch.Tensor,
    maxq: int,
    seed_key: str,
    base_seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(stable_seed(seed_key, base_seed=int(base_seed)))
    q_cpu = torch.randint(
        0,
        int(maxq) + 1,
        tuple(scale_full.shape),
        generator=gen,
        dtype=torch.int64,
        device="cpu",
    )
    q_int = q_cpu.to(device=scale_full.device, dtype=torch.int32, non_blocking=False)
    x_init = scale_full * (q_int.to(dtype=torch.float32) - zero_full)
    return q_int, x_init


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
    H_f32 = H.to(dtype=torch.float32)
    H_f32 = 0.5 * (H_f32 + H_f32.T)
    # Sanitize: replace any NaN/Inf with 0 before eigh. cuSOLVER on H100/H200
    # can crash with CUSOLVER_STATUS_INVALID_VALUE if the input contains NaN
    # (observed on Llama-2-70B mlp.down_proj layer 0). The Hessian shouldn't
    # have NaN under normal conditions; this guard is purely defensive.
    if not torch.isfinite(H_f32).all():
        H_f32 = torch.nan_to_num(H_f32, nan=0.0, posinf=0.0, neginf=0.0)
    try:
        lam, V = torch.linalg.eigh(H_f32)  # ascending order
    except Exception:
        # Fallback: try CPU eigh (more numerically tolerant on some installs).
        lam_cpu, V_cpu = torch.linalg.eigh(H_f32.detach().cpu())
        lam = lam_cpu.to(device=H_f32.device)
        V = V_cpu.to(device=H_f32.device)
    lam = lam.to(dtype=torch.float32)
    V = V.to(dtype=torch.float32)
    max_lam = float(lam.max().item()) if lam.numel() else 0.0
    diag_mean = float(torch.diagonal(H_f32).mean().abs().item()) if H_f32.numel() else 0.0
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
    mpq_cfg = getattr(qcfg, "mpq", None)
    block_cd_backend = str(getattr(mpq_cfg, "block_cd_backend", "") or "").strip().lower()
    if not block_cd_backend:
        block_cd_backend = "compile" if bool(getattr(mpq_cfg, "compile_block_cd", False)) else "eager"
    return UniformVampConfig(
        iters=int(getattr(mpq_cfg, "iters", 10)),
        damping=float(getattr(mpq_cfg, "damping", 0.5)),
        final_decision=str(getattr(mpq_cfg, "final_decision", "map") or "map"),
        init_from="rtn",
        gamma_start=float(getattr(mpq_cfg, "gamma_start", 1.0)),
        gamma_retry_factor=float(getattr(mpq_cfg, "gamma_retry_factor", 4.0)),
        gamma_retry_steps=int(getattr(mpq_cfg, "gamma_retry_steps", 3)),
        prior_tau=float(getattr(mpq_cfg, "prior_tau", 0.2)),
        obs_precision=float(getattr(mpq_cfg, "obs_precision", 0.0)),
        em_obs_precision=bool(getattr(mpq_cfg, "em_obs_precision", True)),
        em_obs_precision_damping=float(getattr(mpq_cfg, "em_obs_precision_damping", 0.3)),
        em_obs_precision_min=float(getattr(mpq_cfg, "em_obs_precision_min", 1e-4)),
        em_obs_precision_max=float(getattr(mpq_cfg, "em_obs_precision_max", 1e8)),
        polish_sweeps=int(getattr(mpq_cfg, "polish_sweeps", 3)),
        polish_mode=str(getattr(mpq_cfg, "polish_mode", "block_cd") or "block_cd"),
        topk_k=int(getattr(mpq_cfg, "topk_k", 16) or 16),
        vamp_backend=str(getattr(mpq_cfg, "vamp_backend", "auto") or "auto"),
        compile_block_cd=bool(getattr(mpq_cfg, "compile_block_cd", False)),
        block_cd_backend=block_cd_backend,
    )


@torch.inference_mode()
def _polish_ojbkq_uniform_solution(
    *,
    W: torch.Tensor,
    H: torch.Tensor,
    Q_init: torch.Tensor,
    q_init: torch.Tensor,
    scale_groups: torch.Tensor,
    zero_groups: torch.Tensor,
    g_idx: torch.Tensor,
    qcfg: Any,
    adapter_cfg: GPTQAdapterConfig,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Apply matched scalar CD directly to an OJBKQ Babai/Klein solution."""

    mpq_cfg = getattr(qcfg, "mpq", None)
    polish_mode = str(
        getattr(mpq_cfg, "polish_mode", "none") or "none"
    ).strip().lower()
    polish_sweeps = int(getattr(mpq_cfg, "polish_sweeps", 0))
    if polish_mode not in {"none", "block_cd"}:
        raise ValueError(
            "OJBKQ scalar postprocessing supports polish_mode in "
            f"{{none, block_cd}}, got {polish_mode!r}."
        )

    bits = int(getattr(qcfg, "bits"))
    maxq = int((2**bits) - 1)
    W_f32 = W.to(dtype=torch.float32)
    H_f32 = H.to(device=W.device, dtype=torch.float32)
    Q_initial = Q_init.to(device=W.device, dtype=torch.float32)
    q_initial = q_init.to(device=W.device, dtype=torch.int32)
    g_idx_device = g_idx.to(device=W.device, dtype=torch.int32)
    scale_full = _expand_group_params_from_gidx(
        scale_groups.to(device=W.device, dtype=torch.float32),
        g_idx_device,
    )
    zero_full = _expand_group_params_from_gidx(
        zero_groups.to(device=W.device, dtype=torch.float32),
        g_idx_device,
    )

    eigenvalues, V, eigen_floor = _vamp_hessian_eigen(
        H_f32,
        jitter_base=float(adapter_cfg.jitter_base),
        floor_scale=float(adapter_cfg.eigen_floor_scale),
    )
    V = V.to(device=W.device, dtype=torch.float32).contiguous()
    eigenvalues = eigenvalues.to(
        device=W.device,
        dtype=torch.float32,
    ).contiguous()
    Vs = V * torch.sqrt(eigenvalues).unsqueeze(0)
    Y_u = W_f32.matmul(Vs)
    err_before = vamp_rankspace_error(Y_u=Y_u, x=Q_initial, Vs=Vs)
    delta_before = W_f32 - Q_initial
    raw_before = torch.einsum(
        "bi,ij,bj->b",
        delta_before,
        H_f32,
        delta_before,
    )

    vamp_cfg = _uniform_vamp_cfg_from_qcfg(qcfg)
    q_polished, x_polished, err_after, polish_stats = (
        polish_uniform_rankspace(
            x_init=Q_initial,
            Y_u=Y_u,
            Vs=Vs,
            scale_full=scale_full,
            zero_full=zero_full,
            maxq=maxq,
            sweeps=polish_sweeps,
            polish_mode=polish_mode,
            compile_block_cd=bool(
                getattr(vamp_cfg, "compile_block_cd", False)
            ),
            block_cd_backend=str(
                getattr(vamp_cfg, "block_cd_backend", "") or ""
            ),
            gram_full=H_f32,
            raw_target=W_f32,
            raw_gram=H_f32,
        )
    )
    improve = err_after + 1e-12 < err_before
    if bool(improve.any().item()):
        Q_final = torch.where(improve.unsqueeze(-1), x_polished, Q_initial)
        q_final = torch.where(improve.unsqueeze(-1), q_polished, q_initial)
    else:
        Q_final = Q_initial
        q_final = q_initial

    err_final = vamp_rankspace_error(Y_u=Y_u, x=Q_final, Vs=Vs)
    before_mean = float(err_before.mean().item()) if err_before.numel() else 0.0
    final_mean = float(err_final.mean().item()) if err_final.numel() else 0.0
    fail_closed = (
        not bool(torch.isfinite(err_final).all().item())
        or final_mean > before_mean + 1e-9
    )
    if fail_closed:
        Q_final = Q_initial
        q_final = q_initial
        err_final = err_before
        final_mean = before_mean

    delta_final = W_f32 - Q_final
    raw_final = torch.einsum(
        "bi,ij,bj->b",
        delta_final,
        H_f32,
        delta_final,
    )
    trace = polish_stats.get("cd_trace") or polish_stats.get("sweep_trace") or []
    if not isinstance(trace, list):
        trace = []
    if not trace:
        trace = [
            {
                "budget": 0,
                "stabilized_objective_mean": before_mean,
                "accepted_updates_sweep": 0,
                "accepted_updates_cumulative": 0,
                "assignment_changes_from_cd0": 0,
                "assignments_total": int(q_initial.numel()),
            }
        ]
    else:
        trace = [dict(item) for item in trace]
        for item in trace:
            if int(item.get("assignments_total", 0)) == 0:
                item["assignments_total"] = int(q_initial.numel())
        if int(trace[-1].get("budget", -1)) == int(polish_sweeps):
            trace[-1]["assignment_changes_from_cd0"] = int(
                (q_final != q_initial).to(dtype=torch.int64).sum().item()
            )
    stats = {
        "post_ojbkq_input": "babai_klein_selected_solution",
        "post_ojbkq_polish_mode": polish_mode,
        "post_ojbkq_polish_sweeps": int(polish_sweeps),
        "post_ojbkq_polish_updates": float(
            polish_stats.get("accepted_updates", 0.0)
        ),
        "post_ojbkq_polish_trace": trace,
        "post_ojbkq_block_cd_backend": str(
            polish_stats.get(
                "block_cd_backend",
                getattr(vamp_cfg, "block_cd_backend", "eager"),
            )
        ),
        "post_ojbkq_stabilized_objective_mean_before": before_mean,
        "post_ojbkq_stabilized_objective_mean_after": float(
            err_after.mean().item()
        )
        if err_after.numel()
        else 0.0,
        "post_ojbkq_stabilized_objective_mean_final": final_mean,
        "post_ojbkq_raw_objective_mean_before": float(raw_before.mean().item())
        if raw_before.numel()
        else 0.0,
        "post_ojbkq_raw_objective_mean_final": float(raw_final.mean().item())
        if raw_final.numel()
        else 0.0,
        "post_ojbkq_assignment_changes": int(
            (q_final != q_initial).to(dtype=torch.int64).sum().item()
        ),
        "post_ojbkq_assignments_total": int(q_initial.numel()),
        "post_ojbkq_fail_closed": bool(fail_closed),
        "post_ojbkq_eigen_floor": float(eigen_floor),
    }
    return Q_final, q_final, stats


@torch.inference_mode()
def _adapter_cfg_from_qcfg(qcfg: Any) -> GPTQAdapterConfig:
    mpq_cfg = getattr(qcfg, "mpq", None)
    return GPTQAdapterConfig(
        hessian_scale=float(getattr(mpq_cfg, "hessian_scale", 0.5)),
        block_rows=int(getattr(mpq_cfg, "block_rows", 32)),
        jitter_base=float(getattr(mpq_cfg, "eigen_jitter", 1e-6)),
        eigen_floor_scale=float(getattr(mpq_cfg, "eigen_floor_scale", 1e-6)),
        rotation_seed=int(getattr(mpq_cfg, "rotation_seed", 0)),
    )


def _extract_layer_index(module_name: str) -> Optional[int]:
    match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", str(module_name))
    if match is None:
        return None
    try:
        return int(match.group(1))
    except (TypeError, ValueError):
        return None


@torch.inference_mode()
def _summarize_uniform_mechanism(
    *,
    module_name: str,
    bits: int,
    group_size: int,
    rows: int,
    cols: int,
    gamma_sum: float,
    gamma_count: int,
    gamma_min: float,
    gamma_max: float,
    changed_count: int,
    changed_total: int,
) -> Dict[str, Any]:
    total_count = int(changed_total) if int(changed_total) > 0 else int(rows) * int(cols)
    changed = int(changed_count)
    gamma_mean = float(gamma_sum / max(int(gamma_count), 1))
    return {
        "regime": "3bit_scalar",
        "cleanup_unit": "scalar_element",
        "module": str(module_name),
        "layer_index": _extract_layer_index(str(module_name)),
        "bits": int(bits),
        "group_size": int(group_size),
        "rows": int(rows),
        "cols": int(cols),
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
def prepare_gptq_internal_hadamard(
    *,
    W: torch.Tensor,
    H: torch.Tensor,
    qcfg: Any,
    module_name: str,
    adapter_cfg: GPTQAdapterConfig = GPTQAdapterConfig(),
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    rotation_plan = _resolve_internal_rotation(
        requested=str(getattr(getattr(qcfg, "mpq", None), "rotation_mode", getattr(qcfg, "rotation", None) or "none")),
        cols=int(W.shape[1]),
        module_name=str(module_name),
        seed=int(adapter_cfg.rotation_seed),
        device=W.device,
    )
    if not bool(rotation_plan["internal"]):
        return W.to(dtype=torch.float32), H.to(dtype=torch.float32), rotation_plan

    had_dim = int(rotation_plan["block_dim"])
    signs = rotation_plan["signs"]
    W_work = apply_right_block_hadamard(W.to(dtype=torch.float32), signs=signs, had_dim=had_dim)
    H_work = apply_right_block_hadamard(H.to(dtype=torch.float32).T.contiguous(), signs=signs, had_dim=had_dim).T.contiguous()
    H_work = apply_right_block_hadamard(H_work, signs=signs, had_dim=had_dim).contiguous()
    # Native GPTQ consumes this Hessian before the later MPQ/CD postprocess.
    # Apply the paired cache here as well so initialization and cleanup are
    # optimized against the same producer matrix. Cache entries use the
    # adapter's normalized Hessian convention; restore GPTQModel's scale on
    # return (the common positive scale does not change the target objective).
    hessian_scale = float(adapter_cfg.hessian_scale)
    if hessian_scale <= 0.0:
        raise ValueError(f"hessian_scale must be positive, got {hessian_scale}")
    W_work, H_metric, _input_fingerprint = _paired_matrix_cache(
        W=W_work,
        H=hessian_scale * H_work,
        qcfg=qcfg,
        module_name=str(module_name),
        basis=str(rotation_plan["effective"]),
    )
    H_work = H_metric / hessian_scale
    return W_work, H_work, rotation_plan


@torch.inference_mode()
def finalize_gptq_internal_hadamard(
    *,
    Q_work: torch.Tensor,
    rotation_plan: Dict[str, Any],
) -> torch.Tensor:
    if not bool(rotation_plan.get("internal", False)):
        return Q_work.to(dtype=torch.float32).contiguous()
    return apply_right_block_hadamard_inverse(
        Q_work.to(dtype=torch.float32),
        signs=rotation_plan["signs"],
        had_dim=int(rotation_plan["block_dim"]),
    ).contiguous()


@torch.inference_mode()
def run_mpq_quantization(
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
    """Run MPQ on a GPTQModel layer given weights and accumulated Hessian.

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

    rotation_plan = _resolve_internal_rotation(
        requested=str(getattr(getattr(qcfg, "mpq", None), "rotation_mode", getattr(qcfg, "rotation", None) or "none")),
        cols=int(cols),
        module_name=str(module_name),
        seed=int(adapter_cfg.rotation_seed),
        device=device,
    )
    use_internal_rotation = bool(rotation_plan["internal"])
    use_activation_order = bool(getattr(qcfg, "desc_act", False))

    H_work = float(adapter_cfg.hessian_scale) * H
    solver_name = str(
        getattr(
            getattr(qcfg, "mpq", None),
            "dispatch_solver",
            getattr(getattr(qcfg, "mpq", None), "solver", "vamp"),
        )
        or "vamp"
    ).strip().lower()

    # Preserve the paper/Table-7 VAMP path: decompose the Hessian in the
    # unrotated basis, then rotate its eigenvectors with the weights. Although
    # decomposing R^T H R is algebraically equivalent, eigenspace choices in
    # finite precision can change the discrete VAMP basin.
    paper_eigen = None
    if solver_name == "vamp" and not use_activation_order:
        paper_eigen = _vamp_hessian_eigen(
            H_work,
            jitter_base=float(adapter_cfg.jitter_base),
            floor_scale=float(adapter_cfg.eigen_floor_scale),
        )
    signs = rotation_plan["signs"]
    H_metric = H_work
    if rotation_plan["mode"] == "block" and signs is not None:
        had_dim = int(rotation_plan["block_dim"])
        W_work = apply_right_block_hadamard(W, signs=signs, had_dim=had_dim)
        H_metric = apply_right_block_hadamard(H_work.T.contiguous(), signs=signs, had_dim=had_dim).T.contiguous()
        H_metric = apply_right_block_hadamard(H_metric, signs=signs, had_dim=had_dim).contiguous()
    else:
        W_work = W
    activation_perm = None
    activation_invperm = None
    if use_activation_order:
        activation_perm = torch.argsort(torch.diagonal(H_metric), descending=True)
        activation_invperm = torch.argsort(activation_perm)
        W_work = W_work.index_select(1, activation_perm)
        H_metric = H_metric.index_select(0, activation_perm).index_select(1, activation_perm)

    # Match GPTQModel's native path before dispatching solver swaps. Otherwise a
    # nominally greedy OJBKQ path does not reproduce the GPTQ candidate it is
    # supposed to include.
    dead = torch.diagonal(H_metric) == 0
    has_dead = bool(dead.any().item())
    if has_dead:
        H_metric = H_metric.clone()
        W_work = W_work.clone()
        H_metric[dead, dead] = 1
        W_work[:, dead] = 0
    W_work, H_metric, input_fingerprint = _paired_matrix_cache(
        W=W_work,
        H=H_metric,
        qcfg=qcfg,
        module_name=str(module_name),
        basis=str(rotation_plan["effective"]),
    )

    if solver_name == "ojbkq":
        from ..baselines.ojbkq import run_ojbkq_gptq_quantization

        hessian_for_inverse = H_metric.clone()
        hessian_diag = torch.diagonal(hessian_for_inverse)
        damp_percent = float(getattr(qcfg, "damp_percent", 0.05))
        hessian_diag.add_(damp_percent * hessian_diag.mean())
        hessian_chol = stabilize_cholesky(hessian_for_inverse)
        hessian_inverse_factor = torch.linalg.cholesky(
            torch.cholesky_inverse(hessian_chol),
            upper=True,
        )
        ojbkq_config = getattr(qcfg, "ojbkq", None)
        if ojbkq_config is None:
            raise RuntimeError("OJBKQ dispatch requires qcfg.ojbkq")
        (
            q_work,
            q_int_work,
            scale_groups,
            zero_groups,
            g_idx,
            avg_loss,
            ojbkq_stats,
        ) = run_ojbkq_gptq_quantization(
            W=W_work,
            Hinv=hessian_inverse_factor,
            quantizer=quantizer,
            qcfg=qcfg,
            config=ojbkq_config,
            nsamples=1,
            module_name=str(module_name),
        )
        mpq_cfg = getattr(qcfg, "mpq", None)
        requested_polish = str(
            getattr(mpq_cfg, "polish_mode", "none") or "none"
        ).strip().lower()
        post_ojbkq_stats: Dict[str, Any] = {
            "post_ojbkq_input": "babai_klein_selected_solution",
            "post_ojbkq_polish_mode": requested_polish,
            "post_ojbkq_polish_sweeps": int(
                getattr(mpq_cfg, "polish_sweeps", 0)
            ),
            "post_ojbkq_polish_updates": 0.0,
            "post_ojbkq_polish_trace": [],
            "post_ojbkq_assignment_changes": 0,
            "post_ojbkq_assignments_total": int(q_int_work.numel()),
        }
        if requested_polish == "block_cd":
            q_work, q_int_work, post_ojbkq_stats = (
                _polish_ojbkq_uniform_solution(
                    W=W_work,
                    H=H_metric,
                    Q_init=q_work,
                    q_init=q_int_work,
                    scale_groups=scale_groups,
                    zero_groups=zero_groups,
                    g_idx=g_idx,
                    qcfg=qcfg,
                    adapter_cfg=adapter_cfg,
                )
            )
        if activation_invperm is not None:
            q_work = q_work.index_select(1, activation_invperm)
            q_int_work = q_int_work.index_select(1, activation_invperm)
            g_idx = g_idx.index_select(0, activation_invperm)
        if rotation_plan["mode"] == "block" and signs is not None:
            q_out = apply_right_block_hadamard_inverse(
                q_work,
                signs=signs,
                had_dim=int(rotation_plan["block_dim"]),
            )
            q_int_out = None
            scale_out = None
            zero_out = None
            g_idx_out = None
        else:
            q_out = q_work
            q_int_out = q_int_work
            scale_out = scale_groups
            zero_out = zero_groups
            g_idx_out = g_idx
        stats = dict(ojbkq_stats)
        stats.update(
            {
                "method": "ojbkq",
                "bits": int(bits),
                "group_size": int(group_size),
                "rows": int(rows),
                "cols": int(cols),
                "rotation_requested": str(rotation_plan["requested"]),
                "rotation_effective": str(rotation_plan["effective"]),
                "rotation_internal": bool(rotation_plan["internal"]),
                "desc_act": bool(use_activation_order),
                "input_fingerprint": input_fingerprint,
                "avg_loss": float(avg_loss),
                "quant_time_s": float(time.time() - start),
                **post_ojbkq_stats,
            }
        )
        ojbkq_bucket = getattr(ojbkq_config, "_layer_stats", None)
        if not isinstance(ojbkq_bucket, list):
            ojbkq_bucket = []
            setattr(ojbkq_config, "_layer_stats", ojbkq_bucket)
        ojbkq_bucket.append(stats)
        return (
            q_out,
            q_int_out,
            scale_out,
            zero_out,
            g_idx_out,
            float(time.time() - start),
            stats,
        )

    use_paper_eigen = (
        paper_eigen is not None
        and not has_dead
        and str(input_fingerprint.get("cache", "disabled")) != "consumer"
    )
    if use_paper_eigen:
        eigenvalues, V, eigen_floor = paper_eigen
        if rotation_plan["mode"] == "block" and signs is not None:
            V = apply_right_block_hadamard(
                V.T.contiguous(),
                signs=signs,
                had_dim=int(rotation_plan["block_dim"]),
            ).T.contiguous()
    else:
        # Baselines and paired-cache consumers decompose the exact Hessian used
        # for their reported objective in the solver basis.
        eigenvalues, V, eigen_floor = _vamp_hessian_eigen(
            H_metric,
            jitter_base=float(adapter_cfg.jitter_base),
            floor_scale=float(adapter_cfg.eigen_floor_scale),
        )
    V = V.to(device=device, dtype=torch.float32).contiguous()
    eigenvalues = eigenvalues.to(device=device, dtype=torch.float32).contiguous()
    Vh = V.T.contiguous()
    S = torch.sqrt(eigenvalues).contiguous()
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
    # Keep the integer assignments even for the internal-Hadamard path.  The
    # dense weight used while GPTQModel propagates calibration activations is
    # inverse-rotated below, while these assignments are what the native GPTQ
    # artifact packs in the solver basis.
    q_int_work = torch.empty((rows, cols), dtype=torch.uint8, device=device)
    layer_err_sum = 0.0
    raw_layer_err_sum = 0.0
    rankspace_raw_abs_diff_sum = 0.0
    rankspace_raw_abs_diff_max = 0.0
    rtn_err_sum = 0.0
    obs_precision_block = float(obs_precision)
    gamma_sum = 0.0
    gamma_count = 0
    gamma_min = float("inf")
    gamma_max = float("-inf")
    changed_count = 0
    changed_total = 0

    block_stats: List[Dict[str, Any]] = []
    for row_start in range(0, rows, block_rows):
        row_end = min(row_start + block_rows, rows)
        rb = slice(row_start, row_end)
        scale_full = _expand_group_params(scale_groups[rb], widths)
        zero_full = _expand_group_params(zero_groups[rb], widths)
        q_rtn = torch.clamp(torch.round(W_work[rb] / scale_full) + zero_full, 0, maxq).to(dtype=torch.float32)
        x_rtn = scale_full * (q_rtn - zero_full)
        if solver_name == "vamp":
            debug_block: Dict[str, Any] = {"summary_only": True}
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
                gram_full=H_metric,
                obs_precision_init=obs_precision_block,
                log_stats=False,
                debug_out=debug_block,
            )
        elif solver_name in {"cdquant", "quantease"}:
            Y_u_block = W_work[rb].matmul(Vs)
            rtn_err = vamp_rankspace_error(Y_u=Y_u_block, x=x_rtn, Vs=Vs)
            if solver_name == "cdquant":
                q_int, x_final, final_err, stats = run_cdquant_uniform_rankspace(
                    x_rtn=x_rtn,
                    Y_u=Y_u_block,
                    Vs=Vs,
                    scale_full=scale_full,
                    zero_full=zero_full,
                    maxq=maxq,
                    sweeps=max(int(getattr(vamp_cfg, "polish_sweeps", 3)), 1),
                    compile_block_cd=bool(getattr(vamp_cfg, "compile_block_cd", False)),
                    block_cd_backend=str(getattr(vamp_cfg, "block_cd_backend", "") or ""),
                    gram_full=H_metric,
                )
            else:
                init_mode = str(getattr(getattr(qcfg, "mpq", None), "init", "float") or "float").strip().lower()
                x_init = None
                init_label = "float"
                if init_mode == "random":
                    _q_init, x_init = _random_uniform_init(
                        scale_full=scale_full,
                        zero_full=zero_full,
                        maxq=maxq,
                        seed_key=f"{module_name}:quantease:{row_start}:{row_end}",
                        base_seed=int(getattr(getattr(qcfg, "mpq", None), "rotation_seed", 0)),
                    )
                    init_label = "random_uniform"
                elif init_mode == "rtn":
                    x_init = x_rtn
                    init_label = "rtn"
                elif init_mode in {"float", "paper", "unquantized"}:
                    x_init = None
                    init_label = "float"
                elif init_mode == "ldlq":
                    raise ValueError("QuantEase/GPTQModel does not support mpq_init='ldlq'; use 'float', 'rtn', or 'random'.")
                else:
                    raise ValueError(f"unknown QuantEase mpq_init={init_mode!r} (expected 'float', 'rtn', or 'random').")
                q_int, x_final, final_err, stats = run_quantease_uniform_hessian(
                    W=W_work[rb],
                    H=H_metric,
                    Y_u=Y_u_block,
                    Vs=Vs,
                    scale_full=scale_full,
                    zero_full=zero_full,
                    maxq=maxq,
                    sweeps=max(int(getattr(vamp_cfg, "iters", 10)), 1),
                    x_init=x_init,
                    init_label=init_label,
                )
            debug_block = {
                "changed_count": int((q_int.to(dtype=torch.int32) != q_rtn.to(dtype=torch.int32)).to(dtype=torch.int64).sum().item()),
                "changed_total": int(q_int.numel()),
            }
        else:
            raise ValueError(f"unsupported GPTQModel MPQ solver={solver_name!r}.")
        Q_work[rb] = x_final
        q_int_work[rb] = q_int.to(dtype=torch.uint8)
        layer_err_sum += float(final_err.sum().item())
        # Audit the final accepted dense weights against the unfloored Hessian.
        delta_final = W_work[rb] - x_final
        raw_final_err = torch.einsum(
            "bi,ij,bj->b",
            delta_final,
            H_metric,
            delta_final,
        )
        raw_layer_err_sum += float(raw_final_err.sum().item())
        rankspace_raw_abs_diff = (final_err - raw_final_err).abs()
        rankspace_raw_abs_diff_sum += float(rankspace_raw_abs_diff.sum().item())
        if rankspace_raw_abs_diff.numel():
            rankspace_raw_abs_diff_max = max(
                rankspace_raw_abs_diff_max,
                float(rankspace_raw_abs_diff.max().item()),
            )
        rtn_err_sum += float(rtn_err.sum().item())
        if solver_name == "vamp" and bool(getattr(vamp_cfg, "em_obs_precision", True)):
            obs_precision_block = float(stats.get("obs_precision_final", obs_precision_block))
        if "gamma1_sum" in debug_block:
            gamma_sum += float(debug_block.get("gamma1_sum", 0.0) or 0.0)
            gamma_count += int(debug_block.get("gamma1_count", 0) or 0)
            gamma_min = min(gamma_min, float(debug_block.get("gamma1_min", float("inf"))))
            gamma_max = max(gamma_max, float(debug_block.get("gamma1_max", float("-inf"))))
        else:
            gamma_block = debug_block.get("gamma1")
            if isinstance(gamma_block, torch.Tensor) and gamma_block.numel() > 0:
                gamma_block = gamma_block.to(dtype=torch.float64)
                gamma_sum += float(gamma_block.sum().item())
                gamma_count += int(gamma_block.numel())
                gamma_min = min(gamma_min, float(gamma_block.min().item()))
                gamma_max = max(gamma_max, float(gamma_block.max().item()))
        if "changed_count" in debug_block:
            changed_count += int(debug_block.get("changed_count", 0) or 0)
            changed_total += int(debug_block.get("changed_total", 0) or 0)
        else:
            changed_mask = debug_block.get("changed_mask")
            if isinstance(changed_mask, torch.Tensor):
                changed_count += int(changed_mask.to(dtype=torch.int64).sum().item())
                changed_total += int(changed_mask.numel())
        block_stats.append(stats)

    g_idx = torch.arange(cols, device=device, dtype=torch.int32) // int(group_size)
    if activation_invperm is not None:
        Q_work = Q_work.index_select(1, activation_invperm)
        q_int_work = q_int_work.index_select(1, activation_invperm)
        g_idx = g_idx.index_select(0, activation_invperm)

    if rotation_plan["mode"] == "block" and signs is not None:
        Q = apply_right_block_hadamard_inverse(Q_work, signs=signs, had_dim=int(rotation_plan["block_dim"]))
        q_int_out = q_int_work
        scale_out = scale_groups
        zero_out = zero_groups
        g_idx_out = g_idx
    else:
        Q = Q_work
        q_int_out = q_int_work
        scale_out = scale_groups
        zero_out = zero_groups
        g_idx_out = g_idx

    mpq_cfg = getattr(qcfg, "mpq", None)
    if mpq_cfg is not None:
        setattr(mpq_cfg, "rotation_requested", str(rotation_plan["requested"]))
        setattr(mpq_cfg, "rotation_effective", str(rotation_plan["effective"]))
        setattr(mpq_cfg, "rotation_internal", bool(rotation_plan["internal"]))
        setattr(mpq_cfg, "rotation_mode_effective", "internal_rotation" if bool(rotation_plan["internal"]) else ("model_rotation" if str(rotation_plan["effective"]).startswith("model_") else "none"))
        setattr(mpq_cfg, "rotation_fallback_reason", str(rotation_plan["fallback_reason"]))
        setattr(mpq_cfg, "rotation_block_dim", int(rotation_plan["block_dim"]))

    stats: Dict[str, Any] = {
        "method": "mpq",
        "solver": str(solver_name),
        "polish_mode": str(getattr(getattr(qcfg, "mpq", None), "polish_mode", getattr(vamp_cfg, "polish_mode", "block_cd"))),
        "module": str(module_name),
        "bits": int(bits),
        "group_size": int(group_size),
        "rotation_requested": str(rotation_plan["requested"]),
        "rotation_effective": str(rotation_plan["effective"]),
        "rotation_mode": "internal_rotation" if bool(rotation_plan["internal"]) else ("model_rotation" if str(rotation_plan["effective"]).startswith("model_") else "none"),
        "rotation_internal": bool(rotation_plan["internal"]),
        "rotation_block_dim": int(rotation_plan["block_dim"]),
        "rotation_fallback_reason": str(rotation_plan["fallback_reason"]),
        "desc_act": bool(use_activation_order),
        "rows": int(rows),
        "cols": int(cols),
        "rank": int(S.numel()),
        "eigen_floor": float(eigen_floor),
        "block_rows": int(block_rows),
        "block_cd_backend": str(getattr(vamp_cfg, "block_cd_backend", "eager")),
        "obs_precision_initial": float(obs_precision),
        "obs_precision_final": float(obs_precision_block),
        "layer_error_sum": float(layer_err_sum),
        "stabilized_objective_mean_final": float(layer_err_sum / max(rows, 1)),
        "raw_objective_mean_final": float(raw_layer_err_sum / max(rows, 1)),
        "rankspace_raw_abs_diff_mean": float(rankspace_raw_abs_diff_sum / max(rows, 1)),
        "rankspace_raw_abs_diff_max": float(rankspace_raw_abs_diff_max),
        "rtn_error_sum": float(rtn_err_sum),
        "block_stats": block_stats,
        "quant_time_s": float(time.time() - start),
        "input_fingerprint": input_fingerprint,
        "mechanism": _summarize_uniform_mechanism(
            module_name=str(module_name),
            bits=int(bits),
            group_size=int(group_size),
            rows=int(rows),
            cols=int(cols),
            gamma_sum=float(gamma_sum),
            gamma_count=int(gamma_count),
            gamma_min=float(gamma_min),
            gamma_max=float(gamma_max),
            changed_count=int(changed_count),
            changed_total=int(changed_total),
        ),
    }
    return Q, q_int_out, scale_out, zero_out, g_idx_out, float(time.time() - start), stats


def install_internal_hadamard_input_hooks(
    model: torch.nn.Module,
    *,
    rotation_seed: int = 0,
) -> int:
    """Install per-linear input rotations for packed internal-Hadamard weights.

    The packed tensors stay in the solver basis.  Applying the same signed
    block-Hadamard transform to each module input therefore reproduces the
    dense inverse-rotated weight used during sequential quantization.
    """

    installed = 0
    for module_name, module in model.named_modules():
        if not module_name or getattr(module, "_mpq_internal_hadamard_hook", None) is not None:
            continue
        if not all(hasattr(module, attr) for attr in ("qweight", "qzeros", "scales")):
            continue

        width = int(
            getattr(module, "in_features", 0)
            or getattr(module, "infeatures", 0)
            or 0
        )
        if width <= 0:
            qweight = getattr(module, "qweight", None)
            bits = int(getattr(module, "bits", 0) or 0)
            pack_dtype = getattr(qweight, "dtype", torch.int32)
            pack_bits = (
                torch.iinfo(pack_dtype).bits
                if isinstance(pack_dtype, torch.dtype) and not pack_dtype.is_floating_point
                else 32
            )
            if isinstance(qweight, torch.Tensor) and bits > 0:
                width = int(qweight.shape[0]) * int(pack_bits // bits)
        if width <= 0:
            raise RuntimeError(f"Cannot determine MPQ input width for quantized module {module_name!r}")

        mode, block_dim = resolve_block_hadamard_plan(width)
        if mode == "none":
            continue

        signs_by_device: Dict[torch.device, torch.Tensor] = {}

        def _rotate_input(
            _module,
            args,
            *,
            _name=module_name,
            _width=width,
            _block_dim=block_dim,
            _cache=signs_by_device,
        ):
            if not args or not isinstance(args[0], torch.Tensor):
                return args
            value = args[0]
            if int(value.shape[-1]) != int(_width):
                raise RuntimeError(
                    f"MPQ input width mismatch for {_name!r}: expected {_width}, got {int(value.shape[-1])}"
                )
            signs = _cache.get(value.device)
            if signs is None:
                signs = sample_rademacher_signs(
                    width=int(_width),
                    key=f"{_name}:full",
                    seed=int(rotation_seed),
                    device=value.device,
                    dtype=torch.float32,
                )
                _cache[value.device] = signs
            rotated = apply_right_block_hadamard(
                value,
                signs=signs,
                had_dim=int(_block_dim),
            ).to(dtype=value.dtype)
            return (rotated, *args[1:])

        handle = module.register_forward_pre_hook(_rotate_input)
        setattr(module, "_mpq_internal_hadamard_hook", handle)
        installed += 1

    return installed


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

    mpq_cfg = getattr(qcfg, "mpq", None)
    H_work = float(adapter_cfg.hessian_scale) * H
    W, H_work, input_fingerprint = _paired_matrix_cache(
        W=W,
        H=H_work,
        qcfg=qcfg,
        module_name=str(module_name),
        basis=str(getattr(mpq_cfg, "rotation_effective", "internal_block_hadamard")),
    )
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

    polish_mode = str(getattr(mpq_cfg, "polish_mode", "none") or "none").strip().lower()
    polish_sweeps = int(getattr(mpq_cfg, "polish_sweeps", 0))
    vamp_cfg = _uniform_vamp_cfg_from_qcfg(qcfg)

    err_before = vamp_rankspace_error(Y_u=Y_u, x=Q_init, Vs=Vs)
    delta_before = W - Q_init
    raw_before = torch.einsum(
        "bi,ij,bj->b",
        delta_before,
        H_work,
        delta_before,
    )

    topk = None
    aux_stats: Dict[str, Any] = {}
    if polish_mode == "topk_block_cd":
        aux_debug: Dict[str, Any] = {"capture_r1": True}
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
            block_cd_backend=vamp_cfg.block_cd_backend,
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
            gram_full=H_work,
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
        compile_block_cd=bool(getattr(vamp_cfg, "compile_block_cd", False)),
        block_cd_backend=str(getattr(vamp_cfg, "block_cd_backend", "") or ""),
        gram_full=H_work,
        raw_target=W,
        raw_gram=H_work,
    )
    improve = err_after + 1e-12 < err_before
    if bool(improve.any().item()):
        Q_final = torch.where(improve.unsqueeze(-1), x_polished, Q_init)
        q_final = torch.where(improve.unsqueeze(-1), q_polished, q_init)
    else:
        Q_final = Q_init
        q_final = q_init

    err_final = vamp_rankspace_error(Y_u=Y_u, x=Q_final, Vs=Vs)
    module_before_mean = float(err_before.mean().item()) if err_before.numel() else 0.0
    module_final_mean = float(err_final.mean().item()) if err_final.numel() else 0.0
    fail_closed = (not torch.isfinite(err_final).all()) or (module_final_mean > module_before_mean + 1e-9)
    if fail_closed:
        Q_final = Q_init
        q_final = q_init
        err_final = err_before
        module_final_mean = module_before_mean
    delta_final = W - Q_final
    raw_final = torch.einsum(
        "bi,ij,bj->b",
        delta_final,
        H_work,
        delta_final,
    )

    stats: Dict[str, Any] = {
        "method": "mpq",
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
        "topk_k": int(getattr(mpq_cfg, "topk_k", 16) or 16),
        "rankspace_error_mean_before": module_before_mean,
        "rankspace_error_mean_after": float(err_after.mean().item()) if err_after.numel() else 0.0,
        "rankspace_error_mean_final": module_final_mean,
        "raw_objective_mean_before": float(raw_before.mean().item())
        if raw_before.numel()
        else 0.0,
        "raw_objective_mean_final": float(raw_final.mean().item())
        if raw_final.numel()
        else 0.0,
        "polish_gain": float((err_before - err_after).mean().item()) if err_after.numel() else 0.0,
        "polish_updates": float(polish_stats.get("accepted_updates", 0.0)),
        "cd_trace": list(
            polish_stats.get("cd_trace")
            or polish_stats.get("sweep_trace")
            or []
        ),
        "polish_assignment_changes": int(
            (
                q_final.to(dtype=torch.int32)
                != q_init.to(dtype=torch.int32)
            )
            .to(dtype=torch.int64)
            .sum()
            .item()
        ),
        "polish_assignments_total": int(q_init.numel()),
        "block_cd_backend": str(polish_stats.get("block_cd_backend", getattr(vamp_cfg, "block_cd_backend", "eager"))),
        "module_fail_closed": bool(fail_closed),
        "aux_vamp": aux_stats if topk is not None else {},
        "quant_time_s": float(time.time() - start),
        "input_fingerprint": input_fingerprint,
    }
    stats_bucket = getattr(mpq_cfg, "_layer_stats", None)
    if not isinstance(stats_bucket, list):
        stats_bucket = []
        setattr(mpq_cfg, "_layer_stats", stats_bucket)
    stats_bucket.append(stats)
    return Q_final, q_final, scale_groups, zero_groups, g_idx, float(time.time() - start), stats
