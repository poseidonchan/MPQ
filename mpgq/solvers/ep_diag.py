from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import torch

from mpgq.linear import vamp_rankspace_error
from mpgq.polish.block_cd import BlockCDConfig, E8P12Provider, block_coordinate_descent_rankspace
from mpgq.priors.e8p12 import prior_scale_from_center, vamp_vector_discrete_mmse_e8p12, vamp_vector_map_e8p12
from mpgq.utils import log_damp_positive


@dataclass(frozen=True)
class E8P12EpDiagConfig:
    """Diagonal-precision EP/EC (site) solver for E8P12 in rank space.

    This is an experimental, more expressive message family than scalar-precision VAMP:
    both modules exchange diagonal Gaussian messages (per-coordinate precision).
    """

    iters: int = 10
    damping: float = 0.5

    gamma_start: float = 1.0
    gamma_min: float = 1e-8
    gamma_max: float = 1e12
    v_min: float = 1e-6
    v_max: float = 1e6

    probes: int = 2
    probe_seed: int = 0

    pcg_max_iters: int = 64
    pcg_tol: float = 1e-4

    prior_tau: float = 0.25
    prior_tau_scale_mode: str = "absmax"
    blocks_per_chunk: int = 16

    obs_precision: float = 0.0  # <=0 means "use obs_precision_init or init-mismatch heuristic"

    # Stability: clip Gaussian cavity means to the codebook's dynamic range.
    # When tau is very small, nu and r can become ill-conditioned; clipping nu
    # to +/- (tau * r_clip) prevents "tiny precision, huge mean" pathologies.
    r_clip_mult: float = 4.0

    # Return the best (lowest surrogate error) hard iterate observed.
    best_iterate: bool = True

    polish_sweeps: int = 0
    polish_metric: str = "gram"  # "gram" | "diag"
    polish_metric_rows_per_chunk: int = 0
    polish_max_blocks: int = 0


@torch.inference_mode()
def _rowwise_mse(w: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return (w - x).square().mean(dim=1, keepdim=True).clamp_min_(1e-12)


@torch.inference_mode()
def _resolve_obs_precision_from_init(
    *,
    w: torch.Tensor,
    x_init: torch.Tensor,
    Vs: torch.Tensor,
    requested: float,
) -> float:
    if float(requested) > 0.0:
        return float(requested)
    if Vs.numel() == 0:
        return 1.0
    diff_u = (w - x_init).matmul(Vs)
    mse = float(diff_u.square().mean().clamp_min_(1e-12).item())
    return float(1.0 / mse)


@torch.inference_mode()
def _e8p12_block_debug_stats(
    *,
    r1: torch.Tensor,  # (B, n)
    gamma1_diag: torch.Tensor,  # (B, n)
    cb: object,
    blocks_per_chunk: int,
    prior_center_blocks: Optional[torch.Tensor],
    prior_scale_blocks: Optional[torch.Tensor],
    prior_tau: float,
    prior_tau_scale_mode: str,
) -> Dict[str, torch.Tensor]:
    B, n = r1.shape
    codesz = int(getattr(cb, "codesz", 8))
    if codesz != 8 or (n % codesz) != 0:
        raise ValueError(f"E8P12 debug expects n divisible by 8; got n={n}, codesz={codesz}.")
    nblocks = n // codesz

    post = vamp_vector_discrete_mmse_e8p12(
        r1_blocks=r1.reshape(B, nblocks, codesz),
        gamma1=gamma1_diag.reshape(B, nblocks, codesz),
        cb=cb,
        blocks_per_chunk=int(blocks_per_chunk),
        need_stats=False,
        need_var_diag=False,
        prior_center_blocks=prior_center_blocks,
        prior_scale_blocks=prior_scale_blocks,
        prior_tau=float(prior_tau),
        prior_tau_scale_mode=str(prior_tau_scale_mode),
        return_logZ_blocks=False,
        return_margin_blocks=True,
        return_entropy_blocks=True,
    )
    gap = post.get("margin_blocks")
    ent = post.get("entropy_blocks")
    if gap is None or ent is None:
        raise RuntimeError("E8P12 oracle did not return margin_blocks/entropy_blocks for debug.")
    var1 = post["var1"].to(dtype=torch.float32)
    cov_trace = (8.0 * var1).to(dtype=torch.float32)
    return {
        "top2_gap": gap.to(dtype=torch.float32),
        "entropy": ent.to(dtype=torch.float32),
        "cov_trace": cov_trace,
    }


@torch.inference_mode()
def _matvec_H(x: torch.Tensor, *, Vs: torch.Tensor, VsT: torch.Tensor) -> torch.Tensor:
    if Vs.numel() == 0:
        return torch.zeros_like(x)
    return x.matmul(Vs).matmul(VsT)


@torch.inference_mode()
def _pcg_solve(
    *,
    b: torch.Tensor,  # (B, n)
    matvec_A: Callable[[torch.Tensor], torch.Tensor],
    M_inv: torch.Tensor,  # (B, n) preconditioner inverse diag approx
    x0: Optional[torch.Tensor] = None,
    max_iters: int = 64,
    tol: float = 1e-4,
) -> torch.Tensor:
    """Batched preconditioned conjugate gradient for SPD systems."""
    if b.numel() == 0:
        return b
    x = x0.clone() if x0 is not None else torch.zeros_like(b)
    r = b - matvec_A(x)
    z = M_inv * r
    p = z.clone()
    rz = (r * z).sum(dim=1, keepdim=True)
    b_norm = b.square().sum(dim=1, keepdim=True).sqrt().clamp_min_(1e-12)

    for _ in range(max(int(max_iters), 1)):
        Ap = matvec_A(p)
        denom = (p * Ap).sum(dim=1, keepdim=True).clamp_min_(1e-20)
        alpha = rz / denom
        x = x + alpha * p
        r = r - alpha * Ap

        r_norm = r.square().sum(dim=1, keepdim=True).sqrt()
        if bool((r_norm <= float(tol) * b_norm).all().item()):
            break

        z = M_inv * r
        rz_new = (r * z).sum(dim=1, keepdim=True)
        beta = rz_new / rz.clamp_min_(1e-20)
        p = z + beta * p
        rz = rz_new

    return x


@torch.inference_mode()
def solve_e8p12_rankspace_ep_diag(
    *,
    Wr: torch.Tensor,  # (B, n)
    hatWr_init: torch.Tensor,  # (B, n)
    cb: object,
    V: torch.Tensor,  # (n, R) (unused; kept for signature parity)
    Vh: torch.Tensor,  # (R, n) (unused)
    S: torch.Tensor,  # (R,) (unused)
    Vs: torch.Tensor,  # (n, R)
    cfg: E8P12EpDiagConfig,
    module_name: str = "",
    obs_precision_init: Optional[float] = None,
    debug_out: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Diagonal-precision EP/EC solver with PCG linear module (experimental)."""
    Wr = Wr.to(dtype=torch.float32)
    hatWr_init = hatWr_init.to(dtype=torch.float32)
    Vs = Vs.to(dtype=torch.float32)
    VsT = Vs.T.contiguous()

    B, n = Wr.shape
    codesz = int(getattr(cb, "codesz", 8))
    if codesz != 8 or (n % codesz) != 0:
        raise ValueError(f"E8P12 EP/diag expects n divisible by 8; got n={n}, codesz={codesz}.")
    nblocks = n // codesz

    iters = max(int(cfg.iters), 0)
    damping = float(cfg.damping)
    gamma_min = float(cfg.gamma_min)
    gamma_max = float(cfg.gamma_max)
    v_min = float(cfg.v_min)
    v_max = float(cfg.v_max)
    blocks_per_chunk = max(int(cfg.blocks_per_chunk), 1)

    prior_tau = float(cfg.prior_tau)
    prior_tau_scale_mode = str(cfg.prior_tau_scale_mode or "absmax").strip().lower()
    W_blocks = Wr.reshape(B, nblocks, codesz)
    prior_scale_blocks = None
    if prior_tau > 0.0:
        prior_scale_blocks = prior_scale_from_center(W_blocks, prior_tau_scale_mode)

    # Rank-space "observations" for the quadratic surrogate.
    Y_u = Wr.matmul(Vs)  # (B,R)
    obs_precision = float(cfg.obs_precision)
    if obs_precision <= 0.0:
        obs_precision = float(obs_precision_init or 0.0)
    obs_precision = _resolve_obs_precision_from_init(w=Wr, x_init=hatWr_init, Vs=Vs, requested=float(obs_precision))

    # Precompute H and diag(H) via Vs (H = Vs Vs^T).
    diag_H = Vs.square().sum(dim=1).clamp_min_(0.0)  # (n,)
    Hw = _matvec_H(Wr, Vs=Vs, VsT=VsT)  # (B,n)

    cb_absmax = 0.0
    grid = getattr(cb, "grid", None)
    if isinstance(grid, torch.Tensor) and grid.numel():
        cb_absmax = float(grid.abs().max().item())
    r_clip = float(cfg.r_clip_mult) * float(cb_absmax) if float(cfg.r_clip_mult) > 0.0 else 0.0

    # EP-style diagonal Gaussian messages are best maintained in *natural* form:
    #   tau: precision (>=0)
    #   nu:  precision-weighted mean
    #
    # This avoids numerical blow-ups when tau is clamped small (EP can propose
    # negative cavity precisions under discrete priors).
    tau1_row = float(cfg.gamma_start) * (0.25 / _rowwise_mse(Wr, hatWr_init))  # (B,1)
    tau1 = tau1_row.expand(B, n).contiguous().clamp(min=gamma_min, max=gamma_max)
    nu1 = tau1 * hatWr_init

    # Initialize the linear-module cavity with the same message.
    tau2 = tau1.clone()
    nu2 = nu1.clone()

    gen = torch.Generator(device=Wr.device)
    gen.manual_seed(int(getattr(cfg, "probe_seed", 0) or 0))

    iter_logs = []
    x2_prev: Optional[torch.Tensor] = None
    best_hat: Optional[torch.Tensor] = None
    best_idxs: Optional[torch.Tensor] = None
    best_err_mean: float = float("inf")
    best_iter: int = -1

    def _clip_nu(nu: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        if r_clip <= 0.0:
            return nu
        bound = tau * float(r_clip)
        return nu.clamp(min=-bound, max=bound)

    for it in range(iters):
        # Convert natural parameters to mean form for the denoiser.
        r1 = nu1 / tau1.clamp_min(gamma_min)
        if r_clip > 0.0:
            r1 = r1.clamp(min=-float(r_clip), max=float(r_clip))

        # Module 1: exact discrete posterior under diagonal gamma1.
        post1 = vamp_vector_discrete_mmse_e8p12(
            r1_blocks=r1.reshape(B, nblocks, codesz),
            gamma1=tau1.reshape(B, nblocks, codesz),
            cb=cb,
            blocks_per_chunk=int(blocks_per_chunk),
            need_stats=False,
            need_var_diag=True,
            prior_center_blocks=W_blocks if prior_tau > 0.0 else None,
            prior_scale_blocks=prior_scale_blocks,
            prior_tau=prior_tau,
            prior_tau_scale_mode=prior_tau_scale_mode,
        )
        x1 = post1["x1_blocks"].reshape(B, n).to(dtype=torch.float32)
        var1_diag = post1.get("var1_diag")
        if var1_diag is None:
            raise RuntimeError("E8P12 oracle did not return var1_diag for EP/diag.")
        v1 = var1_diag.reshape(B, n).to(dtype=torch.float32).clamp(min=v_min, max=v_max)

        if bool(cfg.best_iterate):
            hat_cand = post1["map_vals"].reshape(B, n).to(dtype=torch.float32)
            idxs_cand = post1["map_idxs"].to(dtype=getattr(cb, "idx_dtype", torch.int64))
            err_cand = vamp_rankspace_error(Y_u=Y_u, x=hat_cand, Vs=Vs)
            err_mean = float(err_cand.mean().item()) if err_cand.numel() else 0.0
            if err_mean < best_err_mean:
                best_err_mean = err_mean
                best_hat = hat_cand.detach().clone()
                best_idxs = idxs_cand.detach().clone()
                best_iter = int(it)

        # Extrinsic to module 2 (site natural parameters).
        tau_post1 = (1.0 / v1).clamp(min=gamma_min, max=gamma_max)
        nu_post1 = x1 / v1
        tau2_raw = tau_post1 - tau1
        nu2_raw = nu_post1 - nu1

        tau2_new = tau2_raw.clamp(min=gamma_min, max=gamma_max)
        nu2_new = nu2_raw
        # Project invalid/negative cavity precisions to a weak (but proper) site
        # while matching the first moment (mean) as closely as possible.
        bad2 = (~torch.isfinite(tau2_raw)) | (tau2_raw <= gamma_min) | (~torch.isfinite(nu2_raw))
        if bool(bad2.any().item()):
            tau2_new = torch.where(bad2, torch.full_like(tau2_new, gamma_min), tau2_new)
            nu2_new = torch.where(bad2, tau2_new * x1, nu2_new)

        if it == 0 or damping >= 1.0:
            tau2 = tau2_new
            nu2 = nu2_new
        else:
            tau2 = log_damp_positive(tau2, tau2_new, damping).clamp(min=gamma_min, max=gamma_max)
            nu2 = (1.0 - damping) * nu2 + damping * nu2_new
        nu2 = _clip_nu(nu2, tau2)

        # Module 2: Gaussian posterior with diagonal precision + low-rank Hessian.
        diag_A = (tau2 + float(obs_precision) * diag_H.unsqueeze(0)).clamp_min_(gamma_min)
        M_inv = (1.0 / diag_A).clamp_max_(1.0 / gamma_min)

        def matvec_A(x: torch.Tensor) -> torch.Tensor:
            return tau2 * x + float(obs_precision) * _matvec_H(x, Vs=Vs, VsT=VsT)

        b = nu2 + float(obs_precision) * Hw
        x2 = _pcg_solve(
            b=b,
            matvec_A=matvec_A,
            M_inv=M_inv,
            x0=x2_prev,
            max_iters=int(cfg.pcg_max_iters),
            tol=float(cfg.pcg_tol),
        )
        x2_prev = x2

        # Hutchinson diagonal estimator for v2 = diag(A^{-1}).
        probes = max(int(cfg.probes), 1)
        diag_est = torch.zeros_like(b)
        for _ in range(probes):
            bits = torch.randint(0, 2, (B, n), generator=gen, device=Wr.device, dtype=torch.int8)
            z = bits.to(dtype=torch.float32) * 2.0 - 1.0
            u = _pcg_solve(
                b=z,
                matvec_A=matvec_A,
                M_inv=M_inv,
                x0=None,
                max_iters=int(cfg.pcg_max_iters),
                tol=float(cfg.pcg_tol),
            )
            diag_est = diag_est + z * u

        # The true covariance is Sigma2 = A^{-1}. Since A = diag(tau2) + beta*H
        # with H PSD, we have the PSD ordering A >= diag(tau2), hence
        #   Sigma2 <= diag(tau2)^{-1}  =>  diag(Sigma2) <= 1/tau2.
        #
        # For any SPD matrix A, (A^{-1})_{ii} >= 1/A_{ii}. Here A_{ii} = diag_A.
        # Enforce both bounds to stabilize noisy Hutchinson estimates.
        v2 = (diag_est / float(probes)).clamp(min=v_min, max=v_max)
        v2_lower = (1.0 / diag_A).clamp(min=v_min, max=v_max)
        v2_upper = (1.0 / tau2.clamp_min(gamma_min)).clamp(min=v_min, max=v_max)
        v2 = torch.maximum(v2, v2_lower)
        v2 = torch.minimum(v2, v2_upper)

        tau_post2 = (1.0 / v2).clamp(min=gamma_min, max=gamma_max)
        nu_post2 = x2 / v2
        tau1_raw = tau_post2 - tau2
        nu1_raw = nu_post2 - nu2

        tau1_new = tau1_raw.clamp(min=gamma_min, max=gamma_max)
        nu1_new = nu1_raw
        bad1 = (~torch.isfinite(tau1_raw)) | (tau1_raw <= gamma_min) | (~torch.isfinite(nu1_raw))
        if bool(bad1.any().item()):
            tau1_new = torch.where(bad1, torch.full_like(tau1_new, gamma_min), tau1_new)
            nu1_new = torch.where(bad1, tau1_new * x2, nu1_new)

        if damping >= 1.0:
            tau1 = tau1_new
            nu1 = nu1_new
        else:
            tau1 = log_damp_positive(tau1, tau1_new, damping).clamp(min=gamma_min, max=gamma_max)
            nu1 = (1.0 - damping) * nu1 + damping * nu1_new
        nu1 = _clip_nu(nu1, tau1)

        gap = float(((x1 - x2).norm() / x2.norm().clamp_min(1e-12)).item())
        iter_logs.append(
            {
                "iter": int(it),
                "gamma1_mean": float(tau1.mean().item()),
                "gamma2_mean": float(tau2.mean().item()),
                "v1_mean": float(v1.mean().item()),
                "v2_mean": float(v2.mean().item()),
                "gap": float(gap),
            }
        )

    r1 = nu1 / tau1.clamp_min(gamma_min)
    final_map = vamp_vector_map_e8p12(
        r1_blocks=r1.reshape(B, nblocks, codesz),
        gamma1=tau1.reshape(B, nblocks, codesz),
        cb=cb,
        blocks_per_chunk=int(blocks_per_chunk),
        need_stats=False,
        prior_center_blocks=W_blocks if prior_tau > 0.0 else None,
        prior_scale_blocks=prior_scale_blocks,
        prior_tau=prior_tau,
        prior_tau_scale_mode=prior_tau_scale_mode,
    )
    hat = final_map["map_vals"].reshape(B, n).to(dtype=torch.float32)
    idxs = final_map["map_idxs"].to(dtype=getattr(cb, "idx_dtype", torch.int64))
    err = vamp_rankspace_error(Y_u=Y_u, x=hat, Vs=Vs)
    if best_hat is not None and best_idxs is not None and float(best_err_mean) < float(err.mean().item()):
        hat = best_hat
        idxs = best_idxs
        err = vamp_rankspace_error(Y_u=Y_u, x=hat, Vs=Vs)

    if debug_out is not None:
        dbg_stats = _e8p12_block_debug_stats(
            r1=r1,
            gamma1_diag=tau1,
            cb=cb,
            blocks_per_chunk=int(blocks_per_chunk),
            prior_center_blocks=W_blocks if prior_tau > 0.0 else None,
            prior_scale_blocks=prior_scale_blocks,
            prior_tau=prior_tau,
            prior_tau_scale_mode=prior_tau_scale_mode,
        )
        debug_out["solver"] = "e8p12_ep_diag"
        debug_out["idxs_pre_polish"] = idxs.detach().to(device="cpu")
        debug_out["gamma1_mean"] = float(tau1.mean().item())
        debug_out["gamma2_mean"] = float(tau2.mean().item())
        debug_out["r1_absmax"] = float(r1.abs().amax().item()) if r1.numel() else 0.0
        debug_out["nu1_absmax"] = float(nu1.abs().amax().item()) if nu1.numel() else 0.0
        debug_out["tau1_min"] = float(tau1.min().item()) if tau1.numel() else 0.0
        debug_out["tau1_max"] = float(tau1.max().item()) if tau1.numel() else 0.0
        debug_out["tau2_min"] = float(tau2.min().item()) if tau2.numel() else 0.0
        debug_out["tau2_max"] = float(tau2.max().item()) if tau2.numel() else 0.0
        debug_out["block_stats"] = {k: v.detach().to(device="cpu") for k, v in dbg_stats.items()}
        debug_out["iter_logs"] = list(iter_logs)

    polish_sweeps = max(int(cfg.polish_sweeps), 0)
    polish_metric = str(cfg.polish_metric or "gram").strip().lower()
    if polish_metric in ("isotropic", "diag"):
        polish_metric = "diag"
    elif polish_metric != "gram":
        raise ValueError(f"unsupported polish_metric={polish_metric!r}; expected 'gram'|'diag'.")

    polish_gain = 0.0
    polish_updates = 0.0
    if polish_sweeps > 0 and Vs.numel() > 0:
        err_before = err
        hat, idxs, polish_stats = block_coordinate_descent_rankspace(
            x=hat.to(dtype=torch.float32),
            Y_u=Y_u.to(dtype=torch.float32),
            Vs=Vs.to(dtype=torch.float32),
            block_size=codesz,
            provider=E8P12Provider(cb=cb, metric_rows_per_chunk=int(cfg.polish_metric_rows_per_chunk)),
            cfg=BlockCDConfig(
                sweeps=polish_sweeps,
                metric="gram" if polish_metric == "gram" else "diag",
                accept_tol=0.0,
                metric_rows_per_chunk=int(cfg.polish_metric_rows_per_chunk),
                max_blocks_per_sweep=int(getattr(cfg, "polish_max_blocks", 0) or 0),
                order_mode="greedy_gain",
            ),
            idxs_init=idxs,
        )
        err = vamp_rankspace_error(Y_u=Y_u, x=hat, Vs=Vs)
        polish_gain = float((err_before - err).mean().item()) if err.numel() else 0.0
        polish_updates = float(polish_stats.get("accepted_updates", 0.0))

    if debug_out is not None:
        idxs_post = idxs.detach().to(device="cpu")
        debug_out["idxs_post_polish"] = idxs_post
        idxs_pre = debug_out.get("idxs_pre_polish")
        if isinstance(idxs_pre, torch.Tensor) and idxs_pre.shape == idxs_post.shape:
            debug_out["changed_mask"] = (idxs_pre != idxs_post)

    stats: Dict[str, Any] = {
        "solver": "e8p12_ep_diag",
        "module": str(module_name),
        "iters": int(iters),
        "damping": float(damping),
        "obs_precision": float(obs_precision),
        "r_clip_mult": float(cfg.r_clip_mult),
        "r_clip": float(r_clip),
        "best_iterate": bool(cfg.best_iterate),
        "best_iter": int(best_iter),
        "best_rankspace_error_mean": float(best_err_mean) if math.isfinite(best_err_mean) else float("inf"),
        "probes": int(cfg.probes),
        "pcg_max_iters": int(cfg.pcg_max_iters),
        "pcg_tol": float(cfg.pcg_tol),
        "prior_tau": float(prior_tau),
        "prior_tau_scale_mode": str(prior_tau_scale_mode),
        "polish_sweeps": int(polish_sweeps),
        "polish_metric": str(polish_metric),
        "polish_gain": float(polish_gain),
        "polish_updates": float(polish_updates),
        "rankspace_error_mean": float(err.mean().item()) if err.numel() else 0.0,
    }
    return hat, idxs, err, stats
