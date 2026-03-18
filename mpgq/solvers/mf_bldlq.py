from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch

from ..priors.e8p12 import vamp_vector_map_topk_e8p12


@dataclass(frozen=True)
class MfBldlqConfig:
    """Multi-feedback Block LDLQ (MF-BLDLQ) for E8P12.

    This is a limited-branching augmentation of greedy block LDLQ (reverse-order SIC).

    Current implementation branches at (up to) `branch_per_row` blocks per row by selecting
    the smallest top2 gap blocks under the baseline greedy cavity targets.
    """

    candidates: int = 8
    branch_per_row: int = 1
    gap_thresh: float = 0.0  # only branch if worst gap <= gap_thresh (0 => always branch)
    max_branched_rows: int = 0  # 0 => no cap (per solve() call / chunk)
    max_branched_frac: float = 0.1  # cap branched rows per chunk by fraction of rows (<=0 disables)
    blocks_per_chunk_topk: int = 64
    candidate_metric: str = "isotropic"  # "isotropic" | "mahalanobis"
    metric_rows_per_chunk: int = 0  # for mahalanobis top-k (optional)


@torch.inference_mode()
def _block_quadratic_cost(u: torch.Tensor, Dk: torch.Tensor) -> torch.Tensor:
    """Return 0.5 * u^T Dk u for each row in u."""
    Du = u.matmul(Dk.to(dtype=torch.float32))
    return 0.5 * (Du * u).sum(dim=1)


@torch.inference_mode()
def _greedy_bldlq_with_targets(
    *,
    Wr: torch.Tensor,  # (B,n)
    L: torch.Tensor,  # (n,n) unit block-lower-triangular
    D: torch.Tensor,  # (n_blocks,8,8) SPD blocks
    cb: Any,
    resid_scale_override: Optional[float],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Greedy block LDLQ (reverse order), recording cavity targets and per-block costs."""
    Wr = Wr.to(dtype=torch.float32)
    L = L.to(dtype=torch.float32).contiguous()
    D = D.to(dtype=torch.float32).contiguous()

    B, n = Wr.shape
    codesz = int(getattr(cb, "codesz", 8))
    if codesz != 8 or (n % codesz) != 0:
        raise ValueError(f"MF-BLDLQ expects E8P12 codesz=8 with n divisible by 8; got n={n}, codesz={codesz}.")
    nblocks = n // codesz
    if D.ndim != 3 or D.shape[0] != nblocks or D.shape[1:] != (codesz, codesz):
        raise ValueError(f"expected D shape ({nblocks},8,8), got {tuple(D.shape)}")

    hat = torch.zeros_like(Wr)
    idxs = torch.empty(B, nblocks, dtype=getattr(cb, "idx_dtype", torch.int64), device=Wr.device)
    targets = torch.empty(B, nblocks, codesz, dtype=torch.float32, device=Wr.device)
    costs = torch.empty(B, nblocks, dtype=torch.float32, device=Wr.device)

    # Running cached contribution from already-quantized (future) blocks:
    #   prod[:, :] = sum_{i>k} (Wr_i - hat_i) @ L[i, :]
    # so the current cavity target is:
    #   r_k = Wr_k + prod[:, cols(k)].
    prod = torch.zeros_like(Wr, dtype=torch.float32)
    for k in reversed(range(nblocks)):
        s = codesz * k
        e = s + codesz
        r = Wr[:, s:e] + prod[:, s:e]
        targets[:, k, :] = r
        q, qidx = cb.quantize(r, resid_scale_override=resid_scale_override)
        q = q.to(dtype=Wr.dtype)
        hat[:, s:e] = q
        idxs[:, k] = qidx.to(dtype=idxs.dtype)
        u = (r - q).to(dtype=torch.float32)
        costs[:, k] = _block_quadratic_cost(u, D[k])
        if s > 0:
            delta = (Wr[:, s:e] - q).to(dtype=torch.float32)
            prod[:, :s] = prod[:, :s] + delta.matmul(L[s:e, :s])

    return hat, idxs, targets, costs


@torch.inference_mode()
def _metric_topk_e8p12(
    *,
    target: torch.Tensor,  # (B,8)
    chol: torch.Tensor,  # (8,8) lower-triangular such that D = chol @ chol^T
    cb: Any,
    k: int,
    rows_per_chunk: int,
    cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Top-k under Mahalanobis metric induced by `chol` (exact over full 65536 grid)."""
    grid = cb.grid.to(dtype=torch.float32, device=target.device)
    key = int(chol.data_ptr())
    metric_grid = None
    metric_grid_norm = None
    if key in cache:
        metric_grid, metric_grid_norm = cache[key]
        if metric_grid.device != target.device:
            metric_grid = metric_grid.to(device=target.device)
            metric_grid_norm = metric_grid_norm.to(device=target.device)
    if metric_grid is None or metric_grid_norm is None:
        chol_f32 = chol.to(dtype=torch.float32, device=target.device)
        metric_grid = grid.matmul(chol_f32)
        metric_grid_norm = metric_grid.square().sum(dim=-1).contiguous()
        cache[key] = (metric_grid, metric_grid_norm)

    target_metric = target.to(dtype=torch.float32).matmul(chol.to(dtype=torch.float32))
    rows = int(target_metric.shape[0])
    rpc = rows
    if int(rows_per_chunk) > 0:
        rpc = max(1, min(rows, int(rows_per_chunk)))
    k = min(int(k), int(metric_grid.shape[0]))
    topk_idx = torch.empty(rows, k, dtype=torch.long, device=target.device)
    topk_score = torch.empty(rows, k, dtype=torch.float32, device=target.device)
    for start in range(0, rows, rpc):
        end = min(start + rpc, rows)
        chunk = target_metric[start:end]  # (C,8)
        chunk_norm = chunk.square().sum(dim=-1, keepdim=True)  # (C,1)
        # squared distances in metric space
        dist = chunk_norm + metric_grid_norm.unsqueeze(0) - 2.0 * chunk.matmul(metric_grid.T)  # (C,65536)
        vals, idx = torch.topk(-dist, k=k, dim=1)  # higher is better (negative distance)
        topk_idx[start:end] = idx
        topk_score[start:end] = vals.to(dtype=torch.float32)
    return topk_idx.to(dtype=getattr(cb, "idx_dtype", torch.int64)), topk_score


@torch.inference_mode()
def solve_e8p12_mf_bldlq(
    *,
    Wr: torch.Tensor,  # (B,n)
    L: torch.Tensor,  # (n,n)
    D: torch.Tensor,  # (n_blocks,8,8)
    cb: Any,
    cfg: MfBldlqConfig,
    resid_scale_override: Optional[float] = None,
    debug_out: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """MF-BLDLQ: branch on unreliable blocks and pick the best path by the LDL tree metric."""
    t0 = time.time()
    Wr = Wr.to(dtype=torch.float32)
    device = Wr.device
    B, n = Wr.shape
    codesz = int(getattr(cb, "codesz", 8))
    nblocks = n // codesz

    candidates = max(int(cfg.candidates), 1)
    branch_per_row = max(int(cfg.branch_per_row), 0)

    L_f32 = L.to(device=device, dtype=torch.float32).contiguous()
    D_f32 = D.to(device=device, dtype=torch.float32).contiguous()

    hat_base, idxs_base, targets, costs = _greedy_bldlq_with_targets(
        Wr=Wr,
        L=L_f32,
        D=D_f32,
        cb=cb,
        resid_scale_override=resid_scale_override,
    )
    total_base = float(costs.sum(dim=1).mean().item()) if costs.numel() else 0.0

    if branch_per_row <= 0 or candidates <= 1:
        stats = {
            "solver": "mf-bldlq",
            "branched_rows": 0,
            "branched_blocks": 0,
            "base_cost_mean": total_base,
            "cost_mean": total_base,
            "time_s": float(time.time() - t0),
        }
        return hat_base, idxs_base, stats

    # Reliability = top2 gap under the isotropic factorized E8P12 score at the recorded targets.
    # Use a per-block scalar precision derived from the LDL block metric:
    #   gamma_k ≈ tr(D_k) / codesz
    # This better matches the Mahalanobis cost scale than a fixed gamma=1.
    diag = torch.diagonal(D.to(device=device, dtype=torch.float32), dim1=-2, dim2=-1)
    gamma_k = (diag.sum(dim=-1) / float(codesz)).clamp_min_(1e-6)  # (nblocks,)
    gamma = gamma_k.view(1, nblocks, 1).expand(B, nblocks, 1).contiguous()
    top2 = vamp_vector_map_topk_e8p12(
        r1_blocks=targets,
        gamma1=gamma,
        cb=cb,
        k=2,
        blocks_per_chunk=int(cfg.blocks_per_chunk_topk),
    )
    scores = top2["topk_scores"]  # (B,nblocks,2)
    gap = (scores[:, :, 0] - scores[:, :, 1]).to(dtype=torch.float32)  # (B,nblocks)

    # Select worst blocks per row (currently only supports branch_per_row=1).
    if branch_per_row != 1:
        raise NotImplementedError("MF-BLDLQ currently supports branch_per_row=1 (single worst block per row).")
    worst_k = gap.argmin(dim=1)  # (B,)
    worst_gap = gap.gather(1, worst_k.view(B, 1)).squeeze(1)

    do_branch = torch.ones(B, dtype=torch.bool, device=device)
    if float(cfg.gap_thresh) > 0.0:
        do_branch = worst_gap <= float(cfg.gap_thresh)

    hat_out = hat_base.clone()
    idxs_out = idxs_base.clone()

    rows = torch.arange(B, device=device, dtype=torch.long)
    rows_branch = rows[do_branch]
    # Budget-based gating: only branch on the least reliable rows.
    max_rows = int(getattr(cfg, "max_branched_rows", 0) or 0)
    max_frac = float(getattr(cfg, "max_branched_frac", 0.0) or 0.0)
    budget = int(rows_branch.numel())
    if max_rows > 0:
        budget = min(budget, max_rows)
    if max_frac > 0.0 and max_frac < 1.0:
        budget = min(budget, max(1, int(torch.ceil(torch.tensor(float(B) * max_frac)).item())))
    if budget < int(rows_branch.numel()):
        gaps = worst_gap.index_select(0, rows_branch)
        _, pos = torch.topk(gaps, k=budget, largest=False)
        rows_branch = rows_branch.index_select(0, pos)
    if rows_branch.numel() == 0:
        stats = {
            "solver": "mf-bldlq",
            "branched_rows": 0,
            "branched_blocks": 0,
            "base_cost_mean": total_base,
            "cost_mean": total_base,
            "time_s": float(time.time() - t0),
        }
        return hat_out, idxs_out, stats

    # Group rows by their chosen branch block.
    branched_blocks = 0
    total_cost = costs.sum(dim=1)  # (B,)
    unique_blocks = torch.unique(worst_k[rows_branch])
    metric_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
    for k in unique_blocks.tolist():
        k_int = int(k)
        sel = rows_branch[worst_k[rows_branch] == k_int]
        if sel.numel() == 0:
            continue

        s = codesz * k_int
        e = s + codesz
        Wr_g = Wr.index_select(0, sel)
        hat_base_g = hat_base.index_select(0, sel)
        idxs_base_g = idxs_base.index_select(0, sel)
        targets_g = targets.index_select(0, sel)
        r_k = targets_g[:, k_int, :]  # (G,8)
        cost_suffix = costs.index_select(0, sel)[:, k_int + 1 :].sum(dim=1) if (k_int + 1) < nblocks else torch.zeros(sel.numel(), device=device)

        # Candidate list for this block.
        cand_metric = str(getattr(cfg, "candidate_metric", "isotropic") or "isotropic").strip().lower()
        if cand_metric in ("mahalanobis", "gram", "metric"):
            chol = torch.linalg.cholesky(D[k_int].to(device=device, dtype=torch.float32))
            cand_idxs, _cand_scores = _metric_topk_e8p12(
                target=r_k,
                chol=chol,
                cb=cb,
                k=candidates,
                rows_per_chunk=int(getattr(cfg, "metric_rows_per_chunk", 0) or 0),
                cache=metric_cache,
            )
        else:
            gamma_g = gamma_k[k_int].expand(sel.numel(), 1).to(dtype=torch.float32, device=device)
            topC = vamp_vector_map_topk_e8p12(
                r1_blocks=r_k.unsqueeze(1),
                gamma1=gamma_g,
                cb=cb,
                k=candidates,
                blocks_per_chunk=int(cfg.blocks_per_chunk_topk),
            )
            cand_idxs = topC["topk_idxs"].squeeze(1)  # (G,C)
        C_eff = int(cand_idxs.shape[1])
        cand_vals = cb.grid.index_select(0, cand_idxs.reshape(-1)).to(dtype=torch.float32).view(sel.numel(), C_eff, codesz)

        # Expand batch over candidates, but only recompute the prefix blocks (<k).
        # Use a cheap correction cache relative to the baseline greedy targets:
        #
        #   r_j(candidate) = r_j(base) + sum_{i=j+1..k} (q_i(base)-q_i(candidate)) @ L[i, j]
        #
        # where blocks >k are fixed to the baseline decisions.
        G = int(sel.numel())
        M = G * C_eff
        row_ids = torch.arange(G, device=device, dtype=torch.long).repeat_interleave(C_eff)

        # Fixed suffix cost for blocks >k.
        suffix_flat = cost_suffix.index_select(0, row_ids)

        # Block-k candidate decisions and local cost at k.
        qk_flat = cand_vals.reshape(M, codesz)
        rk_flat = r_k.index_select(0, row_ids)
        uk_flat = (rk_flat - qk_flat).to(dtype=torch.float32)
        prefix_cost = _block_quadratic_cost(uk_flat, D_f32[k_int])

        # Correction cache for columns [0, s) where s = codesz*k.
        s_prefix = int(s)
        corr = None
        hat_prefix = None
        idx_prefix = None
        if s_prefix > 0:
            corr = torch.zeros(M, s_prefix, dtype=torch.float32, device=device)
            qk_base = hat_base_g[:, s:e].index_select(0, row_ids).to(dtype=torch.float32)
            delta_diff = qk_base - qk_flat.to(dtype=torch.float32)  # q_base - q_candidate
            corr = corr + delta_diff.matmul(L_f32[s:e, :s_prefix])
            hat_prefix = torch.empty(M, s_prefix, dtype=torch.float32, device=device)
            idx_prefix = torch.empty(M, k_int, dtype=idxs_base_g.dtype, device=device)

        # Greedy completion for blocks <k, using corrected baseline targets.
        for j in reversed(range(k_int)):
            sj = codesz * j
            ej = sj + codesz
            base_rj = targets_g[:, j, :].index_select(0, row_ids)  # (M,8)
            if corr is None:
                r_j = base_rj
            else:
                r_j = base_rj + corr[:, sj:ej]
            q_j, qidx_j = cb.quantize(r_j, resid_scale_override=resid_scale_override)
            q_j = q_j.to(dtype=torch.float32)
            if hat_prefix is not None and idx_prefix is not None:
                hat_prefix[:, sj:ej] = q_j
                idx_prefix[:, j] = qidx_j.to(dtype=idx_prefix.dtype)
            u_j = (r_j - q_j).to(dtype=torch.float32)
            prefix_cost = prefix_cost + _block_quadratic_cost(u_j, D_f32[j])

            if corr is not None and sj > 0:
                qj_base = hat_base_g[:, sj:ej].index_select(0, row_ids).to(dtype=torch.float32)
                delta_diff_j = qj_base - q_j  # q_base - q_candidate
                corr[:, :sj] = corr[:, :sj] + delta_diff_j.matmul(L_f32[sj:ej, :sj])

        total_flat = suffix_flat + prefix_cost
        total_mat = total_flat.view(G, C_eff)
        best_c = total_mat.argmin(dim=1)
        win = torch.arange(G, device=device) * C_eff + best_c

        # Assemble final solutions from the baseline by replacing prefix blocks and block k.
        hat_best = hat_base_g.clone()
        idx_best = idxs_base_g.clone()
        hat_best[:, s:e] = cand_vals[torch.arange(G, device=device), best_c].to(dtype=hat_best.dtype)
        idx_best[:, k_int] = cand_idxs[torch.arange(G, device=device), best_c].to(dtype=idx_best.dtype)
        if s_prefix > 0 and hat_prefix is not None and idx_prefix is not None:
            hat_best[:, :s_prefix] = hat_prefix.index_select(0, win).to(dtype=hat_best.dtype)
            idx_best[:, :k_int] = idx_prefix.index_select(0, win).to(dtype=idx_best.dtype)

        cost_best = total_flat.index_select(0, win)

        hat_out.index_copy_(0, sel, hat_best)
        idxs_out.index_copy_(0, sel, idx_best)
        total_cost.index_copy_(0, sel, cost_best)
        branched_blocks += int(sel.numel())

    stats = {
        "solver": "mf-bldlq",
        "branched_rows": int(rows_branch.numel()),
        "branched_blocks": int(branched_blocks),
        "base_cost_mean": total_base,
        "cost_mean": float(total_cost.mean().item()) if total_cost.numel() else total_base,
        "time_s": float(time.time() - t0),
    }
    if debug_out is not None:
        debug_out["worst_gap"] = worst_gap.detach().to(device="cpu")
        debug_out["worst_k"] = worst_k.detach().to(device="cpu")
        debug_out["gap"] = gap.detach().to(device="cpu")
    return hat_out, idxs_out, stats
