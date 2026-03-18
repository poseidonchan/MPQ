from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, Optional, Tuple

import torch

from mpgq.solver import E8P12VampConfig, solve_e8p12_rankspace_vamp


@torch.inference_mode()
def solve_e8p12_rankspace_vamp_topk_polish(
    *,
    Wr: torch.Tensor,
    hatWr_init: torch.Tensor,
    cb: object,
    V: torch.Tensor,
    Vh: torch.Tensor,
    S: torch.Tensor,
    Vs: torch.Tensor,
    cfg: E8P12VampConfig,
    candidate_k: int = 8,
    module_name: str = "",
    obs_precision_init: Optional[float] = None,
    log_stats: bool = True,
    debug_out: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    shared_cfg = replace(
        cfg,
        polish_mode="topk_block_cd",
        topk_k=max(int(candidate_k), 1),
    )
    return solve_e8p12_rankspace_vamp(
        Wr=Wr,
        hatWr_init=hatWr_init,
        cb=cb,
        V=V,
        Vh=Vh,
        S=S,
        Vs=Vs,
        cfg=shared_cfg,
        module_name=module_name,
        obs_precision_init=obs_precision_init,
        log_stats=log_stats,
        debug_out=debug_out,
    )
