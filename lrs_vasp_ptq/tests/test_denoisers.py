from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

import torch

from lrs_vasp_ptq.brute_force import brute_force_posterior
from lrs_vasp_ptq.codebooks import FiniteE8PCodebook, ProductGridCodebook
from lrs_vasp_ptq.denoisers.e8p_exact import rs_posterior_blocks
from lrs_vasp_ptq.denoisers.product_grid import rs_posterior_product_grid


def _maybe_reexec_with_conda_ld_library_path() -> None:
    """Match scripts/bench_quip_module_proxy.py behavior for quiptools_cuda import.

    Some environments require CONDA_PREFIX/lib to be on LD_LIBRARY_PATH so that
    extension modules pick up a compatible libstdc++.
    """
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix:
        return
    lib_dir = str(Path(conda_prefix) / "lib")
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    paths = [p for p in existing.split(":") if p]
    if lib_dir in paths:
        return
    if os.environ.get("MPGQ_REEXEC_LD_LIBRARY_PATH") == "1":
        return
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = f"{lib_dir}:{existing}" if existing else lib_dir
    env["MPGQ_REEXEC_LD_LIBRARY_PATH"] = "1"
    argv0 = str(sys.argv[0]) if sys.argv else ""
    argv = [sys.executable, *sys.argv]
    # `conda run python -m unittest ...` can yield argv0 like "python -m unittest".
    # Reconstruct a proper `-m` invocation so Python doesn't interpret argv0 as a filename.
    if argv0.startswith("python") and " -m " in argv0:
        toks = argv0.split()
        if "-m" in toks:
            m_idx = toks.index("-m")
            if m_idx + 1 < len(toks):
                module = toks[m_idx + 1]
                argv = [sys.executable, "-m", module, *sys.argv[1:]]
    os.execvpe(sys.executable, argv, env)


_maybe_reexec_with_conda_ld_library_path()


class TestProductGridDenoiser(unittest.TestCase):
    def test_product_grid_matches_bruteforce(self) -> None:
        torch.manual_seed(0)
        levels = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64)
        cb = ProductGridCodebook(levels=levels)

        center = torch.randn(4, 8, dtype=torch.float64) * 0.3
        gamma = 0.7

        ref_grid = cb.enumerate_grid(max_entries=1_000_000, dtype=torch.float64)
        ref_cb = FiniteE8PCodebook(grid=ref_grid, dedup_unique=False)
        ref = brute_force_posterior(center, gamma, ref_cb, return_cov=False, return_top2=False)
        out = rs_posterior_product_grid(center, gamma, cb, return_cov=False, return_top2=False)

        self.assertTrue(torch.allclose(out["mean"], ref["mean"], rtol=0.0, atol=1e-10))
        self.assertTrue(torch.allclose(out["var"], ref["var"], rtol=0.0, atol=1e-10))
        self.assertTrue(torch.allclose(out["logZ"], ref["logZ"], rtol=0.0, atol=1e-10))
        self.assertTrue(torch.allclose(out["map"], ref["map"], rtol=0.0, atol=1e-10))


class TestE8P12ExactDenoiser(unittest.TestCase):
    @staticmethod
    def _ensure_quip_pythonpath() -> None:
        repo_root = Path(__file__).resolve().parents[2]
        quip_root = repo_root / "third_party" / "quip-sharp"
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        if str(quip_root) not in sys.path:
            sys.path.insert(0, str(quip_root))

    def test_factorized_matches_bruteforce(self) -> None:
        self._ensure_quip_pythonpath()
        from lib import codebook as quip_codebook

        torch.manual_seed(0)
        device = torch.device("cpu")

        cb = quip_codebook.get_codebook("E8P12")
        cb = cb.to(dtype=torch.float32, device=device)

        B = 2
        n_blocks = 3
        center_blocks = torch.randn(B, n_blocks, 8, device=device, dtype=torch.float32) * 0.25
        gamma = torch.tensor([[0.6], [1.2]], device=device, dtype=torch.float32)

        out = rs_posterior_blocks(
            center_blocks=center_blocks,
            gamma=gamma,
            cb=cb,
            blocks_per_chunk=64,
            prior_center_blocks=None,
            prior_scale_blocks=None,
            prior_tau=0.0,
            prior_tau_scale_mode="absmax",
            return_var_diag=False,
            return_logZ=True,
            return_top2=True,
            backend="factorized",
        )

        # Brute-force reference over unique reconstructed vectors.
        ref_cb = FiniteE8PCodebook.from_quip(cb, dedup_unique=True)
        flat = center_blocks.to(dtype=torch.float64).reshape(-1, 8)
        gamma_flat = gamma.repeat_interleave(n_blocks, dim=0).to(dtype=torch.float64)
        ref = brute_force_posterior(flat, gamma_flat, ref_cb, return_cov=False, return_top2=True)

        ref_mean = ref["mean"].reshape(B, n_blocks, 8).to(dtype=torch.float32)
        ref_cov_trace = ref["var"].reshape(B, n_blocks, 8).sum(dim=-1).to(dtype=torch.float32)
        ref_logZ = ref["logZ"].reshape(B, n_blocks).to(dtype=torch.float32)
        ref_gap = ref["top2_gap"].reshape(B, n_blocks).to(dtype=torch.float32)

        self.assertTrue(torch.allclose(out["mean"], ref_mean, rtol=0.0, atol=2e-4))
        self.assertTrue(torch.allclose(out["cov_trace"], ref_cov_trace, rtol=0.0, atol=2e-3))
        self.assertTrue(torch.allclose(out["logZ"], ref_logZ, rtol=0.0, atol=2e-4))
        self.assertTrue(torch.allclose(out["top2_gap"], ref_gap, rtol=0.0, atol=2e-4))


if __name__ == "__main__":
    unittest.main()
