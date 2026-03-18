# Speedup Notes

This file records the current state of speed work on `VAMP + (topk) block CD`, with emphasis on:

- what was measured
- what actually helped
- what did not help
- what remains slow
- what to avoid repeating

All timings below are from the `QUIP#` single-module proxy fixture on `Llama-3.2-1B`, `layer=0`, `suffix=qkv`, using the saved Hessian fixture at:

- `runs/20260313_214017_b2/seed0/hess_c4_train/0_qkv.pt`

Unless otherwise stated, the relevant config is:

- `mpq_iters=16`
- `mpq_damping=0.5`
- `prior_tau=0.25`
- `init=rtn`
- `polish_sweeps=3`
- `topk_k=8`


## 1. What Was Slow

Two different bottlenecks existed:

1. `E8P12` VAMP prior/posterior
2. `block CD polish`

Originally, `block CD` looked like the dominant problem, but part of that was because the compiled prior path was silently not taking effect.


## 2. Compiled Prior: Real Root Cause And Fix

### Problem

The intended `torch.compile` path for the `E8P12` prior was silently failing and falling back to eager mode.

The failure came from `mpq/priors/e8p12.py` inside `_branch_factorized_stats_impl`, specifically the use of:

```python
float(offset) ** 2
```

TorchDynamo hit this during compile and disabled the compiled branch.

### Fix

Replace the problematic expression with explicit float temporaries:

```python
offset_f = float(offset)
offset_sq = offset_f * offset_f
```

and use `offset_f` / `offset_sq` inside the branch code.

This fix is in:

- [mpq/priors/e8p12.py](mpq/priors/e8p12.py)

### Result

After this fix, the compiled prior path actually engaged.

Same-process `no-polish` timings, same fixture:

- `compile_prior=False`
  - rep1: `solver_time_s ≈ 18.62`
  - rep2: `solver_time_s ≈ 18.57`
- `compile_prior=True`
  - rep1: `solver_time_s ≈ 7.85`
  - rep2: `solver_time_s ≈ 3.83`

So the compiled prior acceleration is real and large once the graph is actually used.

### Numerical behavior

The proxy changed only very slightly:

- without compile: around `77.0314`
- with compile: around `77.0303`

This difference was small enough to be acceptable for current use.


## 3. Current Hot Timings After Prior Fix

Same-process hot measurements after the prior compile fix:

- `vamp`, `polish_sweeps=0`
  - hot rep: `solver_time_s ≈ 3.79`
- `vamp`, `polish_sweeps=3`
  - hot rep: `solver_time_s ≈ 18.59`
  - proxy after: `48.66197176774343`
- `vamp-topk-polish`, `polish_sweeps=3`
  - hot rep: `solver_time_s ≈ 18.56`
  - proxy after: `46.14886871973673`

Interpretation:

- `VAMP` itself is now fast enough
- the remaining major cost is still `block CD`
- polish adds roughly `~15s` over hot no-polish VAMP on this fixture


## 4. Block CD: Exact-Equivalent Speedups That Helped

These changes were kept because they improved speed without intentionally changing solver logic.

### 4.1 Batched greedy gain

The old implementation computed gain block-by-block in Python.

This was changed to a batched `_batched_greedy_gain(...)` path.

Location:

- [mpq/polish/block_cd.py](mpq/polish/block_cd.py)

### 4.2 Keep E8P12 top-k data on device

Top-k indices/values were previously incurring unnecessary host/device movement.

Changes:

- keep `topk_idxs` on GPU
- keep decoded `topk_vals` on GPU
- stop moving top-k data to CPU and back

Location:

- [mpq/polish/block_cd.py](mpq/polish/block_cd.py)
- [mpq/solver.py](mpq/solver.py)
- [mpq/adapters/quip_sharp.py](mpq/adapters/quip_sharp.py)

### 4.3 Sparse `index_copy_` writeback

Accepted rows are written back sparsely instead of using heavier generic update patterns.

This reduced overhead without changing update semantics.

### 4.4 Prestore `Vs_blk`

The block metadata now stores contiguous `Vs_blk` directly, so residual update uses:

```python
accepted_delta.matmul(Vs_blk)
```

instead of repeatedly materializing/transposing `Vs_blk_t.T`.

### 4.5 Compiled acceptance helper

The acceptance math:

- `delta`
- `cross`
- `delta_norm`
- `cand_err`
- `accept`

was wrapped in a compiled helper with fallback:

```python
torch.compile(..., mode="max-autotune-no-cudagraphs", fullgraph=False)
```

Location:

- [mpq/polish/block_cd.py](mpq/polish/block_cd.py)

### Aggregate effect before the prior fix confusion was resolved

These block-CD-only optimizations did produce real savings on the proxy fixture. They brought the old `block CD` / `topk block CD` times down materially before the compile-prior diagnosis was corrected.

However, after the prior fix, the correct interpretation is:

- `VAMP` is no longer the bottleneck
- `block CD` still contributes roughly `~15s`


## 5. Block CD Profiling Findings

Measured hot-path breakdowns on the same single-module fixture showed:

- `_batched_greedy_gain` is now negligible
- `torch.cholesky_solve` is not the main bottleneck
- `topk` candidate generation is only part of the remaining cost

Observed measurements:

- `greedy_gain` time: about `0.03s`
- `torch.cholesky_solve` total: about `1.8s`
- `topk propose` total: about `4.4s`

The remaining large chunk is therefore mostly:

- repeated `rhs = residual_u.matmul(Vs_blk_t)`
- per-block target/accept/update bookkeeping
- serial residual maintenance in the Gauss-Seidel loop

This is important: the remaining cost is mostly in the block-sequential loop itself, not in the prior anymore and not in greedy block ordering.


## 6. What Did Not Help Or Was Not Good Enough

These are important because they should not be repeated blindly.

### 6.1 Prior micro-optimizations inside `_branch_factorized_stats`

Caching factor tensors and removing minor allocations gave only about `~1%` improvement. Not worth prioritizing relative to compile.

### 6.2 A batched branch rewrite of the prior

A version that tried to evaluate branches together became slower and also introduced slight numerical drift. It was not kept.

### 6.3 Compiling `_batched_greedy_gain`

This gave essentially no meaningful end-to-end speedup.

### 6.4 Replacing `cholesky_solve` with explicit inverse

This was discussed but not adopted for the strict-equivalence phase, because it changes the numeric path and is not necessary given that `cholesky_solve` is not the main bottleneck.

### 6.5 Full `torch.profiler` on the complete hot path

This was a mistake.

A full profiler run was launched on the hot `vamp-topk-polish` path while also doing:

- model loading
- compiled prior warmup
- full quantization pass
- profiler trace capture

This likely saturated the machine and contributed to a system lockup.

Conclusion:

- do not use full `torch.profiler` on the complete quantization path again
- only use lightweight timing on single-module runs


## 7. Block CD: Phase 2 Optimizations

Three further optimizations reduced polish time by another ~11s.

### 7.1 Replace `cholesky_solve` with precomputed `gram_inv`

The per-block target computation used `torch.cholesky_solve(rhs.T.contiguous(), gram_chol).T` which was measured at ~1.8s total. This was replaced with:

- Precompute `gram_inv = torch.cholesky_inverse(gram_chol)` per block (once, at setup)
- In the inner loop: `target = x_blk + rhs.matmul(gram_inv)` (one small matmul)

This also eliminates the `.T.contiguous()` copy. For 8×8 PD matrices, the inverse is well-conditioned.

The same change was applied to `_batched_greedy_gain`, replacing batched `cholesky_solve` with `torch.bmm(rhs_perm, gram_inv_stack)`.

### 7.2 Unconditional masked writeback (eliminate GPU→CPU sync)

The old writeback path had a `accept.any().item()` early-exit check that forced a GPU→CPU synchronization on every block iteration (~73K times per run). This was removed in favor of unconditional masked operations:

```python
mask = accept.unsqueeze(1)
x_work[:, start:end] = torch.where(mask, cand_vals, x_blk)
delta.mul_(mask)
residual_u.sub_(delta.matmul(Vs_blk))
residual_norm = torch.where(accept, cand_err, residual_norm)
```

This also eliminated `torch.nonzero`, `index_select`, and `index_copy_` (replaced by `torch.where`), reducing per-block kernel launches from ~10 to ~5 for the writeback portion.

### 7.3 Increase default `block_rows` from 32 to 128

The per-block CUDA kernels operate on `(B, 8)` or `(B, R)` tensors where B = `block_rows`. With B=32, every kernel severely underutilizes the GPU. Profiling by sweeping `block_rows` showed:

| block_rows | solver_time_s | proxy |
|---|---|---|
| 32 | 16.41 | 46.151 |
| 64 | 10.93 | 46.172 |
| 128 | 7.57 | 46.192 |
| 256 | 5.94 | 46.239 |

The quality impact is negligible (<0.1% proxy change). Default changed from 32 to 128 across all adapter configs and scripts.

### 7.4 Combined effect

Same-process hot timings, same fixture, `block_rows=128`:

- `vamp`, `polish_sweeps=0`
  - hot rep: `solver_time_s ≈ 3.63`
- `vamp-topk-polish`, `polish_sweeps=3`
  - hot rep: `solver_time_s ≈ 7.27`
  - proxy after: `46.1924`
- `vamp`, `polish_sweeps=3`
  - hot rep: `solver_time_s ≈ 7.45`
  - proxy after: `48.7762`

Polish-only time: `7.27 - 3.63 ≈ 3.64s` (was `~14.8s`, **4.1× faster**).


## 8. Current Code State Relevant To Speed

Speed-related active files:

- [mpq/priors/e8p12.py](mpq/priors/e8p12.py)
- [mpq/polish/block_cd.py](mpq/polish/block_cd.py)

The key points of the current state are:

- compiled prior is fixed and should be kept
- `block CD` went from `~14.8s` to `~3.6s` via gram_inv + masked writeback + larger batch
- VAMP and polish are now roughly equal cost (~3.6s each)


## 9. Best Current Mental Model

At this point the runtime picture is:

1. `VAMP prior/posterior`
   - was the original bottleneck
   - fixed by getting compile to work (~3.6s hot)

2. `block CD polish`
   - was the second bottleneck (~14.8s)
   - now `~3.6s` after gram_inv, masked writeback, and larger block_rows
   - remaining cost is dominated by the serial Gauss-Seidel block loop (per-block CUDA kernel launches)


## 10. Phase 3: Compiled Block Step Functions

The per-block inner loop was restructured to use compiled step functions that fuse target computation + proposal + accept + writeback into a single `torch.compile` graph.

### 10.1 Compiled topk-gram step

For the `E8P12TopKProvider` with `metric="gram"`, a compiled function `_topk_gram_step_impl` replaces:
- rhs + target computation (2 small matmuls)
- topk metric select (elementwise + argmin over k=8 candidates)
- accept step (already had a separate compiled helper)
- masked writeback (torch.where, matmul, sub)

Pre-computation: all per-block `cand_metric` and `cand_metric_norm` tensors are computed once before the sweep loop (~9MB for 256 blocks with B=128, k=8, bs=8).

### 10.2 Compiled CD-gram step

For the `E8P12Provider` with `metric="gram"`, a compiled function `_cd_gram_step_impl` replaces:
- rhs + target computation
- full grid proposal: `grid @ chol` + distance matrix + argmin over G=65536 codewords
- accept step
- masked writeback

The `grid @ chol` matmul (65536×8 × 8×8) is recomputed inside the compiled function per block per sweep. The extra cost is negligible (~45ms total) compared to the memory cost of caching all 256 metric grids (~512MB).

### 10.3 Why this helps

The main benefit comes from:
1. **Fusing small matmuls with elementwise ops** — the tiny (128,8)×(8,8) matmuls (gram_inv, chol, gram_raw) previously launched separate cuBLAS kernels with ~5-10μs overhead each. Triton generates custom kernels that fuse these with surrounding operations.
2. **Eliminating Python method dispatch** — the old code called `provider.propose()` which went through Python dict cache lookups and conditional checks.
3. **Single compiled graph** — the compiler can optimize register usage and memory access patterns across the entire block step.

### 10.4 Results

Same-process hot timings, same Llama-3.2-1B QKV fixture, `block_rows=128`:

- `vamp-topk-polish`, `polish_sweeps=3`
  - hot rep: `solver_time_s ≈ 6.53` (was `~7.27`)
  - proxy: `46.188` (was `46.192`)
  - polish-only: `6.53 - 3.64 ≈ 2.89s` (was `~3.64s`, **1.26× faster**)
- `vamp`, `polish_sweeps=3`
  - hot rep: `solver_time_s ≈ 6.15` (was `~7.45`)
  - proxy: `48.771` (was `48.776`)
  - polish-only: `6.15 - 3.64 ≈ 2.51s` (was `~3.82s`, **1.52× faster**)
- `vamp`, `polish_sweeps=0`
  - hot rep: `solver_time_s ≈ 3.64` (unchanged)

The full CD path benefits more because its original proposal had no compilation at all, while the topk path already had `_topk_metric_select` compiled.

Location:

- [mpq/polish/block_cd.py](mpq/polish/block_cd.py)


## 11. Safe Next Steps

The remaining polish time (~2.5-2.9s) is inherently limited by the serial Gauss-Seidel structure. Further speedups would likely require:

1. **Jacobi-style parallel sweeps** — process all blocks with stale rhs, accept/reject with stale error, recompute residual at end of sweep. This changes algorithm semantics but could vectorize across all 256 blocks within a sweep.

2. **Further increasing `block_rows`** — block_rows=256 gives ~5.9s total. Quality impact remains small. Could be auto-tuned based on available GPU memory.

3. **Reducing sweeps** — if 2 sweeps gives acceptable quality, that directly saves ~33% of polish time.

4. **`max_blocks_per_sweep`** — only polish the most important blocks per sweep. Already implemented as a config knob.

Things that should NOT be repeated:

- Heavy `torch.profiler` traces (risk of system lockup)
- Compiling `_batched_greedy_gain` (no end-to-end gain)
- Micro-optimizing individual tensor allocations (dominated by kernel launch overhead)


## 12. Numbers To Remember

The most important verified numbers are:

- Eager no-polish VAMP: `~18.6s`
- Compiled-prior no-polish VAMP, hot: `~3.6s`
- Compiled-prior VAMP + topk block CD (compiled step, block_rows=128), hot: `~6.5s`
- Compiled-prior VAMP + block CD (compiled step, block_rows=128), hot: `~6.2s`
- Compiled-prior VAMP + topk block CD (block_rows=128, no compiled step): `~7.3s`
- Compiled-prior VAMP + block CD (block_rows=128, no compiled step): `~7.5s`
- Compiled-prior VAMP + topk block CD (block_rows=32, old code): `~18.6s`

Overall speedup from all optimizations: **2.85× on VAMP+topk-polish** (18.6s → 6.5s).

