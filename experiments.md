# MPGQ (VAMP) Experiments — Clean, Reproducible Baselines

Date: 2026-03-15

This repo intentionally keeps a **clean, scalar-precision VAMP** implementation and a
minimal, reproducible QUIP# proxy benchmark. In addition, it includes a **VAMP/VASP-like**
block solver (RS-VAMP / LRS-VASP) for research comparisons. Other experimental solvers
(EP / adaptive accept-reject / best-iterate export) are removed.

## QUIP# E8P12 proxy baseline (deterministic)

### Fixture

- Base model: `meta-llama/Llama-3.2-1B`
- Hessian: `/home/cys/mpgq-unified/runs/20260313_214017_b2/seed0/hess_c4_train/0_qkv.pt`
- Layer / suffix: `0 / qkv`
- Codebook: `E8P12`
- Metric: exact Hessian-weighted surrogate (“proxy”) reported by QUIP#

### Important: QUIP incoherence preprocessing is randomized

QUIP#’s `incoherence_preprocess` samples random Hadamard signs / butterfly factors.
`scripts/bench_quip_module_proxy.py` now seeds torch with `--seed` **right before**
calling `incoherence_preprocess`, so proxy numbers are stable across runs.

### Commands + results (seed=0)

**No polish (~77 proxy)**
```
conda run -n mpgq-quip python scripts/bench_quip_module_proxy.py \
  --base-model meta-llama/Llama-3.2-1B \
  --hessian-path /home/cys/mpgq-unified/runs/20260313_214017_b2/seed0/hess_c4_train/0_qkv.pt \
  --layer 0 --suffix qkv --codebook E8P12 --seed 0 \
  --mpgq-init rtn --mpgq-iters 16 --mpgq-prior-tau 0.25 --mpgq-polish-sweeps 0
```
- LDLQ no-tune proxy: **58.6044**
- LDLQ mean proxy: **48.0162**
- MPGQ (VAMP) mean proxy: **77.0314**

**3-sweep polish (~49 proxy)**
```
conda run -n mpgq-quip python scripts/bench_quip_module_proxy.py \
  --base-model meta-llama/Llama-3.2-1B \
  --hessian-path /home/cys/mpgq-unified/runs/20260313_214017_b2/seed0/hess_c4_train/0_qkv.pt \
  --layer 0 --suffix qkv --codebook E8P12 --seed 0 \
  --mpgq-init rtn --mpgq-iters 16 --mpgq-prior-tau 0.25 --mpgq-polish-sweeps 3
```
- LDLQ mean proxy: **48.0162**
- MPGQ (VAMP + polish) mean proxy: **48.6637**

## RS-VAMP / LRS-VASP (block RS denoiser; non-greedy)

These solvers are **VAMP/VASP-like** (two-module extrinsic recursion). They operate on a
trace-normalized Hessian `Hbar = H / (tr(H)/n)` and use:
- exact finite-codebook RS block posterior moments (E8P12 factorized backend)
- final hard decision = per-block MAP
- optional `polish` = separate rank-space block coordinate descent

### QUIP proxy, seed=0, no polish

**LRS-VASP (auto lambda grid; default `0.25,1.0,4.0`)**
```
conda run -n mpgq-quip python scripts/bench_quip_module_proxy.py \
  --base-model meta-llama/Llama-3.2-1B \
  --hessian-path /home/cys/mpgq-unified/runs/20260313_214017_b2/seed0/hess_c4_train/0_qkv.pt \
  --layer 0 --suffix qkv --codebook E8P12 --seed 0 \
  --mpgq-init rtn --mpgq-method lrs-vasp --mpgq-iters 16 --mpgq-prior-tau 0.25 \
  --mpgq-polish-sweeps 0
```
- MPGQ (LRS-VASP) mean proxy: **74.9640**

**LRS-VASP (fixed `lambda_eff≈58.40`)**
```
conda run -n mpgq-quip python scripts/bench_quip_module_proxy.py \
  --base-model meta-llama/Llama-3.2-1B \
  --hessian-path /home/cys/mpgq-unified/runs/20260313_214017_b2/seed0/hess_c4_train/0_qkv.pt \
  --layer 0 --suffix qkv --codebook E8P12 --seed 0 \
  --mpgq-init rtn --mpgq-method lrs-vasp --mpgq-iters 16 --mpgq-prior-tau 0.25 \
  --mpgq-polish-sweeps 0 --mpgq-lambda-eff-grid 58.4025 --mpgq-lambda-eff-scale-mode none
```
- MPGQ (LRS-VASP) mean proxy: **74.1353**

**RS-VAMP (fixed `lambda_eff=59.0`)**
```
conda run -n mpgq-quip python scripts/bench_quip_module_proxy.py \
  --base-model meta-llama/Llama-3.2-1B \
  --hessian-path /home/cys/mpgq-unified/runs/20260313_214017_b2/seed0/hess_c4_train/0_qkv.pt \
  --layer 0 --suffix qkv --codebook E8P12 --seed 0 \
  --mpgq-init rtn --mpgq-method rs-vamp --mpgq-iters 16 --mpgq-prior-tau 0.25 \
  --mpgq-polish-sweeps 0 --mpgq-lambda-eff-grid 59.0 --mpgq-lambda-eff-scale-mode none
```
- MPGQ (RS-VAMP) mean proxy: **74.3988**

### QUIP proxy, seed=0, 3-sweep polish

With a good `lambda_eff`, all MPGQ methods get close to LDLQ on this fixture:
- LDLQ mean proxy: **48.0162**
- MPGQ (VAMP + polish, default settings): **48.6637**
- MPGQ (VAMP + polish, fixed obs_precision=59, no EM): **48.1284**
- MPGQ (RS-VAMP + polish, fixed lambda_eff=59): **48.1460**
- MPGQ (LRS-VASP + polish, fixed lambda_eff=59): **48.1633**

### Note: lambda grid is resolved once per module (chunk-stable)

For `--mpgq-method rs-vamp|lrs-vasp` with `--mpgq-lambda-eff-scale-mode auto`, the
benchmark now resolves an **absolute** `lambda_eff_grid` once per module before the
row-chunk loop, so row chunking does not change the lambda scale across chunks.

### QUIP proxy with GPTQ-style solver params (iters=10, prior_tau=0.2)

These are the defaults used in `scripts/bench_qwen3_0p6b_3bit.py` (GPTQModel backend),
applied to the QUIP# proxy fixture above.

**No polish**
```
conda run -n mpgq-quip python scripts/bench_quip_module_proxy.py \
  --base-model meta-llama/Llama-3.2-1B \
  --hessian-path /home/cys/mpgq-unified/runs/20260313_214017_b2/seed0/hess_c4_train/0_qkv.pt \
  --layer 0 --suffix qkv --codebook E8P12 --seed 0 \
  --mpgq-init rtn --mpgq-iters 10 --mpgq-prior-tau 0.2 --mpgq-polish-sweeps 0
```
- LDLQ mean proxy: **48.0162**
- MPGQ mean proxy: **78.9062**

**3-sweep polish**
```
conda run -n mpgq-quip python scripts/bench_quip_module_proxy.py \
  --base-model meta-llama/Llama-3.2-1B \
  --hessian-path /home/cys/mpgq-unified/runs/20260313_214017_b2/seed0/hess_c4_train/0_qkv.pt \
  --layer 0 --suffix qkv --codebook E8P12 --seed 0 \
  --mpgq-init rtn --mpgq-iters 10 --mpgq-prior-tau 0.2 --mpgq-polish-sweeps 3
```
- LDLQ mean proxy: **48.0162**
- MPGQ (VAMP + polish) mean proxy: **49.4240**

## Solver notes (what “clean VAMP” means here)

- VAMP iterations use the **MMSE denoiser mean** (`x₁ = E[x|r₁,γ₁]`) and a scalar-precision
  linear MMSE step in rank space.
- The final hard decision is a **MAP** projection to the discrete codebook.
- “Polish” is a separate, explicit **rank-space block coordinate descent** step (Gram or diag metric),
  not part of the VAMP fixed-point equations.

Code pointers:
- QUIP E8P12 scalar VAMP: `mpgq/solver.py` (`solve_e8p12_rankspace_vamp`)
- Rank-space polish: `mpgq/polish/block_cd.py` (`block_coordinate_descent_rankspace`)
- Benchmark driver: `scripts/bench_quip_module_proxy.py`

## Qwen3 0.6B — 3-bit Wikitext2 PPL (GPTQModel backend)

This benchmark runs in-process evaluation (no checkpoint export) using `scripts/backend_gptq.py`.

### Settings (seed=0)

- Model: `Qwen/Qwen3-0.6B`
- Bits / group: `3 / 128`
- Calibration: `c4:train`, `limit=128`, `seqlen=2048`
- Eval: `wikitext2`, `seqlen=2048`, `max_eval_tokens=131072`

Run directory:
- `runs/20260315_181258_qwen3_0p6b_3bit/seed0/results.json`

### Results

- `mpgq_vamp10`: **41.2128**
- `mpgq_vamp10_polish3`: **33.2207**
- `awq_mpgq_vamp10`: **29.2600**
- `awq_mpgq_vamp10_polish3`: **26.1454**
- `gptq_strong_rht_descact`: **44.4748**

### Important: GPTQ checkpoint format vs in-process inference

GPTQModel defaults `QuantizeConfig.format=FORMAT.GPTQ` (v1) for checkpoint interchange. For the
TorchQuantLinear inference path, that v1 setting triggers a v2→v1 `qzeros` conversion during packing,
which makes in-process evaluation wildly incorrect (absurd PPL).

For benchmarking (in-process eval), `scripts/backend_gptq.py` now forces `format=FORMAT.GPTQ_V2` to
keep `qzeros` consistent with TorchQuantLinear inference.
