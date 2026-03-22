# Benchmark Matrix

This is the intended public matrix for benchmark runs.

Model scope for both backend families:
- dense decoder-only HuggingFace causal LMs
- validated target families currently include Llama-like and Qwen-like stacks
- unsupported families should fail early through adapter/model-map checks

## Normal grid: GPTQModel

### Current scalar-grid PPL replication config

The current replicated `3-bit` / `4-bit` Qwen scalar-grid PPL experiments use a fairness-aligned matrix:

- `rotation=internal_hadamard` for both `gptq` and `vamp`
- `preprocess in {none, awq}`
- `solver in {gptq, vamp}`
- `polish_mode in {none, block_cd}`
- `topk_block_cd` is not part of the scalar public matrix
- PPL protocol:
  - `Wikitext2`: full test split
  - `C4`: fixed `256 * seqlen` validation tokens

This is the matrix used by the current `qwen3_*_internalhad_evalfix` runs.

| backend | codebook | preprocess | rotation | solver | polish_mode | notes |
| --- | --- | --- | --- | --- | --- | --- |
| `gptqmodel` | `grid_g128` | `none` | `internal_hadamard` | `gptq` | `none` | GPTQ baseline under the same internal rotation family as VAMP |
| `gptqmodel` | `grid_g128` | `none` | `internal_hadamard` | `gptq` | `block_cd` | GPTQ + block CD under internal rotation |
| `gptqmodel` | `grid_g128` | `none` | `internal_hadamard` | `vamp` | `none` | VAMP baseline (MPQ internal right-side block Hadamard) |
| `gptqmodel` | `grid_g128` | `none` | `internal_hadamard` | `vamp` | `block_cd` | VAMP + block CD |
| `gptqmodel` | `grid_g128` | `awq` | `internal_hadamard` | `gptq` | `none` | AWQ + GPTQ under internal rotation |
| `gptqmodel` | `grid_g128` | `awq` | `internal_hadamard` | `gptq` | `block_cd` | AWQ + GPTQ + block CD under internal rotation |
| `gptqmodel` | `grid_g128` | `awq` | `internal_hadamard` | `vamp` | `none` | AWQ + VAMP |
| `gptqmodel` | `grid_g128` | `awq` | `internal_hadamard` | `vamp` | `block_cd` | AWQ + VAMP + block CD |

### Reproduction command

Example `Qwen3-4B / 4-bit` scalar-grid run:

```bash
conda run -n mpq-gptq python scripts/benchmark.py \
  --bits 4 \
  --model-id Qwen/Qwen3-4B \
  --seeds 0 \
  --quant-preprocesses none awq \
  --quant-rotations internal_hadamard \
  --quant-solvers gptq vamp \
  --quant-polish-modes none block_cd \
  --run-id qwen3_4b_b4_internalhad_evalfix \
  --resume-existing
```

## E8P12: QUIP#

| backend | codebook | preprocess | rotation | solver | polish_mode | notes |
| --- | --- | --- | --- | --- | --- | --- |
| `quip` | `e8p12` | `none` | `had` | `ldlq` | `none` | native LDLQ no tune |
| `quip` | `e8p12` | `none` | `had` | `ldlq` | `block_cd` | LDLQ init + block CD |
| `quip` | `e8p12` | `none` | `had` | `ldlq` | `topk_block_cd` | LDLQ init + aux-VAMP top-k CD |
| `quip` | `e8p12` | `none` | `had` | `vamp` | `none` | VAMP baseline |
| `quip` | `e8p12` | `none` | `had` | `vamp` | `block_cd` | VAMP + block CD |
| `quip` | `e8p12` | `none` | `had` | `vamp` | `topk_block_cd` | VAMP + top-k block CD |
| `quip` | `e8p12` | `none` | `had` | `ldlq_tune10` | `none` | default QUIP# baseline |

## Notes

- Default PPL eval protocol is shared across both backends:
  - `Wikitext2`: full test split
  - `C4`: fixed `256 * seqlen` validation tokens
  - set `--eval-max-eval-tokens N` only for an explicit token-capped ablation
- `topk_block_cd` remains available for manual/debug runs, but the public benchmark matrix only uses it on `e8p12`.
- `topk_block_cd` always uses `topk_source=aux_vamp`.
- `solver=ldlq_tune10` is only valid with `polish_mode=none`.
- `preprocess=awq` is only valid on `backend=gptqmodel`.
- `desc_act=True` is automatic for `solver=gptq`, `False` for `solver=vamp`.
- GPTQModel rotations are explicit:
  - `model_hadamard`: GPTQModel whole-model randomized Hadamard rotation
  - `model_random`: GPTQModel whole-model dense random orthogonal rotation
  - `internal_hadamard`: MPQ/VAMP internal right-side block Hadamard
- `gptq + internal_hadamard` currently runs as a dense path (`packed=false`) and is the scalar-grid comparison path used in the current Qwen PPL runs.
- Other rotations (`none`, `model_random`, `kron`) are available via `--quant-rotations` for ablation but not in the default matrix.
- The benchmark and ablation CLIs validate these combinations through `mpq/combo.py`.
