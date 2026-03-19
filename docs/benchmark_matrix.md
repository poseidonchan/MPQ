# Benchmark Matrix

This is the intended public matrix for benchmark runs.

Model scope for both backend families:
- dense decoder-only HuggingFace causal LMs
- validated target families currently include Llama-like and Qwen-like stacks
- unsupported families should fail early through adapter/model-map checks

## Normal grid: GPTQModel

| backend | codebook | preprocess | rotation | solver | polish_mode | notes |
| --- | --- | --- | --- | --- | --- | --- |
| `gptqmodel` | `grid_g128` | `none` | `hadamard` | `gptq` | `none` | GPTQ baseline (desc_act=True) |
| `gptqmodel` | `grid_g128` | `none` | `hadamard` | `gptq` | `block_cd` | GPTQ + block CD polish |
| `gptqmodel` | `grid_g128` | `none` | `hadamard` | `vamp` | `none` | VAMP baseline |
| `gptqmodel` | `grid_g128` | `none` | `hadamard` | `vamp` | `block_cd` | VAMP + block CD |
| `gptqmodel` | `grid_g128` | `awq` | `hadamard` | `gptq` | `none` | AWQ + GPTQ |
| `gptqmodel` | `grid_g128` | `awq` | `hadamard` | `gptq` | `block_cd` | AWQ + GPTQ + block CD |
| `gptqmodel` | `grid_g128` | `awq` | `hadamard` | `vamp` | `none` | AWQ + VAMP |
| `gptqmodel` | `grid_g128` | `awq` | `hadamard` | `vamp` | `block_cd` | AWQ + VAMP + block CD |

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

- `topk_block_cd` remains available for manual/debug runs, but the public benchmark matrix only uses it on `e8p12`.
- `topk_block_cd` always uses `topk_source=aux_vamp`.
- `solver=ldlq_tune10` is only valid with `polish_mode=none`.
- `preprocess=awq` is only valid on `backend=gptqmodel`.
- `desc_act=True` is automatic for `solver=gptq`, `False` for `solver=vamp`.
- Other rotations (`none`, `random`, `kron`) are available via `--quant-rotations` for ablation but not in the default matrix.
- The benchmark and ablation CLIs validate these combinations through `mpq/combo.py`.
