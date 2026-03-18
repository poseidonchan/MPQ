# Benchmark Matrix

This is the intended public matrix for benchmark and ablation runs.

## Normal grid: GPTQModel

| backend | codebook | preprocess | rotation | solver | polish_mode | notes |
| --- | --- | --- | --- | --- | --- | --- |
| `gptqmodel` | `grid_g128` | `none` | `none`, `hadamard`, `random` | `gptq` | `none` | plain GPTQ |
| `gptqmodel` | `grid_g128` | `none` | `none`, `hadamard`, `random` | `gptq` | `block_cd` | GPTQ init + full-space block CD |
| `gptqmodel` | `grid_g128` | `none` | `none`, `hadamard`, `random` | `gptq` | `topk_block_cd` | GPTQ init + aux-VAMP top-k CD |
| `gptqmodel` | `grid_g128` | `none` | `none`, `hadamard`, `random` | `vamp` | `none` | scalar VAMP |
| `gptqmodel` | `grid_g128` | `none` | `none`, `hadamard`, `random` | `vamp` | `block_cd` | VAMP + block CD |
| `gptqmodel` | `grid_g128` | `none` | `none`, `hadamard`, `random` | `vamp` | `topk_block_cd` | VAMP + top-k block CD |
| `gptqmodel` | `grid_g128` | `awq` | `none`, `hadamard`, `random` | `gptq`, `vamp` | `none`, `block_cd`, `topk_block_cd` | AWQ is GPTQModel-only |

## E8P12: QUIP#

| backend | codebook | preprocess | rotation | solver | polish_mode | notes |
| --- | --- | --- | --- | --- | --- | --- |
| `quip` | `e8p12` | `none` | `had`, `kron` | `ldlq` | `none` | native LDLQ no tune |
| `quip` | `e8p12` | `none` | `had`, `kron` | `ldlq` | `block_cd` | LDLQ init + block CD |
| `quip` | `e8p12` | `none` | `had`, `kron` | `ldlq` | `topk_block_cd` | LDLQ init + aux-VAMP top-k CD |
| `quip` | `e8p12` | `none` | `had`, `kron` | `vamp` | `none` | scalar VAMP |
| `quip` | `e8p12` | `none` | `had`, `kron` | `vamp` | `block_cd` | VAMP + block CD |
| `quip` | `e8p12` | `none` | `had`, `kron` | `vamp` | `topk_block_cd` | VAMP + top-k block CD |
| `quip` | `e8p12` | `none` | `had`, `kron` | `ldlq_tune10` | `none` | default QUIP# baseline |

## Notes

- `topk_block_cd` always uses `topk_source=aux_vamp`.
- `solver=ldlq_tune10` is only valid with `polish_mode=none`.
- `preprocess=awq` is only valid on `backend=gptqmodel`.
- The benchmark and ablation CLIs validate these combinations through `mpq/combo.py`.
