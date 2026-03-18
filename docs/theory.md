# Theory notes (MPQ)

MPQ solves a discrete weight quantization problem by combining:

1. A **message passing / VAMP**-style iteration in a rank-space representation of the quadratic surrogate.
2. A **greedy block-coordinate polish** step that guarantees monotone decrease under the same surrogate.

We unify GPTQ-style scalar quantization and QUIP# E8P12 block codebooks by:

- Sharing the same rank-space objective `||Y_u - X Vs||^2`.
- Sharing the same linear exact-MMSE step in rank space.
- Sharing one polish engine (`mpq/polish/block_cd.py`), parameterized by block partition and candidate provider.

