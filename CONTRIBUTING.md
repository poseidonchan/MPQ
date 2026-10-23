# Contributing

MPQ integrates one solver with two independently versioned quantization
backends. Keep changes within the ownership boundaries below.

## Development Setup

Install the backend relevant to the change:

```bash
bash install.sh --gptq-only
# or
bash install.sh --quip-only
```

The stable user interface is intentionally GPU-free to test:

```bash
python -m unittest discover -s tests -v
python -m compileall -q mpq tests examples
```

## Code Ownership

- Put solver, prior, and refinement logic in `mpq/`.
- Put user-facing commands in `mpq/cli.py` and `mpq/quantization.py`.
- Keep `scripts/backend_gptq.py` and `scripts/backend_quip.py` as backend
  orchestration layers.
- Do not edit `third_party/` directly. Update the corresponding file in
  `patches/` and verify it against the pinned upstream revision.
- Keep paper experiments and result processing outside the public package.

## Before Opening a Pull Request

```bash
python -m unittest discover -s tests -v
python -m compileall -q mpq tests examples
python patches/apply_patches.py --check
git diff --check
```

For backend changes, also run `mpq quantize ... --dry-run` and one small-model
GPU smoke test in the affected environment.
