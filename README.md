# MPQ: A Message-Passing View of Post-Training Quantization

MPQ is a post-training quantization toolkit for dense decoder-only language
models. It uses a message-passing solver followed by coordinate-descent
refinement and provides two deployment paths:

| Weight format | Backend | Typical use |
| --- | --- | --- |
| 3-bit or 4-bit grouped scalar | GPTQModel | broadly supported Hugging Face causal LMs |
| 2-bit E8P12 lattice | QUIP# | aggressive low-bit compression |

The public interface chooses a validated recipe from the requested bit width,
so quantizing a model does not require configuring the solver internals.

## Requirements

- Linux with an NVIDIA GPU and a compatible driver
- Git and conda
- Python 3.10 or newer available as `python` for repository bootstrap

The installer creates isolated Python 3.11 environments and installs the CUDA
toolkit used to build the backend extensions.

## Quick Start

Clone the pinned backends and install the one you need:

```bash
git clone https://github.com/poseidonchan/MPQ.git
cd MPQ

# 3-bit and 4-bit scalar quantization
bash install.sh --gptq-only
conda activate mpq-gptq
```

Quantize a Hugging Face model or a local checkpoint:

```bash
mpq quantize Qwen/Qwen3-0.6B \
  --bits 4 \
  --output-dir quantized/qwen3-0.6b-mpq-4bit
```

For 2-bit E8P12, install and activate the QUIP# environment instead:

```bash
bash install.sh --quip-only
conda activate mpq-quip

mpq quantize meta-llama/Llama-3.2-1B \
  --bits 2 \
  --output-dir quantized/llama-3.2-1b-mpq-e8p12
```

Use `--dry-run` to inspect the fully resolved backend command without loading a
model or requiring a GPU:

```bash
mpq quantize Qwen/Qwen3-0.6B --bits 4 --output-dir /tmp/mpq --dry-run
```

## Default Recipes

`mpq quantize` uses quality-oriented defaults:

| Setting | 3-bit / 4-bit scalar | 2-bit E8P12 |
| --- | --- | --- |
| Preprocessing | AWQ scaling | none |
| Rotation | internal Hadamard | randomized Hadamard |
| Solver | VAMP | VAMP |
| Initialization | RTN | RTN |
| Refinement | 3 block-CD sweeps | 3 block-CD sweeps |
| Calibration | 128 C4 sequences of length 2048 | same |

Common controls remain available directly:

```bash
mpq quantize meta-llama/Meta-Llama-3-8B \
  --bits 3 \
  --output-dir quantized/llama3-8b-mpq-3bit \
  --calibration-samples 128 \
  --calibration-seqlen 2048 \
  --seed 0
```

Evaluation is opt-in so a normal quantization run does not spend additional
time on WikiText-2 and C4. Add `--eval` to report perplexity after saving the
model.

Run `mpq quantize --help` for the stable public options. Backend-specific
options can be appended after `--`:

```bash
mpq quantize Qwen/Qwen3-0.6B \
  --bits 4 \
  --output-dir quantized/qwen3-0.6b \
  --eval \
  -- --device cuda:1
```

## Python API

The same recipes are available from Python:

```python
from mpq import QuantizationConfig, quantize

result = quantize(
    "Qwen/Qwen3-0.6B",
    "quantized/qwen3-0.6b-mpq-4bit",
    QuantizationConfig(bits=4),
)

print(result["paths"]["native_dir"])
```

See [`examples/quantize.py`](examples/quantize.py) for a complete minimal
example.

## Output Layout

Every run writes `backend_results.json` with the resolved recipe, calibration
fingerprint, timing, evaluation metrics, and artifact paths.

Scalar runs save one directly loadable GPTQModel artifact:

```text
quantized/my-model/
├── backend_results.json
└── quantized/   # packed GPTQModel artifact
```

E8P12 runs instead use `ckpt/` for the native QUIP# checkpoints and `hf/` for
the hfized QUIP# deployment artifact. The normalized locations are always
recorded as `paths.hf_dir` and `paths.native_dir` in the result JSON. A dense,
dequantized scalar copy is optional via `-- --export-dense-hf`; it is disabled
by default because it is substantially larger than the packed checkpoint.

Load a scalar artifact with GPTQModel:

```python
from gptqmodel import GPTQModel

model = GPTQModel.load("quantized/my-model/quantized", device="cuda:0")
```

Use the GPTQModel environment prepared by `install.sh`. MPQ stores standard
packed `qweight`/`qzeros`/`scales` tensors plus input-rotation metadata; the
small MPQ compatibility patch restores those rotations when the checkpoint is
loaded. Runtimes that ignore this metadata are not currently supported for the
default internal-Hadamard recipe.

Re-evaluate the saved artifact, including a fresh GPTQModel load, with:

```bash
python scripts/eval_hf_ppl.py \
  --hf_path quantized/my-model/quantized \
  --gptqmodel
```

Load an hfized E8P12 artifact with the QUIP# runtime:

```python
import sys

sys.path.insert(0, "third_party/quip-sharp")
from lib.utils.unsafe_import import model_from_hf_path

model, tokenizer_source = model_from_hf_path(
    "quantized/my-model/hf",
    use_cuda_graph=False,
    device_map="cuda:0",
)
```

The `ckpt/` directory remains the native QUIP# checkpoint used by its
quantization and conversion tools; `hf/` is the directly loadable deployment
artifact.

## Installation Details

MPQ keeps GPTQModel and QUIP# in separate conda environments because their
Torch and Transformers stacks differ:

| Environment | Formats |
| --- | --- |
| `mpq-gptq` | 3-bit and 4-bit grouped scalar |
| `mpq-quip` | 2-bit E8P12 |

Install both with `bash install.sh`, or select one with `--gptq-only` or
`--quip-only`. The installer clones the backends at pinned revisions, verifies
and applies the compatibility patches, installs MPQ in editable mode, and
builds the CUDA extensions.

Useful build overrides:

```bash
ENV_NAME=mpq-gptq-test \
CUDA_TOOLKIT_VERSION=12.8 \
TORCH_CUDA_ARCH_LIST='8.0;8.9;9.0' \
bash install.sh --gptq-only
```

When using a custom environment name, pass it to the CLI with
`--backend-env mpq-gptq-test` or set `MPQ_GPTQ_ENV`. The E8P12 equivalent is
`MPQ_QUIP_ENV`.

Verify an installation with:

```bash
mpq doctor
```

More details are in [`docs/quickstart.md`](docs/quickstart.md).

## Model Support

The public path targets dense decoder-only Hugging Face causal LMs. Llama,
Qwen, and Mistral-style architectures are the primary validated families.
Mixture-of-experts, multimodal, encoder-decoder, and arbitrary remote-code
architectures are currently outside the supported surface.

## Repository Layout

```text
mpq/                 solver, priors, refinement, and backend adapters
scripts/             backend launchers and environment setup
patches/             thin integrations against pinned upstream checkouts
third_party/         pinned backends downloaded by the installer
examples/            user-facing quantization examples
tests/               GPU-free public-interface tests
docs/                installation and implementation notes
```

Algorithm code stays under `mpq/`. Third-party repositories are pinned to
explicit commits and modified only through `patches/`.

## Development

The public-interface test suite does not require a GPU:

```bash
python -m unittest discover -s tests -v
python -m compileall -q mpq tests examples
```

Before changing a backend integration, check patch health:

```bash
python patches/apply_patches.py --check
```

## License

Original MPQ code is released under the Apache License 2.0. The installer
downloads independently licensed backends: GPTQModel (Apache-2.0),
fast-hadamard-transform (BSD-3-Clause), and QUIP# (GPL-3.0). The QUIP#
compatibility patch is provided for use with that GPL-licensed backend. See
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) before redistributing a
combined installation.
