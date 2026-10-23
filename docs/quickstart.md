# Installation and Usage

This guide covers the supported user path for quantizing a dense Hugging Face
causal language model. Benchmark orchestration and paper experiments are not
part of the public workflow.

Before installation, provide Git, conda, and Python 3.10 or newer on a Linux
machine with an NVIDIA GPU and compatible driver. The setup scripts create the
backend environments and install their CUDA build toolkits.

## 1. Choose a Backend

Use the GPTQModel environment for grouped scalar quantization:

```bash
bash install.sh --gptq-only
conda activate mpq-gptq
```

Use the QUIP# environment for E8P12 quantization:

```bash
bash install.sh --quip-only
conda activate mpq-quip
```

Both environments can coexist. Running `bash install.sh` without a selector
installs both.

For a custom `ENV_NAME`, pass the same name as `--backend-env` when running
`mpq quantize`. `MPQ_GPTQ_ENV` and `MPQ_QUIP_ENV` provide persistent overrides.

## 2. Check the Installation

```bash
mpq doctor
mpq --version
```

`mpq doctor` checks the source checkout, pinned backend checkouts, conda environments,
and local GPU tooling. A login node may report `nvidia-smi` as missing even when
compute nodes provide GPUs; run the actual quantization command inside a GPU
allocation.

## 3. Quantize a Model

### Grouped scalar

```bash
mpq quantize Qwen/Qwen3-0.6B \
  --bits 4 \
  --output-dir quantized/qwen3-0.6b-mpq-4bit
```

The same command supports 3-bit scalar quantization by setting `--bits 3`.

### E8P12

```bash
mpq quantize meta-llama/Llama-3.2-1B \
  --bits 2 \
  --output-dir quantized/llama-3.2-1b-mpq-e8p12
```

The bit width determines the backend automatically. A model argument can be a
Hugging Face id or a local directory containing a standard Transformers model.
The default E8P12 recipe uses the paper configuration: RTN initialization,
16 VAMP iterations, damping 0.5, prior temperature 0.2, and three standard
block-CD sweeps.

Perplexity evaluation is disabled by default. Add `--eval` to evaluate
WikiText-2 and C4 after quantization.

## 4. Inspect and Customize

Print the resolved command before starting a long run:

```bash
mpq quantize meta-llama/Meta-Llama-3-8B \
  --bits 3 \
  --output-dir quantized/llama3-8b \
  --dry-run
```

The stable options cover calibration, initialization, VAMP iterations,
damping, prior temperature, and refinement budget. Use explicit backend
arguments only when reproducing a specialized configuration:

```bash
mpq quantize meta-llama/Meta-Llama-3-8B \
  --bits 3 \
  --output-dir quantized/llama3-8b \
  --eval \
  -- --device cuda:0
```

Arguments after `--` are appended to the backend command and therefore override
earlier duplicate options.

## 5. Consume the Result

The run summary is written to `<output-dir>/backend_results.json`. Read the
normalized artifact locations instead of assuming a backend-specific folder:

```python
import json
from pathlib import Path

result = json.loads(Path("quantized/my-model/backend_results.json").read_text())
print(result["paths"]["hf_dir"])
print(result["paths"]["native_dir"])
```

For 3-bit and 4-bit runs, `native_dir` is loadable with
`GPTQModel.load(...)` from the environment installed by MPQ; that patched
loader restores MPQ's per-linear input rotations. For E8P12 runs, load
`hf_dir` with QUIP#'s `lib.utils.unsafe_import.model_from_hf_path`;
`native_dir` contains the QUIP# layer checkpoints.

## Troubleshooting

### A backend environment is missing

Install only the required environment:

```bash
bash install.sh --gptq-only  # 3-bit / 4-bit
bash install.sh --quip-only  # 2-bit E8P12
```

### Third-party backends are missing

```bash
python patches/apply_patches.py --bootstrap
```

### CUDA extensions target the wrong GPU architecture

Set `TORCH_CUDA_ARCH_LIST` before installation. For example, NVIDIA L40S uses
compute capability 8.9:

```bash
TORCH_CUDA_ARCH_LIST=8.9 bash install.sh --gptq-only
```

### Calibration data must be reused exactly

Use the same `--seed`, `--calibration-samples`, and `--calibration-seqlen`.
Backend result JSON files include a token-level calibration fingerprint for
comparison across runs.
