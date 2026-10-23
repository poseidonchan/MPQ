from mpq import QuantizationConfig, quantize


result = quantize(
    "Qwen/Qwen3-0.6B",
    "quantized/qwen3-0.6b-mpq-4bit",
    QuantizationConfig(bits=4),
)

print(result["paths"]["native_dir"])
