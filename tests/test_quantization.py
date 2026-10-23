import tempfile
import unittest
from pathlib import Path

from mpq.quantization import QuantizationConfig, build_quantize_command


REPO_ROOT = Path(__file__).resolve().parents[1]


def _value_after(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


class QuantizationCommandTest(unittest.TestCase):
    def test_scalar_recipe_resolves_to_paper_quality_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            command = build_quantize_command(
                "meta-llama/Meta-Llama-3-8B",
                Path(directory) / "model",
                QuantizationConfig(bits=3),
                repo_root=REPO_ROOT,
                python_executable="python",
            )

        self.assertTrue(command[1].endswith("scripts/backend_gptq.py"))
        self.assertEqual(_value_after(command, "--quant-codebook"), "grid_g128")
        self.assertEqual(_value_after(command, "--quant-preprocess"), "awq")
        self.assertEqual(_value_after(command, "--quant-rotation"), "internal_hadamard")
        self.assertEqual(_value_after(command, "--quant-solver"), "vamp")
        self.assertEqual(_value_after(command, "--quant-polish-mode"), "block_cd")
        self.assertEqual(_value_after(command, "--mpq-init"), "rtn")
        self.assertEqual(_value_after(command, "--mpq-iters"), "16")
        self.assertEqual(_value_after(command, "--mpq-prior-tau"), "0.2")
        self.assertIn("--no-sym", command)
        self.assertIn("--skip-eval", command)

    def test_e8p12_recipe_selects_quip_backend(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            command = build_quantize_command(
                "meta-llama/Meta-Llama-3-8B",
                Path(directory) / "model",
                QuantizationConfig(bits=2),
                repo_root=REPO_ROOT,
                python_executable="python",
            )

        self.assertTrue(command[1].endswith("scripts/backend_quip.py"))
        self.assertEqual(_value_after(command, "--quant-codebook"), "e8p12")
        self.assertEqual(_value_after(command, "--quant-preprocess"), "none")
        self.assertEqual(_value_after(command, "--quant-rotation"), "had")
        self.assertEqual(_value_after(command, "--quant-polish-mode"), "block_cd")
        self.assertEqual(_value_after(command, "--mpq-init"), "rtn")
        self.assertEqual(_value_after(command, "--mpq-prior-tau"), "0.2")
        self.assertEqual(_value_after(command, "--batch-size"), "4")
        self.assertEqual(_value_after(command, "--quip-tune-iters"), "0")
        self.assertEqual(_value_after(command, "--mpq-block-rows"), "128")
        self.assertEqual(_value_after(command, "--mpq-blocks-per-chunk"), "16")
        self.assertIn("--mpq-em-obs-precision", command)
        self.assertIn("--mpq-compile-prior", command)
        self.assertNotIn("--quant-topk-k", command)
        self.assertNotIn("--mpq-topk-k", command)
        self.assertIn("--skip-eval", command)

    def test_advanced_backend_args_are_appended(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            command = build_quantize_command(
                "local-model",
                Path(directory) / "model",
                QuantizationConfig(bits=4),
                backend_args=("--device", "cuda:1"),
                repo_root=REPO_ROOT,
                python_executable="python",
            )
        self.assertEqual(command[-2:], ["--device", "cuda:1"])

    def test_unsupported_bits_fail_before_backend_launch(self) -> None:
        for bits in (0, 1, 5, 8):
            with self.subTest(bits=bits):
                with self.assertRaisesRegex(ValueError, "bits must be one of"):
                    QuantizationConfig(bits=bits).resolve()


if __name__ == "__main__":
    unittest.main()
