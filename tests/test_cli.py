import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from mpq.cli import main


REPO_ROOT = Path(__file__).resolve().parents[1]


class CliTest(unittest.TestCase):
    def _dry_run(self, *, bits: int, model: str) -> tuple[int, str, Path]:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        output_dir = Path(temporary.name) / "model"
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            return_code = main(
                [
                    "quantize",
                    model,
                    "--bits",
                    str(bits),
                    "--output-dir",
                    str(output_dir),
                    "--repo-root",
                    str(REPO_ROOT),
                    "--no-conda",
                    "--dry-run",
                ]
            )
        return return_code, stdout.getvalue(), output_dir

    def test_quantize_dry_run_is_gpu_free(self) -> None:
        return_code, stdout, output_dir = self._dry_run(
            bits=4,
            model="Qwen/Qwen3-0.6B",
        )
        self.assertEqual(return_code, 0)
        self.assertIn("scripts/backend_gptq.py", stdout)
        self.assertIn("--model-id Qwen/Qwen3-0.6B", stdout)
        self.assertFalse(output_dir.exists())

    def test_e8p12_dry_run_uses_quip(self) -> None:
        return_code, stdout, _ = self._dry_run(
            bits=2,
            model="meta-llama/Llama-3.2-1B",
        )
        self.assertEqual(return_code, 0)
        self.assertIn("scripts/backend_quip.py", stdout)
        self.assertIn("--quant-codebook e8p12", stdout)

    def test_doctor_accepts_one_complete_backend(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for script in ("backend_gptq.py", "backend_quip.py"):
                path = root / "scripts" / script
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            for checkout in ("GPTQModel", "fast-hadamard-transform"):
                (root / "third_party" / checkout / ".git").mkdir(parents=True)

            stdout = io.StringIO()
            with (
                mock.patch.dict(os.environ, {"MPQ_GPTQ_ENV": "custom-gptq"}),
                mock.patch(
                    "mpq.cli._conda_env_exists",
                    side_effect=lambda name: name == "custom-gptq",
                ),
                contextlib.redirect_stdout(stdout),
            ):
                return_code = main(["doctor", "--repo-root", str(root)])

        self.assertEqual(return_code, 0)
        self.assertIn("OK      custom-gptq environment", stdout.getvalue())
        self.assertIn("MISSING QUIP# checkout", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
