from __future__ import annotations

import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from polar_code.plots import generate_default_plots


class PolarCodePlotsTestCase(unittest.TestCase):
    def test_generate_default_plots_creates_csv_and_svg(self) -> None:
        with TemporaryDirectory() as temp_dir:
            outputs = generate_default_plots(
                temp_dir,
                trials=10,
                seed=3,
                crossover_probability=0.0,
                block_lengths=(8, 16),
                code_rate=0.5,
                rate_block_length=16,
                message_lengths=(4, 8),
            )

            for path in outputs.values():
                self.assertTrue(path.exists(), f"Expected output file {path} to exist")
                self.assertGreater(path.stat().st_size, 0)

            svg_text = outputs["vs_n_svg"].read_text(encoding="utf-8")
            self.assertIn("BER/FER vs Block Length N", svg_text)
            csv_text = outputs["vs_rate_csv"].read_text(encoding="utf-8")
            self.assertIn("bit_error_rate", csv_text)


if __name__ == "__main__":
    unittest.main()
