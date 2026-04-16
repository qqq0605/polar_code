from __future__ import annotations

import math
import sys
from pathlib import Path
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from polar_code.Bhatta import (
    estimate_bhattacharyya_parameters_bsc,
    select_information_bits_monte_carlo,
)


class MonteCarloBhattaTestCase(unittest.TestCase):
    def test_invalid_samples_raise(self) -> None:
        with self.assertRaises(ValueError):
            estimate_bhattacharyya_parameters_bsc(
                block_length=8,
                crossover_probability=0.1,
                samples=0,
            )

    def test_zero_noise_estimates_are_zero(self) -> None:
        estimates = estimate_bhattacharyya_parameters_bsc(
            block_length=8,
            crossover_probability=0.0,
            samples=8,
            seed=11,
        )
        self.assertEqual(estimates, (0.0,) * 8)

    def test_length_one_matches_bsc_bhattacharyya_parameter(self) -> None:
        crossover_probability = 0.11
        estimate = estimate_bhattacharyya_parameters_bsc(
            block_length=1,
            crossover_probability=crossover_probability,
            samples=20_000,
            seed=7,
        )[0]
        expected = 2.0 * math.sqrt(crossover_probability * (1.0 - crossover_probability))
        self.assertAlmostEqual(estimate, expected, delta=0.03)

    def test_information_set_shape_matches_request(self) -> None:
        info_set = select_information_bits_monte_carlo(
            block_length=8,
            message_length=4,
            crossover_probability=0.11,
            samples=2_000,
            seed=7,
        )
        self.assertEqual(len(info_set), 4)
        self.assertEqual(info_set, tuple(sorted(info_set)))
        self.assertEqual(len(set(info_set)), 4)


if __name__ == "__main__":
    unittest.main()
