from __future__ import annotations

import random
import sys
from pathlib import Path
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from polar_code import PolarCode, bsc_transmit


class PolarCodeTestCase(unittest.TestCase):
    def test_invalid_block_length_raises(self) -> None:
        with self.assertRaises(ValueError):
            PolarCode(block_length=12, message_length=6, crossover_probability=0.1)

    def test_information_set_size_matches_message_length(self) -> None:
        codec = PolarCode(block_length=16, message_length=7, crossover_probability=0.1)
        self.assertEqual(len(codec.info_set), 7)
        self.assertEqual(len(set(codec.info_set)), 7)

    def test_noise_free_round_trip_for_multiple_lengths(self) -> None:
        configurations = [
            (4, 2, 0.11),
            (8, 4, 0.11),
            (16, 8, 0.05),
        ]

        for block_length, message_length, probability in configurations:
            codec = PolarCode(block_length, message_length, probability)
            rng = random.Random(1234 + block_length)
            for _ in range(20):
                message = [rng.randrange(2) for _ in range(message_length)]
                codeword = codec.encode(message)
                decoded = codec.decode(codeword)
                self.assertEqual(tuple(message), decoded.estimated_message)

    def test_bsc_transmit_and_decode_without_flips(self) -> None:
        codec = PolarCode(block_length=8, message_length=4, crossover_probability=0.08)
        rng = random.Random(7)
        message = [1, 0, 1, 1]
        codeword = codec.encode(message)
        received = bsc_transmit(codeword, 0.0, rng=rng)
        decoded = codec.decode(received)
        self.assertEqual(tuple(message), decoded.estimated_message)

    def test_zero_noise_simulation_has_zero_error(self) -> None:
        codec = PolarCode(block_length=8, message_length=4, crossover_probability=0.0)
        result = codec.simulate(trials=25, seed=99)
        self.assertEqual(result.bit_errors, 0)
        self.assertEqual(result.frame_errors, 0)


if __name__ == "__main__":
    unittest.main()
