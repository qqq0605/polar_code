from __future__ import annotations

import sys
from pathlib import Path
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from polar_code.search import find_maximum_message_length


class MessageLengthSearchTestCase(unittest.TestCase):
    def test_zero_noise_search_returns_full_block_length(self) -> None:
        result = find_maximum_message_length(
            block_length=8,
            crossover_probability=0.0,
            trials=5,
            target_frame_error_rate=0.0,
            seed=7,
        )
        self.assertIsNotNone(result.best_pass)
        self.assertEqual(result.best_pass.message_length, 8)
        self.assertIsNone(result.next_fail)

    def test_invalid_target_frame_error_rate_raises(self) -> None:
        with self.assertRaises(ValueError):
            find_maximum_message_length(
                block_length=8,
                crossover_probability=0.1,
                trials=5,
                target_frame_error_rate=1.1,
            )


if __name__ == "__main__":
    unittest.main()
