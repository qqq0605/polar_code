from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"


class PolarCodeCliTestCase(unittest.TestCase):
    def test_encode_then_decode_round_trip(self) -> None:
        env = os.environ.copy()
        existing_path = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(SRC_DIR) if not existing_path else f"{SRC_DIR}{os.pathsep}{existing_path}"

        encode_command = [
            sys.executable,
            "-m",
            "polar_code",
            "encode",
            "--block-length",
            "8",
            "--message-length",
            "4",
            "--crossover-probability",
            "0.11",
            "--message",
            "1011",
        ]
        encode_result = subprocess.run(
            encode_command,
            cwd=PROJECT_ROOT,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        codeword = None
        for line in encode_result.stdout.splitlines():
            if line.startswith("codeword="):
                codeword = line.split("=", 1)[1].strip()
                break

        self.assertIsNotNone(codeword)

        decode_command = [
            sys.executable,
            "-m",
            "polar_code",
            "decode",
            "--block-length",
            "8",
            "--message-length",
            "4",
            "--crossover-probability",
            "0.11",
            "--received",
            codeword,
        ]
        decode_result = subprocess.run(
            decode_command,
            cwd=PROJECT_ROOT,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("estimated_message=1011", decode_result.stdout)


if __name__ == "__main__":
    unittest.main()
