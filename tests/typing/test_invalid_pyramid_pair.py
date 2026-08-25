import subprocess
import sys
import unittest
from pathlib import Path


class InvalidPyramidPairTypingTest(unittest.TestCase):
    def test_mypy_rejects_single_scale_aggregator_for_pyramid_sample(self) -> None:
        fixture = Path(__file__).with_name("invalid_pyramid_pair.py")
        result = subprocess.run(
            [sys.executable, "-m", "mypy", str(fixture)],
            check=False,
            capture_output=True,
            text=True,
        )

        output = result.stdout + result.stderr
        self.assertNotEqual(result.returncode, 0, output)
        self.assertIn('Argument "aggregator"', output)
        self.assertIn("incompatible type", output)


if __name__ == "__main__":
    unittest.main()
