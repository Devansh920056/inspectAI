import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inspectai.device import mps_available, select_device  # noqa: E402


class SelectDeviceTests(unittest.TestCase):
    def test_cpu_always_works(self):
        self.assertEqual(select_device("cpu").type, "cpu")

    def test_auto_returns_known_device(self):
        self.assertIn(select_device("auto").type, {"mps", "cuda", "cpu"})

    def test_invalid_preference(self):
        with self.assertRaises(ValueError):
            select_device("tpu")

    def test_unavailable_mps_raises(self):
        if mps_available():
            self.skipTest("MPS is available on this machine")
        with self.assertRaises(RuntimeError):
            select_device("mps")


if __name__ == "__main__":
    unittest.main()
