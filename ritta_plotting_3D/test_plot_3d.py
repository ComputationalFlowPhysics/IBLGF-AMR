"""Unit tests for the ParaView-independent 3D plotting helpers."""

import argparse
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_3d import output_paths, threshold_fraction


class Plot3DHelperTests(unittest.TestCase):
    def test_threshold_fraction_bounds(self):
        self.assertEqual(threshold_fraction("0.02"), 0.02)
        for invalid_value in ("0", "-0.1", "1.1"):
            with self.assertRaises(argparse.ArgumentTypeError):
                threshold_fraction(invalid_value)

    def test_output_directory_override_is_used(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            folder = Path(temporary_directory)
            snapshot_folder = folder / "tau_1p0" / "output"
            requested_output = folder / "formation_new_tau_1p0_vorticity_0p02"

            paths = output_paths(
                snapshot_folder,
                "vorticity",
                requested_output,
            )

            self.assertEqual(paths[0], requested_output.resolve())
            self.assertEqual(
                paths[2],
                requested_output.resolve() / "tau_1p0_vorticity.gif",
            )


if __name__ == "__main__":
    unittest.main()
