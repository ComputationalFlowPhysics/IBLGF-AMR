"""Unit tests for the ParaView-independent circulation helpers."""

import argparse
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_circulation import lamb_center_from_moments, threshold_fraction


class LambCenterTests(unittest.TestCase):
    def test_calculates_axial_and_radial_center(self):
        center_x, center_y = lamb_center_from_moments(-2.0, -8.0, -24.0)

        self.assertEqual(center_x, 3.0)
        self.assertEqual(center_y, 2.0)

    def test_zero_vorticity_returns_nan_coordinates(self):
        center_x, center_y = lamb_center_from_moments(0.0, 2.0, 3.0)

        self.assertTrue(math.isnan(center_x))
        self.assertTrue(math.isnan(center_y))

    def test_inconsistent_moment_signs_return_nan_coordinates(self):
        center_x, center_y = lamb_center_from_moments(2.0, -8.0, -24.0)

        self.assertTrue(math.isnan(center_x))
        self.assertTrue(math.isnan(center_y))

    def test_threshold_fraction_bounds(self):
        self.assertEqual(threshold_fraction("0.4"), 0.4)
        for invalid_value in ("0", "-0.1", "1.1"):
            with self.assertRaises(argparse.ArgumentTypeError):
                threshold_fraction(invalid_value)


if __name__ == "__main__":
    unittest.main()
