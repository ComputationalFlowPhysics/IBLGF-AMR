"""Tests for upper-half circulation and strong-vorticity center tracking."""

import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from upper_half_circulation import measure_frame, save_combined_plots  # noqa: E402


class UpperHalfCirculationTests(unittest.TestCase):
    def test_signed_integral_and_center_use_separate_domains(self):
        frame = {
            "cells": {
                "x": np.asarray([0.0, 2.0, 4.0, 6.0]),
                "y": np.asarray([0.5, 0.5, -0.5, 0.0]),
                "vorticity": np.asarray([10.0, -2.0, 100.0, 6.0]),
                "area": np.ones(4),
            }
        }

        result = measure_frame(frame, 0.4)

        # The y=0 cell contributes its upper half. The strong lower vortex is ignored.
        self.assertAlmostEqual(result["circulation_upper_half"], 10.0 - 2.0 + 3.0)
        self.assertAlmostEqual(result["peak_upper_vorticity"], 10.0)
        self.assertAlmostEqual(result["center_threshold_vorticity"], 4.0)
        self.assertEqual(result["center_cells"], 2)
        self.assertAlmostEqual(result["center_area"], 1.5)
        self.assertAlmostEqual(result["center_vorticity_weight"], 13.0)
        self.assertAlmostEqual(result["x_center"], 18.0 / 13.0)
        self.assertAlmostEqual(result["y_center"], (5.0 + 0.75) / 13.0)

    def test_zero_vorticity_has_zero_circulation_and_no_center(self):
        frame = {
            "cells": {
                "x": np.asarray([0.0, 1.0]),
                "y": np.asarray([0.5, -0.5]),
                "vorticity": np.zeros(2),
                "area": np.ones(2),
            }
        }

        result = measure_frame(frame, 0.4)

        self.assertEqual(result["circulation_upper_half"], 0.0)
        self.assertFalse(result["center_found"])
        self.assertTrue(math.isnan(result["x_center"]))

    def test_all_combined_plots_are_created(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_folder = Path(temporary_directory)
            run_info = [{"name": "tau_1p0", "tau": 1.0}]
            rows = {
                "tau_1p0": [
                    {"time": 0.0, "circulation_upper_half": 0.0, "x_center": math.nan},
                    {"time": 1.0, "circulation_upper_half": 1.0, "x_center": 2.0},
                ]
            }

            save_combined_plots(output_folder, run_info, rows, (5.0, 3.0))

            expected = (
                "combined_upper_half_circulation_vs_time.png",
                "combined_upper_half_circulation_vs_time_over_tau.png",
                "combined_upper_half_center_x_vs_time.png",
                "combined_upper_half_center_x_vs_time_over_tau.png",
            )
            for filename in expected:
                self.assertGreater((output_folder / filename).stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
