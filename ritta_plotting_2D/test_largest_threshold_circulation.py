"""Tests for the independent 2D largest-threshold circulation method."""

import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from largest_threshold_circulation import (  # noqa: E402
    largest_enclosed_component,
    measure_frame,
    save_combined_plots,
)


class LargestThresholdCirculationTests(unittest.TestCase):
    def test_largest_boundary_touching_component_is_rejected(self):
        omega = np.zeros((7, 7), dtype=float)
        omega[0, 1:6] = 10.0
        omega[3:5, 3:5] = 5.0

        labels, found, enclosed, selected = largest_enclosed_component(omega, 1.0)

        self.assertEqual(found, 2)
        self.assertEqual(enclosed, 1)
        self.assertEqual(np.count_nonzero(labels == selected), 4)

    def test_integral_uses_original_cell_areas(self):
        omega = np.zeros((6, 6), dtype=float)
        omega[2:4, 2:4] = np.asarray([[2.0, 4.0], [6.0, 8.0]])
        x = np.arange(6, dtype=float) + 0.5
        y = np.arange(6, dtype=float) + 0.5
        cell_x, cell_y = np.meshgrid(x, y)
        areas = np.ones_like(omega)
        areas[2:4, 2:4] = np.asarray([[0.25, 0.5], [1.0, 2.0]])
        frame = {
            "x": x,
            "y": y,
            "dx": 1.0,
            "vorticity": omega,
            "cells": {
                "x": cell_x.ravel(),
                "y": cell_y.ravel(),
                "vorticity": omega.ravel(),
                "area": areas.ravel(),
            },
        }

        result = measure_frame(frame, 0.02)

        expected_circulation = 2.0 * 0.25 + 4.0 * 0.5 + 6.0 + 8.0 * 2.0
        expected_x = (
            2.5 * 2.0 * 0.25
            + 3.5 * 4.0 * 0.5
            + 2.5 * 6.0
            + 3.5 * 8.0 * 2.0
        ) / expected_circulation
        self.assertTrue(result["region_found"])
        self.assertAlmostEqual(result["circulation_positive"], expected_circulation)
        self.assertAlmostEqual(result["x_center_positive"], expected_x)
        self.assertAlmostEqual(result["region_area"], 3.75)
        self.assertAlmostEqual(
            result["equivalent_radius"],
            math.sqrt(3.75 / math.pi),
        )

    def test_combined_plots_are_created(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_folder = Path(temporary_directory)
            run_info = [{"name": "tau_1p0", "tau": 1.0}]
            rows = {
                "tau_1p0": [
                    {"time": 0.0, "circulation_positive": 0.0},
                    {"time": 1.0, "circulation_positive": 1.0},
                ]
            }

            save_combined_plots(output_folder, run_info, rows, (5.0, 3.0))

            self.assertGreater(
                (output_folder / "combined_circulation_vs_time.png").stat().st_size,
                0,
            )
            self.assertGreater(
                (
                    output_folder
                    / "combined_circulation_vs_time_over_tau.png"
                ).stat().st_size,
                0,
            )


if __name__ == "__main__":
    unittest.main()
