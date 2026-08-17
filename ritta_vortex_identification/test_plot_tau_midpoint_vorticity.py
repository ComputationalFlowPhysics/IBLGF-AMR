"""Tests for the selected-tau midpoint vorticity comparison."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from plot_tau_midpoint_vorticity import (
    midpoint_frame_index,
    panel_grid_shape,
    validate_positive_values,
)


class MidpointFrameIndexTests(unittest.TestCase):
    def test_uses_only_frame_for_single_frame_run(self):
        self.assertEqual(midpoint_frame_index(1), 0)

    def test_uses_lower_middle_for_even_frame_count(self):
        self.assertEqual(midpoint_frame_index(126), 62)

    def test_uses_middle_for_odd_frame_count(self):
        self.assertEqual(midpoint_frame_index(127), 63)

    def test_rejects_empty_frame_list(self):
        with self.assertRaisesRegex(ValueError, "At least one"):
            midpoint_frame_index(0)


class ValidatePositiveValuesTests(unittest.TestCase):
    def test_sorts_positive_values(self):
        self.assertEqual(validate_positive_values([8, 0.35, 2], "levels"), (0.35, 2.0, 8.0))

    def test_rejects_duplicate_values(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_positive_values([1, 1.0], "levels")

    def test_rejects_nonpositive_values(self):
        with self.assertRaisesRegex(ValueError, "positive finite"):
            validate_positive_values([0.0, 1.0], "levels")


class PanelGridShapeTests(unittest.TestCase):
    def test_defaults_to_one_horizontal_row(self):
        self.assertEqual(panel_grid_shape(4), (1, 4))

    def test_builds_two_by_two_grid(self):
        self.assertEqual(panel_grid_shape(4, 2), (2, 2))

    def test_caps_columns_at_panel_count(self):
        self.assertEqual(panel_grid_shape(3, 5), (1, 3))

    def test_rejects_nonpositive_columns(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            panel_grid_shape(4, 0)


if __name__ == "__main__":
    unittest.main()
