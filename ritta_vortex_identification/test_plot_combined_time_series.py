"""Tests for filtering combined 2D time-series datasets."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from plot_combined_time_series import select_tau_datasets


class SelectTauDatasetsTests(unittest.TestCase):
    def setUp(self):
        self.datasets = [
            {"name": "tau_1p0", "forcing_end_time": 1.0},
            {"name": "tau_5p5", "forcing_end_time": 5.5},
            {"name": "tau_11p0", "forcing_end_time": 11.0},
            {"name": "tau_45p0", "forcing_end_time": 45.0},
        ]

    def test_selects_only_requested_tau_values_in_dataset_order(self):
        selected = select_tau_datasets(self.datasets, [1, 11, 45])

        self.assertEqual(
            [dataset["forcing_end_time"] for dataset in selected],
            [1.0, 11.0, 45.0],
        )

    def test_none_keeps_every_dataset(self):
        self.assertIs(select_tau_datasets(self.datasets, None), self.datasets)

    def test_rejects_missing_tau_value(self):
        with self.assertRaisesRegex(ValueError, "missing.*20"):
            select_tau_datasets(self.datasets, [1, 20])

    def test_rejects_duplicate_tau_value(self):
        with self.assertRaisesRegex(ValueError, "provided more than once"):
            select_tau_datasets(self.datasets, [11, 11.0])


if __name__ == "__main__":
    unittest.main()
