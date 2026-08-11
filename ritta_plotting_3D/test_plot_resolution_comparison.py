"""Unit tests for the ParaView-independent resolution comparison helpers."""

import csv
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import plot_resolution_comparison as comparison


class ResolutionComparisonTests(unittest.TestCase):
    def make_run(self, root, name, dx_base, levels):
        run_folder = root / name
        (run_folder / "output").mkdir(parents=True)
        (run_folder / f"{name}.cfg").write_text(
            "simulation_parameters\n"
            "{\n"
            f"    nLevels={levels};\n"
            "    cfl=0.1;\n"
            "    domain\n"
            "    {\n"
            f"        dx_base={dx_base};\n"
            "    }\n"
            "}\n",
            encoding="utf-8",
        )
        return run_folder

    def make_csv(self, path, fraction=0.4, snapshot_file=""):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=(
                    "time",
                    "circulation",
                    "center_x",
                    "center_threshold_fraction",
                    "snapshot_file",
                ),
            )
            writer.writeheader()
            writer.writerow(
                {
                    "time": "1.0",
                    "circulation": "0.75",
                    "center_x": "2.5",
                    "center_threshold_fraction": str(fraction),
                    "snapshot_file": snapshot_file,
                }
            )

    def test_reads_resolution_metadata_and_analysis_values(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_folder = self.make_run(root, "amr_2", 0.125, 2)
            csv_path = root / "analysis.csv"
            self.make_csv(csv_path)

            with mock.patch.object(comparison, "analysis_csv_path", return_value=csv_path):
                dataset = comparison.read_dataset(run_folder)

            self.assertEqual(dataset["levels"], 2)
            self.assertEqual(dataset["dx_base"], 0.125)
            self.assertEqual(dataset["dx_finest"], 0.03125)
            self.assertEqual(dataset["rows"][0]["circulation"], 0.75)
            self.assertEqual(dataset["rows"][0]["center_x"], 2.5)

    def test_orders_amr_then_uniform_cases_coarse_to_fine(self):
        datasets = [
            {"name": "res_0p0625", "levels": 0, "dx_base": 0.0625},
            {"name": "amr_2", "levels": 2, "dx_base": 0.125},
            {"name": "res_0p125", "levels": 0, "dx_base": 0.125},
            {"name": "amr_1", "levels": 1, "dx_base": 0.0625},
        ]

        ordered = sorted(datasets, key=comparison.dataset_sort_key)

        self.assertEqual(
            [dataset["name"] for dataset in ordered],
            ["amr_1", "amr_2", "res_0p125", "res_0p0625"],
        )

    def test_rejects_mixed_center_thresholds(self):
        datasets = [
            {"name": "one", "center_threshold_fraction": 0.4},
            {"name": "two", "center_threshold_fraction": 0.3},
        ]

        with self.assertRaisesRegex(ValueError, "different center threshold"):
            comparison.validate_center_thresholds(datasets)

    def test_rejects_csv_from_a_different_same_named_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_folder = self.make_run(root, "amr_1", 0.0625, 1)
            csv_path = root / "analysis.csv"
            self.make_csv(
                csv_path,
                snapshot_file="/another/campaign/amr_1/output/flowTime_32.hdf5",
            )

            with mock.patch.object(comparison, "analysis_csv_path", return_value=csv_path):
                with self.assertRaisesRegex(ValueError, "belongs to a different run"):
                    comparison.read_dataset(run_folder)


if __name__ == "__main__":
    unittest.main()
