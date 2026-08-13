"""Unit tests for the ParaView-independent resolution comparison helpers."""

import csv
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import plot_resolution_comparison as comparison


class ResolutionComparisonTests(unittest.TestCase):
    def make_run(self, root, name, dx_base, levels, tau=None):
        run_folder = root / name
        (run_folder / "output").mkdir(parents=True)
        tau_line = "" if tau is None else f"    b_f_tau={tau};\n"
        (run_folder / f"{name}.cfg").write_text(
            "simulation_parameters\n"
            "{\n"
            f"    nLevels={levels};\n"
            "    cfl=0.1;\n"
            f"{tau_line}"
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
                    "vorticity_threshold_fraction",
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
                    "vorticity_threshold_fraction": "0.02",
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
            self.assertEqual(dataset["vorticity_threshold_fraction"], 0.02)

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

    def test_rejects_mixed_vorticity_thresholds(self):
        datasets = [
            {"name": "one", "vorticity_threshold_fraction": 0.02},
            {"name": "two", "vorticity_threshold_fraction": 0.03},
        ]

        with self.assertRaisesRegex(ValueError, "different vorticity threshold"):
            comparison.validate_vorticity_thresholds(datasets)

    def test_tau_legend_is_numeric_and_uses_config_value(self):
        datasets = [
            {
                "name": "tau_11p0",
                "forcing_end_time": 11.0,
                "config_path": Path("tau_11p0.cfg"),
            },
            {
                "name": "tau_3p0",
                "forcing_end_time": 3.0,
                "config_path": Path("tau_3p0.cfg"),
            },
        ]

        title = comparison.prepare_datasets(datasets, "tau")

        self.assertEqual(title, "3D formation-time comparison")
        self.assertEqual([dataset["name"] for dataset in datasets], ["tau_3p0", "tau_11p0"])
        self.assertEqual([dataset["legend_label"] for dataset in datasets], [r"$\tau=3$", r"$\tau=11$"])

    def test_tau_legend_requires_positive_config_value(self):
        datasets = [
            {
                "name": "tau_missing",
                "forcing_end_time": None,
                "config_path": Path("tau_missing.cfg"),
            }
        ]

        with self.assertRaisesRegex(ValueError, "positive b_f_tau"):
            comparison.prepare_datasets(datasets, "tau")

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
