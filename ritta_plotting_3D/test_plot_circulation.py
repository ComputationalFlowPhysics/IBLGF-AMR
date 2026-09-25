"""Unit tests for the ParaView-independent circulation helpers."""

import argparse
import math
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_circulation import (
    CENTER_MARKER_RADIUS,
    CSV_FIELDNAMES,
    TRANSPARENT_GIF_FILTER,
    assigned_snapshots,
    circulation_row,
    lamb_center_from_moments,
    load_resume_rows,
    output_paths,
    physical_time,
    read_worker_rows,
    threshold_fraction,
    worker_csv_folder,
    worker_csv_path,
    write_csv_rows,
)


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

    def test_center_marker_is_small(self):
        self.assertGreater(CENTER_MARKER_RADIUS, 0.0)
        self.assertLess(CENTER_MARKER_RADIUS, 0.1)

    def test_gif_filter_preserves_transparency(self):
        self.assertIn("reserve_transparent=1", TRANSPARENT_GIF_FILTER)
        self.assertIn("alpha_threshold=128", TRANSPARENT_GIF_FILTER)

    def test_resume_reuses_row_with_matching_nonempty_frame(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            folder = Path(temporary_directory)
            snapshot = folder / "flowTime_32.hdf5"
            frames_folder = folder / "frames"
            frames_folder.mkdir()
            frame_path = frames_folder / "flowTime_32.png"
            frame_path.write_bytes(b"png")

            row = {name: "" for name in CSV_FIELDNAMES}
            row.update(
                {
                    "frame_index": "0",
                    "snapshot_step": "32",
                    "time": "0.1",
                    "center_threshold_fraction": "0.4",
                    "snapshot_file": str(snapshot),
                    "png_file": str(frame_path),
                }
            )
            csv_path = folder / "circulation.csv"
            write_csv_rows(csv_path, [row])

            reusable_rows = load_resume_rows(
                csv_path,
                [snapshot],
                frames_folder,
                cfl=0.1,
                dx_base=0.0625,
                levels=1,
                vorticity_threshold_fraction=0.02,
                center_threshold_fraction=0.4,
            )

            self.assertEqual(reusable_rows[32]["snapshot_step"], "32")

    def test_data_only_resume_does_not_require_a_frame(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            folder = Path(temporary_directory)
            snapshot = folder / "flowTime_32.hdf5"
            frames_folder = folder / "frames"
            frames_folder.mkdir()

            row = {name: "" for name in CSV_FIELDNAMES}
            row.update(
                {
                    "frame_index": "0",
                    "snapshot_step": "32",
                    "time": "0.1",
                    "center_threshold_fraction": "0.4",
                    "snapshot_file": str(snapshot),
                }
            )
            csv_path = folder / "circulation.csv"
            write_csv_rows(csv_path, [row])

            reusable_rows = load_resume_rows(
                csv_path,
                [snapshot],
                frames_folder,
                cfl=0.1,
                dx_base=0.0625,
                levels=1,
                vorticity_threshold_fraction=0.02,
                center_threshold_fraction=0.4,
                require_frame=False,
            )

            self.assertEqual(reusable_rows[32]["snapshot_step"], "32")

    def test_output_directory_override_is_used(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            folder = Path(temporary_directory)
            snapshot_folder = folder / "run" / "output"
            requested_output = folder / "campaign_run_circulation"

            paths = output_paths(
                snapshot_folder,
                view_only=False,
                output_dir=requested_output,
            )

            self.assertEqual(paths[0], requested_output.resolve())
            self.assertEqual(
                paths[3],
                requested_output.resolve() / "leading_vortex_circulation.csv",
            )

    def test_parallel_workers_are_disjoint_and_preserve_global_order(self):
        snapshots = [Path(f"flowTime_{step}.hdf5") for step in range(0, 80, 10)]

        assignments = [
            assigned_snapshots(snapshots, worker_index, 3)
            for worker_index in range(3)
        ]

        flattened = sorted(item for worker in assignments for item in worker)
        self.assertEqual(flattened, list(enumerate(snapshots)))
        self.assertEqual(
            assignments[1],
            [(1, snapshots[1]), (4, snapshots[4]), (7, snapshots[7])],
        )

    def test_parallel_worker_csvs_merge_by_numeric_snapshot_order(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_folder = Path(temporary_directory)
            frames_folder = output_folder / "frames"
            worker_csv_folder(output_folder).mkdir()
            snapshots = [
                output_folder / "flowTime_0.hdf5",
                output_folder / "flowTime_10.hdf5",
                output_folder / "flowTime_20.hdf5",
            ]
            metadata = (0.1, 0.0625, 1)
            thresholds = (0.02, 0.4)
            worker_rows = [[], []]
            for frame_index, snapshot in enumerate(snapshots):
                step = frame_index * 10
                row = circulation_row(
                    frame_index,
                    snapshot,
                    physical_time(step, *metadata),
                    (1.0 + frame_index, 2.0, 3, 2, 4.0, 5.0, 2, 1),
                    *thresholds,
                )
                worker_rows[frame_index % 2].append(row)
            for worker_index, rows in enumerate(worker_rows):
                write_csv_rows(worker_csv_path(output_folder, worker_index), rows)

            rows_by_step = read_worker_rows(
                output_folder,
                2,
                snapshots,
                frames_folder,
                metadata,
                thresholds,
            )

            self.assertEqual(sorted(rows_by_step), [0, 10, 20])
            self.assertEqual(rows_by_step[20]["frame_index"], "2")


if __name__ == "__main__":
    unittest.main()
