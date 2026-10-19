"""Unit tests for the ParaView-independent 3D plotting helpers."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "visualization"))

import argparse
import tempfile
import unittest
from unittest.mock import Mock, patch

from plot_3d import (
    assigned_snapshots,
    build_vorticity_threshold,
    build_q_criterion_threshold,
    parse_args,
    frame_manifest_row,
    output_paths,
    read_worker_rows,
    threshold_fraction,
    worker_manifest_folder,
    worker_manifest_path,
    write_manifest,
)


class Plot3DHelperTests(unittest.TestCase):
    def test_contour_cli_and_separate_default_output(self):
        with patch.object(sys, "argv", ["plot_3d.py", "run", "--contour"]):
            self.assertTrue(parse_args().contour)
        original = output_paths(Path("run/output"), "vorticity")
        contour = output_paths(Path("run/output"), "vorticity", contour=True)
        self.assertNotEqual(original[0], contour[0])
        self.assertNotEqual(original[2], contour[2])

    def test_vorticity_contour_uses_requested_fraction(self):
        simple = Mock()
        surface, proxies = build_vorticity_threshold(simple, object(), 0.2, contour=True)
        self.assertIs(surface, simple.Contour.return_value)
        self.assertEqual(surface.ContourBy, ["POINTS", "Normalized Vorticity"])
        self.assertEqual(surface.Isosurfaces, [0.2])
        self.assertEqual(proxies, [surface, simple.PythonCalculator.return_value])
        simple.Threshold.assert_not_called()

    def test_threshold_remains_default(self):
        simple = Mock()
        surface, _ = build_vorticity_threshold(simple, object(), 0.2)
        self.assertIs(surface, simple.Threshold.return_value)
        self.assertEqual(surface.UpperThreshold, 0.2)
        simple.Contour.assert_not_called()

    def test_q_contour_preserves_threshold_cutoff(self):
        simple = Mock()
        threshold, _ = build_q_criterion_threshold(simple, object())
        surface, _ = build_q_criterion_threshold(simple, object(), contour=True)
        self.assertEqual(surface.Isosurfaces, [threshold.UpperThreshold])
        self.assertEqual(surface.ContourBy, ["POINTS", "Q Criterion"])

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

    def test_frame_workers_are_disjoint_and_preserve_global_order(self):
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

    def test_frame_worker_index_must_be_in_range(self):
        with self.assertRaises(ValueError):
            assigned_snapshots([Path("flowTime_0.hdf5")], 2, 2)

    def test_private_worker_manifests_merge_by_snapshot_step(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_folder = Path(temporary_directory)
            worker_manifest_folder(output_folder).mkdir()
            snapshots = [
                output_folder / "flowTime_0.hdf5",
                output_folder / "flowTime_10.hdf5",
            ]
            for worker_index, snapshot in enumerate(snapshots):
                row = frame_manifest_row(
                    worker_index,
                    snapshot,
                    output_folder / f"flowTime_{worker_index * 10}.png",
                    100 + worker_index,
                    200 + worker_index,
                    True,
                )
                write_manifest(
                    worker_manifest_path(output_folder, worker_index),
                    [row],
                )

            rows = read_worker_rows(output_folder, 2)

            self.assertEqual(sorted(rows), [0, 10])
            self.assertEqual(rows[10]["source_cells"], "101")


if __name__ == "__main__":
    unittest.main()
