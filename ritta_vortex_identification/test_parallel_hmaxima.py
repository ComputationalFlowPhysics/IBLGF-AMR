"""Integration test for deterministic parallel h-maxima frame processing."""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np


SCRIPT = Path(__file__).with_name("01_find_hmaxima.py")


def write_frame(path: Path, omega: np.ndarray) -> None:
    boxes_type = np.dtype(
        [
            ("lo_i", "<i4"),
            ("lo_j", "<i4"),
            ("hi_i", "<i4"),
            ("hi_j", "<i4"),
        ]
    )
    ny, nx = omega.shape
    chunk = omega.T[..., np.newaxis]
    raw = np.asarray(chunk).reshape(-1, order="F")

    with h5py.File(path, "w") as handle:
        handle.attrs["num_levels"] = 1
        handle.attrs["num_components"] = 1
        handle.attrs["component_0"] = "edge_aux"
        handle.create_group("Chombo_global").attrs["SpaceDim"] = 2
        level = handle.create_group("level_0")
        level.attrs["dx"] = 1.0
        level.create_dataset(
            "boxes",
            data=np.asarray([(0, 0, nx - 1, ny - 1)], dtype=boxes_type),
        )
        level.create_dataset("data:datatype=0", data=raw)
        data_attributes = level.create_group("data_attributes")
        data_attributes.create_dataset("offsets", data=np.asarray([0, len(raw)]))


def assert_hdf5_equal(test: unittest.TestCase, serial_path: Path, parallel_path: Path) -> None:
    with h5py.File(serial_path, "r") as serial, h5py.File(parallel_path, "r") as parallel:
        test.assertEqual(set(serial.keys()), set(parallel.keys()))
        for name in serial:
            if isinstance(serial[name], h5py.Dataset):
                np.testing.assert_array_equal(serial[name][:], parallel[name][:])
                continue
            test.assertEqual(set(serial[name].keys()), set(parallel[name].keys()))
            for dataset_name in serial[name]:
                np.testing.assert_array_equal(
                    serial[name][dataset_name][:],
                    parallel[name][dataset_name][:],
                )
            test.assertEqual(
                dict(serial[name].attrs.items()),
                dict(parallel[name].attrs.items()),
            )


class ParallelHmaximaTests(unittest.TestCase):
    def test_parallel_output_matches_serial_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_folder = root / "run"
            output_folder = run_folder / "output"
            output_folder.mkdir(parents=True)
            write_frame(
                output_folder / "flowTime_10.hdf5",
                np.asarray(
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 2.0, 1.0, 0.0],
                        [0.0, 2.0, 5.0, 2.0, 0.0],
                        [0.0, 1.0, 2.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                    ]
                ),
            )
            write_frame(
                output_folder / "flowTime_30.hdf5",
                np.asarray(
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 3.0, 1.0, 4.0, 0.0],
                        [0.0, 1.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                    ]
                ),
            )
            config_path = root / "analysis.toml"
            config_path.write_text(
                """
[input]
field_name = "edge_aux"
hdf5_glob = "flowTime_*.hdf5"
cfl = 0.1
dx_base = 1.0
num_amr_levels = 0
origin_x_index = 0.0
origin_y_index = 0.0

[hmaxima]
h = 1.0
connectivity = 8
reconstruction_tolerance = 1.0e-10
h_mask_tolerance = 1.0e-8
merge_distance = 0.0

[region]
[fit]
[plot]
[time_series]
""".strip(),
                encoding="utf-8",
            )

            result_paths = []
            for workers in (1, 2):
                result_folder = root / f"results_{workers}"
                environment = os.environ.copy()
                environment["RITTA_VORTEX_RESULT_FOLDER"] = str(result_folder)
                completed = subprocess.run(
                    [
                        sys.executable,
                        str(SCRIPT),
                        str(run_folder),
                        str(config_path),
                        "--workers",
                        str(workers),
                        "--no-preview",
                    ],
                    env=environment,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(
                    completed.returncode,
                    0,
                    msg=f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
                )
                result_paths.append(result_folder / "hmaxima.h5")

            assert_hdf5_equal(self, result_paths[0], result_paths[1])


if __name__ == "__main__":
    unittest.main()
