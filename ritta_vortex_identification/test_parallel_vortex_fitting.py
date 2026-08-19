"""Integration tests for serial/parallel Gaussian fitting and circulation."""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np


SCRIPT_FOLDER = Path(__file__).resolve().parent
FIT_SCRIPT = SCRIPT_FOLDER / "03_fit_vortices.py"
METRICS_SCRIPT = SCRIPT_FOLDER / "04_positive_vortex_metrics.py"


def write_frame(path: Path, omega: np.ndarray) -> None:
    boxes_type = np.dtype([
        ("lo_i", "<i4"),
        ("lo_j", "<i4"),
        ("hi_i", "<i4"),
        ("hi_j", "<i4"),
    ])
    ny, nx = omega.shape
    raw = np.asarray(omega.T[..., np.newaxis]).reshape(-1, order="F")
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


def dipole_field(x_center: float) -> np.ndarray:
    x = np.arange(9, dtype=float) + 0.5
    y = np.arange(9, dtype=float) - 4.0
    x_grid, y_grid = np.meshgrid(x, y)
    sigma = 1.0
    gamma = 10.0
    coefficient = gamma / (2.0 * np.pi * sigma**2)
    positive = np.exp(-((x_grid - x_center) ** 2 + (y_grid - 1.0) ** 2) / 2.0)
    negative = np.exp(-((x_grid - x_center) ** 2 + (y_grid + 1.0) ** 2) / 2.0)
    return coefficient * (positive - negative)


def write_saved_candidates(folder: Path, frame_names: list[str]) -> None:
    string_type = h5py.string_dtype(encoding="utf-8")
    with (
        h5py.File(folder / "hmaxima.h5", "w") as maxima,
        h5py.File(folder / "regions.h5", "w") as regions,
    ):
        group_names = [Path(frame_name).stem for frame_name in frame_names]
        maxima.attrs["schema"] = "ritta_hmaxima_v1"
        regions.attrs["schema"] = "ritta_regions_v1"
        maxima.create_dataset("frame_order", data=group_names, dtype=string_type)
        regions.create_dataset("frame_order", data=group_names, dtype=string_type)
        for index, frame_name in enumerate(frame_names):
            group_name = Path(frame_name).stem
            x_center = 3.5 + 0.1 * index
            maximum = maxima.create_group(group_name)
            maximum.attrs["source_filename"] = frame_name
            maximum.create_dataset("candidate_ids", data=np.asarray([1], dtype=np.int32))
            maximum.create_dataset("peak_x", data=np.asarray([x_center]))
            maximum.create_dataset("peak_y", data=np.asarray([1.0]))
            maximum.create_dataset("peak_vorticity", data=np.asarray([1.4]))

            region = regions.create_group(group_name)
            region.create_dataset("candidate_ids", data=np.asarray([1], dtype=np.int32))
            bounds = np.asarray([[0.0, 8.5, -4.5, 4.5]])
            region.create_dataset("intended_bounds", data=bounds)
            region.create_dataset("clamped_bounds", data=bounds)


def assert_fit_data_equal(test: unittest.TestCase, first: Path, second: Path) -> None:
    with h5py.File(first, "r") as serial, h5py.File(second, "r") as parallel:
        test.assertEqual(list(serial["frame_order"][:]), list(parallel["frame_order"][:]))
        for group_name in ("flowTime_10", "flowTime_30"):
            test.assertEqual(set(serial[group_name]), set(parallel[group_name]))
            for dataset_name in serial[group_name]:
                serial_values = serial[group_name][dataset_name][:]
                parallel_values = parallel[group_name][dataset_name][:]
                if np.issubdtype(serial_values.dtype, np.number):
                    np.testing.assert_allclose(
                        serial_values,
                        parallel_values,
                        equal_nan=True,
                    )
                else:
                    np.testing.assert_array_equal(serial_values, parallel_values)


class ParallelVortexFittingTests(unittest.TestCase):
    def test_parallel_fit_and_metrics_match_serial(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_folder = root / "run"
            output_folder = run_folder / "output"
            output_folder.mkdir(parents=True)
            frame_names = ["flowTime_10.hdf5", "flowTime_30.hdf5"]
            write_frame(output_folder / frame_names[0], dipole_field(3.5))
            write_frame(output_folder / frame_names[1], dipole_field(3.6))

            saved = root / "saved"
            saved.mkdir()
            write_saved_candidates(saved, frame_names)
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
origin_y_index = 4.5

[hmaxima]
connectivity = 8

[region]
alpha_x = 1.0
alpha_r = 1.0
alpha = 0.5
buffer_multiplier = 3.0

[fit]
boundary_fraction = 0.02
soft_l1_scale = 0.1
gamma_min = 0.001
gamma_max = 100.0
d_min = 0.1
d_max = 5.0
sigma_min = 0.1
sigma_max = 3.0
max_nfev = 200
ftol = 1.0e-9
xtol = 1.0e-9
gtol = 1.0e-9

[tracking]
max_displacement = 1.0
new_track_max_displacement = 1.0

[plot]

[time_series]
""".strip(),
                encoding="utf-8",
            )

            environment = os.environ.copy()
            environment["RITTA_VORTEX_RESULT_FOLDER"] = str(root / "unused")
            environment["MPLCONFIGDIR"] = str(root / "matplotlib")
            fit_paths = []
            csv_paths = []
            for workers in (1, 2):
                fit_path = root / f"fits_{workers}.h5"
                completed = subprocess.run(
                    [
                        sys.executable,
                        str(FIT_SCRIPT),
                        str(run_folder),
                        str(config_path),
                        "--input-results-dir",
                        str(saved),
                        "--output-file",
                        str(fit_path),
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
                fit_paths.append(fit_path)

                csv_path = root / f"metrics_{workers}.csv"
                completed = subprocess.run(
                    [
                        sys.executable,
                        str(METRICS_SCRIPT),
                        str(run_folder),
                        str(config_path),
                        "--input-results-dir",
                        str(saved),
                        "--fits-file",
                        str(fit_path),
                        "--output-file",
                        str(csv_path),
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
                csv_paths.append(csv_path)

            assert_fit_data_equal(self, fit_paths[0], fit_paths[1])
            self.assertEqual(csv_paths[0].read_text(), csv_paths[1].read_text())


if __name__ == "__main__":
    unittest.main()
