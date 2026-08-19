"""Tests for upper-half circulation measured on a 3D meridional slice."""

import sys
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ritta_plotting_2D"))
from upper_half_circulation import load_task_frame, measure_frame  # noqa: E402


def write_3d_frame(path):
    box_dtype = [
        ("lo_i", "i4"),
        ("lo_j", "i4"),
        ("lo_k", "i4"),
        ("hi_i", "i4"),
        ("hi_j", "i4"),
        ("hi_k", "i4"),
    ]
    values = np.empty((2, 2, 2, 2), dtype=float)
    values[..., 0] = -1.0
    values[:, :, 0, 1] = np.asarray([[1.0, 3.0], [2.0, 4.0]])
    values[:, :, 1, 1] = 99.0
    with h5py.File(path, "w") as handle:
        handle.attrs["num_levels"] = 1
        handle.attrs["num_components"] = 2
        handle.attrs["component_0"] = "other"
        handle.attrs["component_1"] = "edge_aux_2"
        handle.create_group("Chombo_global").attrs["SpaceDim"] = 3
        level = handle.create_group("level_0")
        level.create_dataset(
            "boxes",
            data=np.asarray([(0, 0, 0, 1, 1, 1)], dtype=box_dtype),
        )
        level.create_dataset(
            "data:datatype=0",
            data=values.reshape(-1, order="F"),
        )
        level.create_group("data_attributes").create_dataset(
            "offsets",
            data=np.asarray([0, 16]),
        )


class SliceUpperHalfCirculationTests(unittest.TestCase):
    def test_slice_uses_visible_area_and_clips_cells_at_y_zero(self):
        with tempfile.TemporaryDirectory() as temporary:
            frame_path = Path(temporary) / "flowTime_4.hdf5"
            write_3d_frame(frame_path)
            task = {
                "path": frame_path,
                "source_index": 0,
                "config": {"input": {}},
                "metadata": {
                    "cfl": 0.1,
                    "dx_base": 1.0,
                    "num_amr_levels": 0,
                },
                "origin_3d": np.asarray([0.0, 0.5, 0.0]),
                "slice_z": 1.0e-6,
            }

            frame = load_task_frame(task, include_cells=True)
            result = measure_frame(frame, center_fraction=0.4)

            self.assertAlmostEqual(result["circulation_upper_half"], 8.5)
            self.assertAlmostEqual(result["peak_upper_vorticity"], 4.0)
            self.assertAlmostEqual(result["center_threshold_vorticity"], 1.6)
            self.assertEqual(result["center_cells"], 3)
            self.assertAlmostEqual(result["center_vorticity_weight"], 8.0)
            self.assertAlmostEqual(result["x_center"], 1.125)
            self.assertAlmostEqual(result["y_center"], 0.90625)

    def test_end_to_end_creates_slice_gif_and_combined_plots(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_folder = root / "sweep" / "tau_1p0"
            output_folder = run_folder / "output"
            output_folder.mkdir(parents=True)
            write_3d_frame(output_folder / "flowTime_4.hdf5")
            write_3d_frame(output_folder / "flowTime_8.hdf5")
            (run_folder / "tau_1p0.cfg").write_text(
                """
cfl=0.1;
nLevels=0;
Re=300;
b_f_tau=1;
domain {
  dx_base=1;
  block { base=(-1,-0.5,-1); extent=(2,2,2); }
}
""".strip(),
                encoding="utf-8",
            )
            config_path = root / "analysis.toml"
            config_path.write_text(
                """
[input]
field_name = "edge_aux"
hdf5_glob = "flowTime_*.hdf5"
simulation_config = ""
simulation_config_glob = "*.cfg"
fallback_time_spacing = nan
cfl = nan
dx_base = nan
num_amr_levels = -1
origin_x_index = nan
origin_y_index = nan

[hmaxima]
connectivity = 8

[region]

[fit]

[threshold_mask]
positive_color = "#ffb000"

[plot]
colormap = "RdBu_r"
mask_alpha = 0.35
region_line_width = 1.5
marker_size = 30.0
fit_text_size = 8.0
figure_width = 5.0
figure_height = 3.0
x_axis_min = nan
x_axis_max = nan
y_axis_min = nan
y_axis_max = nan

[time_series]
""".strip(),
                encoding="utf-8",
            )
            analysis_output = root / "analysis_output"
            environment = os.environ.copy()
            environment["MPLCONFIGDIR"] = str(root / "matplotlib")
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "ritta_plotting_2D" / "upper_half_circulation.py"),
                    str(root / "sweep"),
                    str(config_path),
                    "--slice-z",
                    "1e-6",
                    "--workers",
                    "2",
                    "--analysis-stride",
                    "1",
                    "--gif-stride",
                    "1",
                    "--output-dir",
                    str(analysis_output),
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
            for filename in (
                "combined_upper_half_circulation_vs_time.png",
                "combined_upper_half_circulation_vs_time_over_tau.png",
                "combined_upper_half_center_x_vs_time.png",
                "combined_upper_half_center_x_vs_time_over_tau.png",
                "method.json",
                "datasets.toml",
            ):
                self.assertGreater((analysis_output / filename).stat().st_size, 0)
            gif_path = (
                analysis_output
                / "gifs"
                / "tau_1p0"
                / "tau_1p0_upper_half_center_0p4.gif"
            )
            self.assertGreater(gif_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
