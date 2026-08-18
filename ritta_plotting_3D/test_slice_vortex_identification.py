"""Tests for direct AMR slice extraction used by 3D vortex identification."""

import tempfile
import unittest
from pathlib import Path
import sys

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import slice_vortex_identification as analysis


class SliceVortexIdentificationTests(unittest.TestCase):
    def test_filters_only_maxima_strictly_below_minimum_peak(self):
        maxima = {
            "candidate_ids": np.asarray([1, 2, 3]),
            "peak_x": np.asarray([10.0, 20.0, 30.0]),
            "peak_y": np.asarray([-1.0, -2.0, -3.0]),
            "peak_vorticity": np.asarray([0.0009, 0.001, 0.002]),
            "hmax_mask": np.ones((2, 2), dtype=bool),
        }

        filtered = analysis.filter_weak_maxima(maxima, 0.001)

        np.testing.assert_array_equal(filtered["candidate_ids"], [2, 3])
        np.testing.assert_array_equal(filtered["peak_x"], [20.0, 30.0])
        np.testing.assert_array_equal(filtered["peak_y"], [-2.0, -3.0])
        np.testing.assert_array_equal(filtered["peak_vorticity"], [0.001, 0.002])
        self.assertIs(filtered["hmax_mask"], maxima["hmax_mask"])

    def test_chunk_boundaries_accept_length_style_offsets(self):
        boxes = np.asarray(
            [
                (0, 0, 0, 1, 1, 0),
                (2, 0, 0, 2, 1, 0),
            ],
            dtype=[
                ("lo_i", "i4"),
                ("lo_j", "i4"),
                ("lo_k", "i4"),
                ("hi_i", "i4"),
                ("hi_j", "i4"),
                ("hi_k", "i4"),
            ],
        )

        boundaries = analysis.chunk_boundaries(
            np.asarray([0, 8, 4]),
            boxes,
            data_size=12,
            components=2,
        )

        np.testing.assert_array_equal(boundaries, [0, 8, 12])

    def test_reads_only_requested_component_and_positive_z_cell_plane(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "flowTime_4.hdf5"
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
                global_group = handle.create_group("Chombo_global")
                global_group.attrs["SpaceDim"] = 3
                level = handle.create_group("level_0")
                level.create_dataset(
                    "boxes",
                    data=np.asarray([(0, 0, 0, 1, 1, 1)], dtype=box_dtype),
                )
                level.create_dataset(
                    "data:datatype=0",
                    data=values.reshape(-1, order="F"),
                )
                attributes = level.create_group("data_attributes")
                attributes.create_dataset("offsets", data=np.asarray([0, 16]))

            tiles = analysis.load_slice_tiles(
                path,
                metadata={"dx_base": 1.0},
                origin_3d=np.asarray([0.0, 0.0, 0.0]),
                slice_z=1.0e-6,
            )

            self.assertEqual(len(tiles), 1)
            np.testing.assert_array_equal(
                tiles[0]["values"],
                [[1.0, 2.0], [3.0, 4.0]],
            )
            self.assertEqual(tiles[0]["bounds"], (0.0, 2.0, 0.0, 2.0))

    def test_reads_three_dimensional_origin_from_block(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary) / "config.cfg"
            config.write_text(
                "domain { block { base=(-8,-6,-4); extent=(16,12,8); } }",
                encoding="utf-8",
            )

            origin = analysis.simulation_origin_3d(config)

            np.testing.assert_array_equal(origin, [0.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
