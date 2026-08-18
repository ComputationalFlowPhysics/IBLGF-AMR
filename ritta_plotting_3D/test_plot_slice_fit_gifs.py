"""Tests for the ParaView-independent 3D-slice fit GIF helpers."""

import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_slice_fit_gifs import axis_limits, save_gif, successful_records


class SliceFitGifTests(unittest.TestCase):
    def test_successful_records_reject_failed_and_nonfinite_fits(self):
        result = {
            "records": [
                {
                    "fit_success": True,
                    "_fit_x": 1.0,
                    "_fit_y": 0.5,
                    "boundary_radius": 0.25,
                },
                {
                    "fit_success": False,
                    "_fit_x": 2.0,
                    "_fit_y": 0.5,
                    "boundary_radius": 0.25,
                },
                {
                    "fit_success": True,
                    "_fit_x": float("nan"),
                    "_fit_y": 0.5,
                    "boundary_radius": 0.25,
                },
            ]
        }

        records = successful_records(result)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0][1:], (1.0, 0.5, 0.25))

    def test_requested_axis_limits_override_config(self):
        config = {
            "plot": {
                "x_axis_min": -1.0,
                "x_axis_max": 8.0,
                "y_axis_min": -3.0,
                "y_axis_max": 3.0,
            }
        }

        limits = axis_limits(config, (0.0, 10.0), (-2.0, 2.0))

        self.assertEqual(limits, ((0.0, 10.0), (-2.0, 2.0)))

    def test_save_gif_preserves_frame_count(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            folder = Path(temporary_directory)
            frames = []
            for index, color in enumerate(("red", "blue")):
                path = folder / f"frame_{index}.png"
                Image.new("RGB", (8, 8), color=color).save(path)
                frames.append(path)
            gif_path = folder / "fits.gif"

            save_gif(frames, gif_path, fps=8)

            with Image.open(gif_path) as image:
                self.assertEqual(image.n_frames, 2)


if __name__ == "__main__":
    unittest.main()
