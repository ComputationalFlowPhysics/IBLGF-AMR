"""Tests for retained h-maximum tracking."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_threshold_masks import assign_extrema_tracks


def record(frame_index: int, time: float, x: float, y: float = 0.0) -> dict:
    return {
        "frame_index": frame_index,
        "frame_name": f"frame_{frame_index}",
        "time": time,
        "candidate_id": 1,
        "track_id": None,
        "x": x,
        "y": y,
        "peak_vorticity": 1.0,
    }


class AssignExtremaTracksTests(unittest.TestCase):
    def test_missing_track_matches_constant_velocity_prediction(self) -> None:
        true_reappearance = record(3, 3.0, 3.0)
        stale_position_distractor = record(3, 3.0, 1.2)
        records_by_frame = [
            [record(0, 0.0, 0.0)],
            [record(1, 1.0, 1.0)],
            [],
            [stale_position_distractor, true_reappearance],
        ]

        assign_extrema_tracks(records_by_frame, [0.0, 1.0, 2.0, 3.0], 0.3, 1.1, 2, 1)

        self.assertEqual(true_reappearance["track_id"], 1)
        self.assertIsNone(stale_position_distractor["track_id"])

    def test_prediction_uses_physical_time_for_nonuniform_frames(self) -> None:
        reappearance = record(3, 7.0, 7.0)
        records_by_frame = [
            [record(0, 0.0, 0.0)],
            [record(1, 2.0, 2.0)],
            [],
            [reappearance],
        ]

        assign_extrema_tracks(records_by_frame, [0.0, 2.0, 5.0, 7.0], 0.1, 2.1, 2, 1)

        self.assertEqual(reappearance["track_id"], 1)

    def test_prediction_averages_requested_velocity_history(self) -> None:
        mean_velocity_reappearance = record(5, 5.0, 11.0)
        last_velocity_distractor = record(5, 5.0, 12.0)
        records_by_frame = [
            [record(0, 0.0, 0.0)],
            [record(1, 1.0, 1.0)],
            [record(2, 2.0, 3.0)],
            [record(3, 3.0, 6.0)],
            [],
            [last_velocity_distractor, mean_velocity_reappearance],
        ]

        assign_extrema_tracks(
            records_by_frame,
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            1.6,
            1.1,
            2,
            2,
        )

        self.assertEqual(mean_velocity_reappearance["track_id"], 1)
        self.assertIsNone(last_velocity_distractor["track_id"])

    def test_frame_times_must_be_strictly_increasing(self) -> None:
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            assign_extrema_tracks([[], []], [1.0, 1.0], 0.5, 0.5, 1, 3)


if __name__ == "__main__":
    unittest.main()
