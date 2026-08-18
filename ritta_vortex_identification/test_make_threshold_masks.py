"""Tests for retained h-maximum tracking."""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_threshold_masks import (
    assign_extrema_tracks,
    first_pair_interactions,
    save_extrema_track_plot,
    save_pair_interaction_plot,
)


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
    def test_interpolates_first_pair_crossover(self) -> None:
        first_start = record(0, 0.0, 0.0)
        second_start = record(0, 0.0, 2.0)
        first_end = record(1, 2.0, 2.0)
        second_end = record(1, 2.0, 0.0)
        for item in (first_start, first_end):
            item["track_id"] = 1
        for item in (second_start, second_end):
            item["track_id"] = 2

        interactions = first_pair_interactions(
            [[first_start, second_start], [first_end, second_end]]
        )

        self.assertEqual(len(interactions), 1)
        self.assertEqual(interactions[0]["kind"], "crossover")
        self.assertAlmostEqual(interactions[0]["time"], 1.0)
        self.assertAlmostEqual(interactions[0]["x"], 1.0)

    def test_uses_closest_approach_when_pair_does_not_cross(self) -> None:
        first_start = record(0, 0.0, 0.0)
        second_start = record(0, 0.0, 3.0)
        first_end = record(1, 1.0, 1.0)
        second_end = record(1, 1.0, 2.2)
        for item in (first_start, first_end):
            item["track_id"] = 1
        for item in (second_start, second_end):
            item["track_id"] = 2

        interactions = first_pair_interactions(
            [[first_start, second_start], [first_end, second_end]]
        )

        self.assertEqual(interactions[0]["kind"], "closest_approach")
        self.assertAlmostEqual(interactions[0]["time"], 1.0)
        self.assertAlmostEqual(interactions[0]["separation"], 1.2)

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

    def test_saves_x_and_y_track_plots(self) -> None:
        first = record(0, 0.0, 1.0, -0.5)
        second = record(1, 1.0, 2.0, 0.5)
        first["track_id"] = 1
        second["track_id"] = 1
        records_by_frame = [[first], [second]]

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_folder = Path(temporary_directory)
            interactions = [
                {
                    "first_track_id": 1,
                    "second_track_id": 2,
                    "time": 0.5,
                    "x": 1.5,
                    "kind": "crossover",
                    "separation": 0.0,
                }
            ]
            for coordinate in ("x", "y"):
                for forcing_frequency, suffix in ((None, "time"), (0.1, "cycles")):
                    output_path = output_folder / f"{coordinate}_vs_{suffix}.png"
                    track_count = save_extrema_track_plot(
                        output_path,
                        records_by_frame,
                        (4.0, 3.0),
                        coordinate,
                        forcing_frequency=forcing_frequency,
                        interactions=interactions,
                    )
                    self.assertEqual(track_count, 1)
                    self.assertGreater(output_path.stat().st_size, 0)
            interaction_path = output_folder / "pair_interactions.png"
            save_pair_interaction_plot(
                interaction_path,
                interactions,
                0.1,
                (4.0, 3.0),
            )
            self.assertGreater(interaction_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
