"""Create positive and negative vorticity-threshold masks for every frame."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from itertools import combinations
import math
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.ndimage import label

from common import (
    load_config,
    read_frame_order,
    require_nonnegative,
    require_positive,
    result_folder,
    simulation_parameter,
    write_string_dataset,
)
from plot_vorticity import browse_frames, image_extent


EIGHT_CONNECTED = np.ones((3, 3), dtype=bool)
TRACK_COLUMNS = (
    "frame_index",
    "frame_name",
    "time",
    "candidate_id",
    "track_id",
    "x",
    "y",
    "peak_vorticity",
)


def remove_small_regions(
    mask: np.ndarray, dx: float, minimum_area: float
) -> Tuple[np.ndarray, int, np.ndarray]:
    """Keep 8-connected regions whose physical area is at least minimum_area."""
    labels, region_count = label(mask, structure=EIGHT_CONNECTED)
    cell_counts = np.bincount(labels.ravel())[1:]
    areas = cell_counts.astype(float) * dx ** 2
    kept_labels = np.flatnonzero(areas >= minimum_area) + 1
    keep = np.zeros(region_count + 1, dtype=bool)
    keep[kept_labels] = True
    return keep[labels], region_count, areas[kept_labels - 1]


def make_masks(frame: dict, threshold: float, minimum_area: float) -> dict:
    """Apply the sign thresholds and discard connected regions that are too small."""
    omega = np.asarray(frame["vorticity"], dtype=float)
    finite = np.isfinite(omega)
    positive, positive_found, positive_areas = remove_small_regions(
        finite & (omega >= threshold), float(frame["dx"]), minimum_area
    )
    negative, negative_found, negative_areas = remove_small_regions(
        finite & (omega <= -threshold), float(frame["dx"]), minimum_area
    )
    return {
        **frame,
        "positive_mask": positive,
        "negative_mask": negative,
        "positive_regions_found": positive_found,
        "negative_regions_found": negative_found,
        "positive_region_areas": positive_areas,
        "negative_region_areas": negative_areas,
    }


def retained_extrema(frame: dict, maxima: h5py.Group) -> dict:
    """Keep saved h-maxima whose physical locations lie inside a retained mask cell."""
    peak_x = maxima["peak_x"][:]
    peak_y = maxima["peak_y"][:]
    retained_mask = frame["positive_mask"] | frame["negative_mask"]
    dx = float(frame["dx"])
    columns = np.floor((peak_x - (frame["x"][0] - 0.5 * dx)) / dx).astype(int)
    rows = np.floor((peak_y - (frame["y"][0] - 0.5 * dx)) / dx).astype(int)
    columns = np.clip(columns, 0, retained_mask.shape[1] - 1)
    rows = np.clip(rows, 0, retained_mask.shape[0] - 1)
    inside = retained_mask[rows, columns]
    return {
        "extrema_candidate_ids": maxima["candidate_ids"][:][inside],
        "extrema_x": peak_x[inside],
        "extrema_y": peak_y[inside],
        "extrema_vorticity": maxima["peak_vorticity"][:][inside],
        "extrema_found": len(peak_x),
    }


def frame_from_hmaxima(group: h5py.Group) -> dict:
    """Load the physical raster already saved by Stage 1."""
    return {
        "source_filename": str(group.attrs["source_filename"]),
        "source_path": str(group.attrs["source_path"]),
        "time": float(group.attrs["simulation_time"]),
        "step": int(group.attrs["time_step"]),
        "dx": float(group.attrs["dx"]),
        "x": group["x"][:],
        "y": group["y"][:],
        "vorticity": group["vorticity"][:],
    }


def save_frame(group: h5py.Group, frame: dict) -> None:
    """Save coordinates, filtered extrema, and the two binary masks."""
    group.attrs["source_filename"] = frame["source_filename"]
    group.attrs["source_path"] = frame["source_path"]
    group.attrs["simulation_time"] = frame["time"]
    group.attrs["time_step"] = frame["step"]
    group.attrs["dx"] = frame["dx"]
    group.attrs["positive_regions_found"] = frame["positive_regions_found"]
    group.attrs["negative_regions_found"] = frame["negative_regions_found"]
    group.attrs["extrema_found"] = frame["extrema_found"]
    group.create_dataset("x", data=frame["x"])
    group.create_dataset("y", data=frame["y"])
    group.create_dataset("positive_mask", data=frame["positive_mask"].astype(np.uint8), compression="gzip")
    group.create_dataset("negative_mask", data=frame["negative_mask"].astype(np.uint8), compression="gzip")
    group.create_dataset("positive_region_areas", data=frame["positive_region_areas"])
    group.create_dataset("negative_region_areas", data=frame["negative_region_areas"])
    group.create_dataset("extrema_candidate_ids", data=frame["extrema_candidate_ids"])
    group.create_dataset("extrema_x", data=frame["extrema_x"])
    group.create_dataset("extrema_y", data=frame["extrema_y"])
    group.create_dataset("extrema_vorticity", data=frame["extrema_vorticity"])


def load_saved_frame(path: Path, hmaxima_path: Path, group_name: str) -> dict:
    """Load one saved frame for the terminal preview without recomputing masks."""
    with h5py.File(path, "r") as handle, h5py.File(hmaxima_path, "r") as maxima:
        group = handle[group_name]
        source = maxima[group_name]
        return {
            "source_filename": str(group.attrs["source_filename"]),
            "time": float(group.attrs["simulation_time"]),
            "dx": float(group.attrs["dx"]),
            "x": group["x"][:],
            "y": group["y"][:],
            "vorticity": source["vorticity"][:],
            "positive_mask": group["positive_mask"][:].astype(bool),
            "negative_mask": group["negative_mask"][:].astype(bool),
            "extrema_x": group["extrema_x"][:],
            "extrema_y": group["extrema_y"][:],
        }


def extrema_records(frame_index: int, frame: dict) -> List[dict]:
    """Convert retained extrema from one frame into tracking records."""
    records = []
    for candidate_id, x_value, y_value, peak_vorticity in zip(
        frame["extrema_candidate_ids"],
        frame["extrema_x"],
        frame["extrema_y"],
        frame["extrema_vorticity"],
    ):
        if not all(math.isfinite(float(value)) for value in (x_value, y_value, peak_vorticity)):
            raise ValueError(
                f"{frame['source_filename']} contains a non-finite retained-extremum value."
            )
        records.append({
            "frame_index": frame_index,
            "frame_name": frame["source_filename"],
            "time": float(frame["time"]),
            "candidate_id": int(candidate_id),
            "track_id": None,
            "x": float(x_value),
            "y": float(y_value),
            "peak_vorticity": float(peak_vorticity),
        })
    return records


def process_mask_to_shard(task: tuple) -> tuple:
    """Calculate one threshold-mask frame and save its large arrays locally."""
    hmaxima_path, group_name, frame_index, threshold, minimum_area, shard_path = task
    with h5py.File(hmaxima_path, "r") as maxima:
        source = maxima[group_name]
        frame = frame_from_hmaxima(source)
        result = make_masks(frame, threshold, minimum_area)
        result.update(retained_extrema(result, source))

    with h5py.File(shard_path, "w") as shard:
        save_frame(shard.create_group(group_name), result)

    return (
        group_name,
        shard_path,
        result["source_filename"],
        len(result["positive_region_areas"]),
        result["positive_regions_found"],
        len(result["negative_region_areas"]),
        result["negative_regions_found"],
        len(result["extrema_x"]),
        result["extrema_found"],
        extrema_records(frame_index, result),
        float(result["time"]),
    )


def calculate_masks_parallel(
    hmaxima_path: Path,
    output_path: Path,
    group_names: List[str],
    threshold: float,
    minimum_area: float,
    workers: int,
) -> Tuple[List[List[dict]], List[float]]:
    """Calculate independent mask frames concurrently and merge in order."""
    worker_count = min(workers, len(group_names))
    with tempfile.TemporaryDirectory(
        prefix="threshold-mask-shards-",
        dir=output_path.parent,
    ) as temporary:
        temporary_folder = Path(temporary)
        tasks = [
            (
                hmaxima_path,
                group_name,
                index,
                threshold,
                minimum_area,
                temporary_folder / f"{index:08d}.h5",
            )
            for index, group_name in enumerate(group_names)
        ]
        completed = []
        records_by_frame = []
        frame_times = []
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            for index, result in enumerate(
                executor.map(process_mask_to_shard, tasks),
                start=1,
            ):
                (
                    group_name,
                    shard_path,
                    filename,
                    positive_kept,
                    positive_found,
                    negative_kept,
                    negative_found,
                    extrema_kept,
                    extrema_found,
                    records,
                    frame_time,
                ) = result
                completed.append((group_name, shard_path))
                records_by_frame.append(records)
                frame_times.append(frame_time)
                print(
                    f"[{index}/{len(group_names)}] {filename}: "
                    f"kept {positive_kept}/{positive_found} positive and "
                    f"{negative_kept}/{negative_found} negative regions; "
                    f"marked {extrema_kept}/{extrema_found} extrema",
                    flush=True,
                )

        with h5py.File(output_path, "a") as output:
            for group_name, shard_path in completed:
                with h5py.File(shard_path, "r") as shard:
                    shard.copy(group_name, output)

    return records_by_frame, frame_times


def nearest_one_to_one_matches(
    source_centers: Dict[int, Tuple[float, float]],
    records: List[dict],
    max_displacement: float,
    unavailable_indices: Optional[Set[int]] = None,
) -> Dict[int, int]:
    """Match each source to its nearest detection, using each detection at most once."""
    if not records:
        return {}

    proposals = []
    for source_id, (source_x, source_y) in source_centers.items():
        nearest_index = min(
            range(len(records)),
            key=lambda index: (
                math.hypot(records[index]["x"] - source_x, records[index]["y"] - source_y),
                index,
            ),
        )
        distance = math.hypot(
            records[nearest_index]["x"] - source_x,
            records[nearest_index]["y"] - source_y,
        )
        if distance <= max_displacement:
            proposals.append((distance, source_id, nearest_index))

    matches = {}
    used_indices = set() if unavailable_indices is None else set(unavailable_indices)
    for _, source_id, candidate_index in sorted(proposals):
        if candidate_index in used_indices:
            continue
        matches[source_id] = candidate_index
        used_indices.add(candidate_index)
    return matches


def assign_extrema_tracks(
    records_by_frame: List[List[dict]],
    frame_times: List[float],
    max_displacement: float,
    new_track_max_displacement: float,
    max_missed_frames: int,
    velocity_history_length: int,
) -> None:
    """Track extrema using recent mean velocity to predict missed positions."""
    if len(frame_times) != len(records_by_frame):
        raise ValueError("frame_times must contain one time for every analyzed frame.")
    if any(not math.isfinite(time) for time in frame_times):
        raise ValueError("frame_times must contain only finite values.")
    if any(current <= previous for previous, current in zip(frame_times, frame_times[1:])):
        raise ValueError("frame_times must be strictly increasing.")
    if (
        isinstance(velocity_history_length, bool)
        or not isinstance(velocity_history_length, int)
        or velocity_history_length < 1
    ):
        raise ValueError("velocity_history_length must be a positive integer.")

    active_tracks = {}
    tentative_records = []
    next_track_id = 1

    for current_time, records in zip(frame_times, records_by_frame):
        active_centers = {}
        for track_id, state in active_tracks.items():
            mean_velocity_x = sum(velocity[0] for velocity in state["velocities"]) / len(
                state["velocities"]
            )
            mean_velocity_y = sum(velocity[1] for velocity in state["velocities"]) / len(
                state["velocities"]
            )
            elapsed_time = current_time - state["time"]
            active_centers[track_id] = (
                state["x"] + mean_velocity_x * elapsed_time,
                state["y"] + mean_velocity_y * elapsed_time,
            )
        existing_matches = nearest_one_to_one_matches(
            active_centers,
            records,
            max_displacement,
        )
        used_indices = set(existing_matches.values())
        next_active_tracks = {}
        for track_id, index in existing_matches.items():
            record = records[index]
            state = active_tracks[track_id]
            elapsed_time = current_time - state["time"]
            record["track_id"] = track_id
            velocity = (
                (record["x"] - state["x"]) / elapsed_time,
                (record["y"] - state["y"]) / elapsed_time,
            )
            next_active_tracks[track_id] = {
                "x": record["x"],
                "y": record["y"],
                "time": current_time,
                "velocities": [*state["velocities"], velocity][-velocity_history_length:],
                "missed_frames": 0,
            }

        for track_id, state in active_tracks.items():
            if track_id in existing_matches:
                continue
            missed_frames = state["missed_frames"] + 1
            if missed_frames <= max_missed_frames:
                next_active_tracks[track_id] = {
                    **state,
                    "missed_frames": missed_frames,
                }

        tentative_centers = {
            index: (record["x"], record["y"])
            for index, record in enumerate(tentative_records)
        }
        confirmed_matches = nearest_one_to_one_matches(
            tentative_centers,
            records,
            new_track_max_displacement,
            unavailable_indices=used_indices,
        )
        for tentative_index, current_index in confirmed_matches.items():
            previous_record = tentative_records[tentative_index]
            current_record = records[current_index]
            elapsed_time = current_time - previous_record["time"]
            initial_velocity = (
                (current_record["x"] - previous_record["x"]) / elapsed_time,
                (current_record["y"] - previous_record["y"]) / elapsed_time,
            )
            track_id = next_track_id
            next_track_id += 1
            previous_record["track_id"] = track_id
            current_record["track_id"] = track_id
            used_indices.add(current_index)
            next_active_tracks[track_id] = {
                "x": current_record["x"],
                "y": current_record["y"],
                "time": current_time,
                "velocities": [initial_velocity],
                "missed_frames": 0,
            }

        tentative_records = [
            records[index]
            for index in range(len(records))
            if index not in used_indices
        ]
        active_tracks = next_active_tracks


def filter_short_tracks(
    records_by_frame: List[List[dict]],
    minimum_track_points: int,
) -> Tuple[int, int]:
    """Discard short tracks and renumber the retained IDs consecutively."""
    track_counts = {}
    for records in records_by_frame:
        for record in records:
            if record["track_id"] is not None:
                track_id = int(record["track_id"])
                track_counts[track_id] = track_counts.get(track_id, 0) + 1

    retained_ids = sorted(
        track_id
        for track_id, count in track_counts.items()
        if count >= minimum_track_points
    )
    new_ids = {
        old_track_id: new_track_id
        for new_track_id, old_track_id in enumerate(retained_ids, start=1)
    }
    for records in records_by_frame:
        for record in records:
            if record["track_id"] is not None:
                record["track_id"] = new_ids.get(int(record["track_id"]))

    return len(retained_ids), len(track_counts) - len(retained_ids)


def write_extrema_tracks(path: Path, records_by_frame: List[List[dict]]) -> None:
    """Write every retained extremum, leaving unconfirmed track IDs blank."""
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRACK_COLUMNS)
        writer.writeheader()
        for records in records_by_frame:
            for record in records:
                row = dict(record)
                row["track_id"] = "" if row["track_id"] is None else row["track_id"]
                writer.writerow(row)


def read_extrema_tracks(path: Path) -> List[List[dict]]:
    """Load saved tracking records without recalculating maxima or masks."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Saved track CSV does not exist: {path}. Run without --plots-only first."
        )

    records_by_index = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing_columns = set(TRACK_COLUMNS) - set(reader.fieldnames or ())
        if missing_columns:
            raise ValueError(
                f"{path} is missing columns: {', '.join(sorted(missing_columns))}"
            )
        for row in reader:
            frame_index = int(row["frame_index"])
            track_text = row["track_id"].strip()
            records_by_index.setdefault(frame_index, []).append({
                "frame_index": frame_index,
                "frame_name": row["frame_name"],
                "time": float(row["time"]),
                "candidate_id": int(row["candidate_id"]),
                "track_id": int(track_text) if track_text else None,
                "x": float(row["x"]),
                "y": float(row["y"]),
                "peak_vorticity": float(row["peak_vorticity"]),
            })
    return [records_by_index[index] for index in sorted(records_by_index)]


def track_colors(track_count: int) -> list:
    """Return stable categorical colors for the plotted track IDs."""
    colors = []
    for name in ("tab20", "tab20b", "tab20c"):
        colors.extend(plt.get_cmap(name).colors)
    if track_count <= len(colors):
        return colors[:track_count]
    return list(plt.get_cmap("turbo")(np.linspace(0.0, 1.0, track_count)))


def all_pair_interactions(records_by_frame: List[List[dict]]) -> List[dict]:
    """Find every x-order crossover among every pair of retained tracks."""
    tracks = {}
    for records in records_by_frame:
        for record in records:
            if record["track_id"] is not None:
                tracks.setdefault(int(record["track_id"]), {})[
                    float(record["time"])
                ] = float(record["x"])

    interactions = []
    for first_id, second_id in combinations(sorted(tracks), 2):
        times = sorted(set(tracks[first_id]) & set(tracks[second_id]))
        if not times:
            continue
        samples = [
            (
                time,
                tracks[first_id][time] - tracks[second_id][time],
                0.5 * (tracks[first_id][time] + tracks[second_id][time]),
            )
            for time in times
        ]

        pair_interactions = []
        in_zero_plateau = False
        for sample_index, sample in enumerate(samples):
            if math.isclose(sample[1], 0.0, abs_tol=1.0e-12):
                if not in_zero_plateau:
                    pair_interactions.append((sample[0], sample[2]))
                in_zero_plateau = True
                continue

            if sample_index > 0 and not in_zero_plateau:
                left = samples[sample_index - 1]
                right = sample
                if left[1] * right[1] < 0.0:
                    fraction = -left[1] / (right[1] - left[1])
                    pair_interactions.append(
                        (
                            left[0] + fraction * (right[0] - left[0]),
                            left[2] + fraction * (right[2] - left[2]),
                        )
                    )
            in_zero_plateau = False

        for interaction_time, interaction_x in pair_interactions:
            interactions.append(
                {
                    "first_track_id": first_id,
                    "second_track_id": second_id,
                    "time": float(interaction_time),
                    "x": float(interaction_x),
                    "kind": "crossover",
                }
            )
    return sorted(
        interactions,
        key=lambda item: (
            item["time"],
            item["first_track_id"],
            item["second_track_id"],
        ),
    )


def forcing_end_cycles_by_track(
    records_by_frame: List[List[dict]],
    forcing_frequency: float,
    forcing_duration: float,
) -> dict:
    """Map each retained track to its pulse's end time in forcing cycles."""
    if not math.isfinite(forcing_frequency) or forcing_frequency <= 0.0:
        raise ValueError("forcing_frequency must be finite and greater than zero")
    if not math.isfinite(forcing_duration) or forcing_duration <= 0.0:
        raise ValueError("forcing_duration must be finite and greater than zero")

    first_time_by_track = {}
    for records in records_by_frame:
        for record in records:
            if record["track_id"] is None:
                continue
            track_id = int(record["track_id"])
            time = float(record["time"])
            first_time_by_track[track_id] = min(
                time,
                first_time_by_track.get(track_id, time),
            )

    forcing_end_by_track = {}
    for track_id, first_time in first_time_by_track.items():
        first_cycle = forcing_frequency * first_time
        pulse_index = max(0, math.floor(first_cycle + 1.0e-9))
        forcing_end_by_track[track_id] = (
            pulse_index + forcing_frequency * forcing_duration
        )
    return forcing_end_by_track


def save_extrema_track_plot(
    path: Path,
    records_by_frame: List[List[dict]],
    figure_size: Tuple[float, float],
    coordinate: str,
    forcing_frequency: Optional[float] = None,
    interactions: Optional[List[dict]] = None,
) -> int:
    """Plot one coordinate versus physical time or forcing cycles."""
    if coordinate not in {"x", "y"}:
        raise ValueError("coordinate must be 'x' or 'y'.")

    tracks = {}
    for records in records_by_frame:
        for record in records:
            if record["track_id"] is not None:
                tracks.setdefault(int(record["track_id"]), []).append(record)

    figure, axis = plt.subplots(figsize=figure_size)
    colors = track_colors(len(tracks))
    for color, track_id in zip(colors, sorted(tracks)):
        records = sorted(tracks[track_id], key=lambda record: record["frame_index"])
        axis.plot(
            [
                record["time"]
                if forcing_frequency is None
                else forcing_frequency * record["time"]
                for record in records
            ],
            [record[coordinate] for record in records],
            color=color,
            marker="o",
            markersize=4,
            linewidth=1.5,
            label=f"track {track_id}",
        )

    if coordinate == "x" and interactions:
        for interaction_index, interaction in enumerate(interactions):
            horizontal = interaction["time"]
            if forcing_frequency is not None:
                horizontal *= forcing_frequency
            axis.scatter(
                horizontal,
                interaction["x"],
                s=75,
                marker="X",
                facecolors="black",
                edgecolors="white",
                linewidths=0.8,
                zorder=5,
                label="interpolated crossover" if interaction_index == 0 else None,
            )
            axis.annotate(
                f"{interaction['first_track_id']}&{interaction['second_track_id']}",
                (horizontal, interaction["x"]),
                xytext=(4, 5),
                textcoords="offset points",
                fontsize=7,
            )

    if forcing_frequency is None:
        horizontal_name = "simulation time"
        axis.set_xlabel(horizontal_name)
    else:
        horizontal_name = "forcing cycles"
        axis.set_xlabel(r"forcing cycles, $t^* = ft = t/T$")
    axis.set_ylabel(f"retained h-maximum {coordinate} coordinate")
    axis.set_title(
        f"Tracked retained h-maxima: {coordinate} coordinate versus {horizontal_name}"
    )
    axis.grid(True, alpha=0.3)
    if tracks:
        legend_columns = max(1, math.ceil(len(tracks) / 20))
        axis.legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            ncol=legend_columns,
            fontsize=7,
        )
    else:
        axis.text(
            0.5,
            0.5,
            "No tracks meet the minimum-point requirement",
            transform=axis.transAxes,
            ha="center",
            va="center",
        )
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return len(tracks)


def save_pair_interaction_plot(
    path: Path,
    interactions: List[dict],
    forcing_frequency: float,
    figure_size: Tuple[float, float],
    forcing_end_by_track: Optional[dict] = None,
) -> None:
    """Plot every normalized crossover and forcing-end time by track pair."""
    figure, axis = plt.subplots(figsize=figure_size)
    labeled = set()
    interactions_by_pair = {}
    for item in interactions:
        pair = (int(item["first_track_id"]), int(item["second_track_id"]))
        interactions_by_pair.setdefault(pair, []).append(item)
    ordered_pairs = sorted(
        interactions_by_pair,
        key=lambda pair: (
            min(item["time"] for item in interactions_by_pair[pair]),
            pair,
        ),
    )

    for index, pair in enumerate(ordered_pairs):
        pair_interactions = sorted(
            interactions_by_pair[pair],
            key=lambda item: item["time"],
        )
        crossover_times = [
            forcing_frequency * item["time"]
            for item in pair_interactions
        ]
        forcing_end_times = []
        if forcing_end_by_track is not None:
            forcing_end_times = [
                forcing_end_by_track[track_id]
                for track_id in pair
                if track_id in forcing_end_by_track
            ]
        connector_times = sorted(crossover_times + forcing_end_times)
        if len(connector_times) >= 2:
            axis.plot(
                [index] * len(connector_times),
                connector_times,
                color="#555555",
                linewidth=1.5,
                linestyle="-",
                zorder=1,
            )

        first_forcing_end = (
            forcing_end_by_track.get(pair[0])
            if forcing_end_by_track is not None
            else None
        )
        for crossover_index, crossover_time in enumerate(crossover_times):
            if crossover_index == 0:
                previous_time = first_forcing_end
            else:
                previous_time = crossover_times[crossover_index - 1]
            if previous_time is None:
                continue
            normalized_delay = crossover_time - previous_time
            axis.annotate(
                f"{normalized_delay:.2f}",
                (index, 0.5 * (crossover_time + previous_time)),
                xytext=(5, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=7,
                color="#333333",
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.75,
                    "pad": 0.5,
                },
                zorder=4,
            )

        for item in pair_interactions:
            label = "interpolated crossover"
            axis.scatter(
                index,
                forcing_frequency * item["time"],
                s=65,
                marker="o",
                facecolors="black",
                edgecolors="black",
                linewidths=1.2,
                zorder=3,
                label=label if label not in labeled else None,
            )
            labeled.add(label)

        if forcing_end_by_track is not None:
            for track_id, marker, color, marker_label in (
                (pair[0], "^", "#1f77b4", "first track forcing end"),
                (pair[1], "v", "#ff7f0e", "second track forcing end"),
            ):
                if track_id not in forcing_end_by_track:
                    continue
                axis.scatter(
                    index,
                    forcing_end_by_track[track_id],
                    s=65,
                    marker=marker,
                    facecolors=color,
                    edgecolors="black",
                    linewidths=0.8,
                    zorder=3,
                    label=marker_label if marker_label not in labeled else None,
                )
                labeled.add(marker_label)

    axis.set_xlabel("track pair")
    axis.set_ylabel(r"normalized time $t^* = ft = t/T$")
    axis.set_xticks(
        range(len(ordered_pairs)),
        [f"{first_id}&{second_id}" for first_id, second_id in ordered_pairs],
    )
    axis.set_title("All crossover and forcing-end times by interacting track pair")
    axis.grid(True, axis="y", alpha=0.3)
    if labeled:
        axis.legend()
    else:
        axis.text(
            0.5,
            0.5,
            "No track-pair crossovers were detected",
            transform=axis.transAxes,
            ha="center",
            va="center",
        )
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description="Create positive and negative vorticity-threshold masks.")
    parser.add_argument("run_folder", type=Path)
    parser.add_argument("config_file", type=Path)
    parser.add_argument("--stride", type=int, default=1, help="Process every Nth sorted HDF5 frame.")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Process this many h-maxima and mask frames concurrently (default: 1).",
    )
    parser.add_argument("--no-preview", action="store_true", help="Skip the terminal preview and preview PNG.")
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Regenerate tracking and interaction PNGs from the saved track CSV only.",
    )
    args = parser.parse_args()
    if args.stride < 1:
        parser.error("--stride must be a positive integer.")
    if args.workers < 1:
        parser.error("--workers must be a positive integer.")

    config = load_config(args.config_file)
    if "threshold_mask" not in config:
        raise ValueError("Missing [threshold_mask] section in the TOML config.")
    threshold = require_positive(config, "threshold_mask", "vorticity_threshold")
    minimum_area = require_nonnegative(config, "threshold_mask", "minimum_region_area")
    output_folder = result_folder(args.run_folder)
    hmaxima_path = output_folder / "hmaxima.h5"
    output_path = output_folder / "threshold_masks.h5"
    track_csv_path = output_folder / "threshold_hmaxima_tracks.csv"
    x_track_plot_path = output_folder / "threshold_hmaxima_x_vs_time.png"
    y_track_plot_path = output_folder / "threshold_hmaxima_y_vs_time.png"
    x_cycle_plot_path = output_folder / "threshold_hmaxima_x_vs_forcing_cycles.png"
    y_cycle_plot_path = output_folder / "threshold_hmaxima_y_vs_forcing_cycles.png"
    interaction_plot_path = output_folder / "threshold_hmaxima_pair_interactions.png"
    records_by_frame = []
    frame_times = []
    try:
        forcing_frequency = simulation_parameter(
            args.run_folder,
            config,
            "b_f_freq",
        )
    except (FileNotFoundError, ValueError) as error:
        frequency_is_unavailable = isinstance(
            error, FileNotFoundError
        ) or "b_f_freq was not found" in str(error)
        if not frequency_is_unavailable:
            raise
        forcing_frequency = math.nan
    forcing_duration = (
        simulation_parameter(args.run_folder, config, "b_f_tau")
        if math.isfinite(forcing_frequency) and forcing_frequency > 0.0
        else math.nan
    )
    has_forcing_cycles = (
        math.isfinite(forcing_frequency)
        and forcing_frequency > 0.0
        and math.isfinite(forcing_duration)
        and forcing_duration > 0.0
    )

    if args.plots_only:
        records_by_frame = read_extrema_tracks(track_csv_path)
        interactions = all_pair_interactions(records_by_frame)
        forcing_end_by_track = (
            forcing_end_cycles_by_track(
                records_by_frame,
                forcing_frequency,
                forcing_duration,
            )
            if has_forcing_cycles
            else {}
        )
        figure_size = (
            float(config["plot"].get("figure_width", 10.0)),
            float(config["plot"].get("figure_height", 7.0)),
        )
        plot_results = [
            (
                x_track_plot_path,
                save_extrema_track_plot(
                    x_track_plot_path, records_by_frame, figure_size, "x"
                ),
            ),
            (
                y_track_plot_path,
                save_extrema_track_plot(
                    y_track_plot_path, records_by_frame, figure_size, "y"
                ),
            ),
        ]
        if has_forcing_cycles:
            plot_results.extend([
                (
                    x_cycle_plot_path,
                    save_extrema_track_plot(
                        x_cycle_plot_path,
                        records_by_frame,
                        figure_size,
                        "x",
                        forcing_frequency,
                        interactions=interactions,
                    ),
                ),
                (
                    y_cycle_plot_path,
                    save_extrema_track_plot(
                        y_cycle_plot_path,
                        records_by_frame,
                        figure_size,
                        "y",
                        forcing_frequency,
                    ),
                ),
            ])
            save_pair_interaction_plot(
                interaction_plot_path,
                interactions,
                forcing_frequency,
                figure_size,
                forcing_end_by_track,
            )
        for plot_path, track_count in plot_results:
            print(f"Saved {plot_path} ({track_count} tracks from saved CSV)")
        if has_forcing_cycles:
            pair_count = len({
                (item["first_track_id"], item["second_track_id"])
                for item in interactions
            })
            print(
                f"Saved {interaction_plot_path} "
                f"({len(interactions)} crossovers among {pair_count} track pairs)"
            )
        print("Reused saved threshold_hmaxima_tracks.csv; maxima and masks were not recalculated.")
        return 0

    # Stage 1 runs first and writes the extrema and physical raster consumed below.
    subprocess.run([
        sys.executable,
        str(Path(__file__).with_name("01_find_hmaxima.py")),
        str(args.run_folder),
        str(args.config_file),
        "--stride",
        str(args.stride),
        "--workers",
        str(args.workers),
        "--no-preview",
    ], check=True)

    with h5py.File(hmaxima_path, "r") as maxima:
        group_names = read_frame_order(maxima)
    worker_count = min(args.workers, len(group_names))
    with h5py.File(output_path, "w") as output:
        output.attrs["schema"] = "ritta_vorticity_threshold_masks_v2"
        output.attrs["run_folder"] = str(args.run_folder.expanduser().resolve())
        output.attrs["config_file"] = str(args.config_file.expanduser().resolve())
        output.attrs["source_hmaxima"] = str(hmaxima_path.resolve())
        output.attrs["vorticity_threshold"] = threshold
        output.attrs["minimum_region_area"] = minimum_area
        output.attrs["stride"] = args.stride
        output.attrs["workers"] = worker_count
        output.attrs["connectivity"] = 8
        write_string_dataset(output, "frame_order", group_names)

    print(f"Threshold-mask workers: {worker_count}", flush=True)
    if worker_count > 1:
        records_by_frame, frame_times = calculate_masks_parallel(
            hmaxima_path,
            output_path,
            group_names,
            threshold,
            minimum_area,
            worker_count,
        )
    else:
        with h5py.File(hmaxima_path, "r") as maxima, h5py.File(output_path, "a") as output:
            for index, group_name in enumerate(group_names):
                source = maxima[group_name]
                frame = frame_from_hmaxima(source)
                result = make_masks(frame, threshold, minimum_area)
                result.update(retained_extrema(result, source))
                save_frame(output.create_group(group_name), result)
                records_by_frame.append(extrema_records(index, result))
                frame_times.append(float(result["time"]))
                print(
                    f"[{index + 1}/{len(group_names)}] {frame['source_filename']}: "
                    f"kept {len(result['positive_region_areas'])}/{result['positive_regions_found']} positive and "
                    f"{len(result['negative_region_areas'])}/{result['negative_regions_found']} negative regions; "
                    f"marked {len(result['extrema_x'])}/{result['extrema_found']} extrema"
                )

    print(f"Saved {output_path}")
    tracking_config = config.get("tracking", {})
    max_displacement = float(tracking_config.get("max_displacement", 0.5))
    new_track_max_displacement = float(
        tracking_config.get("new_track_max_displacement", 0.5)
    )
    max_missed_frames_setting = tracking_config.get("max_missed_frames", 2)
    minimum_track_points_setting = tracking_config.get("minimum_track_points", 5)
    velocity_history_length_setting = tracking_config.get("velocity_history_length", 3)
    if not math.isfinite(max_displacement) or max_displacement <= 0.0:
        raise ValueError("[tracking] max_displacement must be finite and greater than zero.")
    if (
        not math.isfinite(new_track_max_displacement)
        or new_track_max_displacement <= 0.0
    ):
        raise ValueError(
            "[tracking] new_track_max_displacement must be finite and greater than zero."
        )
    if (
        isinstance(max_missed_frames_setting, bool)
        or not isinstance(max_missed_frames_setting, int)
        or max_missed_frames_setting < 0
    ):
        raise ValueError("[tracking] max_missed_frames must be a non-negative integer.")
    if (
        isinstance(minimum_track_points_setting, bool)
        or not isinstance(minimum_track_points_setting, int)
        or minimum_track_points_setting < 1
    ):
        raise ValueError("[tracking] minimum_track_points must be a positive integer.")
    if (
        isinstance(velocity_history_length_setting, bool)
        or not isinstance(velocity_history_length_setting, int)
        or velocity_history_length_setting < 1
    ):
        raise ValueError("[tracking] velocity_history_length must be a positive integer.")

    assign_extrema_tracks(
        records_by_frame,
        frame_times,
        max_displacement,
        new_track_max_displacement,
        max_missed_frames_setting,
        velocity_history_length_setting,
    )
    retained_track_count, discarded_track_count = filter_short_tracks(
        records_by_frame,
        minimum_track_points_setting,
    )
    interactions = all_pair_interactions(records_by_frame)
    forcing_end_by_track = (
        forcing_end_cycles_by_track(
            records_by_frame,
            forcing_frequency,
            forcing_duration,
        )
        if has_forcing_cycles
        else {}
    )
    write_extrema_tracks(track_csv_path, records_by_frame)
    figure_size = (
        float(config["plot"].get("figure_width", 10.0)),
        float(config["plot"].get("figure_height", 7.0)),
    )
    x_track_count = save_extrema_track_plot(
        x_track_plot_path,
        records_by_frame,
        figure_size,
        "x",
    )
    y_track_count = save_extrema_track_plot(
        y_track_plot_path,
        records_by_frame,
        figure_size,
        "y",
    )
    plot_results = [
        (x_track_plot_path, x_track_count),
        (y_track_plot_path, y_track_count),
    ]
    if has_forcing_cycles:
        for coordinate, path in (("x", x_cycle_plot_path), ("y", y_cycle_plot_path)):
            count = save_extrema_track_plot(
                path,
                records_by_frame,
                figure_size,
                coordinate,
                forcing_frequency,
                interactions=interactions if coordinate == "x" else None,
            )
            plot_results.append((path, count))
        save_pair_interaction_plot(
            interaction_plot_path,
            interactions,
            forcing_frequency,
            figure_size,
            forcing_end_by_track,
        )
    print(f"Saved {track_csv_path}")
    for track_plot_path, track_count in plot_results:
        print(
            f"Saved {track_plot_path} "
            f"({track_count} tracks with at least {minimum_track_points_setting} points; "
            f"discarded {discarded_track_count} shorter tracks)"
        )
    if any(count != retained_track_count for _, count in plot_results):
        raise RuntimeError("A plotted track count does not match the filtered track count.")
    if has_forcing_cycles:
        print(
            f"Saved {interaction_plot_path} "
            f"({len(interactions)} crossovers among "
            f"{len(set((item['first_track_id'], item['second_track_id']) for item in interactions))} track pairs)"
        )
    if args.no_preview:
        return 0
    print("Batch calculation complete. Starting terminal frame prompt.")

    positive_color = str(config["threshold_mask"].get("positive_color", "#ffb000"))
    negative_color = str(config["threshold_mask"].get("negative_color", "#00a6ff"))
    marker_color = str(config["plot"].get("marker_color", "black"))
    marker_size = float(config["plot"].get("marker_size", 30.0))

    def load(index: int) -> dict:
        return load_saved_frame(output_path, hmaxima_path, group_names[index])

    def overlay(axis, frame: dict) -> None:
        # Replace the vorticity image with flat colors on a white background.
        axis.images[0].set_visible(False)
        colors = np.ones((*frame["positive_mask"].shape, 3), dtype=float)
        colors[frame["positive_mask"]] = mcolors.to_rgb(positive_color)
        colors[frame["negative_mask"]] = mcolors.to_rgb(negative_color)
        axis.imshow(
            colors,
            origin="lower",
            extent=image_extent(frame),
            interpolation="nearest",
            aspect="equal",
        )
        axis.scatter(
            frame["extrema_x"],
            frame["extrema_y"],
            s=marker_size,
            c=marker_color,
            marker="x",
        )
        axis.set_title(
            f"Vorticity threshold masks | {frame['source_filename']} | "
            f"t = {frame['time']:.8g} | vorticity threshold = {threshold:g} | "
            f"minimum area = {minimum_area:g}"
        )
        axis.legend(handles=(
            Patch(facecolor=positive_color, label=f"ω ≥ {threshold:g}"),
            Patch(facecolor=negative_color, label=f"ω ≤ {-threshold:g}"),
            Line2D([], [], color=marker_color, marker="x", linestyle="none", label="retained h-maximum"),
        ))

    browse_frames(
        len(group_names),
        load,
        config["plot"],
        overlay,
        preview_path=output_folder / "threshold_masks_preview.png",
        show_colorbar=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
