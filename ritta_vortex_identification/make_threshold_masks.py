"""Create positive and negative vorticity-threshold masks for every frame."""

from __future__ import annotations

import argparse
import csv
import math
import subprocess
import sys
from pathlib import Path

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


def remove_small_regions(mask: np.ndarray, dx: float, minimum_area: float) -> tuple[np.ndarray, int, np.ndarray]:
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


def extrema_records(frame_index: int, frame: dict) -> list[dict]:
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


def nearest_one_to_one_matches(
    source_centers: dict[int, tuple[float, float]],
    records: list[dict],
    max_displacement: float,
    unavailable_indices: set[int] | None = None,
    source_displacement_limits: dict[int, float] | None = None,
) -> dict[int, int]:
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
        displacement_limit = (
            source_displacement_limits.get(source_id, max_displacement)
            if source_displacement_limits is not None
            else max_displacement
        )
        if distance <= displacement_limit:
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
    records_by_frame: list[list[dict]],
    max_displacement: float,
    new_track_max_displacement: float,
    max_missed_frames: int,
) -> None:
    """Track retained extrema, bridge short gaps, and confirm new tracks twice."""
    active_tracks = {}
    tentative_records = []
    next_track_id = 1

    for records in records_by_frame:
        active_centers = {
            track_id: (state["x"], state["y"])
            for track_id, state in active_tracks.items()
        }
        # A track missing m frames is now m + 1 frame intervals from its last
        # detection, so allow its displacement gate to grow by that factor.
        active_limits = {
            track_id: max_displacement * (state["missed_frames"] + 1)
            for track_id, state in active_tracks.items()
        }
        existing_matches = nearest_one_to_one_matches(
            active_centers,
            records,
            max_displacement,
            source_displacement_limits=active_limits,
        )
        used_indices = set(existing_matches.values())
        next_active_tracks = {}
        for track_id, index in existing_matches.items():
            records[index]["track_id"] = track_id
            next_active_tracks[track_id] = {
                "x": records[index]["x"],
                "y": records[index]["y"],
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
            track_id = next_track_id
            next_track_id += 1
            tentative_records[tentative_index]["track_id"] = track_id
            records[current_index]["track_id"] = track_id
            used_indices.add(current_index)
            next_active_tracks[track_id] = {
                "x": records[current_index]["x"],
                "y": records[current_index]["y"],
                "missed_frames": 0,
            }

        tentative_records = [
            records[index]
            for index in range(len(records))
            if index not in used_indices
        ]
        active_tracks = next_active_tracks


def filter_short_tracks(
    records_by_frame: list[list[dict]],
    minimum_track_points: int,
) -> tuple[int, int]:
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


def write_extrema_tracks(path: Path, records_by_frame: list[list[dict]]) -> None:
    """Write every retained extremum, leaving unconfirmed track IDs blank."""
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRACK_COLUMNS)
        writer.writeheader()
        for records in records_by_frame:
            for record in records:
                row = dict(record)
                row["track_id"] = "" if row["track_id"] is None else row["track_id"]
                writer.writerow(row)


def track_colors(track_count: int) -> list:
    """Return stable categorical colors for the plotted track IDs."""
    colors = []
    for name in ("tab20", "tab20b", "tab20c"):
        colors.extend(plt.get_cmap(name).colors)
    if track_count <= len(colors):
        return colors[:track_count]
    return list(plt.get_cmap("turbo")(np.linspace(0.0, 1.0, track_count)))


def save_extrema_track_plot(
    path: Path,
    records_by_frame: list[list[dict]],
    figure_size: tuple[float, float],
) -> int:
    """Plot x coordinate versus simulation time with one color per confirmed track."""
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
            [record["time"] for record in records],
            [record["x"] for record in records],
            color=color,
            marker="o",
            markersize=4,
            linewidth=1.5,
            label=f"track {track_id}",
        )

    axis.set_xlabel("simulation time")
    axis.set_ylabel("retained h-maximum x coordinate")
    axis.set_title("Tracked retained h-maxima: x coordinate versus simulation time")
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Create positive and negative vorticity-threshold masks.")
    parser.add_argument("run_folder", type=Path)
    parser.add_argument("config_file", type=Path)
    parser.add_argument("--stride", type=int, default=1, help="Process every Nth sorted HDF5 frame.")
    parser.add_argument("--no-preview", action="store_true", help="Skip the terminal preview and preview PNG.")
    args = parser.parse_args()
    if args.stride < 1:
        parser.error("--stride must be a positive integer.")

    config = load_config(args.config_file)
    if "threshold_mask" not in config:
        raise ValueError("Missing [threshold_mask] section in the TOML config.")
    threshold = require_positive(config, "threshold_mask", "vorticity_threshold")
    minimum_area = require_nonnegative(config, "threshold_mask", "minimum_region_area")
    output_folder = result_folder(args.run_folder)
    hmaxima_path = output_folder / "hmaxima.h5"
    output_path = output_folder / "threshold_masks.h5"
    track_csv_path = output_folder / "threshold_hmaxima_tracks.csv"
    track_plot_path = output_folder / "threshold_hmaxima_x_vs_time.png"
    records_by_frame = []

    # Stage 1 runs first and writes the extrema and physical raster consumed below.
    subprocess.run([
        sys.executable,
        str(Path(__file__).with_name("01_find_hmaxima.py")),
        str(args.run_folder),
        str(args.config_file),
        "--stride",
        str(args.stride),
        "--no-preview",
    ], check=True)

    # Calculate and save every frame before starting the terminal preview.
    with h5py.File(hmaxima_path, "r") as maxima, h5py.File(output_path, "w") as output:
        group_names = read_frame_order(maxima)
        output.attrs["schema"] = "ritta_vorticity_threshold_masks_v2"
        output.attrs["run_folder"] = str(args.run_folder.expanduser().resolve())
        output.attrs["config_file"] = str(args.config_file.expanduser().resolve())
        output.attrs["source_hmaxima"] = str(hmaxima_path.resolve())
        output.attrs["vorticity_threshold"] = threshold
        output.attrs["minimum_region_area"] = minimum_area
        output.attrs["stride"] = args.stride
        output.attrs["connectivity"] = 8
        write_string_dataset(output, "frame_order", group_names)

        for index, group_name in enumerate(group_names):
            source = maxima[group_name]
            frame = frame_from_hmaxima(source)
            result = make_masks(frame, threshold, minimum_area)
            result.update(retained_extrema(result, source))
            save_frame(output.create_group(group_name), result)
            records_by_frame.append(extrema_records(index, result))
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

    assign_extrema_tracks(
        records_by_frame,
        max_displacement,
        new_track_max_displacement,
        max_missed_frames_setting,
    )
    retained_track_count, discarded_track_count = filter_short_tracks(
        records_by_frame,
        minimum_track_points_setting,
    )
    write_extrema_tracks(track_csv_path, records_by_frame)
    track_count = save_extrema_track_plot(
        track_plot_path,
        records_by_frame,
        (
            float(config["plot"].get("figure_width", 10.0)),
            float(config["plot"].get("figure_height", 7.0)),
        ),
    )
    print(f"Saved {track_csv_path}")
    print(
        f"Saved {track_plot_path} "
        f"({track_count} tracks with at least {minimum_track_points_setting} points; "
        f"discarded {discarded_track_count} shorter tracks)"
    )
    if track_count != retained_track_count:
        raise RuntimeError("The plotted track count does not match the filtered track count.")
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
