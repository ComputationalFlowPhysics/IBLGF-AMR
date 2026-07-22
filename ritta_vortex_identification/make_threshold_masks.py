"""Create positive and negative vorticity-threshold masks for every frame."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import h5py
import matplotlib.colors as mcolors
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
            print(
                f"[{index + 1}/{len(group_names)}] {frame['source_filename']}: "
                f"kept {len(result['positive_region_areas'])}/{result['positive_regions_found']} positive and "
                f"{len(result['negative_region_areas'])}/{result['negative_regions_found']} negative regions; "
                f"marked {len(result['extrema_x'])}/{result['extrema_found']} extrema"
            )

    print(f"Saved {output_path}")
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
