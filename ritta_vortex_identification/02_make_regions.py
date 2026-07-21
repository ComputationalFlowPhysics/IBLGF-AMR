"""Stage 2: construct the authoritative rectangle for every saved candidate."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import h5py
import numpy as np
from matplotlib.patches import Rectangle

from common import load_config, read_frame_order, require_positive, result_folder, stage_command, write_string_dataset
from plot_vorticity import browse_frames


def make_regions(peak_x: np.ndarray, peak_y: np.ndarray, x: np.ndarray, y: np.ndarray, dx: float, config: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build intended and domain-clipped rectangles for every saved peak."""
    alpha_x = require_positive(config, "region", "alpha_x")
    alpha_r = require_positive(config, "region", "alpha_r")
    alpha = require_positive(config, "region", "alpha")
    ell_x = 4.0 * alpha_x / math.sqrt(2.0 * alpha)
    ell_y = 4.0 * alpha_r / math.sqrt(2.0 * alpha)

    # Enclose the positive peak, its y-mirrored partner, and both physical buffers.
    intended = np.column_stack((
        peak_x - ell_x,
        peak_x + ell_x,
        -np.abs(peak_y) - ell_y,
        np.abs(peak_y) + ell_y,
    )) if len(peak_x) else np.empty((0, 4), dtype=float)

    domain = np.asarray((x[0] - 0.5 * dx, x[-1] + 0.5 * dx, y[0] - 0.5 * dx, y[-1] + 0.5 * dx))
    # Save the intended rectangle and a second copy clipped to available cells.
    clamped = intended.copy()
    if len(clamped):
        clamped[:, 0] = np.maximum(clamped[:, 0], domain[0])
        clamped[:, 1] = np.minimum(clamped[:, 1], domain[1])
        clamped[:, 2] = np.maximum(clamped[:, 2], domain[2])
        clamped[:, 3] = np.minimum(clamped[:, 3], domain[3])

    positive = np.column_stack((peak_x, peak_y)) if len(peak_x) else np.empty((0, 2), dtype=float)
    negative = np.column_stack((peak_x, -peak_y)) if len(peak_x) else np.empty((0, 2), dtype=float)
    return intended, clamped, positive, negative


def load_preview_frame(hmaxima_path: Path, regions_path: Path, group_name: str) -> dict:
    with h5py.File(hmaxima_path, "r") as maxima, h5py.File(regions_path, "r") as regions:
        source = maxima[group_name]
        region = regions[group_name]
        return {
            "source_filename": str(source.attrs["source_filename"]),
            "time": float(source.attrs["simulation_time"]),
            "dx": float(source.attrs["dx"]),
            "x": source["x"][:],
            "y": source["y"][:],
            "vorticity": source["vorticity"][:],
            "candidate_ids": region["candidate_ids"][:],
            "clamped_bounds": region["clamped_bounds"][:],
            "positive_points": region["positive_points"][:],
            "negative_points": region["negative_points"][:],
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build fitting rectangles from saved h-maxima.")
    parser.add_argument("run_folder", type=Path)
    parser.add_argument("config_file", type=Path)
    parser.add_argument("--no-preview", action="store_true", help="Skip the terminal preview and preview PNG.")
    args = parser.parse_args()

    config = load_config(args.config_file)
    require_positive(config, "region", "alpha_x")
    require_positive(config, "region", "alpha_r")
    require_positive(config, "region", "alpha")
    output_folder = result_folder(args.run_folder)
    hmaxima_path = output_folder / "hmaxima.h5"
    regions_path = output_folder / "regions.h5"

    if not hmaxima_path.is_file():
        command = stage_command("01_find_hmaxima.py", args.run_folder, args.config_file)
        print(f"hmaxima.h5 does not exist. Run this exact command first:\n{command}")
        return 1

    # Read the saved candidates instead of repeating Stage 1 detection.
    with h5py.File(hmaxima_path, "r") as maxima:
        group_names = read_frame_order(maxima)
        with h5py.File(regions_path, "w") as output:
            output.attrs["schema"] = "ritta_regions_v1"
            output.attrs["source_hmaxima"] = str(hmaxima_path.resolve())
            output.attrs["config_file"] = str(args.config_file.expanduser().resolve())
            output.attrs["alpha_x"] = require_positive(config, "region", "alpha_x")
            output.attrs["alpha_r"] = require_positive(config, "region", "alpha_r")
            output.attrs["alpha"] = require_positive(config, "region", "alpha")
            write_string_dataset(output, "frame_order", group_names)
            for index, group_name in enumerate(group_names):
                source = maxima[group_name]
                candidate_ids = source["candidate_ids"][:]
                peak_x = source["peak_x"][:]
                peak_y = source["peak_y"][:]
                x = source["x"][:]
                y = source["y"][:]
                dx = float(source.attrs["dx"])
                intended, clamped, positive, negative = make_regions(
                    peak_x, peak_y, x, y, dx, config
                )

                group = output.create_group(group_name)
                group.attrs["source_filename"] = source.attrs["source_filename"]
                group.attrs["simulation_time"] = source.attrs["simulation_time"]
                group.attrs["time_step"] = source.attrs["time_step"]
                group.create_dataset("candidate_ids", data=candidate_ids)
                group.create_dataset("intended_bounds", data=intended)
                group.create_dataset("clamped_bounds", data=clamped)
                group.create_dataset("positive_points", data=positive)
                group.create_dataset("negative_points", data=negative)
                print(f"[{index + 1}/{len(group_names)}] {source.attrs['source_filename']}: {len(candidate_ids)} regions")

    print(f"Saved {regions_path}")
    if args.no_preview:
        return 0
    print("Batch calculation complete. Starting terminal frame prompt.")

    # Preview data comes entirely from the two saved stage files.
    region_color = str(config["plot"].get("region_color", "#00aa55"))
    region_line_width = float(config["plot"].get("region_line_width", 1.5))
    positive_color = str(config["plot"].get("positive_marker_color", "black"))
    negative_color = str(config["plot"].get("negative_marker_color", "#7b2cbf"))
    marker_size = float(config["plot"].get("marker_size", 48.0))

    def load(index: int) -> dict:
        return load_preview_frame(hmaxima_path, regions_path, group_names[index])

    def overlay(axis, frame: dict) -> None:
        for candidate_id, bounds, positive, negative in zip(
            frame["candidate_ids"],
            frame["clamped_bounds"],
            frame["positive_points"],
            frame["negative_points"],
        ):
            x0, x1, y0, y1 = bounds
            axis.add_patch(Rectangle(
                (x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=region_color, linewidth=region_line_width
            ))
            axis.scatter(*positive, s=marker_size, c=positive_color, marker="x")
            axis.scatter(*negative, s=marker_size, c=negative_color, marker="+")
            axis.annotate(str(int(candidate_id)), positive, xytext=(5, 5), textcoords="offset points")

    browse_frames(
        len(group_names),
        load,
        config["plot"],
        overlay,
        output_folder / "regions_preview.png",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
