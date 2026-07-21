"""Stage 4: measure positive circulation and its vorticity centroid."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import h5py
import numpy as np
from matplotlib.patches import Circle

from common import (
    discover_frames,
    load_config,
    load_vorticity_frame,
    read_frame_order,
    result_folder,
    simulation_metadata,
    stage_command,
)
from plot_vorticity import browse_frames


CSV_COLUMNS = (
    "frame_index",
    "frame_name",
    "time",
    "candidate_id",
    "vortex_id",
    "fit_success",
    "circulation_positive",
    "x_center_positive",
    "y_center_positive",
    "x_displacement",
)


def positive_metrics(cells: dict, x_c: float, y_c: float, radius: float) -> tuple[float, float, float]:
    """Integrate positive original-cell vorticity inside the fitted circle."""
    omega = cells["vorticity"]
    inside = (cells["x"] - x_c) ** 2 + (cells["y"] - y_c) ** 2 <= radius ** 2
    # Negative vorticity and cells outside the fitted positive boundary contribute nothing.
    selected = inside & np.isfinite(omega) & (omega > 0.0)
    if not np.any(selected):
        return math.nan, math.nan, math.nan
    # omega * physical cell area is each cell's discrete circulation contribution.
    weights = omega[selected] * cells["area"][selected]
    circulation = float(np.sum(weights))
    if circulation == 0.0:
        return math.nan, math.nan, math.nan
    x_center = float(np.sum(cells["x"][selected] * weights) / circulation)
    y_center = float(np.sum(cells["y"][selected] * weights) / circulation)
    return circulation, x_center, y_center


def assign_vortex_tracks(records: list[dict], tracks: dict[int, tuple[float, float]], next_id: int) -> int:
    """Match closest centers globally, allowing only nondecreasing x motion."""
    eligible = [
        index
        for index, record in enumerate(records)
        if record["fit_success"] and math.isfinite(record["_fit_x"]) and math.isfinite(record["_fit_y"])
    ]
    # Build every allowed old-track/new-candidate pairing, then take shortest pairs first.
    pairs = []
    for vortex_id, (previous_x, previous_y) in tracks.items():
        for index in eligible:
            current_x = records[index]["_fit_x"]
            current_y = records[index]["_fit_y"]
            if current_x < previous_x:
                continue
            distance = math.hypot(current_x - previous_x, current_y - previous_y)
            pairs.append((distance, vortex_id, index))

    used_tracks = set()
    used_records = set()
    for _, vortex_id, index in sorted(pairs):
        if vortex_id in used_tracks or index in used_records:
            continue
        records[index]["vortex_id"] = vortex_id
        used_tracks.add(vortex_id)
        used_records.add(index)

    for index in eligible:
        if index not in used_records:
            records[index]["vortex_id"] = next_id
            next_id += 1

    for index in eligible:
        vortex_id = records[index]["vortex_id"]
        tracks[vortex_id] = (records[index]["_fit_x"], records[index]["_fit_y"])
    # Unmatched tracks stay in this dictionary so they can reconnect after missing frames.
    return next_id


def fit_rows(group: h5py.Group) -> list[dict]:
    """Convert one fits.h5 frame group into ordinary dictionaries."""
    return [
        {
            "candidate_id": int(candidate_id),
            "fit_success": bool(success),
            "parameters": parameters,
            "radius": float(radius),
            "positive_center": positive_center,
        }
        for candidate_id, success, parameters, radius, positive_center in zip(
            group["candidate_ids"][:],
            group["success"][:],
            group["parameters"][:],
            group["boundary_radius"][:],
            group["positive_centers"][:],
        )
    ]


def write_metrics(path: Path, records: list[dict]) -> None:
    """Write the final per-frame, per-vortex CSV table."""
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for record in records:
            writer.writerow({name: record.get(name, "") for name in CSV_COLUMNS})


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure positive-vortex circulation and centers.")
    parser.add_argument("run_folder", type=Path)
    parser.add_argument("config_file", type=Path)
    parser.add_argument("--no-preview", action="store_true", help="Skip the terminal preview and preview PNG.")
    args = parser.parse_args()

    config = load_config(args.config_file)
    output_folder = result_folder(args.run_folder)
    dependencies = (
        ("hmaxima.h5", "01_find_hmaxima.py"),
        ("regions.h5", "02_make_regions.py"),
        ("fits.h5", "03_fit_vortices.py"),
    )
    for filename, script_name in dependencies:
        if not (output_folder / filename).is_file():
            print(f"{filename} does not exist. Run this exact command first:")
            print(stage_command(script_name, args.run_folder, args.config_file))
            return 1

    hmaxima_path = output_folder / "hmaxima.h5"
    regions_path = output_folder / "regions.h5"
    fits_path = output_folder / "fits.h5"
    csv_path = output_folder / "positive_vortex_metrics.csv"
    frame_paths = discover_frames(args.run_folder, config)
    paths_by_name = {path.name: path for path in frame_paths}
    metadata = simulation_metadata(args.run_folder, config)
    all_records = []
    ordered_frame_names = []
    tracks = {}
    next_vortex_id = 1

    # All three saved stages must describe exactly the same frames and candidate IDs.
    with (
        h5py.File(hmaxima_path, "r") as maxima,
        h5py.File(regions_path, "r") as regions,
        h5py.File(fits_path, "r") as fits,
    ):
        group_names = read_frame_order(maxima)
        if group_names != read_frame_order(regions) or group_names != read_frame_order(fits):
            raise ValueError("Frame order differs among hmaxima.h5, regions.h5, and fits.h5.")

        for frame_index, group_name in enumerate(group_names):
            frame_name = str(maxima[group_name].attrs["source_filename"])
            ordered_frame_names.append(frame_name)
            if frame_name not in paths_by_name:
                raise FileNotFoundError(f"Original frame is missing: {frame_name}")
            maximum_ids = maxima[group_name]["candidate_ids"][:]
            region_ids = regions[group_name]["candidate_ids"][:]
            fitted = fit_rows(fits[group_name])
            fit_ids = np.asarray([item["candidate_id"] for item in fitted])
            if not np.array_equal(maximum_ids, region_ids) or not np.array_equal(maximum_ids, fit_ids):
                raise ValueError(f"Candidate IDs disagree among saved stages for {frame_name}.")

            frame = load_vorticity_frame(
                paths_by_name[frame_name], frame_index, config, metadata, include_cells=True
            )
            frame_records = []
            for fit in fitted:
                parameters = fit["parameters"]
                radius = fit["radius"]
                positive_center = fit["positive_center"]
                usable = fit["fit_success"] and np.all(np.isfinite(parameters)) and math.isfinite(radius)
                if usable:
                    # Measure the original visible AMR cells, never the fitted Gaussian values.
                    circulation, x_center, y_center = positive_metrics(
                        frame["cells"], positive_center[0], positive_center[1], radius
                    )
                else:
                    circulation, x_center, y_center = math.nan, math.nan, math.nan
                frame_records.append({
                    "frame_index": frame_index,
                    "frame_name": frame_name,
                    "time": frame["time"],
                    "candidate_id": fit["candidate_id"],
                    "vortex_id": "",
                    "fit_success": fit["fit_success"],
                    "circulation_positive": circulation,
                    "x_center_positive": x_center,
                    "y_center_positive": y_center,
                    "x_displacement": x_center,
                    "_fit_x": float(positive_center[0]),
                    "_fit_y": float(positive_center[1]),
                })

            # Track every fitted positive candidate using the approved forward-x nearest match.
            next_vortex_id = assign_vortex_tracks(frame_records, tracks, next_vortex_id)
            if not frame_records:
                # Keep an empty row so later plots retain this frame and show a gap.
                frame_records.append({
                    "frame_index": frame_index,
                    "frame_name": frame_name,
                    "time": frame["time"],
                    "candidate_id": "",
                    "vortex_id": "",
                    "fit_success": False,
                    "circulation_positive": math.nan,
                    "x_center_positive": math.nan,
                    "y_center_positive": math.nan,
                    "x_displacement": math.nan,
                })
            all_records.extend(frame_records)
            valid_count = sum(math.isfinite(record["circulation_positive"]) for record in frame_records)
            print(f"[{frame_index + 1}/{len(group_names)}] {frame_name}: {valid_count} measured vortices")

    write_metrics(csv_path, all_records)
    print(f"Saved {csv_path}")
    if args.no_preview:
        return 0
    print("Batch calculation complete. Starting terminal frame prompt.")
    # The preview reads the finished CSV records and saved fits after all calculations end.
    records_by_frame = {}
    for record in all_records:
        records_by_frame.setdefault(record["frame_index"], {})[record["candidate_id"]] = record
    ordered_paths = [paths_by_name[name] for name in ordered_frame_names]

    def load(index: int) -> dict:
        frame = load_vorticity_frame(ordered_paths[index], index, config, metadata)
        with h5py.File(fits_path, "r") as fits:
            group = fits[group_names[index]]
            for name in ("candidate_ids", "success", "boundary_radius", "positive_centers"):
                frame[name] = group[name][:]
        frame["metric_records"] = records_by_frame.get(index, {})
        return frame

    boundary_color = str(config["plot"].get("positive_marker_color", "black"))
    centroid_color = str(config["plot"].get("mask_color", "#ffd400"))
    line_width = float(config["plot"].get("region_line_width", 1.5))
    marker_size = float(config["plot"].get("marker_size", 48.0))
    text_size = float(config["plot"].get("fit_text_size", 8.0))

    def overlay(axis, frame: dict) -> None:
        lines = []
        for candidate_id, success, radius, fitted_center in zip(
            frame["candidate_ids"], frame["success"], frame["boundary_radius"], frame["positive_centers"]
        ):
            record = frame["metric_records"].get(int(candidate_id))
            if bool(success) and np.all(np.isfinite(fitted_center)) and np.isfinite(radius):
                axis.add_patch(Circle(fitted_center, radius, fill=False, edgecolor=boundary_color, linewidth=line_width))
                axis.scatter(*fitted_center, s=marker_size, c=boundary_color, marker="x")
            if record is not None and math.isfinite(record["x_center_positive"]):
                axis.scatter(
                    record["x_center_positive"],
                    record["y_center_positive"],
                    s=marker_size,
                    c=centroid_color,
                    marker="o",
                    edgecolors="black",
                )
                lines.append(
                    f"vortex {record['vortex_id']} / candidate {int(candidate_id)}: "
                    f"Gamma+={record['circulation_positive']:.6g}"
                )
        if lines:
            axis.text(
                0.01,
                0.99,
                "\n".join(lines),
                transform=axis.transAxes,
                va="top",
                fontsize=text_size,
                bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
            )

    browse_frames(
        len(group_names),
        load,
        config["plot"],
        overlay,
        output_folder / "positive_vortex_metrics_preview.png",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
