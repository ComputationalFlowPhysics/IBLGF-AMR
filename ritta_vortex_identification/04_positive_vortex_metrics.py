"""Stage 4: measure positive circulation and its vorticity centroid."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import h5py
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

from common import (
    discover_frames,
    largest_successful_fit_index,
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
    "boundary_radius",
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


def eligible_center_indices(records: list[dict]) -> list[int]:
    """Return fitted candidates that can participate in temporal tracking."""
    return [
        index
        for index, record in enumerate(records)
        if record["fit_success"] and math.isfinite(record["_fit_x"]) and math.isfinite(record["_fit_y"])
    ]


def nearest_one_to_one_matches(
    source_centers: dict[int, tuple[float, float]],
    records: list[dict],
    candidate_indices: list[int],
    max_displacement: float,
    unavailable_indices: set[int] | None = None,
) -> dict[int, int]:
    """Match each source to its nearest candidate, with each candidate used at most once."""
    if not candidate_indices:
        return {}

    proposals = []
    for source_id, (source_x, source_y) in source_centers.items():
        nearest_index = min(
            candidate_indices,
            key=lambda index: (
                math.hypot(records[index]["_fit_x"] - source_x, records[index]["_fit_y"] - source_y),
                index,
            ),
        )
        distance = math.hypot(
            records[nearest_index]["_fit_x"] - source_x,
            records[nearest_index]["_fit_y"] - source_y,
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


def assign_vortex_tracks(
    records_by_frame: list[list[dict]],
    max_displacement: float,
    new_track_max_displacement: float,
) -> None:
    """Track consecutive-frame centers and require two detections to start a track."""
    active_tracks = {}
    tentative_records = []
    next_vortex_id = 1

    for records in records_by_frame:
        eligible = eligible_center_indices(records)

        # Existing tracks link only from the immediately preceding analyzed frame.
        existing_matches = nearest_one_to_one_matches(
            active_tracks,
            records,
            eligible,
            max_displacement,
        )
        used_indices = set(existing_matches.values())
        for vortex_id, index in existing_matches.items():
            records[index]["vortex_id"] = vortex_id

        # A previous unassigned candidate becomes a track only when the current
        # frame contains a unique, still-unassigned nearest neighbor.
        tentative_centers = {
            index: (record["_fit_x"], record["_fit_y"])
            for index, record in enumerate(tentative_records)
        }
        confirmed_matches = nearest_one_to_one_matches(
            tentative_centers,
            records,
            eligible,
            new_track_max_displacement,
            unavailable_indices=used_indices,
        )
        for tentative_index, current_index in confirmed_matches.items():
            vortex_id = next_vortex_id
            next_vortex_id += 1
            tentative_records[tentative_index]["vortex_id"] = vortex_id
            records[current_index]["vortex_id"] = vortex_id
            used_indices.add(current_index)

        tentative_records = [records[index] for index in eligible if index not in used_indices]
        active_tracks = {
            int(records[index]["vortex_id"]): (records[index]["_fit_x"], records[index]["_fit_y"])
            for index in eligible
            if records[index]["vortex_id"] != ""
        }


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
    parser.add_argument(
        "--input-results-dir",
        type=Path,
        help="Directory containing hmaxima.h5 and regions.h5 (defaults to the run's results directory).",
    )
    parser.add_argument(
        "--fits-file",
        type=Path,
        help="Fit HDF5 file to measure (defaults to fits.h5 in the input results directory).",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        help="Output CSV path (defaults to positive_vortex_metrics.csv in the run's results directory).",
    )
    args = parser.parse_args()

    config = load_config(args.config_file)
    output_folder = result_folder(args.run_folder)
    input_folder = args.input_results_dir.expanduser().resolve() if args.input_results_dir else output_folder
    hmaxima_path = input_folder / "hmaxima.h5"
    regions_path = input_folder / "regions.h5"
    fits_path = args.fits_file.expanduser().resolve() if args.fits_file else input_folder / "fits.h5"
    csv_path = (
        args.output_file.expanduser().resolve()
        if args.output_file
        else output_folder / "positive_vortex_metrics.csv"
    )
    dependencies = (
        (hmaxima_path, "01_find_hmaxima.py"),
        (regions_path, "02_make_regions.py"),
        (fits_path, "03_fit_vortices.py"),
    )
    for path, script_name in dependencies:
        if not path.is_file():
            print(f"{path} does not exist.")
            if args.input_results_dir is None and args.fits_file is None:
                print("Run this exact command first:")
                print(stage_command(script_name, args.run_folder, args.config_file))
            return 1

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    frame_paths = discover_frames(args.run_folder, config)
    paths_by_name = {path.name: path for path in frame_paths}
    source_indices_by_name = {path.name: index for index, path in enumerate(frame_paths)}
    metadata = simulation_metadata(args.run_folder, config)
    records_by_frame = []
    ordered_frame_names = []
    tracking_config = config.get("tracking", {})
    max_displacement = float(tracking_config.get("max_displacement", 0.5))
    new_track_max_displacement = float(tracking_config.get("new_track_max_displacement", 0.5))
    if not math.isfinite(max_displacement) or max_displacement <= 0.0:
        raise ValueError("[tracking] max_displacement must be finite and greater than zero.")
    if not math.isfinite(new_track_max_displacement) or new_track_max_displacement <= 0.0:
        raise ValueError("[tracking] new_track_max_displacement must be finite and greater than zero.")

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
                paths_by_name[frame_name],
                source_indices_by_name[frame_name],
                config,
                metadata,
                include_cells=True,
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
                    "boundary_radius": radius,
                    "circulation_positive": circulation,
                    "x_center_positive": x_center,
                    "y_center_positive": y_center,
                    "x_displacement": x_center,
                    "_fit_x": float(positive_center[0]),
                    "_fit_y": float(positive_center[1]),
                })

            if not frame_records:
                # Keep an empty row so later plots retain this frame and show a gap.
                frame_records.append({
                    "frame_index": frame_index,
                    "frame_name": frame_name,
                    "time": frame["time"],
                    "candidate_id": "",
                    "vortex_id": "",
                    "fit_success": False,
                    "boundary_radius": math.nan,
                    "circulation_positive": math.nan,
                    "x_center_positive": math.nan,
                    "y_center_positive": math.nan,
                    "x_displacement": math.nan,
                })
            records_by_frame.append(frame_records)
            valid_count = sum(math.isfinite(record["circulation_positive"]) for record in frame_records)
            print(f"[{frame_index + 1}/{len(group_names)}] {frame_name}: {valid_count} measured vortices")

    assign_vortex_tracks(records_by_frame, max_displacement, new_track_max_displacement)
    all_records = [record for frame_records in records_by_frame for record in frame_records]
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
        index = largest_successful_fit_index(frame["success"], frame["boundary_radius"])
        if index is not None:
            candidate_id = int(frame["candidate_ids"][index])
            radius = frame["boundary_radius"][index]
            fitted_center = frame["positive_centers"][index]
            record = frame["metric_records"].get(candidate_id)
            if np.all(np.isfinite(fitted_center)):
                axis.add_patch(Circle(fitted_center, radius, fill=False, edgecolor=boundary_color, linewidth=line_width))
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
                    f"vortex {record['vortex_id']} / candidate {candidate_id}: "
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
        axis.legend(
            handles=[
                Line2D([], [], color=boundary_color, linewidth=line_width, label="Fitted vortex boundary"),
                Line2D(
                    [],
                    [],
                    linestyle="none",
                    marker="o",
                    markersize=math.sqrt(marker_size),
                    markerfacecolor=centroid_color,
                    markeredgecolor="black",
                    label="Circulation-weighted center",
                ),
            ],
            loc="upper right",
        )

    browse_frames(
        len(group_names),
        load,
        config["plot"],
        overlay,
        csv_path.parent / f"{csv_path.stem}_preview.png",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
