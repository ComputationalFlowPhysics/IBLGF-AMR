"""Stage 1: compute positive h-maxima for every output frame."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.colors as mcolors
import numpy as np
from scipy.ndimage import label, maximum_filter

from common import (
    discover_frames,
    load_config,
    load_vorticity_frame,
    require_nonnegative,
    require_positive,
    result_folder,
    simulation_metadata,
    write_string_dataset,
)
from plot_vorticity import browse_frames, image_extent


EIGHT_CONNECTED = np.ones((3, 3), dtype=bool)


def merge_close_maxima(
    component_labels: np.ndarray,
    peak_x: np.ndarray,
    peak_y: np.ndarray,
    peak_vorticity: np.ndarray,
    merge_distance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Merge connected groups of peaks separated by at most merge_distance."""
    peak_x = np.asarray(peak_x, dtype=float)
    peak_y = np.asarray(peak_y, dtype=float)
    peak_vorticity = np.asarray(peak_vorticity, dtype=float)
    candidate_count = len(peak_x)
    if candidate_count <= 1 or merge_distance == 0.0:
        candidate_ids = np.arange(1, candidate_count + 1, dtype=np.int32)
        return component_labels, candidate_ids, peak_x, peak_y, peak_vorticity

    distances = np.hypot(
        peak_x[:, np.newaxis] - peak_x[np.newaxis, :],
        peak_y[:, np.newaxis] - peak_y[np.newaxis, :],
    )
    adjacent = distances <= merge_distance
    unseen = set(range(candidate_count))
    groups = []
    while unseen:
        first = min(unseen)
        unseen.remove(first)
        group = [first]
        stack = [first]
        while stack:
            current = stack.pop()
            neighbors = [
                candidate
                for candidate in unseen
                if adjacent[current, candidate]
            ]
            for candidate in neighbors:
                unseen.remove(candidate)
                group.append(candidate)
                stack.append(candidate)
        groups.append(sorted(group))

    merged_labels = np.zeros(component_labels.shape, dtype=np.int32)
    merged_x = []
    merged_y = []
    merged_vorticity = []
    for merged_id, group in enumerate(groups, start=1):
        for old_index in group:
            merged_labels[component_labels == old_index + 1] = merged_id
        merged_x.append(float(np.mean(peak_x[group])))
        merged_y.append(float(np.mean(peak_y[group])))
        merged_vorticity.append(float(np.max(peak_vorticity[group])))

    candidate_ids = np.arange(1, len(groups) + 1, dtype=np.int32)
    return (
        merged_labels,
        candidate_ids,
        np.asarray(merged_x, dtype=float),
        np.asarray(merged_y, dtype=float),
        np.asarray(merged_vorticity, dtype=float),
    )


def reconstruct_by_dilation(omega: np.ndarray, h: float, tolerance: float) -> np.ndarray:
    """Apply the specified geodesic dilation until its infinity-norm change is small."""
    valid = np.isfinite(omega)
    # The marker starts h below the vorticity field; the field itself is the upper mask.
    marker = np.where(valid, omega - h, -np.inf)
    mask = np.where(valid, omega, -np.inf)

    # Eight-neighbor dilation propagates peaks without ever exceeding omega.
    while True:
        dilated = maximum_filter(marker, footprint=EIGHT_CONNECTED, mode="constant", cval=-np.inf)
        updated = np.minimum(mask, dilated)
        change = float(np.max(np.abs(updated[valid] - marker[valid]))) if np.any(valid) else 0.0
        marker = updated
        if change < tolerance:
            break
    return np.where(valid, marker, np.nan)


def find_hmaxima(frame: dict, config: dict) -> dict:
    h = require_positive(config, "hmaxima", "h")
    reconstruction_tolerance = require_positive(config, "hmaxima", "reconstruction_tolerance")
    mask_tolerance = require_nonnegative(config, "hmaxima", "h_mask_tolerance")
    merge_distance = require_nonnegative(config, "hmaxima", "merge_distance")
    omega = np.asarray(frame["vorticity"], dtype=float)
    reconstruction = reconstruct_by_dilation(omega, h, reconstruction_tolerance)
    dome = omega - reconstruction
    # Keep only domes that reach the full configured height h.
    candidate_mask = np.isfinite(dome) & (dome >= h - mask_tolerance)
    raw_labels, count = label(candidate_mask, structure=EIGHT_CONNECTED)

    labels = np.zeros(raw_labels.shape, dtype=np.int32)
    candidate_ids = []
    peak_x = []
    peak_y = []
    peak_vorticity = []
    next_id = 1
    for raw_id in range(1, count + 1):
        component = raw_labels == raw_id
        peak = float(np.max(omega[component]))
        if peak <= 0.0:
            continue
        # A flat maximum is represented by the physical centroid of all tied peak cells.
        plateau = component & (omega == peak)
        rows, columns = np.nonzero(plateau)
        labels[component] = next_id
        candidate_ids.append(next_id)
        peak_x.append(float(np.mean(frame["x"][columns])))
        peak_y.append(float(np.mean(frame["y"][rows])))
        peak_vorticity.append(peak)
        next_id += 1

    labels, candidate_ids, peak_x, peak_y, peak_vorticity = merge_close_maxima(
        labels,
        np.asarray(peak_x, dtype=float),
        np.asarray(peak_y, dtype=float),
        np.asarray(peak_vorticity, dtype=float),
        merge_distance,
    )
    return {
        **frame,
        "h_dome": dome,
        "hmax_mask": labels > 0,
        "component_labels": labels,
        "candidate_ids": candidate_ids,
        "peak_x": peak_x,
        "peak_y": peak_y,
        "peak_vorticity": peak_vorticity,
    }


def save_result(group: h5py.Group, result: dict) -> None:
    group.attrs["source_filename"] = result["source_filename"]
    group.attrs["source_path"] = result["source_path"]
    group.attrs["simulation_time"] = result["time"]
    group.attrs["time_step"] = result["step"]
    group.attrs["dx"] = result["dx"]
    group.create_dataset("x", data=result["x"])
    group.create_dataset("y", data=result["y"])
    group.create_dataset("vorticity", data=result["vorticity"], compression="gzip", shuffle=True)
    group.create_dataset("h_dome", data=result["h_dome"], compression="gzip", shuffle=True)
    group.create_dataset("hmax_mask", data=result["hmax_mask"].astype(np.uint8), compression="gzip")
    group.create_dataset("component_labels", data=result["component_labels"], compression="gzip")
    group.create_dataset("candidate_ids", data=result["candidate_ids"])
    group.create_dataset("peak_x", data=result["peak_x"])
    group.create_dataset("peak_y", data=result["peak_y"])
    group.create_dataset("peak_vorticity", data=result["peak_vorticity"])


def load_saved_frame(result_path: Path, group_name: str) -> dict:
    with h5py.File(result_path, "r") as handle:
        group = handle[group_name]
        return {
            "source_filename": str(group.attrs["source_filename"]),
            "time": float(group.attrs["simulation_time"]),
            "dx": float(group.attrs["dx"]),
            "x": group["x"][:],
            "y": group["y"][:],
            "vorticity": group["vorticity"][:],
            "hmax_mask": group["hmax_mask"][:].astype(bool),
            "candidate_ids": group["candidate_ids"][:],
            "peak_x": group["peak_x"][:],
            "peak_y": group["peak_y"][:],
            "peak_vorticity": group["peak_vorticity"][:],
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="Find positive h-maxima in every vorticity frame.")
    parser.add_argument("run_folder", type=Path)
    parser.add_argument("config_file", type=Path)
    parser.add_argument("--stride", type=int, default=1, help="Process every Nth sorted HDF5 frame.")
    parser.add_argument("--no-preview", action="store_true", help="Skip the terminal preview and preview PNG.")
    args = parser.parse_args()
    if args.stride < 1:
        parser.error("--stride must be a positive integer.")

    config = load_config(args.config_file)
    require_positive(config, "hmaxima", "h")
    require_positive(config, "hmaxima", "reconstruction_tolerance")
    require_nonnegative(config, "hmaxima", "h_mask_tolerance")
    require_nonnegative(config, "hmaxima", "merge_distance")
    all_frames = discover_frames(args.run_folder, config)
    selected_frames = list(enumerate(all_frames))[::args.stride]
    metadata = simulation_metadata(args.run_folder, config)
    output_path = result_folder(args.run_folder) / "hmaxima.h5"
    group_names = [path.stem for _, path in selected_frames]

    # Calculate and save every frame before entering the preview prompt.
    with h5py.File(output_path, "w") as output:
        output.attrs["schema"] = "ritta_hmaxima_v1"
        output.attrs["run_folder"] = str(args.run_folder.expanduser().resolve())
        output.attrs["config_file"] = str(args.config_file.expanduser().resolve())
        output.attrs["time_source"] = metadata["source"]
        output.attrs["h"] = require_positive(config, "hmaxima", "h")
        output.attrs["stride"] = args.stride
        output.attrs["connectivity"] = 8
        output.attrs["reconstruction_tolerance"] = require_positive(
            config, "hmaxima", "reconstruction_tolerance"
        )
        output.attrs["h_mask_tolerance"] = require_nonnegative(
            config, "hmaxima", "h_mask_tolerance"
        )
        output.attrs["merge_distance"] = require_nonnegative(
            config, "hmaxima", "merge_distance"
        )
        write_string_dataset(output, "frame_order", group_names)
        for index, ((source_index, path), group_name) in enumerate(zip(selected_frames, group_names)):
            frame = load_vorticity_frame(path, source_index, config, metadata)
            result = find_hmaxima(frame, config)
            save_result(output.create_group(group_name), result)
            print(
                f"[{index + 1}/{len(selected_frames)}] {path.name}: "
                f"{len(result['candidate_ids'])} positive candidates"
            )

    print(f"Saved {output_path}")
    if args.no_preview:
        return 0
    print("Batch calculation complete. Starting terminal frame prompt.")

    # The preview is rebuilt from hmaxima.h5, not by rerunning the detector.
    mask_color = str(config["plot"].get("mask_color", "#ffd400"))
    marker_color = str(config["plot"].get("marker_color", "black"))
    label_color = str(config["plot"].get("label_color", "black"))
    mask_cmap = mcolors.ListedColormap([(0, 0, 0, 0), mcolors.to_rgba(mask_color)])

    def load(index: int) -> dict:
        return load_saved_frame(output_path, group_names[index])

    def overlay(axis, frame: dict) -> None:
        axis.imshow(
            frame["hmax_mask"].astype(int),
            origin="lower",
            extent=image_extent(frame),
            interpolation="nearest",
            cmap=mask_cmap,
            vmin=0,
            vmax=1,
            alpha=float(config["plot"].get("mask_alpha", 0.35)),
        )
        axis.scatter(
            frame["peak_x"],
            frame["peak_y"],
            s=float(config["plot"].get("marker_size", 48.0)),
            c=marker_color,
            marker="x",
        )
        candidates = zip(
            frame["candidate_ids"], frame["peak_x"], frame["peak_y"], frame["peak_vorticity"]
        )
        for label_index, (candidate_id, x_value, y_value, peak_vorticity) in enumerate(candidates):
            label = (
                f"{int(candidate_id)}: ({x_value:.5g}, {y_value:.5g})\n"
                f"ω={peak_vorticity:.5g}"
            )
            axis.annotate(
                label,
                (x_value, y_value),
                color=label_color,
                fontsize=float(config["plot"].get("fit_text_size", 8.0)),
                xytext=(6, 6 + 34 * (label_index % 4)),
                textcoords="offset points",
                bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 1.0},
                arrowprops={"arrowstyle": "-", "color": label_color, "linewidth": 0.5},
            )

    browse_frames(
        len(group_names),
        load,
        config["plot"],
        overlay,
        result_folder(args.run_folder) / "hmaxima_preview.png",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
