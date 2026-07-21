"""Stage 3: fit one circular Gaussian dipole to every saved candidate."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import h5py
import numpy as np
from matplotlib.patches import Circle
from scipy.optimize import least_squares

from common import (
    discover_frames,
    load_config,
    load_vorticity_frame,
    read_frame_order,
    require_positive,
    result_folder,
    simulation_metadata,
    stage_command,
    write_string_dataset,
)
from plot_vorticity import browse_frames


PARAMETER_NAMES = ("gamma", "x_c", "d", "sigma")


def fit_settings(config: dict) -> dict:
    """Validate every configured optimizer constant and bound."""
    settings = {
        "boundary_fraction": require_positive(config, "fit", "boundary_fraction"),
        "soft_l1_scale": require_positive(config, "fit", "soft_l1_scale"),
        "gamma_min": require_positive(config, "fit", "gamma_min"),
        "gamma_max": require_positive(config, "fit", "gamma_max"),
        "d_min": require_positive(config, "fit", "d_min"),
        "d_max": require_positive(config, "fit", "d_max"),
        "sigma_min": require_positive(config, "fit", "sigma_min"),
        "sigma_max": require_positive(config, "fit", "sigma_max"),
        "ftol": require_positive(config, "fit", "ftol"),
        "xtol": require_positive(config, "fit", "xtol"),
        "gtol": require_positive(config, "fit", "gtol"),
    }
    if not 0.0 < settings["boundary_fraction"] < 1.0:
        raise ValueError("[fit] boundary_fraction must satisfy 0 < boundary_fraction < 1.")
    for lower_name, upper_name in (
        ("gamma_min", "gamma_max"),
        ("d_min", "d_max"),
        ("sigma_min", "sigma_max"),
    ):
        if settings[lower_name] >= settings[upper_name]:
            raise ValueError(f"[fit] {lower_name} must be smaller than {upper_name}.")
    max_nfev = int(config["fit"].get("max_nfev", -1))
    if max_nfev <= 0:
        raise ValueError("[fit] max_nfev must be a positive integer.")
    settings["max_nfev"] = max_nfev
    return settings


def dipole_vorticity(x: np.ndarray, y: np.ndarray, parameters: np.ndarray) -> np.ndarray:
    gamma, x_c, separation, sigma = parameters
    # The two circular Gaussians share gamma, x center, separation, and width.
    s_positive = (x - x_c) ** 2 + (y - 0.5 * separation) ** 2
    s_negative = (x - x_c) ** 2 + (y + 0.5 * separation) ** 2
    coefficient = gamma / (2.0 * math.pi * sigma ** 2)
    e_positive = np.exp(-s_positive / (2.0 * sigma ** 2))
    e_negative = np.exp(-s_negative / (2.0 * sigma ** 2))
    return coefficient * (e_positive - e_negative)


def nearest_finite_value(frame: dict, x_target: float, y_target: float) -> float:
    """Evaluate at the nearest available physical grid cell without interpolation."""
    x = frame["x"]
    y = frame["y"]
    omega = frame["vorticity"]
    column = int(np.argmin(np.abs(x - x_target)))
    row = int(np.argmin(np.abs(y - y_target)))
    if np.isfinite(omega[row, column]):
        return float(omega[row, column])

    valid_rows, valid_columns = np.nonzero(np.isfinite(omega))
    if not len(valid_rows):
        raise ValueError("The frame contains no finite vorticity cell.")
    distances = (x[valid_columns] - x_target) ** 2 + (y[valid_rows] - y_target) ** 2
    nearest = int(np.argmin(distances))
    return float(omega[valid_rows[nearest], valid_columns[nearest]])


def rectangle_samples(frame: dict, bounds: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return finite physical samples inside one saved fitting rectangle."""
    x0, x1, y0, y1 = bounds
    columns = np.flatnonzero((frame["x"] >= x0) & (frame["x"] <= x1))
    rows = np.flatnonzero((frame["y"] >= y0) & (frame["y"] <= y1))
    if not len(columns) or not len(rows):
        return np.empty(0), np.empty(0), np.empty(0)
    x_grid, y_grid = np.meshgrid(frame["x"][columns], frame["y"][rows])
    observed = frame["vorticity"][np.ix_(rows, columns)]
    valid = np.isfinite(observed)
    return x_grid[valid], y_grid[valid], observed[valid]


def unavailable_result(candidate_id: int, message: str, initial=None, lower=None, upper=None) -> dict:
    """Return a complete failed-fit record with unavailable values set to NaN."""
    nan_parameters = np.full(4, np.nan)
    return {
        "candidate_id": candidate_id,
        "success": False,
        "message": message,
        "parameters": nan_parameters,
        "initial_parameters": nan_parameters if initial is None else np.asarray(initial, dtype=float),
        "lower_bounds": nan_parameters if lower is None else np.asarray(lower, dtype=float),
        "upper_bounds": nan_parameters if upper is None else np.asarray(upper, dtype=float),
        "boundary_radius": math.nan,
        "normalized_rmse": math.nan,
        "positive_center": np.full(2, np.nan),
        "negative_center": np.full(2, np.nan),
        "omega_scale": math.nan,
        "sample_count": 0,
        "nfev": 0,
    }


def fit_candidate(frame: dict, candidate: dict, config: dict, settings: dict) -> dict:
    candidate_id = int(candidate["candidate_id"])
    x_values, y_values, observed = rectangle_samples(frame, candidate["clamped_bounds"])
    if not len(observed):
        return unavailable_result(candidate_id, "No finite grid points are available inside the rectangle.")
    omega_scale = float(np.max(np.abs(observed)))
    if omega_scale == 0.0:
        return unavailable_result(candidate_id, "Vorticity scale is zero; fit unavailable.")

    alpha_r = require_positive(config, "region", "alpha_r")
    alpha = require_positive(config, "region", "alpha")
    # These initial values come directly from the forcing scale and mirrored peak geometry.
    sigma_initial = alpha_r / math.sqrt(2.0 * alpha)
    x_initial = float(candidate["peak_x"])
    d_initial = 2.0 * abs(float(candidate["peak_y"]))
    mirrored_vorticity = nearest_finite_value(frame, x_initial, -float(candidate["peak_y"]))
    amplitude_initial = 0.5 * (float(candidate["peak_vorticity"]) + abs(mirrored_vorticity))
    gamma_initial = 2.0 * math.pi * sigma_initial ** 2 * amplitude_initial
    initial = np.asarray((gamma_initial, x_initial, d_initial, sigma_initial), dtype=float)

    # x_c is bounded by this candidate's intended rectangle; the other bounds come from TOML.
    x_min, x_max = candidate["intended_bounds"][:2]
    lower = np.asarray((settings["gamma_min"], x_min, settings["d_min"], settings["sigma_min"]))
    upper = np.asarray((settings["gamma_max"], x_max, settings["d_max"], settings["sigma_max"]))
    if np.any(initial < lower) or np.any(initial > upper):
        return unavailable_result(
            candidate_id,
            "The equation-defined initial vector lies outside the configured bounds.",
            initial,
            lower,
            upper,
        )

    def normalized_residual(parameters: np.ndarray) -> np.ndarray:
        # Scaling by the largest observed magnitude makes soft_l1_scale dimensionless.
        return (dipole_vorticity(x_values, y_values, parameters) - observed) / omega_scale

    try:
        optimized = least_squares(
            normalized_residual,
            initial,
            bounds=(lower, upper),
            loss="soft_l1",
            f_scale=settings["soft_l1_scale"],
            max_nfev=settings["max_nfev"],
            ftol=settings["ftol"],
            xtol=settings["xtol"],
            gtol=settings["gtol"],
        )
    except (ValueError, FloatingPointError) as error:
        return unavailable_result(candidate_id, f"SciPy fit unavailable: {error}", initial, lower, upper)

    gamma, x_c, separation, sigma = optimized.x
    # The boundary is the circle where the Gaussian falls to boundary_fraction of its peak.
    boundary_radius = sigma * math.sqrt(-2.0 * math.log(settings["boundary_fraction"]))
    normalized_rmse = float(np.sqrt(np.mean(normalized_residual(optimized.x) ** 2)))
    return {
        "candidate_id": candidate_id,
        "success": bool(optimized.success),
        "message": str(optimized.message),
        "parameters": optimized.x,
        "initial_parameters": initial,
        "lower_bounds": lower,
        "upper_bounds": upper,
        "boundary_radius": boundary_radius,
        "normalized_rmse": normalized_rmse,
        "positive_center": np.asarray((x_c, 0.5 * separation)),
        "negative_center": np.asarray((x_c, -0.5 * separation)),
        "omega_scale": omega_scale,
        "sample_count": len(observed),
        "nfev": int(optimized.nfev),
    }


def save_frame_results(group: h5py.Group, source: dict, candidates: list[dict], results: list[dict]) -> None:
    """Save fitted parameters and diagnostics for one frame."""
    group.attrs["source_filename"] = source["source_filename"]
    group.attrs["source_path"] = source["source_path"]
    group.attrs["simulation_time"] = source["time"]
    group.attrs["time_step"] = source["step"]
    group.create_dataset("candidate_ids", data=[result["candidate_id"] for result in results], dtype=np.int32)
    group.create_dataset("success", data=[result["success"] for result in results], dtype=np.uint8)
    write_string_dataset(group, "optimizer_messages", [result["message"] for result in results])
    group.create_dataset("parameters", data=np.asarray([result["parameters"] for result in results]).reshape(-1, 4))
    group.create_dataset(
        "initial_parameters", data=np.asarray([result["initial_parameters"] for result in results]).reshape(-1, 4)
    )
    group.create_dataset("lower_bounds", data=np.asarray([result["lower_bounds"] for result in results]).reshape(-1, 4))
    group.create_dataset("upper_bounds", data=np.asarray([result["upper_bounds"] for result in results]).reshape(-1, 4))
    group.create_dataset("boundary_radius", data=[result["boundary_radius"] for result in results])
    group.create_dataset("normalized_rmse", data=[result["normalized_rmse"] for result in results])
    group.create_dataset(
        "positive_centers", data=np.asarray([result["positive_center"] for result in results]).reshape(-1, 2)
    )
    group.create_dataset(
        "negative_centers", data=np.asarray([result["negative_center"] for result in results]).reshape(-1, 2)
    )
    group.create_dataset("omega_scale", data=[result["omega_scale"] for result in results])
    group.create_dataset("sample_count", data=[result["sample_count"] for result in results], dtype=np.int64)
    group.create_dataset("nfev", data=[result["nfev"] for result in results], dtype=np.int64)
    group.create_dataset("intended_bounds", data=np.asarray([item["intended_bounds"] for item in candidates]).reshape(-1, 4))
    group.create_dataset("clamped_bounds", data=np.asarray([item["clamped_bounds"] for item in candidates]).reshape(-1, 4))


def read_candidates(maxima: h5py.File, regions: h5py.File, group_name: str) -> list[dict]:
    """Join each saved h-maximum to its Stage 2 rectangle by candidate ID."""
    maximum_group = maxima[group_name]
    region_group = regions[group_name]
    maximum_ids = maximum_group["candidate_ids"][:]
    region_ids = region_group["candidate_ids"][:]
    if not np.array_equal(maximum_ids, region_ids):
        raise ValueError(f"Candidate IDs disagree between hmaxima.h5 and regions.h5 for {group_name}.")
    return [
        {
            "candidate_id": int(candidate_id),
            "peak_x": float(peak_x),
            "peak_y": float(peak_y),
            "peak_vorticity": float(peak_vorticity),
            "intended_bounds": intended,
            "clamped_bounds": clamped,
        }
        for candidate_id, peak_x, peak_y, peak_vorticity, intended, clamped in zip(
            maximum_ids,
            maximum_group["peak_x"][:],
            maximum_group["peak_y"][:],
            maximum_group["peak_vorticity"][:],
            region_group["intended_bounds"][:],
            region_group["clamped_bounds"][:],
        )
    ]


def load_preview_frame(
    frame_path: Path,
    frame_index: int,
    group_name: str,
    config: dict,
    metadata: dict,
    fits_path: Path,
) -> dict:
    frame = load_vorticity_frame(frame_path, frame_index, config, metadata)
    with h5py.File(fits_path, "r") as fits:
        group = fits[group_name]
        for name in (
            "parameters",
            "boundary_radius",
            "positive_centers",
            "negative_centers",
        ):
            frame[name] = group[name][:]
    return frame


def main() -> int:
    parser = argparse.ArgumentParser(description="Fit circular Gaussian dipoles to saved candidates.")
    parser.add_argument("run_folder", type=Path)
    parser.add_argument("config_file", type=Path)
    parser.add_argument("--no-preview", action="store_true", help="Skip the terminal preview and preview PNG.")
    args = parser.parse_args()

    config = load_config(args.config_file)
    output_folder = result_folder(args.run_folder)
    hmaxima_path = output_folder / "hmaxima.h5"
    regions_path = output_folder / "regions.h5"
    fits_path = output_folder / "fits.h5"
    if not hmaxima_path.is_file():
        print("hmaxima.h5 does not exist. Run this exact command first:")
        print(stage_command("01_find_hmaxima.py", args.run_folder, args.config_file))
        return 1
    if not regions_path.is_file():
        print("regions.h5 does not exist. Run this exact command first:")
        print(stage_command("02_make_regions.py", args.run_folder, args.config_file))
        return 1

    settings = fit_settings(config)
    frame_paths = discover_frames(args.run_folder, config)
    paths_by_name = {path.name: path for path in frame_paths}
    metadata = simulation_metadata(args.run_folder, config)
    # Reuse saved candidates/rectangles, then fit every frame before previewing anything.
    with h5py.File(hmaxima_path, "r") as maxima, h5py.File(regions_path, "r") as regions:
        group_names = read_frame_order(maxima)
        if group_names != read_frame_order(regions):
            raise ValueError("Frame order differs between hmaxima.h5 and regions.h5.")

        with h5py.File(fits_path, "w") as output:
            output.attrs["schema"] = "ritta_circular_gaussian_dipole_fits_v1"
            output.attrs["config_file"] = str(args.config_file.expanduser().resolve())
            output.attrs["parameter_order"] = "gamma,x_c,d,sigma"
            output.attrs["loss"] = "soft_l1"
            output.attrs["jacobian"] = "SciPy numerical differentiation"
            for name, value in settings.items():
                output.attrs[name] = value
            write_string_dataset(output, "frame_order", group_names)

            for frame_index, group_name in enumerate(group_names):
                source_filename = str(maxima[group_name].attrs["source_filename"])
                if source_filename not in paths_by_name:
                    raise FileNotFoundError(f"Original frame is missing: {source_filename}")
                frame = load_vorticity_frame(paths_by_name[source_filename], frame_index, config, metadata)
                candidates = read_candidates(maxima, regions, group_name)
                results = [fit_candidate(frame, candidate, config, settings) for candidate in candidates]
                save_frame_results(output.create_group(group_name), frame, candidates, results)
                success_count = sum(result["success"] for result in results)
                print(
                    f"[{frame_index + 1}/{len(group_names)}] {source_filename}: "
                    f"{success_count}/{len(results)} fits successful"
                )

    print(f"Saved {fits_path}")
    if args.no_preview:
        return 0
    print("Batch calculation complete. Starting terminal frame prompt.")
    # The fitting preview intentionally draws only the two fitted boundaries.
    positive_color = str(config["plot"].get("positive_marker_color", "black"))
    negative_color = str(config["plot"].get("negative_marker_color", "#7b2cbf"))
    line_width = float(config["plot"].get("region_line_width", 1.5))

    ordered_paths = []
    with h5py.File(hmaxima_path, "r") as maxima:
        for group_name in group_names:
            ordered_paths.append(paths_by_name[str(maxima[group_name].attrs["source_filename"])])

    def load(index: int) -> dict:
        return load_preview_frame(
            ordered_paths[index], index, group_names[index], config, metadata, fits_path
        )

    def overlay(axis, frame: dict) -> None:
        for parameters, radius, positive, negative in zip(
            frame["parameters"],
            frame["boundary_radius"],
            frame["positive_centers"],
            frame["negative_centers"],
        ):
            if np.all(np.isfinite(parameters)) and np.isfinite(radius):
                axis.add_patch(Circle(positive, radius, fill=False, edgecolor=positive_color, linewidth=line_width))
                axis.add_patch(Circle(negative, radius, fill=False, edgecolor=negative_color, linewidth=line_width))

    browse_frames(
        len(group_names),
        load,
        config["plot"],
        overlay,
        output_folder / "fits_preview.png",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
