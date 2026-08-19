#!/usr/bin/env python3
"""Measure upper-half circulation and track its strong-vorticity center."""

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


SCRIPT_FOLDER = Path(__file__).resolve().parent
VORTEX_FOLDER = SCRIPT_FOLDER.parent / "ritta_vortex_identification"
sys.path.insert(0, str(VORTEX_FOLDER))

from common import (  # noqa: E402
    discover_frames,
    frame_step,
    load_config,
    load_vorticity_frame,
    simulation_metadata,
    simulation_parameter,
)
from time_series_plotting import distinct_line_colors  # noqa: E402

from largest_threshold_circulation import (  # noqa: E402
    discover_runs,
    fraction_tag,
    positive_integer,
    reject_explicit_dt,
    threshold_fraction,
)
from plot_largest_threshold_gifs import (  # noqa: E402
    configured_limits,
    save_gif,
    shard_step,
)


METHOD_VERSION = "2d-upper-half-circulation-strong-center-v1"
SLICE_METHOD_VERSION = "3d-slice-upper-half-circulation-strong-center-v1"
DEFAULT_CENTER_THRESHOLD_FRACTION = 0.4
DEFAULT_GIF_STRIDE = 5
DEFAULT_FPS = 8
CSV_COLUMNS = (
    "frame_index",
    "frame_name",
    "step",
    "time",
    "circulation_upper_half",
    "upper_half_area",
    "upper_half_cells",
    "peak_upper_vorticity",
    "center_threshold_fraction",
    "center_threshold_vorticity",
    "center_found",
    "center_cells",
    "center_area",
    "center_vorticity_weight",
    "x_center",
    "y_center",
    "source_path",
)


def finite_limit(value):
    try:
        number = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("axis limit must be a number") from error
    if not math.isfinite(number):
        raise argparse.ArgumentTypeError("axis limit must be finite")
    return number


def load_task_frame(task, include_cells):
    """Load either a native 2D frame or an AMR-correct 3D meridional slice."""
    if task.get("slice_z") is None:
        return load_vorticity_frame(
            task["path"],
            task["source_index"],
            task["config"],
            task["metadata"],
            include_cells=include_cells,
        )

    three_d_folder = SCRIPT_FOLDER.parent / "ritta_plotting_3D"
    if str(three_d_folder) not in sys.path:
        sys.path.insert(0, str(three_d_folder))
    from slice_vortex_identification import load_slice_frame

    frame = load_slice_frame(
        task["path"],
        task["source_index"],
        task["config"],
        task["metadata"],
        task["origin_3d"],
        task["slice_z"],
    )
    if not include_cells:
        frame = dict(frame)
        frame.pop("cells", None)
    return frame


def upper_half_cell_geometry(cells):
    """Return each square cell's area and y centroid clipped to y > 0."""
    area = np.asarray(cells["area"], dtype=float)
    y = np.asarray(cells["y"], dtype=float)
    width = np.sqrt(area)
    bottom = y - 0.5 * width
    top = y + 0.5 * width
    clipped_bottom = np.maximum(bottom, 0.0)
    height = np.clip(top - clipped_bottom, 0.0, width)
    upper_area = width * height
    upper_y = np.full(y.shape, np.nan, dtype=float)
    present = height > 0.0
    upper_y[present] = 0.5 * (clipped_bottom[present] + top[present])
    return upper_area, upper_y


def measure_frame(frame, center_fraction=DEFAULT_CENTER_THRESHOLD_FRACTION):
    """Integrate signed omega over y > 0 and centroid omega >= fraction*max."""
    cells = frame["cells"]
    x = np.asarray(cells["x"], dtype=float)
    omega = np.asarray(cells["vorticity"], dtype=float)
    upper_area, upper_y = upper_half_cell_geometry(cells)
    valid = (
        np.isfinite(x)
        & np.isfinite(omega)
        & np.isfinite(upper_area)
        & (upper_area > 0.0)
    )
    upper_half_area = float(np.sum(upper_area[valid]))
    circulation = float(np.sum(omega[valid] * upper_area[valid]))
    peak = float(np.max(omega[valid])) if np.any(valid) else math.nan
    threshold = (
        center_fraction * peak
        if math.isfinite(peak) and peak > 0.0
        else math.nan
    )

    result = {
        "circulation_upper_half": circulation,
        "upper_half_area": upper_half_area,
        "upper_half_cells": int(np.count_nonzero(valid)),
        "peak_upper_vorticity": peak,
        "center_threshold_fraction": center_fraction,
        "center_threshold_vorticity": threshold,
        "center_found": False,
        "center_cells": 0,
        "center_area": math.nan,
        "center_vorticity_weight": math.nan,
        "x_center": math.nan,
        "y_center": math.nan,
    }
    if not math.isfinite(threshold) or threshold <= 0.0:
        return result

    selected = valid & (omega >= threshold)
    weights = omega[selected] * upper_area[selected]
    total_weight = float(np.sum(weights))
    if not np.any(selected) or not math.isfinite(total_weight) or total_weight <= 0.0:
        return result

    return {
        **result,
        "center_found": True,
        "center_cells": int(np.count_nonzero(selected)),
        "center_area": float(np.sum(upper_area[selected])),
        "center_vorticity_weight": total_weight,
        "x_center": float(np.sum(x[selected] * weights) / total_weight),
        "y_center": float(np.sum(upper_y[selected] * weights) / total_weight),
    }


def analyze_frame(task):
    frame = load_task_frame(task, include_cells=True)
    return {
        "run_name": task["run_name"],
        "frame_index": task["frame_index"],
        "frame_name": frame["source_filename"],
        "step": frame["step"],
        "time": frame["time"],
        **measure_frame(frame, task["center_fraction"]),
        "source_path": frame["source_path"],
    }


def config_digest(
    analysis_config,
    simulation_config,
    center_fraction,
    slice_z=None,
):
    method_version = (
        METHOD_VERSION if slice_z is None else SLICE_METHOD_VERSION
    )
    digest = hashlib.sha256(method_version.encode("utf-8"))
    digest.update(analysis_config.read_bytes())
    digest.update(simulation_config.read_bytes())
    digest.update(f"{center_fraction:.17g}".encode("ascii"))
    if slice_z is not None:
        digest.update(f"{slice_z:.17g}".encode("ascii"))
    return digest.hexdigest()


def shard_path(output_folder, run_name, step):
    return output_folder / run_name / "frame_results" / f"flowTime_{step}.json"


def reusable_shard(path, task):
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        source_stat = task["path"].stat()
        if (
            payload.get("method_version") != task["method_version"]
            or payload.get("config_digest") != task["config_digest"]
            or payload.get("source_path") != str(task["path"].resolve())
            or payload.get("source_size") != source_stat.st_size
            or payload.get("source_mtime_ns") != source_stat.st_mtime_ns
            or payload.get("frame_index") != task["frame_index"]
            or payload.get("source_index") != task["source_index"]
        ):
            return None
        return payload["result"]
    except (OSError, ValueError, KeyError, TypeError):
        return None


def save_shard(path, task, result):
    source_stat = task["path"].stat()
    payload = {
        "method_version": task["method_version"],
        "config_digest": task["config_digest"],
        "source_path": str(task["path"].resolve()),
        "source_size": source_stat.st_size,
        "source_mtime_ns": source_stat.st_mtime_ns,
        "frame_index": task["frame_index"],
        "source_index": task["source_index"],
        "result": result,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, allow_nan=True), encoding="utf-8")
    temporary.replace(path)


def write_case_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in CSV_COLUMNS})


def plot_series(output_path, run_info, rows_by_run, figure_size, value_name,
                y_label, title, normalized_time=False):
    colors = distinct_line_colors(len(run_info))
    figure, axis = plt.subplots(figsize=figure_size, constrained_layout=True)
    for color, info in zip(colors, run_info):
        rows = rows_by_run[info["name"]]
        times = np.asarray([row["time"] for row in rows], dtype=float)
        if normalized_time:
            times = times / info["tau"]
        values = np.asarray([row[value_name] for row in rows], dtype=float)
        axis.plot(times, values, color=color, label=rf"$\tau={info['tau']:g}$")
    axis.set_xlabel(
        r"normalized simulation time $t/\tau$"
        if normalized_time
        else "simulation time"
    )
    axis.set_ylabel(y_label)
    axis.set_title(title)
    axis.grid(True, alpha=0.3)
    axis.legend()
    if normalized_time:
        axis.set_xlim(0.0, 1.0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError(f"Failed to create plot: {output_path}")
    print(f"Saved {output_path}", flush=True)


def save_combined_plots(output_folder, run_info, rows_by_run, figure_size):
    plot_series(
        output_folder / "combined_upper_half_circulation_vs_time.png",
        run_info,
        rows_by_run,
        figure_size,
        "circulation_upper_half",
        r"upper-half circulation $\Gamma_{y>0}$",
        "Signed circulation over the upper half-domain",
    )
    plot_series(
        output_folder / "combined_upper_half_circulation_vs_time_over_tau.png",
        run_info,
        rows_by_run,
        figure_size,
        "circulation_upper_half",
        r"upper-half circulation $\Gamma_{y>0}$",
        r"Signed upper-half circulation versus normalized time $t/\tau$",
        normalized_time=True,
    )
    plot_series(
        output_folder / "combined_upper_half_center_x_vs_time.png",
        run_info,
        rows_by_run,
        figure_size,
        "x_center",
        "strong-vorticity center x",
        "Upper-half strong-vorticity center versus simulation time",
    )
    plot_series(
        output_folder / "combined_upper_half_center_x_vs_time_over_tau.png",
        run_info,
        rows_by_run,
        figure_size,
        "x_center",
        "strong-vorticity center x",
        r"Upper-half strong-vorticity center versus normalized time $t/\tau$",
        normalized_time=True,
    )


def write_datasets(path, run_info):
    lines = ["# Upper-half circulation and strong-vorticity center results.", ""]
    for info in run_info:
        lines.extend(
            [
                "[[dataset]]",
                f"name = {json.dumps(info['name'])}",
                f"csv = {json.dumps((Path(info['name']) / 'upper_half_circulation.csv').as_posix())}",
                f"run_folder = {json.dumps(str(info['run_folder']))}",
                f"forcing_end_time = {info['tau']:g}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def render_frame(task):
    shard = Path(task["shard_path"])
    payload = json.loads(shard.read_text(encoding="utf-8"))
    if payload.get("method_version") != task["method_version"]:
        raise ValueError(f"Unexpected upper-half result version in {shard}")
    result = payload["result"]
    source_path = Path(payload["source_path"])
    if not source_path.is_file():
        source_path = Path(task["run_folder"]) / "output" / result["frame_name"]
    if not source_path.is_file():
        raise FileNotFoundError(f"Original snapshot is missing: {source_path}")

    png_path = Path(task["png_path"])
    if task["resume"] and png_path.is_file() and png_path.stat().st_size > 0:
        return {
            "case": task["case"],
            "step": int(result["step"]),
            "time": float(result["time"]),
            "source_path": str(source_path.resolve()),
            "png_path": str(png_path.resolve()),
            "center_found": bool(result["center_found"]),
            "reused": True,
        }

    frame_task = {
        **task,
        "path": source_path,
        "source_index": int(payload["source_index"]),
    }
    frame = load_task_frame(frame_task, include_cells=False)
    omega = np.asarray(frame["vorticity"], dtype=float)
    finite_absolute = np.abs(omega[np.isfinite(omega)])
    color_limit = float(np.max(finite_absolute)) if finite_absolute.size else 1.0
    if not math.isfinite(color_limit) or color_limit <= 0.0:
        color_limit = 1.0
    threshold = float(result["center_threshold_vorticity"])
    # Include raster cells whose physical area intersects y > 0, matching the
    # clipped-cell integration used for the saved values.
    upper_rows = (
        np.asarray(frame["y"], dtype=float)[:, None] + 0.5 * float(frame["dx"])
        > 0.0
    )
    center_mask = (
        np.isfinite(omega) & upper_rows & (omega >= threshold)
        if math.isfinite(threshold)
        else np.zeros(omega.shape, dtype=bool)
    )

    figure, axis = plt.subplots(
        figsize=tuple(task["figure_size"]), constrained_layout=True
    )
    dx = float(frame["dx"])
    extent = (
        float(frame["x"][0] - 0.5 * dx),
        float(frame["x"][-1] + 0.5 * dx),
        float(frame["y"][0] - 0.5 * dx),
        float(frame["y"][-1] + 0.5 * dx),
    )
    image = axis.imshow(
        omega,
        origin="lower",
        extent=extent,
        interpolation="nearest",
        cmap=task["colormap"],
        vmin=-color_limit,
        vmax=color_limit,
        aspect="equal",
    )
    figure.colorbar(image, ax=axis, label=r"vorticity $\omega$")
    axis.axhspan(extent[2], 0.0, color="0.7", alpha=0.18, zorder=2)
    axis.axhline(0.0, color="0.2", linestyle="--", linewidth=1.0, zorder=3)

    if np.any(center_mask):
        overlay = np.ma.masked_where(~center_mask, np.ones(center_mask.shape))
        axis.imshow(
            overlay,
            origin="lower",
            extent=extent,
            interpolation="nearest",
            cmap=mcolors.ListedColormap([task["region_color"]]),
            vmin=0.0,
            vmax=1.0,
            alpha=float(task["mask_alpha"]),
            aspect="equal",
            zorder=4,
        )
        axis.contour(
            frame["x"],
            frame["y"],
            center_mask.astype(float),
            levels=[0.5],
            colors=[task["region_color"]],
            linewidths=float(task["line_width"]),
            zorder=5,
        )

    center_x = float(result["x_center"])
    center_y = float(result["y_center"])
    if math.isfinite(center_x) and math.isfinite(center_y):
        axis.scatter(
            center_x,
            center_y,
            color=task["center_color"],
            marker="x",
            s=float(task["marker_size"]),
            linewidths=float(task["line_width"]),
            zorder=6,
        )

    circulation = float(result["circulation_upper_half"])
    annotation = (
        rf"$\Gamma_{{y>0}}$={circulation:.6g}, "
        + (
            rf"center threshold={threshold:.4g}"
            if math.isfinite(threshold)
            else "center unavailable"
        )
    )
    axis.text(
        0.02,
        0.02,
        annotation,
        transform=axis.transAxes,
        va="bottom",
        fontsize=float(task["text_size"]),
        bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none"},
        zorder=7,
    )
    axis.legend(
        handles=[
            Patch(
                facecolor=mcolors.to_rgba(
                    task["region_color"], task["mask_alpha"]
                ),
                edgecolor=task["region_color"],
                label=f"upper-half {task['threshold_label']} center mask",
            ),
            Line2D(
                [0],
                [0],
                color=task["center_color"],
                marker="x",
                linestyle="none",
                label="vorticity-weighted center",
            ),
        ],
        loc="upper right",
        fontsize=float(task["text_size"]),
    )
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_title(
        f"{task['case']}: upper-half circulation and {task['threshold_label']} "
        f"center, t={float(result['time']):.6g}, step={int(result['step'])}"
    )
    if task["x_limits"] is not None:
        axis.set_xlim(*task["x_limits"])
    if task["y_limits"] is not None:
        axis.set_ylim(*task["y_limits"])

    png_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(png_path, dpi=120)
    plt.close(figure)
    if not png_path.is_file() or png_path.stat().st_size == 0:
        raise RuntimeError(f"Failed to create upper-half frame: {png_path}")
    return {
        "case": task["case"],
        "step": int(result["step"]),
        "time": float(result["time"]),
        "source_path": str(source_path.resolve()),
        "png_path": str(png_path.resolve()),
        "center_found": bool(result["center_found"]),
        "reused": False,
    }


def render_gifs(output_folder, run_info, config, workers, stride, fps, resume,
                requested_x_limits, requested_y_limits):
    x_limits, y_limits = configured_limits(
        config, requested_x_limits, requested_y_limits
    )
    plot = config.get("plot", {})
    threshold_plot = config.get("threshold_mask", {})
    method = json.loads((output_folder / "method.json").read_text(encoding="utf-8"))
    method_version = str(method["method_version"])
    slice_z = method.get("slice_z")
    center_fraction = float(method["center_threshold_fraction"])
    threshold_label = f"{100.0 * center_fraction:g}%"
    tasks = []
    case_outputs = {}
    for info in run_info:
        case = info["name"]
        metadata = simulation_metadata(info["run_folder"], config)
        origin_3d = None
        if slice_z is not None:
            three_d_folder = SCRIPT_FOLDER.parent / "ritta_plotting_3D"
            if str(three_d_folder) not in sys.path:
                sys.path.insert(0, str(three_d_folder))
            from slice_vortex_identification import simulation_origin_3d

            origin_3d = simulation_origin_3d(Path(metadata["source"]))
        shards = sorted(
            (output_folder / case / "frame_results").glob("flowTime_*.json"),
            key=shard_step,
        )[::stride]
        if not shards:
            raise FileNotFoundError(f"No completed upper-half shards found for {case}")
        case_output = output_folder / "gifs" / case
        case_outputs[case] = {
            "gif": case_output / f"{case}_upper_half_center_{fraction_tag(center_fraction)}.gif",
            "manifest": case_output / "frame_manifest.csv",
        }
        for shard in shards:
            step = shard_step(shard)
            tasks.append(
                {
                    "case": case,
                    "shard_path": str(shard),
                    "run_folder": str(info["run_folder"]),
                    "png_path": str(case_output / "frames" / f"flowTime_{step}.png"),
                    "resume": resume,
                    "method_version": method_version,
                    "config": config,
                    "metadata": metadata,
                    "slice_z": slice_z,
                    "origin_3d": origin_3d,
                    "threshold_label": threshold_label,
                    "colormap": str(plot.get("colormap", "RdBu_r")),
                    "region_color": str(
                        threshold_plot.get("positive_color", "#ffb000")
                    ),
                    "center_color": str(plot.get("marker_color", "black")),
                    "mask_alpha": float(plot.get("mask_alpha", 0.35)),
                    "line_width": float(plot.get("region_line_width", 1.5)),
                    "marker_size": float(plot.get("marker_size", 30.0)),
                    "text_size": float(plot.get("fit_text_size", 8.0)),
                    "figure_size": (
                        float(plot.get("figure_width", 10.0)),
                        float(plot.get("figure_height", 7.0)),
                    ),
                    "x_limits": x_limits,
                    "y_limits": y_limits,
                }
            )

    print(f"GIF frames:        {len(tasks)}", flush=True)
    print(f"GIF frame workers:{min(workers, len(tasks)):>4}", flush=True)
    print(f"GIF stride:        {stride}", flush=True)
    results = []
    if workers == 1 or len(tasks) == 1:
        rendered = [render_frame(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(tasks))) as executor:
            futures = {executor.submit(render_frame, task): task for task in tasks}
            rendered = [future.result() for future in as_completed(futures)]
    for completed, result in enumerate(rendered, start=1):
        results.append(result)
        action = "reused" if result["reused"] else "rendered"
        print(
            f"[GIF {completed}/{len(tasks)}] {result['case']}/"
            f"flowTime_{result['step']}.png: {action}",
            flush=True,
        )

    for info in run_info:
        case = info["name"]
        ordered = sorted(
            (item for item in results if item["case"] == case),
            key=lambda item: item["step"],
        )
        output = case_outputs[case]
        save_gif([Path(item["png_path"]) for item in ordered], output["gif"], fps)
        output["manifest"].parent.mkdir(parents=True, exist_ok=True)
        with output["manifest"].open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "frame_index",
                    "snapshot_step",
                    "simulation_time",
                    "source_snapshot",
                    "center_found",
                    "png_file",
                ]
            )
            for frame_index, item in enumerate(ordered):
                writer.writerow(
                    [
                        frame_index,
                        item["step"],
                        f"{item['time']:.17g}",
                        item["source_path"],
                        int(item["center_found"]),
                        item["png_path"],
                    ]
                )
        print(f"Saved GIF: {output['gif']}", flush=True)
        print(f"Manifest:  {output['manifest']}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Integrate signed vorticity over y > 0, track the vorticity-weighted "
            "center where omega >= FRACTION * max_{y>0}(omega), then create "
            "combined plots and one GIF per case."
        )
    )
    parser.add_argument("sweep_folder", type=Path)
    parser.add_argument("config_file", type=Path)
    parser.add_argument(
        "--center-threshold-fraction",
        type=threshold_fraction,
        default=DEFAULT_CENTER_THRESHOLD_FRACTION,
    )
    parser.add_argument("--analysis-stride", type=positive_integer, default=1)
    parser.add_argument("--gif-stride", type=positive_integer, default=DEFAULT_GIF_STRIDE)
    parser.add_argument("--workers", type=positive_integer, default=1)
    parser.add_argument("--fps", type=positive_integer, default=DEFAULT_FPS)
    parser.add_argument(
        "--slice-z",
        type=finite_limit,
        help="Analyze edge_aux_2 on this z slice of 3D output.",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-gifs", action="store_true")
    parser.add_argument("--x-limits", nargs=2, type=finite_limit, metavar=("MIN", "MAX"))
    parser.add_argument("--y-limits", nargs=2, type=finite_limit, metavar=("MIN", "MAX"))
    parser.add_argument("--cases", nargs="+", metavar="RUN_NAME")
    args = parser.parse_args()

    sweep_folder = args.sweep_folder.expanduser().resolve()
    if not sweep_folder.is_dir():
        parser.error(f"sweep folder does not exist: {sweep_folder}")
    config_path = args.config_file.expanduser().resolve()
    config = load_config(config_path)
    if args.x_limits is not None and args.x_limits[0] >= args.x_limits[1]:
        parser.error("--x-limits requires MIN < MAX")
    if args.y_limits is not None and args.y_limits[0] >= args.y_limits[1]:
        parser.error("--y-limits requires MIN < MAX")
    if args.output_dir is not None:
        output_folder = args.output_dir.expanduser().resolve()
    else:
        output_parent = (
            SCRIPT_FOLDER
            if args.slice_z is None
            else SCRIPT_FOLDER.parent / "ritta_plotting_3D"
        )
        method_tag = "upper_half" if args.slice_z is None else "slice_upper_half"
        output_folder = (
            output_parent
            / "outputs"
            / f"{sweep_folder.name}_{method_tag}_center_"
            f"{fraction_tag(args.center_threshold_fraction)}"
        )
    output_folder.mkdir(parents=True, exist_ok=True)

    runs = discover_runs(sweep_folder, args.cases)
    if not runs:
        raise ValueError(f"No immediate child runs were found in {sweep_folder}")

    tasks = []
    run_info = []
    method_version = (
        METHOD_VERSION if args.slice_z is None else SLICE_METHOD_VERSION
    )
    for run_folder in runs:
        metadata = simulation_metadata(run_folder, config)
        simulation_config = Path(metadata["source"])
        if (
            not simulation_config.is_file()
            or metadata.get("cfl") is None
            or metadata.get("dx_base") is None
            or int(metadata.get("num_amr_levels", -1)) < 0
        ):
            raise ValueError(f"Incomplete simulation metadata for {run_folder}")
        reject_explicit_dt(simulation_config)
        origin_3d = None
        if args.slice_z is not None:
            three_d_folder = SCRIPT_FOLDER.parent / "ritta_plotting_3D"
            if str(three_d_folder) not in sys.path:
                sys.path.insert(0, str(three_d_folder))
            from slice_vortex_identification import simulation_origin_3d

            origin_3d = simulation_origin_3d(simulation_config)
        tau = simulation_parameter(run_folder, config, "b_f_tau")
        if not math.isfinite(tau) or tau <= 0.0:
            raise ValueError(f"Invalid b_f_tau in {simulation_config}: {tau}")
        frames = discover_frames(run_folder, config)
        selected = list(enumerate(frames))[:: args.analysis_stride]
        digest = config_digest(
            config_path,
            simulation_config,
            args.center_threshold_fraction,
            args.slice_z,
        )
        run_info.append(
            {"name": run_folder.name, "run_folder": run_folder, "tau": tau}
        )
        for frame_index, (source_index, path) in enumerate(selected):
            tasks.append(
                {
                    "run_name": run_folder.name,
                    "frame_index": frame_index,
                    "source_index": source_index,
                    "path": path,
                    "config": config,
                    "metadata": metadata,
                    "method_version": method_version,
                    "slice_z": args.slice_z,
                    "origin_3d": origin_3d,
                    "center_fraction": args.center_threshold_fraction,
                    "config_digest": digest,
                }
            )
    run_info.sort(key=lambda item: (item["tau"], item["name"]))

    print(f"Sweep:             {sweep_folder}", flush=True)
    print(f"Cases:             {len(run_info)}", flush=True)
    print(f"Selected frames:   {len(tasks)}", flush=True)
    print(f"Frame workers:     {min(args.workers, len(tasks))}", flush=True)
    if args.slice_z is not None:
        print(f"Slice:             edge_aux_2 at z={args.slice_z:g}", flush=True)
    print("Circulation:       signed integral over y > 0", flush=True)
    print(
        "Center mask:      "
        f"omega >= {args.center_threshold_fraction:g} * max_y>0(omega)",
        flush=True,
    )
    print("AMR integration:   original visible cells with clipped native dx^2", flush=True)
    print(f"Output:            {output_folder}", flush=True)

    results = []
    pending = []
    for task in tasks:
        path = shard_path(output_folder, task["run_name"], frame_step(task["path"]))
        reused = reusable_shard(path, task) if args.resume else None
        if reused is None:
            pending.append((task, path))
        else:
            results.append(reused)
    print(f"Reused frames:     {len(results)}", flush=True)
    print(f"Frames to process: {len(pending)}", flush=True)

    if pending:
        if args.workers == 1 or len(pending) == 1:
            completed_results = [
                (item, analyze_frame(item[0])) for item in pending
            ]
        else:
            with ProcessPoolExecutor(max_workers=min(args.workers, len(pending))) as executor:
                future_tasks = {
                    executor.submit(analyze_frame, task): (task, path)
                    for task, path in pending
                }
                completed_results = [
                    (future_tasks[future], future.result())
                    for future in as_completed(future_tasks)
                ]
        for completed, ((task, path), result) in enumerate(completed_results, start=1):
            save_shard(path, task, result)
            results.append(result)
            status = "center found" if result["center_found"] else "no center"
            print(
                f"[{completed}/{len(pending)}] {result['run_name']}/"
                f"{result['frame_name']}: {status}",
                flush=True,
            )

    rows_by_run = {info["name"]: [] for info in run_info}
    for result in results:
        rows_by_run[result["run_name"]].append(result)
    for info in run_info:
        rows = sorted(rows_by_run[info["name"]], key=lambda item: item["frame_index"])
        expected = sum(task["run_name"] == info["name"] for task in tasks)
        if len(rows) != expected:
            raise RuntimeError(f"Missing completed frames for {info['name']}")
        rows_by_run[info["name"]] = rows
        csv_path = output_folder / info["name"] / "upper_half_circulation.csv"
        write_case_csv(csv_path, rows)
        print(f"Saved {csv_path}", flush=True)

    write_datasets(output_folder / "datasets.toml", run_info)
    figure_size = (
        float(config["plot"].get("figure_width", 10.0)),
        float(config["plot"].get("figure_height", 7.0)),
    )
    save_combined_plots(output_folder, run_info, rows_by_run, figure_size)
    method_path = output_folder / "method.json"
    method_path.write_text(
        json.dumps(
            {
                "method_version": method_version,
                "slice_z": args.slice_z,
                "vorticity_component": (
                    "edge_aux" if args.slice_z is None else "edge_aux_2"
                ),
                "circulation_domain": "y > 0",
                "circulation_sign": "signed",
                "center_threshold_fraction": args.center_threshold_fraction,
                "center_threshold_reference": "maximum positive vorticity in y > 0 for each frame",
                "center_weighting": "positive vorticity times visible upper-cell area",
                "integration": "original visible AMR cells, clipped at y=0, weighted by native dx^2",
                "analysis_stride": args.analysis_stride,
                "gif_stride": args.gif_stride,
                "workers": min(args.workers, len(tasks)),
                "analysis_config": str(config_path),
                "cases": [info["name"] for info in run_info],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved {method_path}", flush=True)

    if not args.skip_gifs:
        render_gifs(
            output_folder,
            run_info,
            config,
            args.workers,
            args.gif_stride,
            args.fps,
            args.resume,
            args.x_limits,
            args.y_limits,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
