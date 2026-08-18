#!/usr/bin/env python3
"""Render diagnostic GIFs from completed 3D-slice vortex-fit shards."""

import argparse
import csv
import json
import math
import re
import sys
import tomllib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from PIL import Image


SCRIPT_FOLDER = Path(__file__).resolve().parent
VORTEX_FOLDER = SCRIPT_FOLDER.parent / "ritta_vortex_identification"
sys.path.insert(0, str(VORTEX_FOLDER))

from common import load_config, simulation_metadata  # noqa: E402
from slice_vortex_identification import (  # noqa: E402
    load_slice_frame,
    reject_explicit_dt,
    simulation_origin_3d,
)


SHARD_PATTERN = re.compile(r"flowTime_(\d+)\.json$")
FRAME_PATTERN = re.compile(r"flowTime_(\d+)\.png$")
DEFAULT_FPS = 8


def positive_integer(value):
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("value must be an integer") from error
    if number < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return number


def finite_limit(value):
    try:
        number = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("axis limit must be a number") from error
    if not math.isfinite(number):
        raise argparse.ArgumentTypeError("axis limit must be finite")
    return number


def shard_step(path):
    match = SHARD_PATTERN.fullmatch(path.name)
    if match is None:
        raise ValueError(f"Not a fit shard: {path.name}")
    return int(match.group(1))


def load_datasets(path):
    with path.open("rb") as handle:
        document = tomllib.load(handle)
    datasets = document.get("dataset", [])
    if not datasets:
        raise ValueError(f"No [[dataset]] entries were found in {path}")
    return datasets


def axis_limits(config, requested_x, requested_y):
    plot = config.get("plot", {})
    x_limits = requested_x or (
        plot.get("x_axis_min", math.nan),
        plot.get("x_axis_max", math.nan),
    )
    y_limits = requested_y or (
        plot.get("y_axis_min", math.nan),
        plot.get("y_axis_max", math.nan),
    )

    def usable(values):
        return all(math.isfinite(float(value)) for value in values) and float(
            values[0]
        ) < float(values[1])

    return (
        tuple(float(value) for value in x_limits) if usable(x_limits) else None,
        tuple(float(value) for value in y_limits) if usable(y_limits) else None,
    )


def successful_records(result):
    records = []
    for record in result.get("records", []):
        center_x = float(record.get("_fit_x", math.nan))
        center_y = float(record.get("_fit_y", math.nan))
        radius = float(record.get("boundary_radius", math.nan))
        if (
            bool(record.get("fit_success"))
            and math.isfinite(center_x)
            and math.isfinite(center_y)
            and math.isfinite(radius)
            and radius > 0.0
        ):
            records.append((record, center_x, center_y, radius))
    return records


def render_frame(task):
    shard_path = Path(task["shard_path"])
    payload = json.loads(shard_path.read_text(encoding="utf-8"))
    result = payload["result"]
    source_path = Path(payload["source_path"])
    if not source_path.is_file():
        source_path = (
            Path(task["run_folder"])
            / "output"
            / result["source_filename"]
        )
    if not source_path.is_file():
        raise FileNotFoundError(f"Original snapshot is missing: {source_path}")

    png_path = Path(task["png_path"])
    records = successful_records(result)
    time = float(result["records"][0]["time"]) if result.get("records") else math.nan
    if task["resume"] and png_path.is_file() and png_path.stat().st_size > 0:
        return {
            "case": task["case"],
            "step": int(result["step"]),
            "time": time,
            "source_path": str(source_path.resolve()),
            "png_path": str(png_path.resolve()),
            "fit_count": len(records),
            "reused": True,
        }

    frame = load_slice_frame(
        source_path,
        int(payload["source_index"]),
        task["config"],
        task["metadata"],
        task["origin_3d"],
        float(payload["slice_z"]),
    )
    vorticity = np.asarray(frame["vorticity"], dtype=float)
    finite = np.abs(vorticity[np.isfinite(vorticity)])
    color_limit = float(np.max(finite)) if finite.size else 1.0
    if not math.isfinite(color_limit) or color_limit <= 0.0:
        color_limit = 1.0

    figure, axis = plt.subplots(
        figsize=tuple(task["figure_size"]),
        constrained_layout=True,
    )
    dx = float(frame["dx"])
    extent = (
        float(frame["x"][0] - 0.5 * dx),
        float(frame["x"][-1] + 0.5 * dx),
        float(frame["y"][0] - 0.5 * dx),
        float(frame["y"][-1] + 0.5 * dx),
    )
    image = axis.imshow(
        vorticity,
        origin="lower",
        extent=extent,
        interpolation="nearest",
        cmap=task["colormap"],
        vmin=-color_limit,
        vmax=color_limit,
        aspect="equal",
    )
    figure.colorbar(image, ax=axis, label=r"slice vorticity $\omega_z$")

    positive_color = task["positive_color"]
    negative_color = task["negative_color"]
    line_width = float(task["line_width"])
    marker_size = float(task["marker_size"])
    for _, center_x, center_y, radius in records:
        positive_center = (center_x, center_y)
        negative_center = (center_x, -center_y)
        axis.add_patch(
            Circle(
                positive_center,
                radius,
                fill=False,
                edgecolor=positive_color,
                linewidth=line_width,
            )
        )
        axis.add_patch(
            Circle(
                negative_center,
                radius,
                fill=False,
                edgecolor=negative_color,
                linewidth=line_width,
                linestyle="--",
            )
        )
        axis.scatter(
            *positive_center,
            color=positive_color,
            marker="x",
            s=marker_size,
            linewidths=line_width,
        )
        axis.scatter(
            *negative_center,
            color=negative_color,
            marker="x",
            s=marker_size,
            linewidths=line_width,
        )

    if records:
        largest = max(records, key=lambda item: item[3])
        record, center_x, center_y, radius = largest
        circulation = float(record.get("circulation_positive", math.nan))
        annotation = (
            f"largest fit: center=({center_x:.3g}, {center_y:.3g}), "
            f"R={radius:.3g}"
        )
        if math.isfinite(circulation):
            annotation += f", $\\Gamma_+$={circulation:.4g}"
        axis.text(
            0.02,
            0.02,
            annotation,
            transform=axis.transAxes,
            fontsize=float(task["text_size"]),
            va="bottom",
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )
    else:
        axis.text(
            0.5,
            0.5,
            "No successful fit",
            transform=axis.transAxes,
            ha="center",
            va="center",
            color="black",
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )

    axis.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=positive_color,
                marker="x",
                label="positive fitted center and boundary",
            ),
            Line2D(
                [0],
                [0],
                color=negative_color,
                marker="x",
                linestyle="--",
                label="mirrored negative fitted center and boundary",
            ),
        ],
        loc="upper right",
        fontsize=float(task["text_size"]),
    )
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_title(
        f"{task['case']}: Gaussian-dipole fit, "
        f"t={frame['time']:.6g}, step={frame['step']}"
    )
    if task["x_limits"] is not None:
        axis.set_xlim(*task["x_limits"])
    if task["y_limits"] is not None:
        axis.set_ylim(*task["y_limits"])

    png_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(png_path, dpi=120)
    plt.close(figure)
    if not png_path.is_file() or png_path.stat().st_size == 0:
        raise RuntimeError(f"Failed to create fit diagnostic frame: {png_path}")
    return {
        "case": task["case"],
        "step": int(frame["step"]),
        "time": float(frame["time"]),
        "source_path": str(source_path.resolve()),
        "png_path": str(png_path.resolve()),
        "fit_count": len(records),
        "reused": False,
    }


def save_gif(frame_paths, output_path, fps):
    if not frame_paths:
        raise ValueError(f"No frames are available for {output_path}")
    images = []
    for path in frame_paths:
        with Image.open(path) as image:
            images.append(image.convert("RGB").copy())
    duration_ms = max(1, round(1000 / fps))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError(f"Failed to create fit diagnostic GIF: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Render vorticity-slice GIFs with saved Gaussian-dipole fit "
            "centers and boundary circles; fitting is not repeated."
        )
    )
    parser.add_argument("analysis_output", type=Path)
    parser.add_argument("--workers", type=positive_integer, default=1)
    parser.add_argument("--stride", type=positive_integer, default=1)
    parser.add_argument("--fps", type=positive_integer, default=DEFAULT_FPS)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--x-limits", nargs=2, type=finite_limit, metavar=("MIN", "MAX"))
    parser.add_argument("--y-limits", nargs=2, type=finite_limit, metavar=("MIN", "MAX"))
    parser.add_argument("--cases", nargs="+", metavar="RUN_NAME")
    args = parser.parse_args()

    analysis_output = args.analysis_output.expanduser().resolve()
    method_path = analysis_output / "method.json"
    datasets_path = analysis_output / "datasets.toml"
    if not method_path.is_file() or not datasets_path.is_file():
        parser.error(
            "analysis output must contain method.json and datasets.toml: "
            f"{analysis_output}"
        )
    method = json.loads(method_path.read_text(encoding="utf-8"))
    config_path = (
        args.config.expanduser().resolve()
        if args.config is not None
        else Path(method["analysis_config"]).expanduser().resolve()
    )
    if not config_path.is_file():
        parser.error(f"analysis configuration does not exist: {config_path}")
    config = load_config(config_path)
    x_limits, y_limits = axis_limits(config, args.x_limits, args.y_limits)
    if args.x_limits is not None and args.x_limits[0] >= args.x_limits[1]:
        parser.error("--x-limits requires MIN < MAX")
    if args.y_limits is not None and args.y_limits[0] >= args.y_limits[1]:
        parser.error("--y-limits requires MIN < MAX")

    datasets = load_datasets(datasets_path)
    available = {str(item["name"]): item for item in datasets}
    selected_cases = args.cases or [str(item["name"]) for item in datasets]
    missing = [name for name in selected_cases if name not in available]
    if missing:
        parser.error("requested cases are absent from datasets.toml: " + ", ".join(missing))

    plot = config.get("plot", {})
    tasks = []
    case_outputs = {}
    for case in selected_cases:
        dataset = available[case]
        run_folder = Path(dataset["run_folder"]).expanduser().resolve()
        metadata = simulation_metadata(run_folder, config)
        simulation_config = Path(metadata["source"])
        reject_explicit_dt(simulation_config)
        origin_3d = simulation_origin_3d(simulation_config)
        shards = sorted(
            (analysis_output / case / "frame_results").glob("flowTime_*.json"),
            key=shard_step,
        )[:: args.stride]
        if not shards:
            raise FileNotFoundError(f"No completed fit shards were found for {case}")
        case_output = analysis_output / "fit_diagnostics" / case
        frames_folder = case_output / "frames"
        case_outputs[case] = {
            "folder": case_output,
            "frames": frames_folder,
            "gif": case_output / f"{case}_fit_diagnostics.gif",
            "manifest": case_output / "frame_manifest.csv",
        }
        for shard in shards:
            step = shard_step(shard)
            tasks.append(
                {
                    "case": case,
                    "shard_path": str(shard),
                    "run_folder": str(run_folder),
                    "png_path": str(frames_folder / f"flowTime_{step}.png"),
                    "resume": args.resume,
                    "config": config,
                    "metadata": metadata,
                    "origin_3d": origin_3d,
                    "colormap": str(plot.get("colormap", "RdBu_r")),
                    "positive_color": str(plot.get("positive_marker_color", "black")),
                    "negative_color": str(plot.get("negative_marker_color", "#7b2cbf")),
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

    print(f"Analysis output: {analysis_output}", flush=True)
    print(f"Cases:           {len(selected_cases)}", flush=True)
    print(f"Frames:          {len(tasks)}", flush=True)
    print(f"Frame workers:   {min(args.workers, len(tasks))}", flush=True)
    print("Fit calculation: reused from completed JSON shards", flush=True)

    results = []
    with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks))) as executor:
        futures = {executor.submit(render_frame, task): task for task in tasks}
        for completed, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            action = "reused" if result["reused"] else "rendered"
            print(
                f"[{completed}/{len(tasks)}] {result['case']}/"
                f"flowTime_{result['step']}.png: {action}, "
                f"{result['fit_count']} successful fits",
                flush=True,
            )

    for case in selected_cases:
        ordered = sorted(
            (item for item in results if item["case"] == case),
            key=lambda item: item["step"],
        )
        paths = [Path(item["png_path"]) for item in ordered]
        output = case_outputs[case]
        save_gif(paths, output["gif"], args.fps)
        with output["manifest"].open("w", newline="") as manifest_file:
            writer = csv.writer(manifest_file)
            writer.writerow(
                [
                    "frame_index",
                    "snapshot_step",
                    "simulation_time",
                    "source_snapshot",
                    "successful_fits",
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
                        item["fit_count"],
                        item["png_path"],
                    ]
                )
        print(f"Saved GIF: {output['gif']}", flush=True)
        print(f"Manifest:  {output['manifest']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
