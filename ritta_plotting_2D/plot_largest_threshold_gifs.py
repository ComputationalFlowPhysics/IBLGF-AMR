#!/usr/bin/env python3
"""Render GIFs from completed largest-enclosed-threshold circulation shards."""

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
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from PIL import Image


SCRIPT_FOLDER = Path(__file__).resolve().parent
VORTEX_FOLDER = SCRIPT_FOLDER.parent / "ritta_vortex_identification"
sys.path.insert(0, str(VORTEX_FOLDER))

from common import load_config, load_vorticity_frame, simulation_metadata  # noqa: E402
from largest_threshold_circulation import (  # noqa: E402
    METHOD_VERSION,
    fraction_tag,
    largest_enclosed_component,
)


SHARD_PATTERN = re.compile(r"flowTime_(\d+)\.json$")
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
        raise ValueError(f"Not a threshold-result shard: {path.name}")
    return int(match.group(1))


def load_datasets(path):
    with path.open("rb") as handle:
        document = tomllib.load(handle)
    datasets = document.get("dataset", [])
    if not datasets:
        raise ValueError(f"No [[dataset]] entries were found in {path}")
    return datasets


def configured_limits(config, requested_x, requested_y):
    plot = config.get("plot", {})

    def choose(requested, minimum_name, maximum_name):
        values = requested or (
            plot.get(minimum_name, math.nan),
            plot.get(maximum_name, math.nan),
        )
        values = tuple(float(value) for value in values)
        if all(math.isfinite(value) for value in values) and values[0] < values[1]:
            return values
        return None

    return (
        choose(requested_x, "x_axis_min", "x_axis_max"),
        choose(requested_y, "y_axis_min", "y_axis_max"),
    )


def render_frame(task):
    shard = Path(task["shard_path"])
    payload = json.loads(shard.read_text(encoding="utf-8"))
    if payload.get("method_version") != METHOD_VERSION:
        raise ValueError(f"Unexpected threshold-result version in {shard}")
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
            "region_found": bool(result["region_found"]),
            "reused": True,
        }

    frame = load_vorticity_frame(
        source_path,
        int(payload["source_index"]),
        task["config"],
        task["metadata"],
        include_cells=False,
    )
    vorticity = np.asarray(frame["vorticity"], dtype=float)
    finite = np.abs(vorticity[np.isfinite(vorticity)])
    color_limit = float(np.max(finite)) if finite.size else 1.0
    if not math.isfinite(color_limit) or color_limit <= 0.0:
        color_limit = 1.0
    threshold = float(result["threshold_vorticity"])
    selected_mask = np.zeros(vorticity.shape, dtype=bool)
    if bool(result["region_found"]):
        labels, _, _, component_id = largest_enclosed_component(
            vorticity,
            threshold,
        )
        if component_id is None:
            raise RuntimeError(
                f"Saved enclosed region could not be reconstructed for {source_path}"
            )
        selected_mask = labels == component_id

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
    figure.colorbar(image, ax=axis, label=r"vorticity $\omega$")

    region_color = task["region_color"]
    if np.any(selected_mask):
        overlay = np.ma.masked_where(~selected_mask, np.ones(selected_mask.shape))
        axis.imshow(
            overlay,
            origin="lower",
            extent=extent,
            interpolation="nearest",
            cmap=mcolors.ListedColormap([region_color]),
            vmin=0.0,
            vmax=1.0,
            alpha=float(task["mask_alpha"]),
            aspect="equal",
        )
        axis.contour(
            frame["x"],
            frame["y"],
            selected_mask.astype(float),
            levels=[0.5],
            colors=[region_color],
            linewidths=float(task["line_width"]),
        )

    center_x = float(result["x_center_positive"])
    center_y = float(result["y_center_positive"])
    if math.isfinite(center_x) and math.isfinite(center_y):
        axis.scatter(
            center_x,
            center_y,
            color=task["center_color"],
            marker="x",
            s=float(task["marker_size"]),
            linewidths=float(task["line_width"]),
            zorder=5,
        )

    circulation = float(result["circulation_positive"])
    area = float(result["region_area"])
    if bool(result["region_found"]):
        annotation = (
            f"threshold={threshold:.4g}, area={area:.4g}, "
            f"$\\Gamma_+$={circulation:.5g}"
        )
    else:
        annotation = "No enclosed threshold region"
    axis.text(
        0.02,
        0.02,
        annotation,
        transform=axis.transAxes,
        va="bottom",
        fontsize=float(task["text_size"]),
        bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none"},
    )
    axis.legend(
        handles=[
            Patch(
                facecolor=mcolors.to_rgba(region_color, task["mask_alpha"]),
                edgecolor=region_color,
                label=f"largest enclosed {task['threshold_label']} region",
            ),
            Line2D(
                [0],
                [0],
                color=task["center_color"],
                marker="x",
                linestyle="none",
                label="saved circulation centroid",
            ),
        ],
        loc="upper right",
        fontsize=float(task["text_size"]),
    )
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_title(
        f"{task['case']}: largest enclosed {task['threshold_label']} "
        "vorticity region, "
        f"t={float(result['time']):.6g}, step={int(result['step'])}"
    )
    if task["x_limits"] is not None:
        axis.set_xlim(*task["x_limits"])
    if task["y_limits"] is not None:
        axis.set_ylim(*task["y_limits"])

    png_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(png_path, dpi=120)
    plt.close(figure)
    if not png_path.is_file() or png_path.stat().st_size == 0:
        raise RuntimeError(f"Failed to create threshold frame: {png_path}")
    return {
        "case": task["case"],
        "step": int(result["step"]),
        "time": float(result["time"]),
        "source_path": str(source_path.resolve()),
        "png_path": str(png_path.resolve()),
        "region_found": bool(result["region_found"]),
        "reused": False,
    }


def save_gif(frame_paths, output_path, fps):
    if not frame_paths:
        raise ValueError(f"No frames are available for {output_path}")
    images = []
    for path in frame_paths:
        with Image.open(path) as image:
            images.append(image.convert("RGB").copy())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=max(1, round(1000 / fps)),
        loop=0,
        optimize=False,
    )
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError(f"Failed to create threshold GIF: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Render GIFs from saved largest-enclosed 2% circulation results. "
            "The circulation integral is not repeated."
        )
    )
    parser.add_argument("analysis_output", type=Path)
    parser.add_argument("--workers", type=positive_integer, default=1)
    parser.add_argument("--stride", type=positive_integer, default=1)
    parser.add_argument("--fps", type=positive_integer, default=DEFAULT_FPS)
    parser.add_argument("--resume", action="store_true")
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
    if method.get("method_version") != METHOD_VERSION:
        parser.error(f"unexpected analysis method in {method_path}")
    config_path = Path(method["analysis_config"]).expanduser().resolve()
    if not config_path.is_file():
        parser.error(f"analysis configuration does not exist: {config_path}")
    config = load_config(config_path)
    if args.x_limits is not None and args.x_limits[0] >= args.x_limits[1]:
        parser.error("--x-limits requires MIN < MAX")
    if args.y_limits is not None and args.y_limits[0] >= args.y_limits[1]:
        parser.error("--y-limits requires MIN < MAX")
    x_limits, y_limits = configured_limits(config, args.x_limits, args.y_limits)

    datasets = load_datasets(datasets_path)
    available = {str(item["name"]): item for item in datasets}
    selected_cases = args.cases or [str(item["name"]) for item in datasets]
    missing = [name for name in selected_cases if name not in available]
    if missing:
        parser.error("requested cases are absent from datasets.toml: " + ", ".join(missing))

    fraction = float(method["threshold_fraction"])
    threshold_label = f"{100.0 * fraction:g}%"
    plot = config.get("plot", {})
    threshold_plot = config.get("threshold_mask", {})
    tasks = []
    case_outputs = {}
    for case in selected_cases:
        run_folder = Path(available[case]["run_folder"]).expanduser().resolve()
        metadata = simulation_metadata(run_folder, config)
        shards = sorted(
            (analysis_output / case / "frame_results").glob("flowTime_*.json"),
            key=shard_step,
        )[:: args.stride]
        if not shards:
            raise FileNotFoundError(f"No completed threshold shards were found for {case}")
        case_output = analysis_output / "threshold_gifs" / case
        frames_folder = case_output / "frames"
        gif_path = case_output / (
            f"{case}_largest_enclosed_threshold_{fraction_tag(fraction)}.gif"
        )
        case_outputs[case] = {
            "gif": gif_path,
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
                    "threshold_label": threshold_label,
                    "colormap": str(plot.get("colormap", "RdBu_r")),
                    "region_color": str(threshold_plot.get("positive_color", "#ffb000")),
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

    print(f"Analysis output: {analysis_output}", flush=True)
    print(f"Cases:           {len(selected_cases)}", flush=True)
    print(f"Frames:          {len(tasks)}", flush=True)
    print(f"Frame workers:   {min(args.workers, len(tasks))}", flush=True)
    print(f"GIF stride:      {args.stride}", flush=True)
    print("Circulation:     reused from completed JSON shards", flush=True)
    print("Display mask:    reconstructed from saved per-frame threshold", flush=True)

    results = []
    with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks))) as executor:
        futures = {executor.submit(render_frame, task): task for task in tasks}
        for completed, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            action = "reused" if result["reused"] else "rendered"
            print(
                f"[{completed}/{len(tasks)}] {result['case']}/"
                f"flowTime_{result['step']}.png: {action}",
                flush=True,
            )

    for case in selected_cases:
        ordered = sorted(
            (item for item in results if item["case"] == case),
            key=lambda item: item["step"],
        )
        output = case_outputs[case]
        save_gif([Path(item["png_path"]) for item in ordered], output["gif"], args.fps)
        output["manifest"].parent.mkdir(parents=True, exist_ok=True)
        with output["manifest"].open("w", newline="") as manifest_file:
            writer = csv.writer(manifest_file)
            writer.writerow(
                [
                    "frame_index",
                    "snapshot_step",
                    "simulation_time",
                    "source_snapshot",
                    "region_found",
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
                        int(item["region_found"]),
                        item["png_path"],
                    ]
                )
        print(f"Saved GIF: {output['gif']}", flush=True)
        print(f"Manifest:  {output['manifest']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
