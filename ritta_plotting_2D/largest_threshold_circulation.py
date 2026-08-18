#!/usr/bin/env python3
"""Measure 2D circulation in the largest enclosed positive-threshold region."""

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_dilation, label


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


METHOD_VERSION = "2d-largest-enclosed-positive-threshold-v1"
DEFAULT_THRESHOLD_FRACTION = 0.02
EIGHT_CONNECTED = np.ones((3, 3), dtype=bool)
CSV_COLUMNS = (
    "frame_index",
    "frame_name",
    "step",
    "time",
    "peak_absolute_vorticity",
    "threshold_fraction",
    "threshold_vorticity",
    "components_found",
    "enclosed_components",
    "region_found",
    "threshold_cells",
    "region_area",
    "equivalent_radius",
    "circulation_positive",
    "x_center_positive",
    "y_center_positive",
    "source_path",
)


def positive_integer(value):
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("value must be an integer") from error
    if number < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return number


def threshold_fraction(value):
    try:
        fraction = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("threshold fraction must be a number") from error
    if not math.isfinite(fraction) or not 0.0 < fraction <= 1.0:
        raise argparse.ArgumentTypeError(
            "threshold fraction must be greater than 0 and at most 1"
        )
    return fraction


def strip_comments(text):
    return re.sub(r"//.*?$|#.*?$", "", text, flags=re.MULTILINE)


def reject_explicit_dt(simulation_config):
    text = strip_comments(simulation_config.read_text(encoding="utf-8"))
    if re.search(r"\bdt\s*=", text):
        raise ValueError(
            f"{simulation_config} sets an explicit dt; the project CFL time "
            "conversion is not automatically valid"
        )


def discover_runs(sweep_folder, case_names):
    available = {
        folder.name: folder
        for folder in sweep_folder.iterdir()
        if folder.is_dir() and (folder / "output").is_dir()
    }
    if case_names:
        if len(case_names) != len(set(case_names)):
            raise ValueError("Case names must not be repeated")
        missing = [name for name in case_names if name not in available]
        if missing:
            raise ValueError("Requested case folders are missing: " + ", ".join(missing))
        return [available[name] for name in case_names]
    return [available[name] for name in sorted(available)]


def component_touches_available_boundary(component, finite):
    if (
        np.any(component[0, :])
        or np.any(component[-1, :])
        or np.any(component[:, 0])
        or np.any(component[:, -1])
    ):
        return True
    return bool(np.any(binary_dilation(component, structure=EIGHT_CONNECTED) & ~finite))


def largest_enclosed_component(vorticity, threshold):
    omega = np.asarray(vorticity, dtype=float)
    finite = np.isfinite(omega)
    mask = finite & (omega >= threshold)
    labels, component_count = label(mask, structure=EIGHT_CONNECTED)
    enclosed = []
    for component_id in range(1, component_count + 1):
        component = labels == component_id
        if not component_touches_available_boundary(component, finite):
            enclosed.append((int(np.count_nonzero(component)), component_id))
    if not enclosed:
        return labels, component_count, 0, None
    _, selected_id = max(enclosed, key=lambda item: (item[0], -item[1]))
    return labels, component_count, len(enclosed), selected_id


def original_cells_in_component(frame, labels, component_id, threshold):
    cells = frame["cells"]
    dx = float(frame["dx"])
    x_left = float(frame["x"][0] - 0.5 * dx)
    y_bottom = float(frame["y"][0] - 0.5 * dx)
    columns = np.floor((cells["x"] - x_left) / dx).astype(int)
    rows = np.floor((cells["y"] - y_bottom) / dx).astype(int)
    inside_raster = (
        (rows >= 0)
        & (rows < labels.shape[0])
        & (columns >= 0)
        & (columns < labels.shape[1])
    )
    selected = np.zeros(len(cells["x"]), dtype=bool)
    valid_indices = np.flatnonzero(inside_raster)
    selected[valid_indices] = (
        labels[rows[valid_indices], columns[valid_indices]] == component_id
    )
    selected &= np.isfinite(cells["vorticity"])
    selected &= cells["vorticity"] >= threshold
    return selected


def measure_frame(frame, fraction):
    omega = np.asarray(frame["vorticity"], dtype=float)
    finite = omega[np.isfinite(omega)]
    peak = float(np.max(np.abs(finite))) if finite.size else math.nan
    threshold = fraction * peak if math.isfinite(peak) and peak > 0.0 else math.nan
    empty = {
        "peak_absolute_vorticity": peak,
        "threshold_fraction": fraction,
        "threshold_vorticity": threshold,
        "components_found": 0,
        "enclosed_components": 0,
        "region_found": False,
        "threshold_cells": 0,
        "region_area": math.nan,
        "equivalent_radius": math.nan,
        "circulation_positive": math.nan,
        "x_center_positive": math.nan,
        "y_center_positive": math.nan,
    }
    if not math.isfinite(threshold) or threshold <= 0.0:
        return empty

    labels, component_count, enclosed_count, component_id = (
        largest_enclosed_component(omega, threshold)
    )
    empty["components_found"] = component_count
    empty["enclosed_components"] = enclosed_count
    if component_id is None:
        return empty

    selected = original_cells_in_component(frame, labels, component_id, threshold)
    if not np.any(selected):
        return empty
    cells = frame["cells"]
    area = float(np.sum(cells["area"][selected]))
    weights = cells["vorticity"][selected] * cells["area"][selected]
    circulation = float(np.sum(weights))
    if not math.isfinite(circulation) or circulation <= 0.0:
        return empty

    return {
        **empty,
        "region_found": True,
        "threshold_cells": int(np.count_nonzero(selected)),
        "region_area": area,
        "equivalent_radius": math.sqrt(area / math.pi),
        "circulation_positive": circulation,
        "x_center_positive": float(np.sum(cells["x"][selected] * weights) / circulation),
        "y_center_positive": float(np.sum(cells["y"][selected] * weights) / circulation),
    }


def analyze_frame(task):
    frame = load_vorticity_frame(
        task["path"],
        task["source_index"],
        task["config"],
        task["metadata"],
        include_cells=True,
    )
    return {
        "run_name": task["run_name"],
        "frame_index": task["frame_index"],
        "frame_name": frame["source_filename"],
        "step": frame["step"],
        "time": frame["time"],
        **measure_frame(frame, task["threshold_fraction"]),
        "source_path": frame["source_path"],
    }


def config_digest(analysis_config, simulation_config, threshold):
    digest = hashlib.sha256(METHOD_VERSION.encode("utf-8"))
    digest.update(analysis_config.read_bytes())
    digest.update(simulation_config.read_bytes())
    digest.update(f"{threshold:.17g}".encode("ascii"))
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
            payload.get("method_version") != METHOD_VERSION
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
        "method_version": METHOD_VERSION,
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


def save_combined_plots(
    output_folder,
    run_info,
    rows_by_run,
    figure_size,
    fraction=DEFAULT_THRESHOLD_FRACTION,
):
    threshold_label = f"{100.0 * fraction:g}%"
    colors = distinct_line_colors(len(run_info))
    figures = (
        (
            "combined_circulation_vs_time.png",
            f"Largest enclosed {threshold_label}-threshold circulation versus simulation time",
            "simulation time",
            False,
        ),
        (
            "combined_circulation_vs_time_over_tau.png",
            f"Largest enclosed {threshold_label}-threshold circulation versus "
            r"normalized time $t/\tau$",
            r"normalized simulation time $t/\tau$",
            True,
        ),
    )
    output_folder.mkdir(parents=True, exist_ok=True)
    for filename, title, x_label, normalized in figures:
        figure, axis = plt.subplots(figsize=figure_size, constrained_layout=True)
        for color, info in zip(colors, run_info):
            rows = rows_by_run[info["name"]]
            times = np.asarray([row["time"] for row in rows], dtype=float)
            if normalized:
                times = times / info["tau"]
            circulation = np.asarray(
                [row["circulation_positive"] for row in rows], dtype=float
            )
            axis.plot(times, circulation, color=color, label=rf"$\tau={info['tau']:g}$")
        axis.set_xlabel(x_label)
        axis.set_ylabel("positive circulation")
        axis.set_title(title)
        axis.grid(True, alpha=0.3)
        axis.legend()
        if normalized:
            axis.set_xlim(0.0, 1.0)
        path = output_folder / filename
        figure.savefig(path, dpi=160)
        plt.close(figure)
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Failed to create plot: {path}")
        print(f"Saved {path}", flush=True)


def write_datasets(path, run_info):
    lines = ["# Independent 2D largest-enclosed-threshold circulation results.", ""]
    for info in run_info:
        lines.extend(
            [
                "[[dataset]]",
                f"name = {json.dumps(info['name'])}",
                f"csv = {json.dumps((Path(info['name']) / 'largest_threshold_circulation.csv').as_posix())}",
                f"run_folder = {json.dumps(str(info['run_folder']))}",
                f"forcing_end_time = {info['tau']:g}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def fraction_tag(fraction):
    return f"{fraction:.12g}".replace("-", "m").replace(".", "p")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "For every 2D frame, integrate positive vorticity in the largest "
            "enclosed positive 8-connected region above "
            "FRACTION * max(abs(omega))."
        )
    )
    parser.add_argument("sweep_folder", type=Path)
    parser.add_argument("config_file", type=Path)
    parser.add_argument(
        "--threshold-fraction",
        type=threshold_fraction,
        default=DEFAULT_THRESHOLD_FRACTION,
    )
    parser.add_argument("--stride", type=positive_integer, default=1)
    parser.add_argument("--workers", type=positive_integer, default=1)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cases", nargs="+", metavar="RUN_NAME")
    args = parser.parse_args()

    sweep_folder = args.sweep_folder.expanduser().resolve()
    if not sweep_folder.is_dir():
        parser.error(f"sweep folder does not exist: {sweep_folder}")
    config_path = args.config_file.expanduser().resolve()
    config = load_config(config_path)
    output_folder = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else SCRIPT_FOLDER
        / "outputs"
        / (
            f"{sweep_folder.name}_largest_enclosed_threshold_"
            f"{fraction_tag(args.threshold_fraction)}"
        )
    )
    output_folder.mkdir(parents=True, exist_ok=True)

    runs = discover_runs(sweep_folder, args.cases)
    if not runs:
        raise ValueError(f"No immediate child runs were found in {sweep_folder}")

    tasks = []
    run_info = []
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
        tau = simulation_parameter(run_folder, config, "b_f_tau")
        if not math.isfinite(tau) or tau <= 0.0:
            raise ValueError(f"Invalid b_f_tau in {simulation_config}: {tau}")
        frames = discover_frames(run_folder, config)
        selected = list(enumerate(frames))[:: args.stride]
        digest = config_digest(config_path, simulation_config, args.threshold_fraction)
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
                    "threshold_fraction": args.threshold_fraction,
                    "config_digest": digest,
                }
            )
    run_info.sort(key=lambda item: (item["tau"], item["name"]))

    print(f"Sweep:             {sweep_folder}", flush=True)
    print(f"Cases:             {len(run_info)}", flush=True)
    print(f"Selected frames:   {len(tasks)}", flush=True)
    print(f"Frame workers:     {min(args.workers, len(tasks))}", flush=True)
    print(
        f"Threshold:         {args.threshold_fraction:g} * max(abs(vorticity))",
        flush=True,
    )
    print("Region selection:  largest enclosed 8-connected component", flush=True)
    print("Integral:          original visible AMR cells with native dx^2", flush=True)
    print(f"Output:            {output_folder}", flush=True)

    results = []
    pending = []
    for task in tasks:
        path = shard_path(
            output_folder,
            task["run_name"],
            frame_step(task["path"]),
        )
        reused = reusable_shard(path, task) if args.resume else None
        if reused is None:
            pending.append((task, path))
        else:
            results.append(reused)
    print(f"Reused frames:     {len(results)}", flush=True)
    print(f"Frames to process: {len(pending)}", flush=True)

    if pending:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(pending))) as executor:
            future_tasks = {
                executor.submit(analyze_frame, task): (task, path)
                for task, path in pending
            }
            for completed, future in enumerate(as_completed(future_tasks), start=1):
                task, path = future_tasks[future]
                result = future.result()
                save_shard(path, task, result)
                results.append(result)
                status = (
                    "enclosed region"
                    if result["region_found"]
                    else "no enclosed region"
                )
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
        csv_path = output_folder / info["name"] / "largest_threshold_circulation.csv"
        write_case_csv(csv_path, rows)
        print(f"Saved {csv_path}", flush=True)

    write_datasets(output_folder / "datasets.toml", run_info)
    figure_size = (
        float(config["plot"].get("figure_width", 10.0)),
        float(config["plot"].get("figure_height", 7.0)),
    )
    save_combined_plots(
        output_folder,
        run_info,
        rows_by_run,
        figure_size,
        args.threshold_fraction,
    )
    method_path = output_folder / "method.json"
    method_path.write_text(
        json.dumps(
            {
                "method_version": METHOD_VERSION,
                "threshold_fraction": args.threshold_fraction,
                "threshold_reference": "maximum absolute vorticity in each frame",
                "connectivity": 8,
                "requires_enclosed_component": True,
                "selection": "largest threshold-raster area",
                "integration": "original visible AMR cells weighted by native dx^2",
                "stride": args.stride,
                "workers": min(args.workers, len(tasks)),
                "analysis_config": str(config_path),
                "cases": [info["name"] for info in run_info],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved {method_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
