#!/usr/bin/env python3
"""Apply the 2D vortex-identification method to z slices of 3D output."""

import argparse
import hashlib
import importlib
import json
import math
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np


SCRIPT_FOLDER = Path(__file__).resolve().parent
VORTEX_FOLDER = SCRIPT_FOLDER.parent / "ritta_vortex_identification"
sys.path.insert(0, str(VORTEX_FOLDER))

from common import (  # noqa: E402
    _component_index,
    _decode,
    _rasterize_vorticity,
    _visible_amr_cells,
    discover_frames,
    load_config,
    simulation_metadata,
    simulation_parameter,
    simulation_time,
)


hmaxima_module = importlib.import_module("01_find_hmaxima")
regions_module = importlib.import_module("02_make_regions")
fits_module = importlib.import_module("03_fit_vortices")
metrics_module = importlib.import_module("04_positive_vortex_metrics")

METHOD_VERSION = "3d-slice-vortex-identification-v1"
VORTICITY_COMPONENT = "edge_aux_2"
DEFAULT_SLICE_Z = 1.0e-6


def positive_integer(value):
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("value must be an integer") from error
    if number < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return number


def finite_float(value):
    try:
        number = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("value must be a number") from error
    if not math.isfinite(number):
        raise argparse.ArgumentTypeError("value must be finite")
    return number


def box_bounds(record, dimensions=3):
    names = getattr(record.dtype, "names", None)
    values = (
        [int(record[name]) for name in names]
        if names
        else [int(value) for value in np.asarray(record).reshape(-1)]
    )
    if len(values) < 2 * dimensions:
        raise ValueError("HDF5 box record does not contain enough 3D bounds")
    return (
        np.asarray(values[:dimensions], dtype=int),
        np.asarray(values[dimensions : 2 * dimensions], dtype=int),
    )


def chunk_boundaries(offsets, boxes, data_size, components):
    sizes = []
    for box in boxes:
        lower, upper = box_bounds(box)
        sizes.append(int(np.prod(upper - lower + 1)) * components)

    candidates = []
    if len(offsets) == len(sizes) + 1:
        candidates.append(offsets)
        if offsets[0] == 0:
            candidates.append(np.cumsum(offsets))
    if len(offsets) == len(sizes):
        candidates.append(np.concatenate(([0], np.cumsum(offsets))))

    for candidate in candidates:
        candidate = np.asarray(candidate, dtype=np.int64)
        if (
            len(candidate) == len(sizes) + 1
            and candidate[0] == 0
            and candidate[-1] == data_size
            and np.array_equal(np.diff(candidate), sizes)
        ):
            return candidate
    raise ValueError("Could not interpret 3D HDF5 chunk offsets")


def strip_cpp_comments(text):
    return re.sub(r"//.*?$|#.*?$", "", text, flags=re.MULTILINE)


def read_vector(text, name):
    match = re.search(rf"\b{re.escape(name)}\s*=\s*\(([^)]*)\)\s*;", text)
    if match is None:
        return None
    try:
        return np.asarray(
            [float(item.strip()) for item in match.group(1).split(",")],
            dtype=float,
        )
    except ValueError:
        return None


def simulation_origin_3d(simulation_config):
    text = strip_cpp_comments(simulation_config.read_text(encoding="utf-8"))
    block_match = re.search(r"\bblock\s*\{([^{}]*)\}", text, re.DOTALL)
    block_text = block_match.group(1) if block_match else ""
    base = read_vector(block_text, "base")
    extent = read_vector(block_text, "extent")
    if base is None or extent is None:
        base = read_vector(text, "bd_base")
        extent = read_vector(text, "bd_extent")
    if base is None or extent is None or len(base) < 3 or len(extent) < 3:
        raise ValueError(
            f"Could not read a 3D domain base and extent from {simulation_config}"
        )
    return base[:3] + 0.5 * extent[:3]


def reject_explicit_dt(simulation_config):
    text = strip_cpp_comments(simulation_config.read_text(encoding="utf-8"))
    if re.search(r"\bdt\s*=", text):
        raise ValueError(
            f"{simulation_config} sets an explicit dt; the project CFL time "
            "conversion is not automatically valid"
        )


def load_slice_tiles(path, metadata, origin_3d, slice_z):
    """Read only the edge_aux_2 cell plane intersecting the requested z value."""
    tiles = []
    with h5py.File(path, "r") as handle:
        dimensions = int(_decode(handle["Chombo_global"].attrs["SpaceDim"]))
        if dimensions != 3:
            raise ValueError(f"{path.name} is {dimensions}D; expected 3D output")
        levels = int(_decode(handle.attrs["num_levels"]))
        components = int(_decode(handle.attrs["num_components"]))
        component = _component_index(handle, VORTICITY_COMPONENT)

        for level in range(levels):
            group = handle[f"level_{level}"]
            dx = float(metadata["dx_base"]) / (2**level)
            origin = np.asarray(origin_3d, dtype=float) * (2**level)
            target_k = math.floor(origin[2] + slice_z / dx)
            boxes = group["boxes"]
            data = group["data:datatype=0"]
            if "data_attributes" in group and "offsets" in group["data_attributes"]:
                offsets = np.asarray(
                    group["data_attributes/offsets"], dtype=np.int64
                )
            else:
                offsets = np.asarray(group["offsets"], dtype=np.int64)
            boundaries = chunk_boundaries(
                offsets,
                boxes,
                len(data),
                components,
            )

            for chunk_index, box in enumerate(boxes):
                lower, upper = box_bounds(box)
                if target_k < lower[2] or target_k > upper[2]:
                    continue
                counts = upper - lower + 1
                nx, ny, nz = (int(value) for value in counts)
                plane_size = nx * ny
                local_k = target_k - int(lower[2])
                start = int(boundaries[chunk_index])
                start += component * nx * ny * nz + local_k * plane_size
                raw = np.asarray(data[start : start + plane_size], dtype=float)
                if len(raw) != plane_size:
                    raise ValueError(
                        f"Incomplete slice plane in {path.name}, level {level}, "
                        f"chunk {chunk_index}"
                    )
                values = raw.reshape(nx, ny, order="F").T
                tiles.append(
                    {
                        "level": level,
                        "dx": dx,
                        "values": values,
                        "bounds": (
                            (lower[0] - origin[0]) * dx,
                            (upper[0] + 1 - origin[0]) * dx,
                            (lower[1] - origin[1]) * dx,
                            (upper[1] + 1 - origin[1]) * dx,
                        ),
                    }
                )

    if not tiles:
        raise ValueError(f"No AMR cells intersect z={slice_z:g} in {path}")
    return tiles


def load_slice_frame(path, source_index, config, metadata, origin_3d, slice_z):
    tiles = load_slice_tiles(path, metadata, origin_3d, slice_z)
    x, y, vorticity, finest_dx = _rasterize_vorticity(tiles)
    return {
        "source_filename": path.name,
        "source_path": str(path.resolve()),
        "step": int(re.fullmatch(r"flowTime_(\d+)\.hdf5", path.name).group(1)),
        "time": simulation_time(path, source_index, config, metadata),
        "dx": finest_dx,
        "x": x,
        "y": y,
        "vorticity": vorticity,
        "cells": _visible_amr_cells(tiles),
    }


def frame_candidates(maxima, regions):
    intended, clamped, _, _ = regions
    return [
        {
            "candidate_id": int(candidate_id),
            "peak_x": float(peak_x),
            "peak_y": float(peak_y),
            "peak_vorticity": float(peak_vorticity),
            "intended_bounds": intended_bounds,
            "clamped_bounds": clamped_bounds,
        }
        for candidate_id, peak_x, peak_y, peak_vorticity, intended_bounds, clamped_bounds in zip(
            maxima["candidate_ids"],
            maxima["peak_x"],
            maxima["peak_y"],
            maxima["peak_vorticity"],
            intended,
            clamped,
        )
    ]


def empty_record(frame_index, frame):
    return {
        "frame_index": frame_index,
        "frame_name": frame["source_filename"],
        "time": frame["time"],
        "candidate_id": "",
        "vortex_id": "",
        "fit_success": False,
        "boundary_radius": math.nan,
        "circulation_positive": math.nan,
        "x_center_positive": math.nan,
        "y_center_positive": math.nan,
        "x_displacement": math.nan,
    }


def analyze_frame(task):
    frame = load_slice_frame(
        task["path"],
        task["source_index"],
        task["config"],
        task["metadata"],
        task["origin_3d"],
        task["slice_z"],
    )
    maxima = hmaxima_module.find_hmaxima(frame, task["config"])
    regions = regions_module.make_regions(
        maxima["peak_x"],
        maxima["peak_y"],
        frame["x"],
        frame["y"],
        frame["dx"],
        frame["time"],
        task["reynolds_number"],
        task["config"],
    )
    candidates = frame_candidates(maxima, regions)
    fit_results = [
        fits_module.fit_candidate(
            frame,
            candidate,
            task["config"],
            task["fit_settings"],
        )
        for candidate in candidates
    ]

    records = []
    for fit in fit_results:
        parameters = fit["parameters"]
        radius = float(fit["boundary_radius"])
        positive_center = fit["positive_center"]
        usable = (
            fit["success"]
            and np.all(np.isfinite(parameters))
            and math.isfinite(radius)
        )
        if usable:
            circulation, center_x, center_y = metrics_module.positive_metrics(
                frame["cells"],
                positive_center[0],
                positive_center[1],
                radius,
            )
        else:
            circulation, center_x, center_y = math.nan, math.nan, math.nan
        records.append(
            {
                "frame_index": task["frame_index"],
                "frame_name": frame["source_filename"],
                "time": frame["time"],
                "candidate_id": fit["candidate_id"],
                "vortex_id": "",
                "fit_success": bool(fit["success"]),
                "boundary_radius": radius,
                "circulation_positive": circulation,
                "x_center_positive": center_x,
                "y_center_positive": center_y,
                "x_displacement": center_x,
                "_fit_x": float(positive_center[0]),
                "_fit_y": float(positive_center[1]),
            }
        )
    if not records:
        records.append(empty_record(task["frame_index"], frame))

    return {
        "run_name": task["run_name"],
        "frame_index": task["frame_index"],
        "step": frame["step"],
        "source_filename": frame["source_filename"],
        "candidate_count": len(candidates),
        "successful_fits": sum(bool(fit["success"]) for fit in fit_results),
        "records": records,
    }


def config_digest(analysis_config, simulation_config):
    digest = hashlib.sha256(METHOD_VERSION.encode("utf-8"))
    digest.update(analysis_config.read_bytes())
    digest.update(simulation_config.read_bytes())
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
            or not math.isclose(
                float(payload.get("slice_z", math.nan)),
                task["slice_z"],
                rel_tol=0.0,
                abs_tol=1.0e-15,
            )
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
        "slice_z": task["slice_z"],
        "result": result,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, allow_nan=True), encoding="utf-8")
    temporary.replace(path)


def discover_runs(sweep_folder, case_names):
    available = {
        folder.name: folder
        for folder in sweep_folder.iterdir()
        if folder.is_dir() and (folder / "output").is_dir()
    }
    if case_names:
        if len(set(case_names)) != len(case_names):
            raise ValueError("Case folder names must not be repeated")
        missing = [name for name in case_names if name not in available]
        if missing:
            raise ValueError("Requested case folders are missing: " + ", ".join(missing))
        return [available[name] for name in case_names]
    return [available[name] for name in sorted(available)]


def write_datasets_manifest(path, run_info):
    lines = [
        "# 3D z-slice results produced with the 2D vortex-identification method.",
        "",
    ]
    for info in sorted(run_info, key=lambda item: (item["tau"], item["name"])):
        lines.extend(
            [
                "[[dataset]]",
                f"name = {json.dumps(info['name'])}",
                f"csv = {json.dumps((Path(info['name']) / 'positive_vortex_metrics.csv').as_posix())}",
                f"run_folder = {json.dumps(str(info['run_folder']))}",
                f"forcing_end_time = {info['tau']:g}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Apply the standalone 2D h-maxima/rectangle/Gaussian-fit/circle-"
            "integration method to edge_aux_2 on a 3D meridional slice."
        )
    )
    parser.add_argument("sweep_folder", type=Path)
    parser.add_argument("config_file", type=Path)
    parser.add_argument("--stride", type=positive_integer, default=1)
    parser.add_argument("--workers", type=positive_integer, default=1)
    parser.add_argument("--slice-z", type=finite_float, default=DEFAULT_SLICE_Z)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--cases",
        nargs="+",
        metavar="RUN_NAME",
        help="process only these immediate child run folders",
    )
    args = parser.parse_args()

    sweep_folder = args.sweep_folder.expanduser().resolve()
    if not sweep_folder.is_dir():
        parser.error(f"sweep folder does not exist: {sweep_folder}")
    analysis_config_path = args.config_file.expanduser().resolve()
    config = load_config(analysis_config_path)
    fit_settings = fits_module.fit_settings(config)
    output_folder = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else SCRIPT_FOLDER
        / "outputs"
        / f"{sweep_folder.name}_slice_vortex_identification"
    )
    output_folder.mkdir(parents=True, exist_ok=True)

    runs = discover_runs(sweep_folder, args.cases)
    if not runs:
        raise ValueError(f"No immediate child runs were found in {sweep_folder}")

    tasks = []
    run_info = []
    for run_folder in runs:
        frames = discover_frames(run_folder, config)
        selected = list(enumerate(frames))[:: args.stride]
        metadata = simulation_metadata(run_folder, config)
        if (
            metadata.get("cfl") is None
            or metadata.get("dx_base") is None
            or int(metadata.get("num_amr_levels", -1)) < 0
        ):
            raise ValueError(f"Incomplete time metadata for {run_folder}")
        simulation_config = Path(metadata["source"])
        reject_explicit_dt(simulation_config)
        origin_3d = simulation_origin_3d(simulation_config)
        reynolds_number = simulation_parameter(run_folder, config, "Re")
        tau = simulation_parameter(run_folder, config, "b_f_tau")
        if not math.isfinite(reynolds_number) or reynolds_number <= 0.0:
            raise ValueError(f"Invalid Re in {simulation_config}: {reynolds_number}")
        if not math.isfinite(tau) or tau <= 0.0:
            raise ValueError(f"Invalid b_f_tau in {simulation_config}: {tau}")
        digest = config_digest(analysis_config_path, simulation_config)
        run_info.append(
            {
                "name": run_folder.name,
                "run_folder": run_folder,
                "tau": tau,
            }
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
                    "origin_3d": origin_3d,
                    "slice_z": args.slice_z,
                    "reynolds_number": reynolds_number,
                    "fit_settings": fit_settings,
                    "config_digest": digest,
                }
            )

    print(f"Sweep:            {sweep_folder}", flush=True)
    print(f"Cases:            {len(runs)}", flush=True)
    print(f"Selected frames:  {len(tasks)}", flush=True)
    print(f"Frame workers:    {min(args.workers, len(tasks))}", flush=True)
    print(f"Slice:            z={args.slice_z:g}", flush=True)
    print(
        "Method:           positive h-maxima -> diffusive rectangle -> "
        "circular Gaussian dipole fit -> positive AMR-cell circulation",
        flush=True,
    )
    print(
        f"Fit boundary:     {fit_settings['boundary_fraction']:g} of fitted peak",
        flush=True,
    )
    print(f"Output:           {output_folder}", flush=True)

    results = []
    pending = []
    for task in tasks:
        path = shard_path(output_folder, task["run_name"], int(re.search(r"(\d+)$", task["path"].stem).group(1)))
        reused = reusable_shard(path, task) if args.resume else None
        if reused is None:
            pending.append((task, path))
        else:
            results.append(reused)
    print(f"Reused frames:    {len(results)}", flush=True)
    print(f"Frames to process:{len(pending):>6}", flush=True)

    if pending:
        worker_count = min(args.workers, len(pending))
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            future_tasks = {
                executor.submit(analyze_frame, task): (task, path)
                for task, path in pending
            }
            for completed, future in enumerate(as_completed(future_tasks), start=1):
                task, path = future_tasks[future]
                result = future.result()
                save_shard(path, task, result)
                results.append(result)
                print(
                    f"[{completed}/{len(pending)}] {result['run_name']}/"
                    f"{result['source_filename']}: "
                    f"{result['successful_fits']}/{result['candidate_count']} fits",
                    flush=True,
                )

    results_by_run = {info["name"]: [] for info in run_info}
    for result in results:
        results_by_run[result["run_name"]].append(result)

    tracking = config.get("tracking", {})
    max_displacement = float(tracking.get("max_displacement", 0.5))
    new_track_max_displacement = float(
        tracking.get("new_track_max_displacement", 0.5)
    )
    if not math.isfinite(max_displacement) or max_displacement <= 0.0:
        raise ValueError("[tracking] max_displacement must be finite and positive")
    if (
        not math.isfinite(new_track_max_displacement)
        or new_track_max_displacement <= 0.0
    ):
        raise ValueError(
            "[tracking] new_track_max_displacement must be finite and positive"
        )
    for info in run_info:
        ordered = sorted(
            results_by_run[info["name"]],
            key=lambda item: item["frame_index"],
        )
        if len(ordered) != sum(task["run_name"] == info["name"] for task in tasks):
            raise RuntimeError(f"Missing completed frames for {info['name']}")
        records_by_frame = [item["records"] for item in ordered]
        metrics_module.assign_vortex_tracks(
            records_by_frame,
            max_displacement,
            new_track_max_displacement,
        )
        metrics_path = output_folder / info["name"] / "positive_vortex_metrics.csv"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_module.write_metrics(
            metrics_path,
            [record for frame_records in records_by_frame for record in frame_records],
        )
        print(f"Saved {metrics_path}", flush=True)

    manifest_path = output_folder / "datasets.toml"
    write_datasets_manifest(manifest_path, run_info)
    metadata_path = output_folder / "method.json"
    metadata_path.write_text(
        json.dumps(
            {
                "method_version": METHOD_VERSION,
                "slice_z": args.slice_z,
                "stride": args.stride,
                "workers": min(args.workers, len(tasks)),
                "analysis_config": str(analysis_config_path),
                "cases": [info["name"] for info in run_info],
                "vorticity_component": VORTICITY_COMPONENT,
                "fit_boundary_fraction": fit_settings["boundary_fraction"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved {manifest_path}", flush=True)
    print(f"Saved {metadata_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
