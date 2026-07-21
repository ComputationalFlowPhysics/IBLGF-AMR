"""Shared CSV loading and headless time-series plotting."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def finite_setting(config: dict, name: str) -> float:
    value = float(config["time_series"].get(name, math.nan))
    if not math.isfinite(value):
        raise ValueError(f"[time_series] {name} must be finite.")
    return value


def read_metrics(path: str | Path) -> list[dict]:
    """Load only the columns needed by the time-series plots."""
    path = Path(path).expanduser().resolve()
    rows = []
    with path.open(newline="") as handle:
        for raw in csv.DictReader(handle):
            rows.append({
                "frame_index": int(raw["frame_index"]),
                "time": float(raw["time"]),
                "vortex_id": int(raw["vortex_id"]) if raw["vortex_id"] else None,
                "circulation_positive": float(raw["circulation_positive"]),
                "x_center_positive": float(raw["x_center_positive"]),
                "x_displacement": float(raw["x_displacement"]),
            })
    if not rows:
        raise ValueError(f"Metrics file contains no frame rows: {path}")
    return rows


def track_series(rows: list[dict], value_name: str, dataset_name: str | None = None) -> list[dict]:
    """Build one NaN-gapped line for each saved vortex ID."""
    frame_times = {}
    tracks = {}
    for row in rows:
        frame_times.setdefault(row["frame_index"], row["time"])
        if row["vortex_id"] is not None:
            tracks.setdefault(row["vortex_id"], {})[row["frame_index"]] = row

    frame_indices = sorted(frame_times)
    times = np.asarray([frame_times[index] for index in frame_indices])
    series = []
    for vortex_id in sorted(tracks):
        # A dataset name stays readable while still distinguishing multiple vortices.
        if dataset_name is None:
            label = f"vortex {vortex_id}"
        elif len(tracks) == 1:
            label = dataset_name
        else:
            label = f"{dataset_name} (vortex {vortex_id})"
        values = np.asarray([
            tracks[vortex_id].get(index, {}).get(value_name, math.nan)
            for index in frame_indices
        ])
        series.append({"label": label, "times": times, "values": values})
    return series


def rightmost_series(rows: list[dict], value_name: str, dataset_name: str) -> list[dict]:
    """Build one line using the valid vortex with the largest x center in each frame."""
    frame_times = {}
    rightmost = {}
    for row in rows:
        frame_index = row["frame_index"]
        frame_times.setdefault(frame_index, row["time"])
        x_center = row["x_center_positive"]
        if not math.isfinite(x_center):
            continue
        previous = rightmost.get(frame_index)
        if previous is None or x_center > previous["x_center_positive"]:
            rightmost[frame_index] = row

    frame_indices = sorted(frame_times)
    return [{
        "label": dataset_name,
        "times": np.asarray([frame_times[index] for index in frame_indices]),
        "values": np.asarray([
            rightmost.get(index, {}).get(value_name, math.nan)
            for index in frame_indices
        ]),
    }]


def configured_figure_size(config: dict) -> tuple[float, float]:
    return (
        float(config["plot"].get("figure_width", 10.0)),
        float(config["plot"].get("figure_height", 7.0)),
    )


def configured_time_limits(config: dict) -> tuple[float | None, float | None]:
    """Read optional simulation-time limits; NaN leaves an end automatic."""
    minimum = float(config["time_series"].get("time_axis_min", math.nan))
    maximum = float(config["time_series"].get("time_axis_max", math.nan))
    minimum = minimum if math.isfinite(minimum) else None
    maximum = maximum if math.isfinite(maximum) else None
    if minimum is not None and maximum is not None and minimum >= maximum:
        raise ValueError("[time_series] time_axis_min must be smaller than time_axis_max.")
    return minimum, maximum


def save_time_series_plot(
    output_path: str | Path,
    series: list[dict],
    ylabel: str,
    title: str,
    figure_size: tuple[float, float],
    reference: tuple[float, float, float] | None = None,
    reference_times=None,
    time_limits: tuple[float | None, float | None] = (None, None),
) -> None:
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=figure_size)
    # Values already contain NaNs at missed detections, so Matplotlib leaves gaps.
    for item in series:
        times = np.asarray(item["times"], dtype=float)
        values = np.asarray(item["values"], dtype=float)
        visible = np.ones(times.shape, dtype=bool)
        if time_limits[0] is not None:
            visible &= times >= time_limits[0]
        if time_limits[1] is not None:
            visible &= times <= time_limits[1]
        axis.plot(times[visible], values[visible], marker="o", label=item["label"])

    if reference is not None:
        if reference_times is None:
            reference_times = sorted({float(time) for item in series for time in item["times"]})
        times = np.asarray(reference_times, dtype=float)
        if time_limits[0] is not None:
            times = times[times >= time_limits[0]]
        if time_limits[1] is not None:
            times = times[times <= time_limits[1]]
        slope, anchor_time, anchor_displacement = reference
        # x_ref(t) = x_anchor + slope * (t - t_anchor)
        values = anchor_displacement + slope * (times - anchor_time)
        axis.plot(times, values, color="black", linestyle=":", label=f"reference slope {slope:g}")

    axis.set_xlabel("simulation time")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.set_xlim(left=time_limits[0], right=time_limits[1])
    axis.grid(True, alpha=0.3)
    handles, _ = axis.get_legend_handles_labels()
    if handles:
        axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
