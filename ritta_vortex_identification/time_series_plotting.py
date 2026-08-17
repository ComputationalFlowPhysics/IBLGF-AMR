"""Shared CSV loading and headless time-series plotting."""

import csv
import math
from pathlib import Path
from typing import List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Put the ten darker Tableau colors first, followed by their lighter partners.
# This avoids Matplotlib's default ten-color repetition in larger tau sweeps.
DISTINCT_LINE_COLORS = tuple(
    plt.get_cmap("tab20").colors[index]
    for index in (0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19)
)


def distinct_line_colors(count: int) -> list:
    """Return a nonrepeating categorical color sequence."""
    if count <= len(DISTINCT_LINE_COLORS):
        return list(DISTINCT_LINE_COLORS[:count])
    return list(plt.get_cmap("turbo")(np.linspace(0.0, 1.0, count)))


def finite_setting(config: dict, name: str) -> float:
    value = float(config["time_series"].get(name, math.nan))
    if not math.isfinite(value):
        raise ValueError(f"[time_series] {name} must be finite.")
    return value


def read_metrics(path: Union[str, Path]) -> List[dict]:
    """Load only the columns needed by the time-series plots."""
    path = Path(path).expanduser().resolve()
    rows = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if "boundary_radius" not in (reader.fieldnames or []):
            raise ValueError(
                f"Metrics file is missing boundary_radius: {path}. "
                "Rerun 04_positive_vortex_metrics.py with the existing fits.h5."
            )
        for raw in reader:
            rows.append({
                "frame_index": int(raw["frame_index"]),
                "time": float(raw["time"]),
                "vortex_id": int(raw["vortex_id"]) if raw["vortex_id"] else None,
                "boundary_radius": float(raw["boundary_radius"]),
                "circulation_positive": float(raw["circulation_positive"]),
                "x_center_positive": float(raw["x_center_positive"]),
                "x_displacement": float(raw["x_displacement"]),
            })
    if not rows:
        raise ValueError(f"Metrics file contains no frame rows: {path}")
    return rows


def track_series(
    rows: List[dict],
    value_name: str,
    dataset_name: Optional[str] = None,
) -> List[dict]:
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


def largest_radius_series(
    rows: List[dict], value_name: str, dataset_name: str
) -> List[dict]:
    """Build one line using the tracked fit with the largest boundary radius per frame."""
    frame_times = {}
    selected = {}
    for row in rows:
        frame_index = row["frame_index"]
        frame_times.setdefault(frame_index, row["time"])
        if row["vortex_id"] is None:
            continue
        radius = row["boundary_radius"]
        if not math.isfinite(radius):
            continue
        previous = selected.get(frame_index)
        if previous is None or radius > previous["boundary_radius"]:
            selected[frame_index] = row

    frame_indices = sorted(frame_times)
    return [{
        "label": dataset_name,
        "times": np.asarray([frame_times[index] for index in frame_indices]),
        "values": np.asarray([
            selected.get(index, {}).get(value_name, math.nan)
            for index in frame_indices
        ]),
    }]


def configured_figure_size(config: dict) -> Tuple[float, float]:
    return (
        float(config["plot"].get("figure_width", 10.0)),
        float(config["plot"].get("figure_height", 7.0)),
    )


def configured_time_limits(config: dict) -> Tuple[Optional[float], Optional[float]]:
    """Read optional simulation-time limits; NaN leaves an end automatic."""
    minimum = float(config["time_series"].get("time_axis_min", math.nan))
    maximum = float(config["time_series"].get("time_axis_max", math.nan))
    minimum = minimum if math.isfinite(minimum) else None
    maximum = maximum if math.isfinite(maximum) else None
    if minimum is not None and maximum is not None and minimum >= maximum:
        raise ValueError("[time_series] time_axis_min must be smaller than time_axis_max.")
    return minimum, maximum


def line_value_at_time(
    times: np.ndarray, values: np.ndarray, target_time: float
) -> Optional[float]:
    """Interpolate within one valid line segment without crossing a NaN gap."""
    order = np.argsort(times)
    times = times[order]
    values = values[order]
    exact = np.flatnonzero(np.isclose(times, target_time, rtol=1.0e-12, atol=1.0e-12))
    for index in exact:
        if math.isfinite(values[index]):
            return float(values[index])

    right = int(np.searchsorted(times, target_time, side="right"))
    if right == 0 or right == len(times):
        return None
    left = right - 1
    if not math.isfinite(values[left]) or not math.isfinite(values[right]):
        return None
    time_span = times[right] - times[left]
    if time_span <= 0.0:
        return None
    fraction = (target_time - times[left]) / time_span
    return float(values[left] + fraction * (values[right] - values[left]))


def save_time_series_plot(
    output_path: Union[str, Path],
    series: List[dict],
    ylabel: str,
    title: str,
    figure_size: Tuple[float, float],
    reference: Optional[Tuple[float, float, float]] = None,
    reference_times=None,
    time_limits: Tuple[Optional[float], Optional[float]] = (None, None),
    value_limits: Tuple[Optional[float], Optional[float]] = (None, None),
    xlabel: str = "simulation time",
    inset_limits: Optional[
        Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]
    ] = None,
) -> None:
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=figure_size)
    axis.set_prop_cycle(color=distinct_line_colors(len(series)))
    has_event_marker = False
    has_breakpoint_marker = False
    # Values already contain NaNs at missed detections, so Matplotlib leaves gaps.
    for item in series:
        times = np.asarray(item["times"], dtype=float)
        values = np.asarray(item["values"], dtype=float)
        visible = np.ones(times.shape, dtype=bool)
        if time_limits[0] is not None:
            visible &= times >= time_limits[0]
        if time_limits[1] is not None:
            visible &= times <= time_limits[1]
        line, = axis.plot(times[visible], values[visible], marker="o", label=item["label"])
        event_time = item.get("event_time")
        if event_time is not None:
            event_time = float(event_time)
            event_is_visible = (
                math.isfinite(event_time)
                and (time_limits[0] is None or event_time >= time_limits[0])
                and (time_limits[1] is None or event_time <= time_limits[1])
            )
            if event_is_visible:
                event_value = line_value_at_time(times, values, event_time)
                if event_value is not None:
                    axis.scatter(
                        event_time,
                        event_value,
                        color=line.get_color(),
                        edgecolors="black",
                        marker="X",
                        s=110,
                        linewidths=0.8,
                        zorder=line.get_zorder() + 2,
                        label="_nolegend_",
                    )
                    has_event_marker = True

        breakpoint_time = item.get("breakpoint_time")
        breakpoint_value = item.get("breakpoint_value")
        if breakpoint_time is not None and breakpoint_value is not None:
            breakpoint_time = float(breakpoint_time)
            breakpoint_value = float(breakpoint_value)
            breakpoint_is_visible = (
                math.isfinite(breakpoint_time)
                and math.isfinite(breakpoint_value)
                and (time_limits[0] is None or breakpoint_time >= time_limits[0])
                and (time_limits[1] is None or breakpoint_time <= time_limits[1])
            )
            if breakpoint_is_visible:
                axis.scatter(
                    breakpoint_time,
                    breakpoint_value,
                    color=line.get_color(),
                    edgecolors="black",
                    marker="D",
                    s=75,
                    linewidths=0.8,
                    zorder=line.get_zorder() + 3,
                    label="_nolegend_",
                )
                has_breakpoint_marker = True

    if has_event_marker:
        axis.scatter(
            [],
            [],
            color="black",
            marker="X",
            s=110,
            label="forcing ends",
        )

    if has_breakpoint_marker:
        axis.scatter(
            [],
            [],
            color="black",
            marker="D",
            s=75,
            label=r"fitted breakpoint $t_b$",
        )

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

    if inset_limits is not None:
        inset_axis = axis.inset_axes([0.47, 0.12, 0.50, 0.42])
        inset_axis.set_prop_cycle(color=distinct_line_colors(len(series)))
        inset_x_min, inset_x_max, inset_y_min, inset_y_max = inset_limits
        for item in series:
            times = np.asarray(item["times"], dtype=float)
            values = np.asarray(item["values"], dtype=float)
            visible = np.ones(times.shape, dtype=bool)
            if inset_x_min is not None:
                visible &= times >= inset_x_min
            if inset_x_max is not None:
                visible &= times <= inset_x_max
            inset_axis.plot(
                times[visible],
                values[visible],
                marker="o",
                markersize=3,
                linewidth=1.0,
            )
        inset_axis.set_xlim(left=inset_x_min, right=inset_x_max)
        inset_axis.set_ylim(bottom=inset_y_min, top=inset_y_max)
        inset_axis.set_title("plateau zoom", fontsize=9)
        inset_axis.grid(True, alpha=0.3)
        inset_axis.tick_params(labelsize=8)
        axis.indicate_inset_zoom(inset_axis, edgecolor="0.35")

    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.set_xlim(left=time_limits[0], right=time_limits[1])
    axis.set_ylim(bottom=value_limits[0], top=value_limits[1])
    axis.grid(True, alpha=0.3)
    handles, _ = axis.get_legend_handles_labels()
    if handles:
        axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
