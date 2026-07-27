"""Fit and plot circulation slope-change times for several pipeline datasets."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from common import load_config
from plot_combined_time_series import load_datasets
from time_series_plotting import (
    configured_figure_size,
    configured_time_limits,
    line_value_at_time,
    read_metrics,
    rightmost_series,
    save_time_series_plot,
)


def fit_circulation_breakpoint(
    times,
    circulations,
    initial_break_time: float,
    tolerance: float = 1.0e-8,
    max_iterations: int = 100,
) -> dict:
    """Fit Gamma_0 + m_1 t + delta_m max(0, t - t_b) by iteration."""
    times = np.asarray(times, dtype=float)
    circulations = np.asarray(circulations, dtype=float)
    if times.shape != circulations.shape:
        raise ValueError("Times and circulations must have the same shape.")

    finite = np.isfinite(times) & np.isfinite(circulations)
    times = times[finite]
    circulations = circulations[finite]
    if len(times) < 4:
        raise ValueError("At least four finite circulation observations are required.")
    if not math.isfinite(initial_break_time):
        raise ValueError("The initial breakpoint time must be finite.")
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("The convergence tolerance must be finite and greater than zero.")
    if max_iterations < 1:
        raise ValueError("The maximum iteration count must be at least one.")

    order = np.argsort(times)
    times = times[order]
    circulations = circulations[order]
    break_time = float(initial_break_time)

    for iteration in range(1, max_iterations + 1):
        positive_part = np.maximum(0.0, times - break_time)
        break_derivative = np.where(times <= break_time, 0.0, -1.0)
        design = np.column_stack((
            np.ones_like(times),
            times,
            positive_part,
            break_derivative,
        ))
        parameters, _, rank, _ = np.linalg.lstsq(design, circulations, rcond=None)
        if rank < design.shape[1]:
            raise ValueError(
                f"Breakpoint fit is underdetermined at t_b={break_time:g}; "
                "the data need at least two times on each side."
            )

        slope_change = float(parameters[2])
        correction = float(parameters[3])
        minimum_slope_change = 1.0e-12 * max(1.0, abs(float(parameters[1])))
        if abs(slope_change) <= minimum_slope_change:
            raise ValueError(
                f"Cannot update t_b at {break_time:g} because the fitted slope change is zero."
            )

        next_break_time = break_time + correction / slope_change
        if not math.isfinite(next_break_time):
            raise ValueError("The breakpoint iteration produced a non-finite time.")
        if next_break_time <= times[0] or next_break_time >= times[-1]:
            raise ValueError(
                f"The breakpoint iteration left the data range: t_b={next_break_time:g}."
            )

        converged = abs(next_break_time - break_time) < tolerance
        break_time = next_break_time
        if converged:
            break
    else:
        raise RuntimeError(
            f"Breakpoint fit did not converge within {max_iterations} iterations."
        )

    # Refit at the converged time so the reported slopes and Gamma(t_b) all use
    # the same breakpoint.
    positive_part = np.maximum(0.0, times - break_time)
    break_derivative = np.where(times <= break_time, 0.0, -1.0)
    design = np.column_stack((
        np.ones_like(times),
        times,
        positive_part,
        break_derivative,
    ))
    parameters, _, rank, _ = np.linalg.lstsq(design, circulations, rcond=None)
    if rank < design.shape[1]:
        raise ValueError(f"Final breakpoint fit is underdetermined at t_b={break_time:g}.")

    intercept, first_slope, slope_change, _ = map(float, parameters)
    return {
        "break_time": break_time,
        "break_circulation": intercept + first_slope * break_time,
        "intercept": intercept,
        "first_slope": first_slope,
        "second_slope": first_slope + slope_change,
        "iterations": iteration,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fit circulation breakpoints and overlay them on one combined plot."
    )
    parser.add_argument(
        "datasets_file",
        type=Path,
        help="datasets.toml written by run_all.py; each entry identifies a metrics CSV.",
    )
    parser.add_argument("config_file", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Defaults to the folder containing datasets.toml.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1.0e-8,
        help="Absolute convergence tolerance for successive t_b values (default: 1e-8).",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=100,
        help="Maximum breakpoint iterations per dataset (default: 100).",
    )
    args = parser.parse_args()

    config = load_config(args.config_file)
    datasets_file = args.datasets_file.expanduser().resolve()
    datasets = load_datasets(datasets_file, config)
    circulation_series = []

    for dataset in datasets:
        rows = read_metrics(dataset["csv"])
        forcing_end_time = dataset["forcing_end_time"]
        label = rf"$\tau={forcing_end_time:g}$"
        item = rightmost_series(rows, "circulation_positive", label)[0]
        fit = fit_circulation_breakpoint(
            item["times"],
            item["values"],
            initial_break_time=forcing_end_time,
            tolerance=args.tolerance,
            max_iterations=args.max_iterations,
        )
        breakpoint_value = line_value_at_time(
            item["times"],
            item["values"],
            fit["break_time"],
        )
        if breakpoint_value is None:
            raise ValueError(
                f"Cannot place the t_b marker for {dataset['name']} because "
                "the fitted time lies outside a valid plotted line segment."
            )
        item["event_time"] = forcing_end_time
        item["breakpoint_time"] = fit["break_time"]
        item["breakpoint_value"] = breakpoint_value
        circulation_series.append(item)
        print(
            f"{dataset['name']}: t_b={fit['break_time']:.10g}, "
            f"Gamma_curve(t_b)={breakpoint_value:.10g}, "
            f"slopes={fit['first_slope']:.10g} -> {fit['second_slope']:.10g}, "
            f"iterations={fit['iterations']}"
        )

    output_folder = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else datasets_file.parent
    )
    output_path = output_folder / "combined_circulation_with_breakpoints.png"
    save_time_series_plot(
        output_path,
        circulation_series,
        "positive circulation",
        "Positive-vortex circulation with fitted slope-change times",
        configured_figure_size(config),
        time_limits=configured_time_limits(config),
    )
    print(f"Saved {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
