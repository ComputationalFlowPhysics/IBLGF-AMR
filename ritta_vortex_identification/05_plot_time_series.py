"""Stage 5: plot circulation and x displacement of the rightmost vortex."""

from __future__ import annotations

import argparse
from pathlib import Path

from common import load_config, result_folder, stage_command
from time_series_plotting import (
    configured_figure_size,
    configured_time_limits,
    finite_setting,
    read_metrics,
    rightmost_series,
    save_time_series_plot,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot rightmost-vortex circulation and x displacement over time.")
    parser.add_argument("run_folder", type=Path)
    parser.add_argument("config_file", type=Path)
    args = parser.parse_args()

    config = load_config(args.config_file)
    output_folder = result_folder(args.run_folder)
    metrics_path = output_folder / "positive_vortex_metrics.csv"
    if not metrics_path.is_file():
        print("positive_vortex_metrics.csv does not exist. Run this exact command first:")
        print(stage_command("04_positive_vortex_metrics.py", args.run_folder, args.config_file))
        return 1

    slope = finite_setting(config, "reference_slope")
    anchor_time = finite_setting(config, "reference_anchor_time")
    anchor_displacement = finite_setting(config, "reference_anchor_displacement")
    # Stage 5 only reads the saved Stage 4 table; it performs no vortex calculations.
    rows = read_metrics(metrics_path)
    figure_size = configured_figure_size(config)
    time_limits = configured_time_limits(config)

    circulation_path = output_folder / "circulation_vs_time.png"
    displacement_path = output_folder / "x_displacement_vs_time.png"
    # The selected track may change between frames; missing detections remain gaps.
    save_time_series_plot(
        circulation_path,
        rightmost_series(rows, "circulation_positive", "rightmost vortex"),
        "positive circulation",
        "Rightmost positive-vortex circulation versus simulation time",
        figure_size,
        time_limits=time_limits,
    )
    save_time_series_plot(
        displacement_path,
        rightmost_series(rows, "x_displacement", "rightmost vortex"),
        "positive-vortex x displacement",
        "Rightmost positive-vortex x displacement versus simulation time",
        figure_size,
        reference=(slope, anchor_time, anchor_displacement),
        time_limits=time_limits,
    )
    print(f"Saved {circulation_path}")
    print(f"Saved {displacement_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
