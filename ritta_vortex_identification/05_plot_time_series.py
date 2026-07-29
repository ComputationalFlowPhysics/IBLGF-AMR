"""Stage 5: plot circulation and x displacement of the largest fitted vortex."""

from __future__ import annotations

import argparse
from pathlib import Path

from common import load_config, result_folder, stage_command
from time_series_plotting import (
    configured_figure_size,
    configured_time_limits,
    finite_setting,
    largest_radius_series,
    read_metrics,
    save_time_series_plot,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot circulation and x displacement of the largest-radius fitted vortex over time."
    )
    parser.add_argument("run_folder", type=Path)
    parser.add_argument("config_file", type=Path)
    parser.add_argument(
        "--metrics-file",
        type=Path,
        help="Stage 4 CSV to plot (defaults to positive_vortex_metrics.csv in the run's results directory).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for plot PNGs (defaults to the run's results directory).",
    )
    parser.add_argument(
        "--filename-prefix",
        default="",
        help="Prefix added to plot filenames, for example tau_1p0_boundary_fraction_0p02.",
    )
    parser.add_argument("--title-label", help="Optional label appended to plot titles.")
    parser.add_argument(
        "--circulation-only",
        action="store_true",
        help="Create only the circulation plot, without the x-displacement plot.",
    )
    args = parser.parse_args()

    config = load_config(args.config_file)
    default_output_folder = result_folder(args.run_folder)
    output_folder = args.output_dir.expanduser().resolve() if args.output_dir else default_output_folder
    metrics_path = (
        args.metrics_file.expanduser().resolve()
        if args.metrics_file
        else default_output_folder / "positive_vortex_metrics.csv"
    )
    if not metrics_path.is_file():
        print(f"{metrics_path} does not exist.")
        if args.metrics_file is None:
            print("Run this exact command first:")
            print(stage_command("04_positive_vortex_metrics.py", args.run_folder, args.config_file))
        return 1
    if args.filename_prefix and Path(args.filename_prefix).name != args.filename_prefix:
        parser.error("--filename-prefix must be a filename prefix, not a path")
    output_folder.mkdir(parents=True, exist_ok=True)

    slope = finite_setting(config, "reference_slope")
    anchor_time = finite_setting(config, "reference_anchor_time")
    anchor_displacement = finite_setting(config, "reference_anchor_displacement")
    # Stage 5 only reads the saved Stage 4 table; it performs no vortex calculations.
    rows = read_metrics(metrics_path)
    figure_size = configured_figure_size(config)
    time_limits = configured_time_limits(config)

    prefix = f"{args.filename_prefix}_" if args.filename_prefix else ""
    title_suffix = f" | {args.title_label}" if args.title_label else ""
    circulation_path = output_folder / f"{prefix}circulation_vs_time.png"
    displacement_path = output_folder / f"{prefix}x_displacement_vs_time.png"
    # The selected track may change between frames; missing detections remain gaps.
    save_time_series_plot(
        circulation_path,
        largest_radius_series(rows, "circulation_positive", "largest-radius vortex"),
        "positive circulation",
        f"Largest-radius positive-vortex circulation versus simulation time{title_suffix}",
        figure_size,
        time_limits=time_limits,
    )
    print(f"Saved {circulation_path}")
    if args.circulation_only:
        return 0
    save_time_series_plot(
        displacement_path,
        largest_radius_series(rows, "x_displacement", "largest-radius vortex"),
        "positive-vortex x displacement",
        f"Largest-radius positive-vortex x displacement versus simulation time{title_suffix}",
        figure_size,
        reference=(slope, anchor_time, anchor_displacement),
        time_limits=time_limits,
    )
    print(f"Saved {displacement_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
