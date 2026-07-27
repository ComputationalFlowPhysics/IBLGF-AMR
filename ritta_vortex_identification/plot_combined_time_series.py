"""Plot circulation and x displacement from several final metrics CSV files."""

from __future__ import annotations

import argparse
import math
import tomllib
from pathlib import Path

from common import load_config, simulation_parameter
from time_series_plotting import (
    configured_figure_size,
    configured_time_limits,
    finite_setting,
    read_metrics,
    rightmost_series,
    save_time_series_plot,
)


def load_datasets(path: str | Path, config: dict) -> list[dict]:
    """Resolve editable legend names and CSV paths from datasets.toml."""
    path = Path(path).expanduser().resolve()
    with path.open("rb") as handle:
        manifest = tomllib.load(handle)
    datasets = manifest.get("dataset", [])
    if not datasets:
        raise ValueError(f"No [[dataset]] entries were found in {path}")

    names = []
    resolved = []
    for entry in datasets:
        name = str(entry.get("name", "")).strip()
        if not name:
            raise ValueError(f"Every dataset in {path} needs a nonempty name.")
        csv_path = Path(str(entry.get("csv", ""))).expanduser()
        if not csv_path.is_absolute():
            csv_path = path.parent / csv_path
        csv_path = csv_path.resolve()
        if not csv_path.is_file():
            raise FileNotFoundError(f"Metrics CSV does not exist: {csv_path}")
        forcing_end_time = entry.get("forcing_end_time")
        if forcing_end_time is None:
            run_folder = str(entry.get("run_folder", "")).strip()
            if not run_folder:
                raise ValueError(
                    f"Dataset {name!r} needs forcing_end_time or run_folder in {path}."
                )
            forcing_end_time = simulation_parameter(run_folder, config, "b_f_tau")
        forcing_end_time = float(forcing_end_time)
        if not math.isfinite(forcing_end_time) or forcing_end_time < 0.0:
            raise ValueError(
                f"Dataset {name!r} has an invalid forcing_end_time: {forcing_end_time}"
            )
        names.append(name)
        resolved.append({
            "name": name,
            "csv": csv_path,
            "forcing_end_time": forcing_end_time,
        })
    if len(set(names)) != len(names):
        raise ValueError("Dataset names in the manifest must be unique.")
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser(description="Create two combined plots from pipeline metrics CSV files.")
    parser.add_argument("datasets_file", type=Path, help="Editable datasets.toml written by run_all.py.")
    parser.add_argument("config_file", type=Path)
    parser.add_argument("--output-dir", type=Path, help="Defaults to the folder containing datasets.toml.")
    args = parser.parse_args()

    config = load_config(args.config_file)
    datasets_file = args.datasets_file.expanduser().resolve()
    datasets = load_datasets(datasets_file, config)
    circulation_series = []
    displacement_series = []
    all_times = set()
    # Keep the same dataset order so both figures use matching colors.
    for dataset in datasets:
        rows = read_metrics(dataset["csv"])
        forcing_end_time = dataset["forcing_end_time"]
        label = rf"$\tau={forcing_end_time:g}$"
        circulation_item = rightmost_series(rows, "circulation_positive", label)[0]
        circulation_item["event_time"] = forcing_end_time
        circulation_series.append(circulation_item)
        displacement_item = rightmost_series(rows, "x_displacement", label)[0]
        displacement_item["event_time"] = forcing_end_time
        displacement_series.append(displacement_item)
        all_times.update(row["time"] for row in rows)

    output_folder = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else datasets_file.parent
    )
    figure_size = configured_figure_size(config)
    time_limits = configured_time_limits(config)
    circulation_path = output_folder / "combined_circulation_vs_time.png"
    displacement_path = output_folder / "combined_x_displacement_vs_time.png"
    # These are the only two artifacts made by the combined plotting command.
    save_time_series_plot(
        circulation_path,
        circulation_series,
        "positive circulation",
        "Positive-vortex circulation versus simulation time",
        figure_size,
        time_limits=time_limits,
    )
    save_time_series_plot(
        displacement_path,
        displacement_series,
        "positive-vortex x displacement",
        "Positive-vortex x displacement versus simulation time",
        figure_size,
        reference=(
            finite_setting(config, "reference_slope"),
            finite_setting(config, "reference_anchor_time"),
            finite_setting(config, "reference_anchor_displacement"),
        ),
        reference_times=sorted(all_times),
        time_limits=time_limits,
    )
    print(f"Saved {circulation_path}")
    print(f"Saved {displacement_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
