"""Plot circulation and x displacement from several final metrics CSV files."""

import argparse
import math
from pathlib import Path
from typing import List, Union

from common import load_config, simulation_parameter
from toml_compat import tomllib
from time_series_plotting import (
    configured_figure_size,
    configured_time_limits,
    finite_setting,
    largest_radius_series,
    read_metrics,
    save_time_series_plot,
)


def load_datasets(path: Union[str, Path], config: dict) -> List[dict]:
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
        run_folder = str(entry.get("run_folder", "")).strip()
        forcing_end_time = entry.get("forcing_end_time")
        if forcing_end_time is None:
            if not run_folder:
                raise ValueError(
                    f"Dataset {name!r} needs forcing_end_time or run_folder in {path}."
                )
            forcing_end_time = simulation_parameter(run_folder, config, "b_f_tau")
        forcing_end_time = float(forcing_end_time)
        if not math.isfinite(forcing_end_time) or forcing_end_time <= 0.0:
            raise ValueError(
                f"Dataset {name!r} has an invalid forcing_end_time: {forcing_end_time}"
            )
        names.append(name)
        resolved.append({
            "name": name,
            "csv": csv_path,
            "run_folder": run_folder,
            "forcing_end_time": forcing_end_time,
        })
    if len(set(names)) != len(names):
        raise ValueError("Dataset names in the manifest must be unique.")
    resolved.sort(key=lambda dataset: (dataset["forcing_end_time"], dataset["name"]))
    return resolved


def configure_legend_labels(datasets: List[dict], config: dict, legend_by: str) -> List[dict]:
    """Attach parameter labels and put datasets in the matching numeric order."""
    if legend_by == "tau":
        for dataset in datasets:
            dataset["legend_label"] = rf"$\tau={dataset['forcing_end_time']:g}$"
        datasets.sort(key=lambda dataset: (dataset["forcing_end_time"], dataset["name"]))
        return datasets

    if legend_by == "reynolds":
        for dataset in datasets:
            if not dataset["run_folder"]:
                raise ValueError(
                    f"Dataset {dataset['name']!r} needs run_folder to label by Reynolds number."
                )
            reynolds_number = simulation_parameter(dataset["run_folder"], config, "Re")
            if not math.isfinite(reynolds_number) or reynolds_number <= 0.0:
                raise ValueError(
                    f"Dataset {dataset['name']!r} has an invalid Reynolds number: "
                    f"{reynolds_number}"
                )
            dataset["reynolds_number"] = reynolds_number
            dataset["legend_label"] = rf"$\mathrm{{Re}}={reynolds_number:g}$"
        datasets.sort(key=lambda dataset: (dataset["reynolds_number"], dataset["name"]))
        return datasets

    for dataset in datasets:
        if not dataset["run_folder"]:
            raise ValueError(
                f"Dataset {dataset['name']!r} needs run_folder to label by dx_base."
            )
        dx_base = simulation_parameter(dataset["run_folder"], config, "dx_base")
        if not math.isfinite(dx_base) or dx_base <= 0.0:
            raise ValueError(
                f"Dataset {dataset['name']!r} has an invalid dx_base: {dx_base}"
            )
        dataset["dx_base"] = dx_base
        dataset["legend_label"] = rf"$\Delta x_{{\mathrm{{base}}}}={dx_base:g}$"
    datasets.sort(key=lambda dataset: (-dataset["dx_base"], dataset["name"]))
    return datasets


def main() -> int:
    parser = argparse.ArgumentParser(description="Create combined plots from pipeline metrics CSV files.")
    parser.add_argument("datasets_file", type=Path, help="Editable datasets.toml written by run_all.py.")
    parser.add_argument("config_file", type=Path)
    parser.add_argument("--output-dir", type=Path, help="Defaults to the folder containing datasets.toml.")
    parser.add_argument(
        "--legend-by",
        choices=("tau", "dx-base", "reynolds"),
        default="tau",
        help="Legend parameter and ordering (default: tau).",
    )
    parser.add_argument(
        "--circulation-inset",
        action="store_true",
        help="Add a zoomed plateau inset to the simulation-time circulation plot.",
    )
    parser.add_argument("--inset-x-min", type=float, default=3.0)
    parser.add_argument("--inset-x-max", type=float, default=25.0)
    parser.add_argument("--inset-y-min", type=float, default=0.78)
    parser.add_argument("--inset-y-max", type=float, default=0.81)
    args = parser.parse_args()
    inset_values = (
        args.inset_x_min,
        args.inset_x_max,
        args.inset_y_min,
        args.inset_y_max,
    )
    if not all(math.isfinite(value) for value in inset_values):
        parser.error("circulation inset limits must be finite")
    if args.inset_x_min >= args.inset_x_max:
        parser.error("--inset-x-min must be smaller than --inset-x-max")
    if args.inset_y_min >= args.inset_y_max:
        parser.error("--inset-y-min must be smaller than --inset-y-max")

    config = load_config(args.config_file)
    datasets_file = args.datasets_file.expanduser().resolve()
    datasets = configure_legend_labels(
        load_datasets(datasets_file, config),
        config,
        args.legend_by,
    )
    circulation_series = []
    normalized_circulation_series = []
    displacement_series = []
    all_times = set()
    # Keep the same dataset order so both figures use matching colors.
    for dataset in datasets:
        rows = read_metrics(dataset["csv"])
        forcing_end_time = dataset["forcing_end_time"]
        label = dataset["legend_label"]
        circulation_item = largest_radius_series(rows, "circulation_positive", label)[0]
        normalized_circulation_series.append({
            **circulation_item,
            "times": circulation_item["times"] / forcing_end_time,
        })
        circulation_item["event_time"] = forcing_end_time
        circulation_series.append(circulation_item)
        displacement_item = largest_radius_series(rows, "x_displacement", label)[0]
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
    normalized_circulation_path = output_folder / "combined_circulation_vs_time_over_tau.png"
    displacement_path = output_folder / "combined_x_displacement_vs_time.png"
    save_time_series_plot(
        circulation_path,
        circulation_series,
        "positive circulation",
        "Positive-vortex circulation versus simulation time",
        figure_size,
        time_limits=time_limits,
        inset_limits=(
            args.inset_x_min,
            args.inset_x_max,
            args.inset_y_min,
            args.inset_y_max,
        ) if args.circulation_inset else None,
    )
    save_time_series_plot(
        normalized_circulation_path,
        normalized_circulation_series,
        "positive circulation",
        r"Positive-vortex circulation versus normalized time $t/\tau$",
        figure_size,
        time_limits=(0.0, 1.0),
        xlabel=r"normalized simulation time $t/\tau$",
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
    print(f"Saved {normalized_circulation_path}")
    print(f"Saved {displacement_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
