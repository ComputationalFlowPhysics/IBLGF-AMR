#!/usr/bin/env python3
"""Combine 3D circulation and axial-center CSV files for a parameter sweep."""

import argparse
import csv
import math
from pathlib import Path


SCRIPT_FOLDER = Path(__file__).resolve().parent
CSV_NAME = "leading_vortex_circulation.csv"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot circulation and Lamb-center x coordinate from every 3D run "
            "directly below a sweep folder."
        )
    )
    parser.add_argument(
        "sweep_folder",
        type=Path,
        help="folder whose immediate children are 3D run folders",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "comparison output folder; defaults to "
            "ritta_plotting_3D/outputs/<sweep-name>_comparison"
        ),
    )
    parser.add_argument(
        "--legend-by",
        choices=("resolution", "tau"),
        default="resolution",
        help="legend parameter and curve ordering (default: resolution)",
    )
    return parser.parse_args()


def discover_run_folders(sweep_folder):
    sweep_folder = sweep_folder.expanduser().resolve()
    if not sweep_folder.is_dir():
        raise ValueError(f"Sweep folder does not exist: {sweep_folder}")

    run_folders = sorted(
        folder
        for folder in sweep_folder.iterdir()
        if folder.is_dir() and (folder / "output").is_dir()
    )
    if not run_folders:
        raise ValueError(
            f"No immediate child run folders with output/ were found in {sweep_folder}"
        )
    return sweep_folder, run_folders


def config_from_meta(run_folder):
    meta_path = run_folder / "meta.txt"
    if not meta_path.is_file():
        return None

    for line in meta_path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition(":")
        if not separator or key.strip() != "config":
            continue
        config_path = Path(value.strip())
        if not config_path.is_absolute():
            config_path = run_folder / config_path
        if config_path.is_file():
            return config_path.resolve()
    return None


def find_run_config(run_folder):
    exact_config = run_folder / f"{run_folder.name}.cfg"
    if exact_config.is_file():
        return exact_config.resolve()

    meta_config = config_from_meta(run_folder)
    if meta_config is not None:
        return meta_config

    candidates = {
        path.resolve()
        for path in run_folder.iterdir()
        if path.is_file() and (path.suffix == ".cfg" or path.name.startswith("config"))
    }
    if len(candidates) == 1:
        return candidates.pop()
    if not candidates:
        raise ValueError(f"No simulation config was found in {run_folder}")
    names = ", ".join(sorted(path.name for path in candidates))
    raise ValueError(f"Multiple simulation configs were found in {run_folder}: {names}")


def read_config_scalar(config_path, name, required=True):
    import re

    text = re.sub(
        r"//.*?$|#.*?$",
        "",
        config_path.read_text(encoding="utf-8"),
        flags=re.MULTILINE,
    )
    match = re.search(
        rf"\b{re.escape(name)}\s*=\s*"
        r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*;",
        text,
    )
    if match is None:
        if not required:
            return None
        raise ValueError(f"Could not read {name} from {config_path}")
    return float(match.group(1))


def analysis_csv_path(run_folder):
    return SCRIPT_FOLDER / "outputs" / f"{run_folder.name}_circulation" / CSV_NAME


def finite_float(row, column, csv_path):
    try:
        value = float(row[column])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid {column} value in {csv_path}: {row.get(column)!r}") from error
    return value


def read_dataset(run_folder):
    config_path = find_run_config(run_folder)
    dx_base = read_config_scalar(config_path, "dx_base")
    levels_value = read_config_scalar(config_path, "nLevels")
    forcing_end_time = read_config_scalar(config_path, "b_f_tau", required=False)
    levels = int(levels_value)
    if levels_value != levels or levels < 0:
        raise ValueError(f"nLevels must be a nonnegative integer in {config_path}")
    if dx_base <= 0.0:
        raise ValueError(f"dx_base must be positive in {config_path}")

    csv_path = analysis_csv_path(run_folder)
    if not csv_path.is_file():
        raise ValueError(
            f"Missing analysis CSV for {run_folder.name}: {csv_path}\n"
            "Run ritta_plotting_3D/run_circulation_analysis.sh for this case first."
        )

    rows = []
    vorticity_threshold_fractions = set()
    threshold_fractions = set()
    expected_snapshot_folder = (run_folder / "output").resolve()
    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        required = {"time", "circulation", "center_x"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"{csv_path} is missing columns: {', '.join(sorted(missing))}"
            )
        for row in reader:
            snapshot_file = row.get("snapshot_file", "").strip()
            if snapshot_file:
                snapshot_path = Path(snapshot_file).expanduser().resolve()
                try:
                    snapshot_path.relative_to(expected_snapshot_folder)
                except ValueError as error:
                    raise ValueError(
                        f"{csv_path} belongs to a different run: {snapshot_path}"
                    ) from error
            rows.append(
                {
                    "time": finite_float(row, "time", csv_path),
                    "circulation": finite_float(row, "circulation", csv_path),
                    "center_x": finite_float(row, "center_x", csv_path),
                }
            )
            fraction = row.get("center_threshold_fraction", "").strip()
            if fraction:
                threshold_fractions.add(float(fraction))
            vorticity_fraction = row.get(
                "vorticity_threshold_fraction", ""
            ).strip()
            vorticity_threshold_fractions.add(
                float(vorticity_fraction) if vorticity_fraction else 0.02
            )

    if not rows:
        raise ValueError(f"Analysis CSV has no data rows: {csv_path}")
    if len(threshold_fractions) > 1:
        raise ValueError(f"Center threshold changes within {csv_path}")
    if len(vorticity_threshold_fractions) > 1:
        raise ValueError(f"Vorticity threshold changes within {csv_path}")

    rows.sort(key=lambda row: row["time"])
    return {
        "name": run_folder.name,
        "run_folder": run_folder,
        "config_path": config_path,
        "csv_path": csv_path,
        "dx_base": dx_base,
        "levels": levels,
        "dx_finest": dx_base / (2**levels),
        "forcing_end_time": forcing_end_time,
        "vorticity_threshold_fraction": next(
            iter(vorticity_threshold_fractions)
        ),
        "center_threshold_fraction": (
            next(iter(threshold_fractions)) if threshold_fractions else None
        ),
        "rows": rows,
    }


def dataset_sort_key(dataset):
    name = dataset["name"]
    if name.startswith("amr_"):
        return 0, dataset["levels"], name
    if name.startswith("res_"):
        return 1, -dataset["dx_base"], name
    return 2, 0, name


def dataset_label(dataset):
    return (
        f"{dataset['name']} "
        rf"($\Delta x_{{base}}={dataset['dx_base']:g}$, "
        rf"$L={dataset['levels']}$, "
        rf"$\Delta x_{{min}}={dataset['dx_finest']:g}$)"
    )


def prepare_datasets(datasets, legend_by):
    if legend_by == "resolution":
        datasets.sort(key=dataset_sort_key)
        for dataset in datasets:
            dataset["legend_label"] = dataset_label(dataset)
        return "3D resolution comparison"

    for dataset in datasets:
        forcing_end_time = dataset["forcing_end_time"]
        if (
            forcing_end_time is None
            or not math.isfinite(forcing_end_time)
            or forcing_end_time <= 0.0
        ):
            raise ValueError(
                f"Could not read a positive b_f_tau from {dataset['config_path']}"
            )
        dataset["legend_label"] = rf"$\tau={forcing_end_time:g}$"
    datasets.sort(key=lambda dataset: (dataset["forcing_end_time"], dataset["name"]))
    return "3D formation-time comparison"


def validate_center_thresholds(datasets):
    known = {
        round(dataset["center_threshold_fraction"], 15)
        for dataset in datasets
        if dataset["center_threshold_fraction"] is not None
    }
    if len(known) > 1:
        details = ", ".join(
            f"{dataset['name']}={dataset['center_threshold_fraction']}"
            for dataset in datasets
        )
        raise ValueError(
            "The center-x CSVs use different center threshold fractions: " + details
        )


def validate_vorticity_thresholds(datasets):
    known = {
        round(dataset["vorticity_threshold_fraction"], 15)
        for dataset in datasets
    }
    if len(known) > 1:
        details = ", ".join(
            f"{dataset['name']}={dataset['vorticity_threshold_fraction']}"
            for dataset in datasets
        )
        raise ValueError(
            "The circulation CSVs use different vorticity threshold fractions: "
            + details
        )


def save_plot(datasets, column, path, title, y_label):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError(
            "Matplotlib is required. Source "
            "ritta_vortex_identification/open_viewer_venv.sh first."
        ) from error

    figure, axis = plt.subplots(figsize=(11, 7))
    plotted = 0
    for dataset in datasets:
        points = [
            (row["time"], row[column])
            for row in dataset["rows"]
            if math.isfinite(row["time"]) and math.isfinite(row[column])
        ]
        if not points:
            continue
        times, values = zip(*points)
        axis.plot(times, values, linewidth=2.2, label=dataset["legend_label"])
        plotted += 1

    if plotted == 0:
        plt.close(figure)
        raise ValueError(f"No finite {column} values were available to plot")

    axis.set_title(title)
    axis.set_xlabel("Simulation time")
    axis.set_ylabel(y_label)
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize=9)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Failed to create plot: {path}")


def main():
    args = parse_args()
    try:
        sweep_folder, run_folders = discover_run_folders(args.sweep_folder)
        datasets = [read_dataset(run_folder) for run_folder in run_folders]
        validate_center_thresholds(datasets)
        validate_vorticity_thresholds(datasets)
        title_prefix = prepare_datasets(datasets, args.legend_by)

        output_folder = (
            args.output_dir.expanduser().resolve()
            if args.output_dir is not None
            else SCRIPT_FOLDER / "outputs" / f"{sweep_folder.name}_comparison"
        )
        output_folder.mkdir(parents=True, exist_ok=True)
        circulation_path = output_folder / "combined_circulation_vs_time.png"
        center_x_path = output_folder / "combined_center_x_vs_time.png"

        save_plot(
            datasets,
            "circulation",
            circulation_path,
            f"{title_prefix}: leading-vortex circulation",
            "Circulation",
        )
        save_plot(
            datasets,
            "center_x",
            center_x_path,
            f"{title_prefix}: leading-vortex axial center",
            "Lamb center x-coordinate",
        )

        for dataset in datasets:
            print(
                f"{dataset['name']}: dx_base={dataset['dx_base']:g}, "
                f"nLevels={dataset['levels']}, "
                f"dx_finest={dataset['dx_finest']:g}\n"
                f"  {dataset['csv_path']}"
            )
        print(f"Circulation comparison: {circulation_path}")
        print(f"Center-x comparison:    {center_x_path}")
        return 0
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
