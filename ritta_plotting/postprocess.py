#!/usr/bin/env python3
"""
Simple CLI post-processing plots using the same backend processing logic as the viewer.

Examples:
    python3 ritta_plotting/postprocess.py /path/to/run/output single
    python3 ritta_plotting/postprocess.py /path/to/batch_parent batch
"""

import argparse
import importlib
import os
import re
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# User settings: edit these to match the viewer post-processing settings.
# ---------------------------------------------------------------------------

BOUNDARY_MODE = "vorticity_threshold"  # vorticity_threshold, computed_q_positive, elliptical_gaussian
THRESHOLD_FACTOR = 0.30
MIN_REGION_AREA = 0.25
LARGEST_REGION_ONLY = False
INITIAL_FIT_FRACTION = 0.30
BOUNDARY_FRACTION = 0.05
VIEW_LIMITS = None
PLOT_FIGSIZE = (9.2, 6.4)
LINE_WIDTH = 2.2
MARKER_SIZE = 5.5
SAVE_DPI = 220
AUTO_EXPORT_PNG = False
PNG_PREFIX = "postprocess"


AGGREGATE_TERMS = [
    ("circulation", "Circulation"),
    ("center_x", "Vorticity centroid x"),
    ("center_y", "Vorticity centroid y"),
    ("vorticity_centroid_x", "Vorticity centroid x"),
    ("vorticity_centroid_y", "Vorticity centroid y"),
    ("fitted_center_x", "Fitted center x"),
    ("fitted_center_y", "Fitted center y"),
    ("area", "Area"),
    ("count", "Count"),
    ("amplitude", "Fit amplitude"),
    ("sigma_1", "Fit sigma 1"),
    ("sigma_2", "Fit sigma 2"),
    ("theta", "Fit orientation"),
    ("omega_peak", "Region vorticity peak"),
    ("threshold", "Fitted boundary level"),
]

REGION_TERMS = [
    ("circulation", "Region circulation"),
    ("center_x", "Vorticity centroid x"),
    ("center_y", "Vorticity centroid y"),
    ("vorticity_centroid_x", "Vorticity centroid x"),
    ("vorticity_centroid_y", "Vorticity centroid y"),
    ("fitted_center_x", "Fitted center x"),
    ("fitted_center_y", "Fitted center y"),
    ("area", "Region area"),
    ("amplitude", "Fit amplitude"),
    ("sigma_1", "Fit sigma 1"),
    ("sigma_2", "Fit sigma 2"),
    ("theta", "Fit orientation"),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Render post-process line plots without the browser.")
    parser.add_argument("path", type=Path, help="Run/output folder for single mode, or parent folder for batch mode.")
    parser.add_argument("mode", choices=["single", "batch"], help="Whether to process one run or a batch of runs.")
    return parser.parse_args()


def repo_root():
    return Path(__file__).resolve().parents[1]


def outputs_dir():
    path = Path(__file__).resolve().parent / "outputs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def find_viewer_root():
    candidates = []
    env_root = os.environ.get("IBLGF_VIEWER_ROOT", "").strip()
    if env_root:
        candidates.append(Path(env_root).expanduser())

    root = repo_root()
    candidates.extend([
        root.parent / "iblgf-viewer",
        root.parent / "iblgf-viewer" / "IBLGF-viewer",
        root.parent / "IBLGF-viewer",
        root / "iblgf-viewer",
        root / "iblgf-viewer" / "IBLGF-viewer",
    ])

    for candidate in candidates:
        if (candidate / "viewer_plotting.py").is_file():
            return candidate
        nested = candidate / "IBLGF-viewer"
        if (nested / "viewer_plotting.py").is_file():
            return nested

    raise FileNotFoundError(
        "Could not find viewer_plotting.py. Set IBLGF_VIEWER_ROOT to the folder that contains it."
    )


def load_viewer_plotting():
    viewer_root = find_viewer_root()
    if str(viewer_root) not in sys.path:
        sys.path.insert(0, str(viewer_root))
    return importlib.import_module("viewer_plotting")


def resolve_run_dir(path):
    path = path.expanduser().resolve()
    if path.is_file():
        path = path.parent
    if path.name == "output":
        return path.parent
    return path


def snapshot_timestep(snapshot_name):
    match = re.fullmatch(r"flowTime_(\d+)\.(?:hdf5|h5)", Path(snapshot_name).name)
    return int(match.group(1)) if match else None


def snapshot_rows(summary):
    names = list(summary.get("time_series_snapshot_files") or summary.get("snapshot_files") or [])
    scale = float(summary.get("simulation_time_scale") or 1.0)
    rows = []
    for index, name in enumerate(names):
        timestep = snapshot_timestep(name)
        time_value = (timestep if timestep is not None else index) * scale
        rows.append({
            "index": index,
            "snapshot_name": name,
            "timestep": timestep,
            "time_value": time_value,
        })
    return rows


def finite_number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def chart_color(sign):
    if sign == "positive":
        return "#d1495b"
    if sign == "negative":
        return "#0077b6"
    return "#0a9396"


def run_series_color(index):
    colors = [
        "#d1495b",
        "#0077b6",
        "#0a9396",
        "#f4a261",
        "#6d597a",
        "#2a9d8f",
        "#e76f51",
        "#4d908e",
        "#577590",
        "#bc4749",
    ]
    return colors[index % len(colors)]


def can_show_figures():
    return not (sys.platform.startswith("linux") and not os.environ.get("DISPLAY"))


def directory_has_snapshot_files(directory):
    try:
        with os.scandir(directory) as entries:
            for entry in entries:
                if entry.is_file() and entry.name.lower().endswith((".hdf5", ".h5")):
                    return True
    except OSError:
        return False
    return False


def discover_batch_runs(upload_root):
    upload_root = upload_root.expanduser().resolve()
    candidates = {}
    for directory, dirnames, _ in os.walk(upload_root):
        path = Path(directory)
        if directory_has_snapshot_files(path):
            run_dir = path.parent if path.name == "output" else path
            try:
                label = path.parent.relative_to(upload_root).as_posix() if path.name == "output" else path.relative_to(upload_root).as_posix()
            except ValueError:
                label = run_dir.name
            candidates[run_dir.resolve()] = {"label": label or run_dir.name, "run_dir": run_dir}
            dirnames[:] = []
            continue

        output_dir = path / "output"
        if output_dir.is_dir() and directory_has_snapshot_files(output_dir):
            run_dir = path
            try:
                label = run_dir.relative_to(upload_root).as_posix()
            except ValueError:
                label = run_dir.name
            candidates[run_dir.resolve()] = {"label": label or run_dir.name, "run_dir": run_dir}
            try:
                dirnames.remove("output")
            except ValueError:
                pass

    ordered = [candidates[key] for key in sorted(candidates, key=lambda item: str(candidates[item]["label"]))]
    for index, record in enumerate(ordered):
        record["run_key"] = str(index)
    return ordered


def process_single_run(vp, run_dir):
    summary = vp.inspect_run_folder(run_dir)
    rows = snapshot_rows(summary)
    results = []
    previous_parameters = None

    for row in rows:
        print("Processing {} ...".format(row["snapshot_name"]))
        if BOUNDARY_MODE == "elliptical_gaussian":
            result = vp.fit_snapshot_vortex_core(
                run_dir,
                row["snapshot_name"],
                initial_fit_fraction=INITIAL_FIT_FRACTION,
                boundary_fraction=BOUNDARY_FRACTION,
                min_region_area=MIN_REGION_AREA,
                previous_parameters=previous_parameters,
                coordinate_dx_base=summary.get("coordinate_dx_base"),
                coordinate_origin_index=summary.get("coordinate_origin_index"),
            )
            if isinstance(result.get("parameters"), dict):
                previous_parameters = result.get("parameters")
        else:
            result = vp.postprocess_vorticity_threshold(
                run_dir,
                row["snapshot_name"],
                threshold_factor=THRESHOLD_FACTOR,
                min_region_area=MIN_REGION_AREA,
                largest_region_only=LARGEST_REGION_ONLY,
                boundary_mode=BOUNDARY_MODE,
                selected_levels=list(summary.get("default_checked_levels") or []),
                coordinate_dx_base=summary.get("coordinate_dx_base"),
                coordinate_origin_index=summary.get("coordinate_origin_index"),
                include_preview=False,
                view_limits=VIEW_LIMITS,
            )
        results.append(result)

    return summary, rows, results


def process_batch(vp, batch_root):
    batch_runs = []
    for run_record in discover_batch_runs(batch_root):
        print("\n=== {} ===".format(run_record["label"]))
        summary, rows, results = process_single_run(vp, run_record["run_dir"])
        batch_runs.append({
            "run": run_record,
            "summary": summary,
            "rows": rows,
            "results": results,
        })
    return batch_runs


def chart_definitions_for_results(results, rows):
    definitions = []
    time_lookup = {row["snapshot_name"]: row["time_value"] for row in rows}

    for sign in ("positive", "negative"):
        for term, label in AGGREGATE_TERMS:
            if not any(finite_number(result.get("aggregates", {}).get(sign, {}).get(term)) is not None for result in results):
                continue
            definitions.append({
                "kind": "aggregate",
                "key": "aggregate:{}:{}".format(sign, term),
                "term": term,
                "label": label,
                "name": "{} - {}".format(label, sign),
                "sign": sign,
                "points": [
                    {
                        "snapshot_name": result["snapshot_name"],
                        "x": time_lookup.get(result["snapshot_name"]),
                        "y": finite_number(result.get("aggregates", {}).get(sign, {}).get(term)),
                    }
                    for result in results
                ],
            })

    region_keys = set()
    for result in results:
        for region in result.get("regions", []) or []:
            region_keys.add((str(region.get("sign") or "unknown"), int(region.get("index") or 0)))

    for sign, index in sorted(region_keys, key=lambda item: (item[0], item[1])):
        for term, label in REGION_TERMS:
            has_value = False
            for result in results:
                for region in result.get("regions", []) or []:
                    if str(region.get("sign")) == sign and int(region.get("index") or 0) == index:
                        if finite_number(region.get(term)) is not None:
                            has_value = True
                            break
                if has_value:
                    break
            if not has_value:
                continue
            definitions.append({
                "kind": "region",
                "key": "region:{}:{}:{}".format(sign, index, term),
                "term": term,
                "label": label,
                "name": "{} - {} #{}".format(label, sign, index + 1),
                "sign": sign,
                "region_index": index,
                "points": [
                    {
                        "snapshot_name": result["snapshot_name"],
                        "x": time_lookup.get(result["snapshot_name"]),
                        "y": finite_number(next(
                            (
                                region.get(term)
                                for region in (result.get("regions", []) or [])
                                if str(region.get("sign")) == sign and int(region.get("index") or 0) == index
                            ),
                            None,
                        )),
                    }
                    for result in results
                ],
            })

    if any(finite_number(result.get("normalized_rmse")) is not None for result in results):
        definitions.append({
            "kind": "special",
            "key": "fit:normalized_rmse",
            "term": "normalized_rmse",
            "label": "Fit normalized RMSE",
            "name": "Fit normalized RMSE",
            "points": [
                {
                    "snapshot_name": result["snapshot_name"],
                    "x": time_lookup.get(result["snapshot_name"]),
                    "y": finite_number(result.get("normalized_rmse")),
                }
                for result in results
            ],
        })

    return definitions


def parse_tau_value(text):
    text = str(text or "")
    explicit = re.search(r"(?:b[_-]?f[_-]?tau|tau)\s*[=_-]?\s*([0-9]+(?:[p.][0-9]+)?)", text, re.IGNORECASE)
    if explicit:
        return float(explicit.group(1).replace("p", "."))
    config = re.search(r"config[_-]?tau([0-9]+(?:[p.][0-9]+)?)", text, re.IGNORECASE)
    if config:
        return float(config.group(1).replace("p", "."))
    return None


def batch_chart_definitions(batch_runs):
    definitions = []

    for sign in ("positive", "negative"):
        for term, label in AGGREGATE_TERMS:
            if not any(
                any(finite_number(result.get("aggregates", {}).get(sign, {}).get(term)) is not None for result in batch_run["results"])
                for batch_run in batch_runs
            ):
                continue

            definitions.append({
                "kind": "aggregate",
                "key": "batch:{}:{}".format(sign, term),
                "term": term,
                "label": label,
                "name": "{} - {}".format(label, sign),
                "sign": sign,
                "series": [
                    {
                        "run_label": batch_run["run"]["label"],
                        "points": [
                            {
                                "snapshot_name": result["snapshot_name"],
                                "x": next(
                                    (row["time_value"] for row in batch_run["rows"] if row["snapshot_name"] == result["snapshot_name"]),
                                    None,
                                ),
                                "y": finite_number(result.get("aggregates", {}).get(sign, {}).get(term)),
                            }
                            for result in batch_run["results"]
                        ],
                    }
                    for batch_run in batch_runs
                ],
            })

    region_keys = set()
    for batch_run in batch_runs:
        for result in batch_run["results"]:
            for region in result.get("regions", []) or []:
                region_keys.add((str(region.get("sign") or "unknown"), int(region.get("index") or 0)))

    for sign, index in sorted(region_keys, key=lambda item: (item[0], item[1])):
        for term, label in REGION_TERMS:
            has_value = False
            for batch_run in batch_runs:
                for result in batch_run["results"]:
                    for region in result.get("regions", []) or []:
                        if str(region.get("sign")) == sign and int(region.get("index") or 0) == index:
                            if finite_number(region.get(term)) is not None:
                                has_value = True
                                break
                    if has_value:
                        break
                if has_value:
                    break
            if not has_value:
                continue
            definitions.append({
                "kind": "region",
                "key": "batch-region:{}:{}:{}".format(sign, index, term),
                "term": term,
                "label": label,
                "name": "{} - {} #{}".format(label, sign, index + 1),
                "sign": sign,
                "region_index": index,
                "series": [
                    {
                        "run_label": batch_run["run"]["label"],
                        "points": [
                            {
                                "snapshot_name": result["snapshot_name"],
                                "x": next(
                                    (row["time_value"] for row in batch_run["rows"] if row["snapshot_name"] == result["snapshot_name"]),
                                    None,
                                ),
                                "y": finite_number(next(
                                    (
                                        region.get(term)
                                        for region in (result.get("regions", []) or [])
                                        if str(region.get("sign")) == sign and int(region.get("index") or 0) == index
                                    ),
                                    None,
                                )),
                            }
                            for result in batch_run["results"]
                        ],
                    }
                    for batch_run in batch_runs
                ],
            })

    if any(any(finite_number(result.get("normalized_rmse")) is not None for result in batch_run["results"]) for batch_run in batch_runs):
        definitions.append({
            "kind": "special",
            "key": "batch:fit:normalized_rmse",
            "term": "normalized_rmse",
            "label": "Fit normalized RMSE",
            "name": "Fit normalized RMSE",
            "series": [
                {
                    "run_label": batch_run["run"]["label"],
                    "points": [
                        {
                            "snapshot_name": result["snapshot_name"],
                            "x": next(
                                (row["time_value"] for row in batch_run["rows"] if row["snapshot_name"] == result["snapshot_name"]),
                                None,
                            ),
                            "y": finite_number(result.get("normalized_rmse")),
                        }
                        for result in batch_run["results"]
                    ],
                }
                for batch_run in batch_runs
            ],
        })

    positive_points = []
    negative_points = []
    for batch_run in batch_runs:
        tau = parse_tau_value(batch_run["run"]["label"])
        if tau is None:
            continue
        max_positive = None
        max_negative_magnitude = None
        for result in batch_run["results"]:
            positive = finite_number(result.get("aggregates", {}).get("positive", {}).get("circulation"))
            if positive is not None:
                max_positive = positive if max_positive is None else max(max_positive, positive)
            negative = finite_number(result.get("aggregates", {}).get("negative", {}).get("circulation"))
            if negative is not None:
                magnitude = abs(negative)
                max_negative_magnitude = magnitude if max_negative_magnitude is None else max(max_negative_magnitude, magnitude)
        positive_points.append({"run_label": batch_run["run"]["label"], "snapshot_name": "max over snapshots", "x": tau, "y": max_positive})
        negative_points.append({"run_label": batch_run["run"]["label"], "snapshot_name": "max over snapshots", "x": tau, "y": max_negative_magnitude})

    if positive_points:
        definitions.append({
            "kind": "tau_summary",
            "key": "batch:positive:max_circulation_by_tau",
            "term": "max_circulation_by_tau",
            "label": "Max circulation vs tau - positive",
            "name": "Max circulation vs tau - positive",
            "sign": "positive",
            "x_label": "tau",
            "points": sorted(positive_points, key=lambda point: point["x"]),
        })
    if negative_points:
        definitions.append({
            "kind": "tau_summary",
            "key": "batch:negative:max_circulation_by_tau",
            "term": "max_circulation_by_tau",
            "label": "Max circulation magnitude vs tau - negative",
            "name": "Max circulation magnitude vs tau - negative",
            "sign": "negative",
            "x_label": "tau",
            "points": sorted(negative_points, key=lambda point: point["x"]),
        })

    return definitions


def unique_term_menu(definitions, kind):
    unique = []
    seen = set()
    for definition in definitions:
        if definition.get("kind") != kind:
            continue
        key = (definition.get("term"), definition.get("label"))
        if key in seen:
            continue
        seen.add(key)
        unique.append({"term": definition.get("term"), "label": definition.get("label")})
    return unique


def prompt_choice(items, header):
    print("\n" + header)
    for index, item in enumerate(items, start=1):
        print("  {}. {}".format(index, item["label"]))
    print()
    while True:
        text = input("Choose a number (or q): ").strip().lower()
        if text in {"q", "quit", "exit"}:
            return None
        try:
            choice = int(text)
        except ValueError:
            print("Please enter a number.")
            continue
        if 1 <= choice <= len(items):
            return items[choice - 1]
        print("That choice is out of range.")


def prompt_sign(definitions, term):
    signs = []
    for sign in ("positive", "negative"):
        if any(definition.get("term") == term and definition.get("sign") == sign for definition in definitions):
            signs.append({"label": sign, "sign": sign})
    if len(signs) == 1:
        return signs[0]["sign"]
    choice = prompt_choice(signs, "Choose the vortex sign:")
    return None if choice is None else choice["sign"]


def prompt_region_index(definitions, term, sign):
    indices = []
    seen = set()
    for definition in definitions:
        if definition.get("kind") != "region":
            continue
        if definition.get("term") != term or definition.get("sign") != sign:
            continue
        index = int(definition.get("region_index"))
        if index in seen:
            continue
        seen.add(index)
        indices.append({"label": "vortex #{}".format(index + 1), "index": index})
    choice = prompt_choice(indices, "Choose which vortex to plot:")
    return None if choice is None else choice["index"]


def choose_definition(definitions, mode):
    aggregate_items = unique_term_menu(definitions, "aggregate")
    region_items = unique_term_menu(definitions, "region")
    special_items = [
        {"label": definition["label"], "definition": definition}
        for definition in definitions
        if definition.get("kind") in {"special", "tau_summary"}
    ]

    print("\nWhat do you want to plot?")
    sections = []
    if aggregate_items:
        sections.append({"label": "aggregate variable", "kind": "aggregate"})
    if region_items:
        sections.append({"label": "vortex-specific variable", "kind": "region"})
    if special_items:
        sections.append({"label": "special variable", "kind": "special"})

    section_choice = prompt_choice(sections, "Choose the plot type:")
    if section_choice is None:
        return None

    kind = section_choice["kind"]
    if kind == "special":
        choice = prompt_choice(special_items, "Choose the special variable:")
        return None if choice is None else choice["definition"]

    items = aggregate_items if kind == "aggregate" else region_items
    choice = prompt_choice(items, "Choose the variable:")
    if choice is None:
        return None

    term = choice["term"]
    sign = prompt_sign(definitions, term)
    if sign is None:
        return None

    if kind == "aggregate":
        return next(
            definition
            for definition in definitions
            if definition.get("kind") == "aggregate"
            and definition.get("term") == term
            and definition.get("sign") == sign
        )

    region_index = prompt_region_index(definitions, term, sign)
    if region_index is None:
        return None

    return next(
        definition
        for definition in definitions
        if definition.get("kind") == "region"
        and definition.get("term") == term
        and definition.get("sign") == sign
        and int(definition.get("region_index")) == region_index
    )


def show_or_save_figure(fig, export_path):
    import matplotlib.pyplot as plt

    if AUTO_EXPORT_PNG or not can_show_figures():
        fig.savefig(export_path, dpi=SAVE_DPI, bbox_inches="tight")
        print("Saved PNG:", export_path)
        plt.close(fig)
        return

    plt.show()
    answer = input("Export this plot to {} ? [y/N]: ".format(export_path.name)).strip().lower()
    if answer in {"y", "yes"}:
        fig.savefig(export_path, dpi=SAVE_DPI, bbox_inches="tight")
        print("Saved PNG:", export_path)
    plt.close(fig)


def plot_single_definition(definition, export_path):
    import matplotlib.pyplot as plt

    x_values = [point["x"] for point in definition["points"] if point["y"] is not None and point["x"] is not None]
    y_values = [point["y"] for point in definition["points"] if point["y"] is not None and point["x"] is not None]
    snapshot_labels = [point["snapshot_name"] for point in definition["points"] if point["y"] is not None and point["x"] is not None]

    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)
    if hasattr(fig.canvas.manager, "set_window_title"):
        fig.canvas.manager.set_window_title("IBLGF postprocess plot")
    ax.plot(
        x_values,
        y_values,
        marker="o",
        linewidth=LINE_WIDTH,
        markersize=MARKER_SIZE,
        color=chart_color(definition.get("sign")),
    )
    ax.set_title(definition["name"])
    ax.set_xlabel(definition.get("x_label", "simulation time"))
    ax.set_ylabel(definition["label"])
    ax.grid(True, alpha=0.25)
    for x_value, y_value, snapshot_name in zip(x_values, y_values, snapshot_labels):
        ax.annotate(snapshot_name, (x_value, y_value), fontsize=8, alpha=0.6)
    fig.tight_layout()
    show_or_save_figure(fig, export_path)


def plot_batch_definition(definition, export_path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)
    if hasattr(fig.canvas.manager, "set_window_title"):
        fig.canvas.manager.set_window_title("IBLGF batch postprocess plot")

    if definition.get("kind") == "tau_summary":
        x_values = [point["x"] for point in definition["points"] if point["y"] is not None and point["x"] is not None]
        y_values = [point["y"] for point in definition["points"] if point["y"] is not None and point["x"] is not None]
        ax.plot(
            x_values,
            y_values,
            marker="o",
            linewidth=LINE_WIDTH,
            markersize=MARKER_SIZE,
            color=chart_color(definition.get("sign")),
        )
        for point in definition["points"]:
            if point["x"] is None or point["y"] is None:
                continue
            ax.annotate(point["run_label"], (point["x"], point["y"]), fontsize=8, alpha=0.7)
        ax.set_xlabel(definition.get("x_label", "tau"))
    else:
        for series_index, series in enumerate(definition["series"]):
            points = [point for point in series["points"] if point["y"] is not None and point["x"] is not None]
            x_values = [point["x"] for point in points]
            y_values = [point["y"] for point in points]
            ax.plot(
                x_values,
                y_values,
                marker="o",
                linewidth=LINE_WIDTH,
                markersize=MARKER_SIZE - 1,
                color=run_series_color(series_index),
                label=series["run_label"],
            )
        ax.set_xlabel(definition.get("x_label", "simulation time"))
        ax.legend(loc="best", fontsize=9)

    ax.set_title(definition["name"])
    ax.set_ylabel(definition["label"])
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    show_or_save_figure(fig, export_path)


def main():
    args = parse_args()
    vp = load_viewer_plotting()

    if args.mode == "single":
        run_dir = resolve_run_dir(args.path)
        summary, rows, results = process_single_run(vp, run_dir)
        definitions = chart_definitions_for_results(results, rows)
        print("\nLoaded run:", run_dir)
        print("Processed {} snapshots.".format(len(rows)))

        while True:
            definition = choose_definition(definitions, "single")
            if definition is None:
                break
            export_name = "{}_{}.png".format(PNG_PREFIX, definition["key"].replace(":", "_"))
            plot_single_definition(definition, outputs_dir() / export_name)

    else:
        batch_runs = process_batch(vp, args.path)
        definitions = batch_chart_definitions(batch_runs)
        print("\nLoaded {} batch outputs.".format(len(batch_runs)))

        while True:
            definition = choose_definition(definitions, "batch")
            if definition is None:
                break
            export_name = "{}_{}.png".format(PNG_PREFIX, definition["key"].replace(":", "_"))
            plot_batch_definition(definition, outputs_dir() / export_name)


if __name__ == "__main__":
    main()
