#!/usr/bin/env python3
"""
Simple CLI snapshot plots using the same backend rendering logic as the viewer.

Example:
    python3 ritta_plotting/plots.py /path/to/run/output
"""

import argparse
import importlib
import io
import os
import re
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# User settings: edit these to match the viewer plot settings you want.
# ---------------------------------------------------------------------------

SELECTED_FIELDS = ["edge_aux"]  # Examples: ["edge_aux"], ["phi"], ["u_0"], ["u_1"]
SELECTED_LEVELS = None  # None = use all default levels. Example manual input: [0], [0, 1], [1, 2]
SHOW_GRID_LINES = False  # True or False
SMOOTH_PLOTS = False  # True or False
DRAW_SAME_LEVEL_CONTOURS = False  # True or False
CONTOUR_LEVEL_COUNT = 15  # Integer example: 10, 15, 20
CONTOUR_LEVEL_SPEC = None  # Leave as None, or use {"edge_aux": [-1.0, -0.5, 0.0, 0.5, 1.0]}
SHOW_VORTEX_BOUNDARY = False  # True or False
VORTEX_BOUNDARY_MODE = "vorticity_threshold"  # Options: "vorticity_threshold", "computed_q_positive", "elliptical_gaussian"
VORTEX_BOUNDARY_THRESHOLD_FACTOR = 0.30  # Float example: 0.2, 0.3, 0.5
VORTEX_MIN_REGION_AREA = None  # None to use snapshot default, or a float like 0.25 or 0.5
OVERLAY_LEVELS = True  # True = stacked AMR levels in one panel, False = separate panel per level
COLOR_SCALE_MODE = "composite_symmetric"  # Options: "composite_symmetric", "composite_linear", "per_level_autoscale"
COLOR_SCHEME = "blue_white_red"  # Examples: "blue_white_red", "viridis", "plasma", "original"
POINT_OVERLAY_OPTIONS = None  # None for viewer defaults, or {} to suppress, or {"vortex_center_positive": True}
POINT_OVERLAY_SIZE = 48.0  # Marker size float example: 32.0, 48.0, 64.0
SUPPRESS_ZERO_OMEGA_EXTREMA = True  # True or False
VIEW_LIMITS = None  # None for auto. Example: {"x": (-4, 8), "y": (-3, 3)}
RENDER_SCALE = 1.0  # Float example: 1.0, 1.5, 2.0
AUTO_EXPORT_PNG = False  # True = save automatically, False = ask after showing plot
PNG_PREFIX = "plot"  # Example output names: plot_flowTime_120.png


def parse_args():
    parser = argparse.ArgumentParser(description="Render viewer-style plots without the browser.")
    parser.add_argument("output_folder", type=Path, help="Run folder or output folder.")
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


def discover_primary_snapshot_paths(run_dir):
    snapshots = list(run_dir.rglob("flowTime_*.hdf5")) + list(run_dir.rglob("flowTime_*.h5"))
    unique = {}
    for path in snapshots:
        unique[path.resolve()] = path
    ordered = sorted(unique.values(), key=lambda path: snapshot_timestep(path.name) or -1)
    if ordered:
        return ordered
    fallback = sorted(vp_path for vp_path in run_dir.rglob("*.hdf5")) + sorted(run_dir.rglob("*.h5"))
    return fallback


def build_fast_summary(vp, run_dir):
    snapshot_paths = discover_primary_snapshot_paths(run_dir)
    if snapshot_paths:
        snapshot_paths = [path for path in snapshot_paths if vp.IBLGFSnapshot.is_plottable_file(path)]
    if not snapshot_paths:
        snapshot_paths = sorted(vp.find_snapshot_files(run_dir), key=vp.snapshot_sort_key)
    if not snapshot_paths:
        raise ValueError("No plottable .hdf5 or .h5 files were found in the selected folder.")

    time_series_paths = [path for path in snapshot_paths if snapshot_timestep(path) is not None]
    primary_paths = time_series_paths or snapshot_paths
    first_snapshot = primary_paths[0]
    snapshot = vp._snapshot_for_render(first_snapshot)
    coordinate_metadata = vp._resolved_coordinate_metadata_for_run(run_dir)
    simulation_time_metadata = vp.infer_simulation_time_metadata(run_dir)

    return {
        "run_dir_name": run_dir.name,
        "snapshot_files": [path.name for path in primary_paths],
        "time_series_snapshot_files": [path.name for path in primary_paths],
        "default_checked_levels": list(range(snapshot.n_levels)),
        "coordinate_dx_base": coordinate_metadata["coordinate_dx_base"],
        "coordinate_dx_base_source": coordinate_metadata["coordinate_dx_base_source"],
        "coordinate_origin_index": coordinate_metadata["coordinate_origin_index"],
        "coordinate_origin_index_source": coordinate_metadata["coordinate_origin_index_source"],
        "space_dim": snapshot.dims,
        **simulation_time_metadata,
    }


def snapshot_timestep(snapshot_name):
    match = re.fullmatch(r"flowTime_(\d+)\.(?:hdf5|h5)", Path(snapshot_name).name)
    return int(match.group(1)) if match else None


def snapshot_rows(summary):
    raw_names = list(summary.get("time_series_snapshot_files") or summary.get("snapshot_files") or [])
    names = [name for name in raw_names if snapshot_timestep(name) is not None]
    if not names:
        names = raw_names
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


def print_snapshot_table(rows):
    print("\nAvailable snapshots:")
    print("  idx    reported_step    simulation_time    snapshot")
    for row in rows:
        step_text = "-" if row["timestep"] is None else str(row["timestep"])
        print(
            "  {idx:>3}    {step:>12}    {time:>15.8g}    {name}".format(
                idx=row["index"],
                step=step_text,
                time=row["time_value"],
                name=row["snapshot_name"],
            )
        )
    print()


def nearest_snapshot(rows, requested_time):
    return min(rows, key=lambda row: abs(row["time_value"] - requested_time))


def panel_meta_text(plot, overlay_mode):
    if overlay_mode:
        levels = ", ".join(str(level) for level in plot.get("levels", []))
        return "Levels {} (stacked)".format(levels)
    return "Level {}".format(plot.get("level"))


def build_export_groups(rendered):
    groups = {}
    overlay_mode = bool(rendered.get("overlay_levels", True))
    for plot in rendered.get("plots", []):
        groups.setdefault(plot["field_name"], []).append(plot)

    payload = []
    for field_name, field_plots in groups.items():
        payload.append({
            "field_name": field_name,
            "full_width": len(field_plots) > 1,
            "panels": [
                {
                    "meta": panel_meta_text(plot, overlay_mode),
                    "image": plot["image"],
                }
                for plot in field_plots
            ],
        })
    return payload


def sheet_png_bytes(vp, rendered, title, subtitle):
    return vp.compose_export_sheet(
        build_export_groups(rendered),
        title=title,
        subtitle=subtitle,
        resolution_scale=RENDER_SCALE,
    )


def can_show_figures():
    return not (sys.platform.startswith("linux") and not os.environ.get("DISPLAY"))


def show_png_bytes(png_bytes, window_title):
    from PIL import Image
    import matplotlib.pyplot as plt

    image = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    figure, axis = plt.subplots(figsize=(10.5, 7.5))
    if hasattr(figure.canvas.manager, "set_window_title"):
        figure.canvas.manager.set_window_title(window_title)
    axis.imshow(image)
    axis.axis("off")
    figure.tight_layout()
    plt.show()


def save_png_bytes(png_bytes, path):
    path.write_bytes(png_bytes)
    print("Saved PNG:", path)


def render_one_snapshot(vp, run_dir, summary, row):
    selected_levels = (
        list(summary.get("default_checked_levels") or [])
        if SELECTED_LEVELS is None
        else list(SELECTED_LEVELS)
    )

    rendered = vp.render_snapshot(
        run_dir,
        row["snapshot_name"],
        selected_level=selected_levels[0] if selected_levels else 0,
        selected_levels=selected_levels,
        selected_fields=SELECTED_FIELDS,
        show_grid_lines=SHOW_GRID_LINES,
        smooth_plots=SMOOTH_PLOTS,
        draw_same_level_contours=DRAW_SAME_LEVEL_CONTOURS,
        contour_level_count=CONTOUR_LEVEL_COUNT,
        contour_level_spec=CONTOUR_LEVEL_SPEC,
        show_vortex_boundary=SHOW_VORTEX_BOUNDARY,
        vortex_boundary_mode=VORTEX_BOUNDARY_MODE,
        vortex_boundary_threshold_factor=VORTEX_BOUNDARY_THRESHOLD_FACTOR,
        vortex_min_region_area=VORTEX_MIN_REGION_AREA,
        overlay_levels=OVERLAY_LEVELS,
        color_scale_mode=COLOR_SCALE_MODE,
        color_scheme=COLOR_SCHEME,
        point_overlay_options=POINT_OVERLAY_OPTIONS,
        point_overlay_size=POINT_OVERLAY_SIZE,
        suppress_zero_omega_extrema_overlays=SUPPRESS_ZERO_OMEGA_EXTREMA,
        view_limits=VIEW_LIMITS,
        coordinate_dx_base=summary.get("coordinate_dx_base"),
        render_scale=RENDER_SCALE,
    )

    title = "{} | {}".format(summary.get("run_dir_name", run_dir.name), row["snapshot_name"])
    subtitle = "t = {:.8g}".format(row["time_value"])
    png_bytes = sheet_png_bytes(vp, rendered, title, subtitle)
    return rendered, png_bytes


def main():
    args = parse_args()
    run_dir = resolve_run_dir(args.output_folder)
    vp = load_viewer_plotting()
    summary = build_fast_summary(vp, run_dir)
    if int(summary.get("space_dim") or 0) != 2:
        raise SystemExit("plots.py is 2D-only. This run is not a 2D run.")
    rows = snapshot_rows(summary)

    print("Loaded run:", run_dir)
    print("simulation_time_scale =", summary.get("simulation_time_scale"))
    print("time formula uses cfl={}, dx_base={}, nLevels={}".format(
        summary.get("simulation_time_cfl"),
        summary.get("simulation_time_dx_base"),
        summary.get("simulation_time_amr_levels"),
    ))
    print_snapshot_table(rows)

    while True:
        user_text = input("Simulation time to plot (list / q / number): ").strip()
        if not user_text:
            continue
        if user_text.lower() in {"q", "quit", "exit"}:
            break
        if user_text.lower() in {"list", "ls"}:
            print_snapshot_table(rows)
            continue

        try:
            requested_time = float(user_text)
        except ValueError:
            print("Please enter a floating-point simulation time, `list`, or `q`.")
            continue

        row = nearest_snapshot(rows, requested_time)
        print(
            "Using {} (reported step {}, simulation time {:.8g})".format(
                row["snapshot_name"],
                row["timestep"],
                row["time_value"],
            )
        )

        _, png_bytes = render_one_snapshot(vp, run_dir, summary, row)
        export_path = outputs_dir() / "{}_{}.png".format(
            PNG_PREFIX,
            Path(row["snapshot_name"]).stem,
        )

        if AUTO_EXPORT_PNG or not can_show_figures():
            save_png_bytes(png_bytes, export_path)
            if not can_show_figures():
                print("No display detected, so the plot was saved instead of opened.")
            continue

        show_png_bytes(png_bytes, "IBLGF plot")
        answer = input("Export this plot to {} ? [y/N]: ".format(export_path.name)).strip().lower()
        if answer in {"y", "yes"}:
            save_png_bytes(png_bytes, export_path)


if __name__ == "__main__":
    main()
