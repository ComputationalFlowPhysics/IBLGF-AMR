#!/usr/bin/env python3
"""
Simple CLI animation export using the same backend rendering logic as the viewer.

Example:
    python3 ritta_plotting/animation.py /path/to/run/output
"""

import argparse
import importlib
import io
import os
import re
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# User settings: edit these to match the viewer animation settings you want.
# ---------------------------------------------------------------------------

SELECTED_FIELDS = ["edge_aux"]  # Examples: ["edge_aux"], ["phi"], ["u_0"], ["u_1"]
SELECTED_LEVELS = None  # None = use all default levels. Example manual input: [0], [0, 1], [1, 2]
SHOW_GRID_LINES = False  # True or False
SMOOTH_PLOTS = True  # True or False
DRAW_SAME_LEVEL_CONTOURS = False  # True or False
CONTOUR_LEVEL_SPEC = None  # Leave as None, or use {"edge_aux": [-1.0, -0.5, 0.0, 0.5, 1.0]}
SHOW_VORTEX_BOUNDARY = False  # True or False
VORTEX_BOUNDARY_MODE = "vorticity_threshold"  # Options: "vorticity_threshold", "computed_q_positive", "elliptical_gaussian"
VORTEX_BOUNDARY_THRESHOLD_FACTOR = 0.30  # Float example: 0.2, 0.3, 0.5
VORTEX_MIN_REGION_AREA = None  # None to use snapshot default, or a float like 0.25 or 0.5
OVERLAY_LEVELS = True  # True = stacked AMR levels in one panel, False = separate panel per level
COLOR_SCALE_MODE = "composite_linear"  # Options: "composite_symmetric", "composite_linear", "per_level_autoscale"
COLOR_SCHEME = "blue_white_red"  # Examples: "blue_white_red", "viridis", "plasma", "original"
POINT_OVERLAY_OPTIONS = None  # None for viewer defaults, or {} to suppress, or {"vortex_center_positive": True}
POINT_OVERLAY_SIZE = 48.0  # Marker size float example: 32.0, 48.0, 64.0
SUPPRESS_ZERO_OMEGA_EXTREMA = True  # True or False
VIEW_LIMITS = {"x": (-1, 40), "y": (-3, 3)}  # None for auto. Example: {"x": (-4, 8), "y": (-3, 3)}
RENDER_SCALE = 1.0  # Float example: 1.0, 1.5, 2.0
GIF_FPS = 6  # Integer example: 4, 5, 8, 10
GIF_LOOP = True  # True = loop forever, False = play once
AUTO_EXPORT_GIF = True  # True = save automatically, False = ask after preview
GIF_PREFIX = "animation"  # Example output names: animation_flowTime_0_to_flowTime_300.gif
PLAN_FRAME_LIMIT = 24  # Use at most this many frames when computing shared view/color plans. Lower = faster startup.


def parse_args():
    parser = argparse.ArgumentParser(description="Render viewer-style animations without the browser.")
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
    fallback = sorted(list(run_dir.rglob("*.hdf5")) + list(run_dir.rglob("*.h5")))
    return fallback


def build_fast_summary(vp, run_dir):
    snapshot_paths = discover_primary_snapshot_paths(run_dir)
    if snapshot_paths:
        snapshot_paths = [path for path in snapshot_paths if vp.IBLGFSnapshot.is_plottable_file(path)]
    if not snapshot_paths:
        snapshot_paths = sorted(vp.find_snapshot_files(run_dir), key=vp.snapshot_sort_key)
    if not snapshot_paths:
        raise ValueError("No plottable .hdf5 or .h5 files were found in the selected folder.")

    time_series_paths = [path for path in snapshot_paths if vp.snapshot_timestep(path) is not None]
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


def selected_rows_for_range(rows, start_time, end_time, stride):
    selected = [row for row in rows if start_time <= row["time_value"] <= end_time]
    return selected[::stride]


def sample_snapshot_names_for_planning(snapshot_names, limit):
    if limit is None or limit <= 0 or len(snapshot_names) <= limit:
        return list(snapshot_names)
    if limit == 1:
        return [snapshot_names[0]]
    indices = []
    last_index = len(snapshot_names) - 1
    for slot in range(limit):
        index = round(slot * last_index / (limit - 1))
        if not indices or index != indices[-1]:
            indices.append(index)
    return [snapshot_names[index] for index in indices]


def can_show_figures():
    return not (sys.platform.startswith("linux") and not os.environ.get("DISPLAY"))


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


def compose_frame_sheet(vp, rendered, summary, row):
    title = "{} | {}".format(summary.get("run_dir_name", ""), row["snapshot_name"])
    subtitle = "t = {:.8g}".format(row["time_value"])
    return vp.compose_export_sheet(
        build_export_groups(rendered),
        title=title,
        subtitle=subtitle,
        resolution_scale=RENDER_SCALE,
    )


def preview_animation(frame_pngs, fps):
    from PIL import Image
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    import numpy as np

    arrays = [np.asarray(Image.open(io.BytesIO(png)).convert("RGBA")) for png in frame_pngs]
    figure, axis = plt.subplots(figsize=(10.5, 7.5))
    if hasattr(figure.canvas.manager, "set_window_title"):
        figure.canvas.manager.set_window_title("IBLGF animation preview")
    axis.axis("off")
    artist = axis.imshow(arrays[0])

    def update(frame_index):
        artist.set_data(arrays[frame_index])
        return [artist]

    animation = FuncAnimation(
        figure,
        update,
        frames=len(arrays),
        interval=1000.0 / max(fps, 1),
        blit=True,
        repeat=GIF_LOOP,
    )
    figure.tight_layout()
    plt.show()
    return animation


def save_gif_bytes(gif_bytes, path):
    path.write_bytes(gif_bytes)
    print("Saved GIF:", path)


def main():
    args = parse_args()
    run_dir = resolve_run_dir(args.output_folder)
    vp = load_viewer_plotting()
    summary = build_fast_summary(vp, run_dir)
    if int(summary.get("space_dim") or 0) != 2:
        raise SystemExit("animation.py is 2D-only. This run is not a 2D run.")
    rows = snapshot_rows(summary)

    print("Loaded run:", run_dir)
    print_snapshot_table(rows)

    while True:
        start_text = input("Animation start time (blank=first, list, q): ").strip()
        if start_text.lower() in {"q", "quit", "exit"}:
            break
        if start_text.lower() in {"list", "ls"}:
            print_snapshot_table(rows)
            continue

        end_text = input("Animation end time (blank=last): ").strip()
        stride_text = input("Snapshot stride (blank=1): ").strip()

        start_time = rows[0]["time_value"] if not start_text else float(start_text)
        end_time = rows[-1]["time_value"] if not end_text else float(end_text)
        stride = max(int(stride_text) if stride_text else 1, 1)

        selected_rows = selected_rows_for_range(rows, min(start_time, end_time), max(start_time, end_time), stride)
        if not selected_rows:
            print("No snapshots matched that time range.")
            continue

        selected_levels = (
            list(summary.get("default_checked_levels") or [])
            if SELECTED_LEVELS is None
            else list(SELECTED_LEVELS)
        )
        snapshot_names = [row["snapshot_name"] for row in selected_rows]
        planning_snapshot_names = sample_snapshot_names_for_planning(snapshot_names, PLAN_FRAME_LIMIT)

        print("Preparing animation for {} frames...".format(len(snapshot_names)))
        if VIEW_LIMITS is None:
            view_plan = vp.compute_animation_view_plan(
                run_dir,
                planning_snapshot_names,
                selected_level=selected_levels[0] if selected_levels else 0,
                selected_levels=selected_levels,
                selected_fields=SELECTED_FIELDS,
                overlay_levels=OVERLAY_LEVELS,
                view_limits=VIEW_LIMITS,
                coordinate_dx_base=summary.get("coordinate_dx_base"),
            )
        else:
            view_plan = {"view_limits": VIEW_LIMITS}

        field_norm_overrides = None
        if COLOR_SCALE_MODE != "per_level_autoscale":
            color_plan = vp.compute_animation_color_plan(
                run_dir,
                planning_snapshot_names,
                selected_level=selected_levels[0] if selected_levels else 0,
                selected_levels=selected_levels,
                selected_fields=SELECTED_FIELDS,
                overlay_levels=OVERLAY_LEVELS,
                color_scale_mode=COLOR_SCALE_MODE,
            )
            field_norm_overrides = color_plan.get("field_norms")

        rendered_frames = []
        frame_pngs = []
        frame_payloads = []
        for frame_index, row in enumerate(selected_rows, start=1):
            print("Rendering frame {}/{}: {}".format(frame_index, len(selected_rows), row["snapshot_name"]))
            rendered = vp.render_snapshot(
                run_dir,
                row["snapshot_name"],
                selected_level=selected_levels[0] if selected_levels else 0,
                selected_levels=selected_levels,
                selected_fields=SELECTED_FIELDS,
                show_grid_lines=SHOW_GRID_LINES,
                smooth_plots=SMOOTH_PLOTS,
                draw_same_level_contours=DRAW_SAME_LEVEL_CONTOURS,
                contour_level_spec=CONTOUR_LEVEL_SPEC,
                show_vortex_boundary=SHOW_VORTEX_BOUNDARY,
                vortex_boundary_mode=VORTEX_BOUNDARY_MODE,
                vortex_boundary_threshold_factor=VORTEX_BOUNDARY_THRESHOLD_FACTOR,
                vortex_min_region_area=VORTEX_MIN_REGION_AREA,
                overlay_levels=OVERLAY_LEVELS,
                color_scale_mode=COLOR_SCALE_MODE,
                color_scheme=COLOR_SCHEME,
                field_norm_overrides=field_norm_overrides,
                point_overlay_options=POINT_OVERLAY_OPTIONS,
                point_overlay_size=POINT_OVERLAY_SIZE,
                suppress_zero_omega_extrema_overlays=SUPPRESS_ZERO_OMEGA_EXTREMA,
                view_limits=view_plan.get("view_limits"),
                coordinate_dx_base=summary.get("coordinate_dx_base"),
                render_scale=RENDER_SCALE,
            )
            rendered_frames.append(rendered)
            frame_png = compose_frame_sheet(vp, rendered, summary, row)
            frame_pngs.append(frame_png)
            frame_payloads.append({
                "title": summary.get("run_dir_name", "IBLGF animation"),
                "subtitle": "Snapshot: {} | t = {:.8g}".format(row["snapshot_name"], row["time_value"]),
                "groups": build_export_groups(rendered),
            })

        gif_bytes = vp.compose_animation_gif(
            frame_payloads,
            title=summary.get("run_dir_name", "IBLGF animation"),
            subtitle="{} to {}".format(
                selected_rows[0]["snapshot_name"],
                selected_rows[-1]["snapshot_name"],
            ),
            fps=GIF_FPS,
            loop=GIF_LOOP,
            resolution_scale=RENDER_SCALE,
        )

        gif_name = "{}_{}_to_{}.gif".format(
            GIF_PREFIX,
            Path(selected_rows[0]["snapshot_name"]).stem,
            Path(selected_rows[-1]["snapshot_name"]).stem,
        )
        export_path = outputs_dir() / gif_name

        if can_show_figures():
            preview_animation(frame_pngs, GIF_FPS)
        else:
            print("No display detected, so only GIF export is available.")

        if AUTO_EXPORT_GIF or not can_show_figures():
            save_gif_bytes(gif_bytes, export_path)
            continue

        answer = input("Export this animation to {} ? [Y/n]: ".format(export_path.name)).strip().lower()
        if answer in {"", "y", "yes"}:
            save_gif_bytes(gif_bytes, export_path)


if __name__ == "__main__":
    main()
