"""Export beginning, middle, and end PNG previews for vortex-fit stages."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch, Rectangle
from PIL import Image

from common import load_config, read_frame_order, result_folder
from plot_vorticity import plot_vorticity_frame


PREVIEW_POSITIONS = ("beginning", "middle", "end")
LOCAL_MAXIMUM_SIZE_SCALE = 0.65


def base_frame(group: h5py.Group) -> dict:
    """Load the saved vorticity raster and physical metadata for one frame."""
    return {
        "source_filename": str(group.attrs["source_filename"]),
        "time": float(group.attrs["simulation_time"]),
        "dx": float(group.attrs["dx"]),
        "x": group["x"][:],
        "y": group["y"][:],
        "vorticity": group["vorticity"][:],
    }


def preview_indices(fits: h5py.File, group_names: list[str]) -> list[int]:
    """Choose early, midpoint-nearest, and late frames containing a successful fit."""
    eligible = [
        index
        for index, group_name in enumerate(group_names)
        if np.any(fits[group_name]["success"][:].astype(bool))
    ]
    if len(eligible) < 3:
        raise ValueError("At least three frames with successful fits are required for previews.")

    targets = (0.0, 0.5 * (len(group_names) - 1), float(len(group_names) - 1))
    selected = []
    for target in targets:
        available = [index for index in eligible if index not in selected]
        selected.append(min(available, key=lambda index: abs(index - target)))
    return selected


def save_preview(
    output_path: Path,
    frame: dict,
    plot_config: dict,
    title: str,
    overlay,
    legend_handles: list | None = None,
) -> None:
    """Render one saved vorticity frame and its requested stage overlay."""
    figure, axis = plt.subplots(
        figsize=(
            float(plot_config.get("figure_width", 10.0)),
            float(plot_config.get("figure_height", 7.0)),
        )
    )
    image = plot_vorticity_frame(axis, frame, plot_config)
    overlay(axis)
    axis.set_title(
        f"{title} | {frame['source_filename']} | t = {frame['time']:.8g}"
    )
    if legend_handles:
        axis.legend(handles=legend_handles, loc="upper right")
    figure.colorbar(image, ax=axis, label="vorticity")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def scatter_outlined_marker(
    axis,
    x,
    y,
    marker_size: float,
    color: str,
    marker: str,
):
    """Draw a marker with a white visibility halo."""
    markers = axis.scatter(
        x,
        y,
        s=marker_size,
        c=color,
        marker=marker,
        linewidths=1.25,
        zorder=5,
    )
    markers.set_path_effects([
        path_effects.Stroke(linewidth=3.5, foreground="white"),
        path_effects.Normal(),
    ])
    return markers


def scatter_local_maxima(axis, x, y, marker_size: float, color: str):
    """Draw a compact local-maximum marker with a white visibility halo."""
    return scatter_outlined_marker(
        axis,
        x,
        y,
        LOCAL_MAXIMUM_SIZE_SCALE * marker_size,
        color,
        "x",
    )


def outlined_legend_handle(
    color: str,
    marker: str,
    label: str,
    linestyle: str = "none",
) -> Line2D:
    """Create a legend marker with the same white visibility halo."""
    handle = Line2D(
        [0],
        [0],
        color=color,
        marker=marker,
        linestyle=linestyle,
        markeredgewidth=1.25,
        markersize=6,
        label=label,
    )
    handle.set_path_effects([
        path_effects.Stroke(linewidth=3.5, foreground="white"),
        path_effects.Normal(),
    ])
    return handle


def local_maximum_legend_handle(color: str) -> Line2D:
    """Create a legend marker matching the outlined local maxima."""
    return outlined_legend_handle(
        color,
        "x",
        "detected local maximum",
    )


def stack_previews(output_folder: Path, prefix: str) -> Path:
    """Stack the beginning, middle, and end previews from one stage horizontally."""
    input_paths = [
        output_folder / f"{prefix}_{position}.png"
        for position in PREVIEW_POSITIONS
    ]
    images = []
    try:
        for path in input_paths:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        output_path = output_folder / f"{prefix}_combined.png"
        canvas = Image.new(
            "RGB",
            (sum(image.width for image in images), max(image.height for image in images)),
            "white",
        )
        x_offset = 0
        for image in images:
            canvas.paste(image, (x_offset, 0))
            x_offset += image.width
        canvas.save(output_path)
        canvas.close()
        return output_path
    finally:
        for image in images:
            image.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export three representative PNGs for each of fit Stages 1-3."
    )
    parser.add_argument("run_folder", type=Path)
    parser.add_argument("config_file", type=Path)
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        help="Folder containing hmaxima.h5 and regions.h5.",
    )
    parser.add_argument("--fits-file", type=Path, help="Use this fits HDF5 file.")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--fits-only",
        action="store_true",
        help="Export only the beginning/middle/end combined fit preview.",
    )
    parser.add_argument(
        "--filename-prefix",
        help="Prefix for the combined fit PNG, such as tau_1p0_boundary_fraction_0p05_fits.",
    )
    parser.add_argument("--y-axis-min", type=float, default=-2.0)
    parser.add_argument("--y-axis-max", type=float, default=2.0)
    args = parser.parse_args()
    if not np.isfinite(args.y_axis_min) or not np.isfinite(args.y_axis_max):
        parser.error("--y-axis-min and --y-axis-max must be finite")
    if args.y_axis_min >= args.y_axis_max:
        parser.error("--y-axis-min must be smaller than --y-axis-max")
    if args.filename_prefix and Path(args.filename_prefix).name != args.filename_prefix:
        parser.error("--filename-prefix must be a filename, not a path")

    config = load_config(args.config_file)
    analysis_folder = (
        args.analysis_dir.expanduser().resolve()
        if args.analysis_dir is not None
        else result_folder(args.run_folder)
    )
    output_folder = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else analysis_folder / "fit_previews"
    )
    output_folder.mkdir(parents=True, exist_ok=True)

    hmaxima_path = analysis_folder / "hmaxima.h5"
    regions_path = analysis_folder / "regions.h5"
    fits_path = (
        args.fits_file.expanduser().resolve()
        if args.fits_file is not None
        else analysis_folder / "fits.h5"
    )
    missing = [
        path.name for path in (hmaxima_path, regions_path, fits_path) if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError("Run fit Stages 1-3 first; missing: " + ", ".join(missing))

    plot_config = dict(config["plot"])
    plot_config["y_axis_min"] = args.y_axis_min
    plot_config["y_axis_max"] = args.y_axis_max
    marker_size = float(plot_config.get("marker_size", 48.0))
    positive_color = str(plot_config.get("positive_marker_color", "black"))
    negative_color = str(plot_config.get("negative_marker_color", "#7b2cbf"))
    region_color = str(plot_config.get("region_color", "#00aa55"))
    line_width = float(plot_config.get("region_line_width", 1.5))

    with (
        h5py.File(hmaxima_path, "r") as maxima,
        h5py.File(regions_path, "r") as regions,
        h5py.File(fits_path, "r") as fits,
    ):
        group_names = read_frame_order(maxima)
        if group_names != read_frame_order(regions) or group_names != read_frame_order(fits):
            raise ValueError("Frame order differs among hmaxima.h5, regions.h5, and fits.h5.")

        boundary_fraction = float(fits.attrs["boundary_fraction"])
        fit_prefix = args.filename_prefix or "03_fits"
        selected_indices = preview_indices(fits, group_names)
        for position, index in zip(PREVIEW_POSITIONS, selected_indices):
            group_name = group_names[index]
            maximum = maxima[group_name]
            region = regions[group_name]
            fit = fits[group_name]
            frame = base_frame(maximum)

            peak_x = maximum["peak_x"][:]
            peak_y = maximum["peak_y"][:]

            def draw_maxima(axis) -> None:
                scatter_local_maxima(
                    axis, peak_x, peak_y, marker_size, positive_color
                )

            if not args.fits_only:
                save_preview(
                    output_folder / f"01_local_maxima_{position}.png",
                    frame,
                    plot_config,
                    f"Detected local maxima ({position})",
                    draw_maxima,
                    [local_maximum_legend_handle(positive_color)],
                )

            positive_points = region["positive_points"][:]
            negative_points = region["negative_points"][:]
            clamped_bounds = region["clamped_bounds"][:]

            def draw_regions(axis) -> None:
                for bounds in clamped_bounds:
                    x0, x1, y0, y1 = bounds
                    axis.add_patch(
                        Rectangle(
                            (x0, y0),
                            x1 - x0,
                            y1 - y0,
                            fill=False,
                            edgecolor=region_color,
                            linewidth=line_width,
                        )
                    )
                if len(positive_points):
                    scatter_local_maxima(
                        axis,
                        positive_points[:, 0],
                        positive_points[:, 1],
                        marker_size,
                        positive_color,
                    )
                    axis.scatter(
                        negative_points[:, 0],
                        negative_points[:, 1],
                        s=marker_size,
                        c=negative_color,
                        marker="+",
                    )

            if not args.fits_only:
                save_preview(
                    output_folder / f"02_regions_{position}.png",
                    frame,
                    plot_config,
                    f"Local maxima, mirrored points, and fitting rectangles ({position})",
                    draw_regions,
                    [
                        local_maximum_legend_handle(positive_color),
                        Line2D(
                            [0], [0], color=negative_color, marker="+", linestyle="none",
                            markersize=7, label="mirrored point",
                        ),
                        Patch(
                            facecolor="none", edgecolor=region_color, linewidth=line_width,
                            label="fitting rectangle",
                        ),
                    ],
                )

            success = fit["success"][:].astype(bool)
            radii = fit["boundary_radius"][:]
            positive_centers = fit["positive_centers"][:]
            negative_centers = fit["negative_centers"][:]

            def draw_fits(axis) -> None:
                for radius, positive, negative in zip(
                    radii[success],
                    positive_centers[success],
                    negative_centers[success],
                ):
                    if not np.isfinite(radius) or not np.all(np.isfinite(positive)):
                        continue
                    axis.add_patch(
                        Circle(
                            positive,
                            radius,
                            fill=False,
                            edgecolor=positive_color,
                            linewidth=line_width,
                        )
                    )
                    axis.add_patch(
                        Circle(
                            negative,
                            radius,
                            fill=False,
                            edgecolor=negative_color,
                            linewidth=line_width,
                        )
                    )
                    scatter_outlined_marker(
                        axis,
                        positive[0],
                        positive[1],
                        marker_size,
                        positive_color,
                        "x",
                    )
                    scatter_outlined_marker(
                        axis,
                        negative[0],
                        negative[1],
                        marker_size,
                        negative_color,
                        "+",
                    )

            save_preview(
                output_folder / f"{fit_prefix}_{position}.png",
                frame,
                plot_config,
                f"Fitted vortex regions and centers ({position}; f_b = {boundary_fraction:g})",
                draw_fits,
                [
                    outlined_legend_handle(
                        positive_color,
                        "x",
                        "positive fitted center and boundary",
                        "-",
                    ),
                    outlined_legend_handle(
                        negative_color,
                        "+",
                        "negative fitted center and boundary",
                        "-",
                    ),
                ],
            )

            print(
                f"{position}: frame {index}, {frame['source_filename']}, "
                f"t={frame['time']:.8g}"
            )

    prefixes = (
        [fit_prefix]
        if args.fits_only
        else ["01_local_maxima", "02_regions", fit_prefix]
    )
    combined_paths = [stack_previews(output_folder, prefix) for prefix in prefixes]
    individual_paths = [
        output_folder / f"{prefix}_{position}.png"
        for prefix in prefixes
        for position in PREVIEW_POSITIONS
    ]
    for path in individual_paths:
        path.unlink()

    print(f"Saved {len(combined_paths)} combined preview PNG(s): {output_folder}")
    for path in combined_paths:
        print(f"Combined: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
