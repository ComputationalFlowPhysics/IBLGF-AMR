"""Render saved vortex-analysis HDF5 frames to PNG files and a GIF."""

import argparse
import math
from pathlib import Path
from typing import List, Optional

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Patch, Rectangle
from PIL import Image

from common import largest_successful_fit_index, load_config, read_frame_order
from plot_vorticity import image_extent, plot_vorticity_frame


SUPPORTED_SCHEMAS = {
    "ritta_hmaxima_v1",
    "ritta_regions_v1",
    "ritta_circular_gaussian_dipole_fits_v1",
    "ritta_vorticity_threshold_masks_v1",
    "ritta_vorticity_threshold_masks_v2",
}


def text_value(value) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def saved_config(handle: h5py.File) -> dict:
    """Use the config recorded in the HDF5 file when it is still available."""
    path = Path(text_value(handle.attrs.get("config_file", ""))).expanduser()
    if path.is_file():
        return load_config(path)
    return {
        "plot": {
            "colormap": "RdBu_r",
            "symmetric_color_limits": True,
            "color_limit": math.nan,
            "figure_width": 10.0,
            "figure_height": 7.0,
        },
        "threshold_mask": {},
    }


def background_file(input_path: Path, schema: str) -> Optional[Path]:
    """Locate saved Stage 1 rasters needed by region and fit outputs."""
    if schema not in {
        "ritta_regions_v1",
        "ritta_circular_gaussian_dipole_fits_v1",
        "ritta_vorticity_threshold_masks_v2",
    }:
        return None
    path = input_path.with_name("hmaxima.h5")
    if not path.is_file():
        raise FileNotFoundError(
            f"{input_path.name} does not contain the vorticity raster. "
            f"Place its matching hmaxima.h5 beside it: {path}"
        )
    return path


def base_frame(group: h5py.Group) -> dict:
    return {
        "source_filename": text_value(group.attrs["source_filename"]),
        "time": float(group.attrs["simulation_time"]),
        "dx": float(group.attrs["dx"]),
        "x": group["x"][:],
        "y": group["y"][:],
        "vorticity": group["vorticity"][:],
    }


def draw_hmaxima(axis, group: h5py.Group, frame: dict, config: dict) -> None:
    color = str(config["plot"].get("mask_color", "#ffd400"))
    colormap = mcolors.ListedColormap([(0, 0, 0, 0), mcolors.to_rgba(color)])
    axis.imshow(
        group["hmax_mask"][:],
        origin="lower",
        extent=image_extent(frame),
        interpolation="nearest",
        cmap=colormap,
        vmin=0,
        vmax=1,
        alpha=float(config["plot"].get("mask_alpha", 0.35)),
    )
    axis.scatter(
        group["peak_x"][:],
        group["peak_y"][:],
        s=float(config["plot"].get("marker_size", 30.0)),
        c=str(config["plot"].get("marker_color", "black")),
        marker="x",
    )


def draw_regions(axis, group: h5py.Group, config: dict) -> None:
    color = str(config["plot"].get("region_color", "#00aa55"))
    width = float(config["plot"].get("region_line_width", 1.5))
    for bounds in group["clamped_bounds"][:]:
        x0, x1, y0, y1 = bounds
        axis.add_patch(Rectangle(
            (x0, y0), x1 - x0, y1 - y0,
            fill=False, edgecolor=color, linewidth=width,
        ))


def draw_fits(axis, group: h5py.Group, config: dict) -> None:
    positive_color = str(config["plot"].get("positive_marker_color", "black"))
    negative_color = str(config["plot"].get("negative_marker_color", "#7b2cbf"))
    width = float(config["plot"].get("region_line_width", 1.5))
    radii = group["boundary_radius"][:]
    index = largest_successful_fit_index(group["success"][:], radii)
    if index is None:
        return
    radius = radii[index]
    positive = group["positive_centers"][index]
    negative = group["negative_centers"][index]
    if np.all(np.isfinite(positive)) and np.all(np.isfinite(negative)):
        axis.add_patch(Circle(positive, radius, fill=False, edgecolor=positive_color, linewidth=width))
        axis.add_patch(Circle(negative, radius, fill=False, edgecolor=negative_color, linewidth=width))


def draw_threshold_masks(axis, group: h5py.Group, frame: dict, config: dict, handle: h5py.File) -> None:
    positive_color = str(config.get("threshold_mask", {}).get("positive_color", "#ffb000"))
    negative_color = str(config.get("threshold_mask", {}).get("negative_color", "#00a6ff"))
    positive = group["positive_mask"][:].astype(bool)
    negative = group["negative_mask"][:].astype(bool)
    colors = np.ones((*positive.shape, 3), dtype=float)
    colors[positive] = mcolors.to_rgb(positive_color)
    colors[negative] = mcolors.to_rgb(negative_color)
    axis.images[0].set_visible(False)
    axis.imshow(colors, origin="lower", extent=image_extent(frame), interpolation="nearest", aspect="equal")
    if "extrema_x" in group and "extrema_y" in group:
        axis.scatter(
            group["extrema_x"][:],
            group["extrema_y"][:],
            s=float(config["plot"].get("marker_size", 30.0)),
            c=str(config["plot"].get("marker_color", "black")),
            marker="x",
        )
    threshold = float(handle.attrs["vorticity_threshold"])
    axis.legend(handles=(
        Patch(facecolor=positive_color, label=f"ω ≥ {threshold:g}"),
        Patch(facecolor=negative_color, label=f"ω ≤ {-threshold:g}"),
    ), loc="upper right")
    axis.set_title(
        f"Vorticity threshold masks | {frame['source_filename']} | t = {frame['time']:.8g}"
    )


def draw_threshold_extrema(axis, group: h5py.Group, frame: dict, config: dict) -> None:
    """Overlay threshold-retained h-maxima on the vorticity field."""
    if "extrema_x" in group and "extrema_y" in group:
        axis.scatter(
            group["extrema_x"][:],
            group["extrema_y"][:],
            s=float(config["plot"].get("marker_size", 30.0)),
            c=str(config["plot"].get("marker_color", "black")),
            marker="x",
        )
    axis.set_title(
        f"Vorticity field with retained h-maxima | {frame['source_filename']} | "
        f"t = {frame['time']:.8g}"
    )


def render_frame(
    output_path: Path,
    schema: str,
    handle: h5py.File,
    background: Optional[h5py.File],
    group_name: str,
    config: dict,
    threshold_vorticity_background: bool,
) -> None:
    group = handle[group_name]
    source = background[group_name] if background is not None else group
    frame = base_frame(source)
    plot_config = config["plot"]
    figure, axis = plt.subplots(figsize=(
        float(plot_config.get("figure_width", 10.0)),
        float(plot_config.get("figure_height", 7.0)),
    ))
    image = plot_vorticity_frame(axis, frame, plot_config)

    if schema == "ritta_hmaxima_v1":
        draw_hmaxima(axis, group, frame, config)
    elif schema == "ritta_regions_v1":
        draw_regions(axis, group, config)
    elif schema == "ritta_circular_gaussian_dipole_fits_v1":
        draw_fits(axis, group, config)
    elif (
        schema in {"ritta_vorticity_threshold_masks_v1", "ritta_vorticity_threshold_masks_v2"}
        and threshold_vorticity_background
    ):
        draw_threshold_extrema(axis, group, frame, config)
    elif schema in {"ritta_vorticity_threshold_masks_v1", "ritta_vorticity_threshold_masks_v2"}:
        draw_threshold_masks(axis, group, frame, config, handle)

    if (
        schema not in {"ritta_vorticity_threshold_masks_v1", "ritta_vorticity_threshold_masks_v2"}
        or threshold_vorticity_background
    ):
        figure.colorbar(image, ax=axis, label="vorticity")
    figure.tight_layout()
    figure.savefig(output_path, dpi=120)
    plt.close(figure)


def save_gif(frame_paths: List[Path], output_path: Path, duration_ms: int) -> None:
    """Load the generated PNG files and save a looping GIF."""
    images = []
    for path in frame_paths:
        with Image.open(path) as image:
            images.append(image.convert("P", palette=Image.Palette.ADAPTIVE))
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
        disposal=2,
    )
    for image in images:
        image.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Render vortex-analysis HDF5 output to PNG frames and a GIF.")
    parser.add_argument("h5_file", type=Path)
    parser.add_argument("stride", type=int)
    parser.add_argument("--duration-ms", type=int, default=150, help="Display time per GIF frame.")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--x-axis-min", type=float, help="Override the saved plot's lower x-axis limit.")
    parser.add_argument("--x-axis-max", type=float, help="Override the saved plot's upper x-axis limit.")
    parser.add_argument(
        "--threshold-vorticity-background",
        action="store_true",
        help="For threshold_masks.h5, show vorticity behind the retained extrema instead of the mask.",
    )
    args = parser.parse_args()

    if args.stride <= 0:
        parser.error("stride must be a positive integer")
    if args.duration_ms <= 0:
        parser.error("--duration-ms must be a positive integer")
    if args.x_axis_min is not None and not math.isfinite(args.x_axis_min):
        parser.error("--x-axis-min must be finite")
    if args.x_axis_max is not None and not math.isfinite(args.x_axis_max):
        parser.error("--x-axis-max must be finite")
    input_path = args.h5_file.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"HDF5 file does not exist: {input_path}")
    output_folder = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else input_path.parent / f"{input_path.stem}_stride_{args.stride}"
    )
    frame_folder = output_folder / "frames"
    frame_folder.mkdir(parents=True, exist_ok=True)

    with h5py.File(input_path, "r") as handle:
        schema = text_value(handle.attrs.get("schema", ""))
        if schema not in SUPPORTED_SCHEMAS:
            supported = ", ".join(sorted(SUPPORTED_SCHEMAS))
            raise ValueError(f"Unsupported HDF5 schema '{schema}'. Supported schemas: {supported}")
        if args.threshold_vorticity_background and schema not in {
            "ritta_vorticity_threshold_masks_v1",
            "ritta_vorticity_threshold_masks_v2",
        }:
            parser.error("--threshold-vorticity-background requires a threshold_masks.h5 input")
        config = saved_config(handle)
        if args.x_axis_min is not None:
            config["plot"]["x_axis_min"] = args.x_axis_min
        if args.x_axis_max is not None:
            config["plot"]["x_axis_max"] = args.x_axis_max
        x_axis_min = float(config["plot"].get("x_axis_min", math.nan))
        x_axis_max = float(config["plot"].get("x_axis_max", math.nan))
        if (
            math.isfinite(x_axis_min)
            and math.isfinite(x_axis_max)
            and x_axis_min >= x_axis_max
        ):
            parser.error("the effective x-axis minimum must be smaller than the maximum")
        group_names = read_frame_order(handle)
        selected = list(enumerate(group_names))[::args.stride]
        if not selected:
            raise ValueError("The HDF5 file contains no saved frames.")

        background_path = background_file(input_path, schema)
        background = h5py.File(background_path, "r") if background_path is not None else None
        try:
            if background is not None and read_frame_order(background) != group_names:
                raise ValueError(f"Frame order differs between {input_path.name} and {background_path.name}.")
            frame_paths = []
            for output_index, (source_index, group_name) in enumerate(selected):
                frame_path = frame_folder / f"frame_{source_index:06d}_{group_name}.png"
                render_frame(
                    frame_path,
                    schema,
                    handle,
                    background,
                    group_name,
                    config,
                    args.threshold_vorticity_background,
                )
                frame_paths.append(frame_path)
                print(f"[{output_index + 1}/{len(selected)}] Saved {frame_path}")
        finally:
            if background is not None:
                background.close()

    gif_stem = (
        f"{input_path.stem}_vorticity"
        if args.threshold_vorticity_background
        else input_path.stem
    )
    gif_path = output_folder / f"{gif_stem}.gif"
    save_gif(frame_paths, gif_path, args.duration_ms)
    print(f"Saved GIF: {gif_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
