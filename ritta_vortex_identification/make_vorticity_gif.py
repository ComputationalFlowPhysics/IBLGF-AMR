"""Render raw IBLGF vorticity frames to PNG files and a looping GIF."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import (
    discover_frames,
    load_config,
    load_vorticity_frame,
    result_folder,
    simulation_metadata,
)
from make_h5_gif import save_gif
from plot_vorticity import plot_vorticity_frame


def render_frame(output_path: Path, frame: dict, plot_config: dict) -> None:
    """Render one raw vorticity frame with the standalone plot styling."""
    figure, axis = plt.subplots(figsize=(
        float(plot_config.get("figure_width", 10.0)),
        float(plot_config.get("figure_height", 7.0)),
    ))
    image = plot_vorticity_frame(axis, frame, plot_config)
    figure.colorbar(image, ax=axis, label="vorticity")
    figure.tight_layout()
    figure.savefig(output_path, dpi=120)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render raw edge_aux vorticity frames to PNG files and a GIF."
    )
    parser.add_argument("run_folder", type=Path)
    parser.add_argument("config_file", type=Path)
    parser.add_argument("--stride", type=int, default=1, help="Render every Nth sorted HDF5 frame.")
    parser.add_argument("--duration-ms", type=int, default=150, help="Display time per GIF frame.")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    if args.stride < 1:
        parser.error("--stride must be a positive integer.")
    if args.duration_ms < 1:
        parser.error("--duration-ms must be a positive integer.")

    config = load_config(args.config_file)
    frames = discover_frames(args.run_folder, config)
    metadata = simulation_metadata(args.run_folder, config)
    selected = list(enumerate(frames))[::args.stride]
    output_folder = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else result_folder(args.run_folder) / f"vorticity_stride_{args.stride}"
    )
    frame_folder = output_folder / "frames"
    frame_folder.mkdir(parents=True, exist_ok=True)

    frame_paths = []
    for output_index, (source_index, source_path) in enumerate(selected, start=1):
        frame = load_vorticity_frame(source_path, source_index, config, metadata)
        frame_path = frame_folder / f"frame_{source_index:06d}_{source_path.stem}.png"
        render_frame(frame_path, frame, config["plot"])
        frame_paths.append(frame_path)
        print(
            f"[{output_index}/{len(selected)}] Saved {frame_path} "
            f"(step {frame['step']}, t = {frame['time']:.8g})"
        )

    gif_path = output_folder / "vorticity.gif"
    save_gif(frame_paths, gif_path, args.duration_ms)
    print(f"Saved GIF: {gif_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
