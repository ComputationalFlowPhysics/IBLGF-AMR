"""Reusable physical-coordinate vorticity plotting and frame browsing."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import discover_frames, load_config, load_vorticity_frame, result_folder, simulation_metadata


def image_extent(frame: dict) -> tuple[float, float, float, float]:
    """Convert cell-center coordinates to the outer physical image edges."""
    dx = float(frame["dx"])
    return (
        float(frame["x"][0] - 0.5 * dx),
        float(frame["x"][-1] + 0.5 * dx),
        float(frame["y"][0] - 0.5 * dx),
        float(frame["y"][-1] + 0.5 * dx),
    )


def plot_vorticity_frame(ax, frame: dict, plot_config: dict):
    """Plot one cell-centered vorticity array in physical coordinates."""
    omega = np.asarray(frame["vorticity"], dtype=float)
    finite = omega[np.isfinite(omega)]
    if not finite.size:
        raise ValueError(f"{frame['source_filename']} contains no finite vorticity values.")

    fixed_limit = float(plot_config.get("color_limit", np.nan))
    symmetric = bool(plot_config.get("symmetric_color_limits", True))
    # Prefer a configured limit; otherwise derive readable limits from this frame.
    if np.isfinite(fixed_limit) and fixed_limit > 0.0:
        vmin, vmax = (-fixed_limit, fixed_limit) if symmetric else (float(np.min(finite)), fixed_limit)
    elif symmetric:
        limit = float(np.max(np.abs(finite)))
        vmin, vmax = -limit, limit
    else:
        vmin, vmax = float(np.min(finite)), float(np.max(finite))

    image = ax.imshow(
        np.ma.masked_invalid(omega),
        origin="lower",
        extent=image_extent(frame),
        interpolation="nearest",
        cmap=str(plot_config.get("colormap", "RdBu_r")),
        vmin=vmin,
        vmax=vmax,
        aspect="equal",
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Vorticity field | {frame['source_filename']} | t = {frame['time']:.8g}")
    return image


def browse_frames(
    frame_count: int,
    frame_loader,
    plot_config: dict,
    overlay_drawer=None,
    preview_path: str | Path = "frame_preview.png",
) -> None:
    """Prompt for frames in the terminal and render the selection to one PNG."""
    if frame_count < 1:
        return
    width = float(plot_config.get("figure_width", 10.0))
    height = float(plot_config.get("figure_height", 7.0))
    preview_path = Path(preview_path).expanduser().resolve()
    preview_path.parent.mkdir(parents=True, exist_ok=True)

    def render(index: int) -> dict:
        # Each selection replaces one lightweight PNG instead of opening a GUI window.
        frame = frame_loader(index)
        figure, axis = plt.subplots(figsize=(width, height))
        image = plot_vorticity_frame(axis, frame, plot_config)
        if overlay_drawer is not None:
            overlay_drawer(axis, frame)
        figure.colorbar(image, ax=axis, label="vorticity")
        figure.tight_layout()
        figure.savefig(preview_path, dpi=150)
        plt.close(figure)
        print(
            f"Frame {index}/{frame_count - 1}: {frame['source_filename']} | "
            f"t = {frame['time']:.8g}\nPreview: {preview_path}"
        )
        return frame

    index = 0
    while True:
        render(index)
        try:
            command = input(f"Frame 0-{frame_count - 1} [n=next, p=previous, q=quit]: ").strip().lower()
        except EOFError:
            print("No interactive terminal input is available; leaving the current preview saved.")
            return
        if command in {"q", "quit", "exit"}:
            return
        if command in {"", "n", "next"}:
            index = min(frame_count - 1, index + 1)
        elif command in {"p", "prev", "previous"}:
            index = max(0, index - 1)
        else:
            try:
                requested = int(command)
            except ValueError:
                print("Enter n, p, q, or a frame number.")
                continue
            if not 0 <= requested < frame_count:
                print(f"Frame number must be between 0 and {frame_count - 1}.")
                continue
            index = requested


def main() -> int:
    parser = argparse.ArgumentParser(description="Browse standalone IBLGF vorticity output.")
    parser.add_argument("run_folder", type=Path)
    parser.add_argument("config_file", type=Path)
    args = parser.parse_args()

    config = load_config(args.config_file)
    frames = discover_frames(args.run_folder, config)
    metadata = simulation_metadata(args.run_folder, config)

    print(f"Checking {len(frames)} vorticity frames...")
    # Check every input first so the prompt never starts with a partly validated run.
    for index, path in enumerate(frames):
        load_vorticity_frame(path, index, config, metadata)
        print(f"[{index + 1}/{len(frames)}] {path.name}")
    print("Batch check complete. Starting terminal frame prompt.")

    def load(index: int) -> dict:
        return load_vorticity_frame(frames[index], index, config, metadata)

    browse_frames(
        len(frames),
        load,
        config["plot"],
        preview_path=result_folder(args.run_folder) / "vorticity_preview.png",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
