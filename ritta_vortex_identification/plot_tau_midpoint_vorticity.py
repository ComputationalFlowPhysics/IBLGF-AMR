"""Compare selected-time vorticity fields from several tau-sweep runs."""

import argparse
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import numpy as np

from common import (
    discover_frames,
    load_config,
    load_vorticity_frame,
    simulation_metadata,
    simulation_parameter,
)
from plot_vorticity import image_extent


DEFAULT_TAU_VALUES = (1.0, 8.0, 30.0)
DEFAULT_CONTOUR_MAGNITUDES = (0.35, 1.0, 2.0, 4.0, 8.0)


def validate_positive_values(values: Sequence[float], name: str) -> Tuple[float, ...]:
    """Return unique, increasing, positive finite values."""
    numbers = tuple(float(value) for value in values)
    if not numbers:
        raise ValueError(f"{name} must contain at least one value.")
    if any(not math.isfinite(value) or value <= 0.0 for value in numbers):
        raise ValueError(f"{name} must contain only positive finite values.")
    if len(set(numbers)) != len(numbers):
        raise ValueError(f"{name} must not contain duplicate values.")
    return tuple(sorted(numbers))


def selected_frame_index(frame_count: int, frame_fraction: float) -> int:
    """Choose the nearest saved-sequence position, breaking ties earlier."""
    if frame_count < 1:
        raise ValueError("At least one frame is required.")
    if not math.isfinite(frame_fraction) or not 0.0 <= frame_fraction <= 1.0:
        raise ValueError("The frame fraction must be finite and between 0 and 1.")
    target = frame_fraction * (frame_count - 1)
    return min(frame_count - 1, int(math.floor(target + 0.5 - 1.0e-12)))


def midpoint_frame_index(frame_count: int) -> int:
    """Match the existing previews by choosing the lower central frame."""
    return selected_frame_index(frame_count, 0.5)


def panel_grid_shape(
    panel_count: int,
    columns: Optional[int] = None,
) -> Tuple[int, int]:
    """Return a compact grid, using one row when columns is not specified."""
    if panel_count < 1:
        raise ValueError("At least one panel is required.")
    if columns is None:
        columns = panel_count
    if columns < 1:
        raise ValueError("The column count must be positive.")
    columns = min(columns, panel_count)
    rows = math.ceil(panel_count / columns)
    return rows, columns


def resolve_tau_runs(
    campaign_dir: Path,
    config: dict,
    tau_values: Sequence[float],
) -> List[Tuple[float, Path]]:
    """Match requested tau values to immediate child run folders."""
    discovered = []
    for run_dir in sorted(path for path in campaign_dir.iterdir() if path.is_dir()):
        if not (run_dir / "output").is_dir():
            continue
        try:
            tau = simulation_parameter(run_dir, config, "b_f_tau")
        except (FileNotFoundError, ValueError):
            continue
        discovered.append((tau, run_dir))

    selected = []
    for requested_tau in tau_values:
        matches = [
            (tau, run_dir)
            for tau, run_dir in discovered
            if math.isclose(tau, requested_tau, rel_tol=1.0e-12, abs_tol=1.0e-12)
        ]
        if not matches:
            available = ", ".join(f"{tau:g}" for tau, _ in discovered) or "none"
            raise ValueError(
                f"No run with tau={requested_tau:g} was found in {campaign_dir}. "
                f"Available values: {available}"
            )
        if len(matches) > 1:
            folders = ", ".join(str(run_dir) for _, run_dir in matches)
            raise ValueError(f"Multiple runs have tau={requested_tau:g}: {folders}")
        selected.append(matches[0])
    return selected


def load_selected_panel(task) -> dict:
    """Load one run's selected edge_aux frame in a worker process."""
    tau, run_dir, config_file, frame_fraction = task
    config = load_config(config_file)
    frames = discover_frames(run_dir, config)
    frame_index = selected_frame_index(len(frames), frame_fraction)
    metadata = simulation_metadata(run_dir, config)
    frame = load_vorticity_frame(frames[frame_index], frame_index, config, metadata)
    return {
        "tau": tau,
        "run_dir": str(run_dir),
        "frame_index": frame_index,
        "frame_count": len(frames),
        "frame": frame,
    }


def load_panels(
    selected_runs: Sequence[Tuple[float, Path]],
    config_file: Path,
    frame_fraction: float,
    workers: int,
) -> List[dict]:
    tasks = [
        (tau, run_dir, config_file, frame_fraction)
        for tau, run_dir in selected_runs
    ]
    if workers == 1:
        return [load_selected_panel(task) for task in tasks]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(load_selected_panel, tasks))


def save_comparison(
    panels: Sequence[dict],
    output_path: Path,
    *,
    contour_magnitudes: Sequence[float],
    color_limit: float,
    x_limits: Tuple[float, float],
    y_limits: Tuple[float, float],
    colormap: str,
    columns: Optional[int],
    frame_fraction: float,
    dpi: int,
) -> None:
    """Render a shared-scale vorticity comparison in the requested grid."""
    rows, columns = panel_grid_shape(len(panels), columns)
    figure, axes_grid = plt.subplots(
        rows,
        columns,
        figsize=(5.4 * columns + 0.8, 4.4 * rows),
        sharex=True,
        sharey=True,
        squeeze=False,
        layout="constrained",
    )
    axes = axes_grid.ravel()
    active_axes = axes[:len(panels)]
    for axis in axes[len(panels):]:
        axis.set_visible(False)
    normalization = Normalize(vmin=-color_limit, vmax=color_limit)
    positive_levels = np.asarray(contour_magnitudes, dtype=float)
    negative_levels = -positive_levels[::-1]
    image = None

    for panel_index, (axis, panel) in enumerate(zip(active_axes, panels)):
        frame = panel["frame"]
        omega = np.ma.masked_invalid(np.asarray(frame["vorticity"], dtype=float))
        image = axis.imshow(
            omega,
            origin="lower",
            extent=image_extent(frame),
            interpolation="nearest",
            cmap=colormap,
            norm=normalization,
            aspect="equal",
        )
        axis.contour(
            frame["x"],
            frame["y"],
            omega,
            levels=positive_levels,
            colors="black",
            linewidths=0.8,
            linestyles="solid",
        )
        axis.contour(
            frame["x"],
            frame["y"],
            omega,
            levels=negative_levels,
            colors="black",
            linewidths=0.8,
            linestyles="dashed",
        )
        panel_letter = chr(ord("a") + panel_index)
        axis.set_title(
            rf"({panel_letter}) $\tau={panel['tau']:g}$, $t={frame['time']:.3g}$"
        )
        axis.set_xlim(*x_limits)
        axis.set_ylim(*y_limits)
        axis.set_xlabel("x")
        if panel_index % columns == 0:
            axis.set_ylabel("y")

    active_axes[0].legend(
        handles=[
            Line2D([0], [0], color="black", linewidth=0.9, label=r"$\omega>0$"),
            Line2D(
                [0],
                [0],
                color="black",
                linewidth=0.9,
                linestyle="--",
                label=r"$\omega<0$",
            ),
        ],
        loc="upper left",
        framealpha=0.9,
    )
    position_label = (
        "Midpoint vorticity fields"
        if math.isclose(frame_fraction, 0.5, rel_tol=0.0, abs_tol=1.0e-12)
        else f"Vorticity fields at {100.0 * frame_fraction:g}% of each saved sequence"
    )
    figure.suptitle(f"{position_label} and absolute-vorticity contours")
    colorbar = figure.colorbar(image, ax=active_axes.tolist(), pad=0.02, shrink=0.92)
    colorbar.set_label(r"vorticity $\omega$")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot selected tau-sweep vorticity fields in a shared grid."
    )
    parser.add_argument("campaign_dir", type=Path)
    parser.add_argument("config_file", type=Path)
    parser.add_argument("--taus", nargs="+", type=float, default=DEFAULT_TAU_VALUES)
    parser.add_argument(
        "--contour-magnitudes",
        nargs="+",
        type=float,
        default=DEFAULT_CONTOUR_MAGNITUDES,
    )
    parser.add_argument("--color-limit", type=float, default=12.0)
    parser.add_argument("--x-limits", nargs=2, type=float, default=(-1.0, 5.0))
    parser.add_argument("--y-limits", nargs=2, type=float, default=(-1.5, 1.5))
    parser.add_argument("--colormap", default="RdBu_r")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument(
        "--frame-fraction",
        type=float,
        default=0.5,
        help="Saved-sequence position from 0 (first) to 1 (last); default: 0.5.",
    )
    parser.add_argument(
        "--columns",
        type=int,
        help="Number of panel columns; defaults to one horizontal row.",
    )
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent
        / "outputs"
        / "paper_previews"
        / "tau_1_8_30_midpoint_vorticity_contours_horizontal.png",
    )
    args = parser.parse_args()

    try:
        tau_values = validate_positive_values(args.taus, "--taus")
        contour_magnitudes = validate_positive_values(
            args.contour_magnitudes,
            "--contour-magnitudes",
        )
    except ValueError as error:
        parser.error(str(error))
    if not math.isfinite(args.color_limit) or args.color_limit <= 0.0:
        parser.error("--color-limit must be positive and finite")
    if contour_magnitudes[-1] >= args.color_limit:
        parser.error("Every contour magnitude must be smaller than --color-limit")
    if args.workers < 1:
        parser.error("--workers must be positive")
    if not math.isfinite(args.frame_fraction) or not 0.0 <= args.frame_fraction <= 1.0:
        parser.error("--frame-fraction must be finite and between 0 and 1")
    if args.columns is not None and args.columns < 1:
        parser.error("--columns must be positive")
    if args.dpi < 1:
        parser.error("--dpi must be positive")
    if not all(math.isfinite(value) for value in (*args.x_limits, *args.y_limits)):
        parser.error("Axis limits must be finite")
    if args.x_limits[0] >= args.x_limits[1]:
        parser.error("--x-limits minimum must be smaller than maximum")
    if args.y_limits[0] >= args.y_limits[1]:
        parser.error("--y-limits minimum must be smaller than maximum")

    campaign_dir = args.campaign_dir.expanduser().resolve()
    config_file = args.config_file.expanduser().resolve()
    if not campaign_dir.is_dir():
        parser.error(f"Campaign directory does not exist: {campaign_dir}")
    if not config_file.is_file():
        parser.error(f"Plot configuration does not exist: {config_file}")

    config = load_config(config_file)
    selected_runs = resolve_tau_runs(campaign_dir, config, tau_values)
    worker_count = min(args.workers, len(selected_runs))
    panels = load_panels(
        selected_runs,
        config_file,
        args.frame_fraction,
        worker_count,
    )
    output_path = args.output.expanduser().resolve()
    save_comparison(
        panels,
        output_path,
        contour_magnitudes=contour_magnitudes,
        color_limit=args.color_limit,
        x_limits=tuple(args.x_limits),
        y_limits=tuple(args.y_limits),
        colormap=args.colormap,
        columns=args.columns,
        frame_fraction=args.frame_fraction,
        dpi=args.dpi,
    )

    signed_levels = [-value for value in reversed(contour_magnitudes)]
    signed_levels.extend(contour_magnitudes)
    print(f"Selected frames near {100.0 * args.frame_fraction:g}% of each sequence:")
    for panel in panels:
        frame = panel["frame"]
        print(
            f"  tau={panel['tau']:g}: frame {panel['frame_index']} of "
            f"{panel['frame_count'] - 1}, {frame['source_filename']}, "
            f"t={frame['time']:.8g}"
        )
    print("Contour levels: " + ", ".join(f"{value:g}" for value in signed_levels))
    print(f"Shared color limits: {-args.color_limit:g}, {args.color_limit:g}")
    print(f"Saved {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
