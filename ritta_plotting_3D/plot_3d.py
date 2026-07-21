#!/usr/bin/env pvpython
"""Render Q-criterion PNG frames and combine them into a GIF.

Usage:
    pvpython plot_3d.py OUTPUT_FOLDER [STRIDE]

``OUTPUT_FOLDER`` may be the folder containing ``flowTime_*.hdf5`` files or
the run folder containing an ``output`` subfolder. Generated files are saved
under ``ritta_plotting_3D/outputs``.
"""

import argparse
import csv
import re
import shutil
import subprocess
import sys
from pathlib import Path


SNAPSHOT_PATTERN = re.compile(r"flowTime_(\d+)\.hdf5$")
VELOCITY_COMPONENTS = ("u_0", "u_1", "u_2")

# Rendering settings. Edit these values to change the visualization.
CONTOUR_VALUE = 2.5
FPS = 8
IMAGE_RESOLUTION = [1280, 720]
CAMERA_POSITION = [17.139627675282647, 11.400499110766626, -22.814391556735938]
CAMERA_FOCAL_POINT = [3.0625, 0.0, 0.0]
CAMERA_VIEW_UP = [-0.12585964286795603, 0.9162931489586358, 0.38021864166373776]
CAMERA_PARALLEL_SCALE = 7.539738473581163


def positive_integer(value):
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("stride must be an integer") from error
    if number < 1:
        raise argparse.ArgumentTypeError("stride must be at least 1")
    return number


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render 3D Q-criterion PNG frames and an animated GIF."
    )
    parser.add_argument(
        "output_folder",
        type=Path,
        help=(
            "folder containing flowTime_*.hdf5 snapshots, or a run folder "
            "containing an output subfolder"
        ),
    )
    parser.add_argument(
        "stride",
        nargs="?",
        type=positive_integer,
        default=1,
        help="render every STRIDE-th snapshot (default: 1)",
    )
    return parser.parse_args()


def snapshot_step(path):
    match = SNAPSHOT_PATTERN.fullmatch(path.name)
    if match is None:
        raise ValueError(f"Not a flowTime snapshot: {path.name}")
    return int(match.group(1))


def find_snapshot_folder(folder):
    folder = folder.expanduser().resolve()
    if not folder.is_dir():
        raise ValueError(f"Input folder does not exist: {folder}")

    if any(SNAPSHOT_PATTERN.fullmatch(path.name) for path in folder.iterdir()):
        return folder

    nested_output = folder / "output"
    if nested_output.is_dir() and any(
        SNAPSHOT_PATTERN.fullmatch(path.name) for path in nested_output.iterdir()
    ):
        return nested_output

    raise ValueError(
        "No flowTime_<step>.hdf5 snapshots found in "
        f"{folder} or {nested_output}"
    )


def discover_snapshots(snapshot_folder, stride):
    snapshots = [
        path
        for path in snapshot_folder.iterdir()
        if path.is_file() and SNAPSHOT_PATTERN.fullmatch(path.name)
    ]
    snapshots.sort(key=snapshot_step)
    if not snapshots:
        raise ValueError(f"No snapshots found in {snapshot_folder}")
    return snapshots[::stride]


def output_paths(snapshot_folder):
    run_name = (
        snapshot_folder.parent.name
        if snapshot_folder.name == "output"
        else snapshot_folder.name
    )
    output_folder = Path(__file__).resolve().parent / "outputs" / f"{run_name}_qcriterion"
    frames_folder = output_folder / "frames"
    gif_path = output_folder / f"{run_name}_qcriterion.gif"
    manifest_path = output_folder / "frame_manifest.csv"
    return output_folder, frames_folder, gif_path, manifest_path


def prepare_output_folders(output_folder, frames_folder):
    output_folder.mkdir(parents=True, exist_ok=True)
    frames_folder.mkdir(parents=True, exist_ok=True)

    # Remove only frames produced by an earlier invocation of this script.
    for old_frame in frames_folder.iterdir():
        if old_frame.is_file() and re.fullmatch(r"flowTime_\d+\.png", old_frame.name):
            old_frame.unlink()


def load_paraview():
    try:
        from paraview import simple
    except ImportError as error:
        raise RuntimeError(
            "ParaView's Python module is unavailable. Run this script with "
            "pvpython or pvbatch, not a regular Python interpreter."
        ) from error
    return simple


def render_snapshot(simple, snapshot, png_path):
    # Start every frame from a completely empty ParaView session. In
    # particular, do not reuse a reader or render view across snapshots.
    simple.ResetSession()
    simple._DisableFirstRenderCameraReset()

    source = simple.VisItChomboReader(
        registrationName=snapshot.name,
        FileName=[str(snapshot)],
    )
    source.CellArrayStatus = list(VELOCITY_COMPONENTS)
    source.UpdatePipeline()

    cell_data = source.GetCellDataInformation()
    missing_components = [
        name
        for name in VELOCITY_COMPONENTS
        if cell_data.GetArray(name) is None
    ]
    if missing_components:
        raise RuntimeError(
            f"{snapshot} is missing required cell arrays: "
            f"{', '.join(missing_components)}"
        )

    source_info = source.GetDataInformation()
    source_cells = source_info.GetNumberOfCells()
    source_points = source_info.GetNumberOfPoints()

    cell_to_point = simple.CellDatatoPointData(
        registrationName="CellDataToPointData",
        Input=source,
    )
    cell_to_point.UpdatePipeline()

    velocity = simple.MergeVectorComponents(
        registrationName="Velocity",
        Input=cell_to_point,
    )
    velocity.XArray = "u_0"
    velocity.YArray = "u_1"
    velocity.ZArray = "u_2"
    velocity.OutputVectorName = "Velocity"
    velocity.UpdatePipeline()

    gradient = simple.Gradient(registrationName="VelocityGradient", Input=velocity)
    gradient.ScalarArray = ["POINTS", "Velocity"]
    gradient.ComputeQCriterion = 1
    gradient.QCriterionArrayName = "Q Criterion"
    gradient.UpdatePipeline()

    contour = simple.Contour(registrationName="QCriterionContour", Input=gradient)
    contour.ContourBy = ["POINTS", "Q Criterion"]
    contour.Isosurfaces = [CONTOUR_VALUE]
    contour.UpdatePipeline()

    render_view = simple.GetActiveViewOrCreate("RenderView")
    contour_display = simple.Show(contour, render_view, "GeometryRepresentation")
    contour_display.Representation = "Surface"
    render_view.Update()

    render_view.CameraPosition = CAMERA_POSITION
    render_view.CameraFocalPoint = CAMERA_FOCAL_POINT
    render_view.CameraViewUp = CAMERA_VIEW_UP
    render_view.CameraParallelScale = CAMERA_PARALLEL_SCALE
    render_view.ViewSize = IMAGE_RESOLUTION

    simple.Render(render_view)
    simple.SaveScreenshot(
        str(png_path),
        render_view,
        ImageResolution=IMAGE_RESOLUTION,
    )

    if not png_path.is_file() or png_path.stat().st_size == 0:
        raise RuntimeError(f"ParaView did not create a valid PNG: {png_path}")

    # ParaView can retain representations and reader caches until their
    # proxies are explicitly unregistered. Clear the view, delete the entire
    # pipeline in reverse order, and reset once more before the next frame.
    simple.HideAll(render_view)
    render_view.Update()
    for proxy in (contour, gradient, velocity, cell_to_point, source):
        simple.Delete(proxy)
    simple.RemoveViewsAndLayouts()
    simple.ResetSession()

    return source_cells, source_points


def build_gif(frames_folder, snapshots, gif_path):
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            "ffmpeg was not found. PNG frames were saved, but the GIF could "
            "not be created. Install ffmpeg and rerun the script."
        )

    staging_folder = gif_path.parent / "_gif_frames"
    shutil.rmtree(staging_folder, ignore_errors=True)
    staging_folder.mkdir(parents=True)

    try:
        for frame_index, snapshot in enumerate(snapshots):
            source_png = frames_folder / f"flowTime_{snapshot_step(snapshot)}.png"
            shutil.copy2(source_png, staging_folder / f"frame_{frame_index:05d}.png")

        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-framerate",
                str(FPS),
                "-i",
                str(staging_folder / "frame_%05d.png"),
                "-filter_complex",
                (
                    "[0:v]split[gif][palette_source];"
                    "[palette_source]palettegen=stats_mode=diff[palette];"
                    "[gif][palette]paletteuse=dither=sierra2_4a"
                ),
                "-loop",
                "0",
                str(gif_path),
            ],
            check=True,
        )
    finally:
        shutil.rmtree(staging_folder, ignore_errors=True)

    if not gif_path.is_file() or gif_path.stat().st_size == 0:
        raise RuntimeError(f"ffmpeg did not create a valid GIF: {gif_path}")


def main():
    args = parse_args()

    try:
        snapshot_folder = find_snapshot_folder(args.output_folder)
        snapshots = discover_snapshots(snapshot_folder, args.stride)
        output_folder, frames_folder, gif_path, manifest_path = output_paths(
            snapshot_folder
        )
        prepare_output_folders(output_folder, frames_folder)
        simple = load_paraview()

        print(f"Snapshot folder: {snapshot_folder}", flush=True)
        print(f"Snapshots used:  {len(snapshots)}", flush=True)
        print(f"Stride:          {args.stride}", flush=True)
        print(f"PNG folder:      {frames_folder}", flush=True)
        print(f"Output GIF:      {gif_path}", flush=True)

        with manifest_path.open("w", newline="") as manifest_file:
            manifest_writer = csv.writer(manifest_file)
            manifest_writer.writerow(
                [
                    "frame_index",
                    "snapshot_step",
                    "snapshot_file",
                    "source_cells",
                    "source_points",
                    "png_file",
                ]
            )

            for frame_index, snapshot in enumerate(snapshots):
                step = snapshot_step(snapshot)
                png_path = frames_folder / f"flowTime_{step}.png"
                print(
                    f"[{frame_index + 1}/{len(snapshots)}] Rendering "
                    f"{snapshot.name}",
                    flush=True,
                )
                source_cells, source_points = render_snapshot(
                    simple, snapshot.resolve(), png_path
                )
                manifest_writer.writerow(
                    [
                        frame_index,
                        step,
                        str(snapshot.resolve()),
                        source_cells,
                        source_points,
                        str(png_path.resolve()),
                    ]
                )
                manifest_file.flush()

        print("All PNG frames rendered. Building GIF...", flush=True)
        build_gif(frames_folder, snapshots, gif_path)
        print(f"Done: {gif_path}", flush=True)
        print(f"PNG frames: {frames_folder}", flush=True)
        print(f"Manifest: {manifest_path}", flush=True)
        return 0
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        print(f"Error: {error}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
