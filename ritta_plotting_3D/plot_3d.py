#!/usr/bin/env pvpython
"""Render 3D vortex-diagnostic PNG frames and combine them into a GIF.

Usage:
    pvpython plot_3d.py OUTPUT_FOLDER [STRIDE]
        [--field {q-criterion,vorticity}]
        [--vorticity-threshold-fraction FRACTION]
        [--output-dir FOLDER] [--resume]

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
VORTICITY_COMPONENTS = ("edge_aux_0", "edge_aux_1", "edge_aux_2")
BRIDGES_FFMPEG_IMAGE = Path(
    "/opt/packages/ffmpeg/4.3.1/singularity-ffmpeg-4.3.1.sif"
)

# Rendering settings. Edit these values to change the visualization.
Q_CRITERION_THRESHOLD = 1.0
VORTICITY_THRESHOLD_FRACTION = 0.2
FPS = 8
IMAGE_RESOLUTION = [1280, 720]
# View from positive z so increasing x runs from screen-left to screen-right.
# The oblique x offset keeps part of the ring's y-z face visible, while the
# downstream focal point and larger distance keep later ring motion in frame.
CAMERA_POSITION = [15.25, 7, 14]
CAMERA_FOCAL_POINT = [7, 1, 2]
CAMERA_VIEW_UP = [0, 1, -0.2]
CAMERA_PARALLEL_SCALE = 12.0
TRANSPARENT_GIF_FILTER = (
    "[0:v]split[gif][palette_source];"
    "[palette_source]palettegen=stats_mode=diff:reserve_transparent=1[palette];"
    "[gif][palette]paletteuse=dither=sierra2_4a:alpha_threshold=128"
)
MANIFEST_COLUMNS = (
    "frame_index",
    "snapshot_step",
    "snapshot_file",
    "source_cells",
    "source_points",
    "domain_boundary",
    "png_file",
)


def positive_integer(value):
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("stride must be an integer") from error
    if number < 1:
        raise argparse.ArgumentTypeError("stride must be at least 1")
    return number


def nonnegative_integer(value):
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("value must be an integer") from error
    if number < 0:
        raise argparse.ArgumentTypeError("value must be nonnegative")
    return number


def threshold_fraction(value):
    try:
        fraction = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "threshold fraction must be a number"
        ) from error
    if not 0.0 < fraction <= 1.0:
        raise argparse.ArgumentTypeError(
            "threshold fraction must be greater than 0 and at most 1"
        )
    return fraction


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Render 3D Q-criterion or normalized-vorticity PNG frames and "
            "an animated GIF."
        )
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
    parser.add_argument(
        "--field",
        choices=("q-criterion", "vorticity"),
        default="q-criterion",
        help="vortex diagnostic used for thresholding (default: q-criterion)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse existing nonempty PNG frames instead of rendering them again",
    )
    parser.add_argument(
        "--vorticity-threshold-fraction",
        type=threshold_fraction,
        default=VORTICITY_THRESHOLD_FRACTION,
        help=(
            "3D normalized-vorticity cutoff as a fraction of maximum |vorticity| "
            f"(default: {VORTICITY_THRESHOLD_FRACTION:g})"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="output folder; defaults to outputs/<run-name>_<field>",
    )
    parser.add_argument(
        "--show-domain-boundary",
        action="store_true",
        help=(
            "overlay the current Chombo data-domain outline so adaptive-domain "
            "motion is visible"
        ),
    )
    parallel_mode = parser.add_mutually_exclusive_group()
    parallel_mode.add_argument(
        "--prepare-only",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parallel_mode.add_argument(
        "--frames-only",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parallel_mode.add_argument(
        "--assemble-only",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--worker-index",
        type=nonnegative_integer,
        default=0,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--worker-count",
        type=positive_integer,
        default=1,
        help=argparse.SUPPRESS,
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


def output_paths(snapshot_folder, field, output_dir=None):
    run_name = (
        snapshot_folder.parent.name
        if snapshot_folder.name == "output"
        else snapshot_folder.name
    )
    field_name = field.replace("-", "")
    output_folder = (
        output_dir.expanduser().resolve()
        if output_dir is not None
        else Path(__file__).resolve().parent
        / "outputs"
        / f"{run_name}_{field_name}"
    )
    frames_folder = output_folder / "frames"
    gif_path = output_folder / f"{run_name}_{field_name}.gif"
    manifest_path = output_folder / "frame_manifest.csv"
    return output_folder, frames_folder, gif_path, manifest_path


def prepare_output_folders(output_folder, frames_folder, resume):
    output_folder.mkdir(parents=True, exist_ok=True)
    frames_folder.mkdir(parents=True, exist_ok=True)

    if resume:
        return

    # Remove only frames produced by an earlier invocation of this script.
    for old_frame in frames_folder.iterdir():
        if old_frame.is_file() and re.fullmatch(r"flowTime_\d+\.png", old_frame.name):
            old_frame.unlink()


def worker_manifest_folder(output_folder):
    return output_folder / "worker_manifests"


def worker_manifest_path(output_folder, worker_index):
    return worker_manifest_folder(output_folder) / f"worker_{worker_index:04d}.csv"


def prepare_parallel_output(output_folder, frames_folder, resume, worker_count):
    prepare_output_folders(output_folder, frames_folder, resume)
    manifests_folder = worker_manifest_folder(output_folder)
    manifests_folder.mkdir(parents=True, exist_ok=True)
    for worker_index in range(worker_count):
        path = worker_manifest_path(output_folder, worker_index)
        if path.is_file():
            path.unlink()


def assigned_snapshots(snapshots, worker_index, worker_count):
    if worker_index >= worker_count:
        raise ValueError(
            f"worker index {worker_index} must be less than worker count "
            f"{worker_count}"
        )
    return list(enumerate(snapshots))[worker_index::worker_count]


def write_manifest(path, rows):
    with path.open("w", newline="") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def read_worker_rows(output_folder, worker_count):
    rows_by_step = {}
    for worker_index in range(worker_count):
        path = worker_manifest_path(output_folder, worker_index)
        if not path.is_file():
            raise RuntimeError(f"Missing frame-worker manifest: {path}")
        with path.open(newline="") as manifest_file:
            reader = csv.DictReader(manifest_file)
            if reader.fieldnames != list(MANIFEST_COLUMNS):
                raise RuntimeError(f"Invalid frame-worker manifest columns: {path}")
            for row in reader:
                step = int(row["snapshot_step"])
                if step in rows_by_step:
                    raise RuntimeError(
                        f"Snapshot step {step} appears in multiple worker manifests"
                    )
                rows_by_step[step] = row
    return rows_by_step


def existing_manifest_rows(manifest_path):
    """Keep cell and point counts when resume mode reuses a saved PNG."""
    if not manifest_path.is_file():
        return {}
    with manifest_path.open(newline="") as manifest_file:
        return {
            int(row["snapshot_step"]): row
            for row in csv.DictReader(manifest_file)
        }


def reusable_png(path):
    return path.is_file() and path.stat().st_size > 0


def load_paraview():
    try:
        from paraview import simple
    except ImportError as error:
        raise RuntimeError(
            "ParaView's Python module is unavailable. Run this script with "
            "pvpython or pvbatch, not a regular Python interpreter."
        ) from error
    return simple


def merge_vector(simple, input_data, components, vector_name):
    vector = simple.MergeVectorComponents(
        registrationName=vector_name,
        Input=input_data,
    )
    vector.XArray, vector.YArray, vector.ZArray = components
    vector.OutputVectorName = vector_name
    return vector


def build_vector_data(simple, point_data):
    velocity = merge_vector(
        simple,
        point_data,
        VELOCITY_COMPONENTS,
        "Velocity",
    )
    vorticity = merge_vector(
        simple,
        velocity,
        VORTICITY_COMPONENTS,
        "Vorticity",
    )
    return vorticity, [vorticity, velocity]


def build_q_criterion_threshold(simple, vector_data):
    gradient = simple.Gradient(
        registrationName="VelocityGradient",
        Input=vector_data,
    )
    gradient.ScalarArray = ["POINTS", "Velocity"]
    gradient.ComputeQCriterion = 1
    gradient.QCriterionArrayName = "Q Criterion"

    # Previous contour visualization:
    # contour = simple.Contour(registrationName="QCriterionContour", Input=gradient)
    # contour.ContourBy = ["POINTS", "Q Criterion"]
    # contour.Isosurfaces = [1.5]

    threshold = simple.Threshold(
        registrationName="QCriterionThreshold",
        Input=gradient,
    )
    threshold.Scalars = ["POINTS", "Q Criterion"]
    threshold.UpperThreshold = Q_CRITERION_THRESHOLD
    threshold.ThresholdMethod = "Above Upper Threshold"
    return threshold, [threshold, gradient]


def build_vorticity_threshold(
    simple,
    vector_data,
    vorticity_threshold_fraction=VORTICITY_THRESHOLD_FRACTION,
):
    calculator = simple.PythonCalculator(
        registrationName="NormalizedVorticity",
        Input=vector_data,
    )
    calculator.Expression = "mag(Vorticity) / max(mag(Vorticity))"
    calculator.ArrayName = "Normalized Vorticity"

    # Previous contour visualization:
    # contour = simple.Contour(
    #     registrationName="NormalizedVorticityContour",
    #     Input=calculator,
    # )
    # contour.ContourBy = ["POINTS", "Normalized Vorticity"]
    # contour.Isosurfaces = [VORTICITY_THRESHOLD_FRACTION]

    threshold = simple.Threshold(
        registrationName="NormalizedVorticityThreshold",
        Input=calculator,
    )
    threshold.Scalars = ["POINTS", "Normalized Vorticity"]
    threshold.UpperThreshold = vorticity_threshold_fraction
    threshold.ThresholdMethod = "Above Upper Threshold"
    return threshold, [threshold, calculator]


def render_snapshot(
    simple,
    snapshot,
    png_path,
    field,
    vorticity_threshold_fraction=VORTICITY_THRESHOLD_FRACTION,
    show_domain_boundary=False,
):
    # Start every frame from a completely empty ParaView session. In
    # particular, do not reuse a reader or render view across snapshots.
    simple.ResetSession()
    simple._DisableFirstRenderCameraReset()

    required_components = VELOCITY_COMPONENTS + VORTICITY_COMPONENTS
    source = simple.VisItChomboReader(
        registrationName=snapshot.name,
        FileName=[str(snapshot)],
    )
    source.CellArrayStatus = list(required_components)
    source.UpdatePipeline()

    cell_data = source.GetCellDataInformation()
    missing_components = [
        name
        for name in required_components
        if cell_data.GetArray(name) is None
    ]
    if missing_components:
        raise RuntimeError(
            f"{snapshot} is missing required cell arrays for {field}: "
            f"{', '.join(missing_components)}"
        )

    source_info = source.GetDataInformation()
    source_cells = source_info.GetNumberOfCells()
    source_points = source_info.GetNumberOfPoints()

    cell_to_point = simple.CellDatatoPointData(
        registrationName="CellDataToPointData",
        Input=source,
    )
    vector_data, vector_proxies = build_vector_data(simple, cell_to_point)
    vector_data.UpdatePipeline()
    vector_point_data = vector_data.GetPointDataInformation()
    missing_vectors = [
        name
        for name in ("Velocity", "Vorticity")
        if vector_point_data.GetArray(name) is None
    ]
    if missing_vectors:
        raise RuntimeError(
            f"{field} vector assembly is missing point arrays: "
            f"{', '.join(missing_vectors)}"
        )

    if field == "q-criterion":
        threshold, field_proxies = build_q_criterion_threshold(simple, vector_data)
    else:
        threshold, field_proxies = build_vorticity_threshold(
            simple,
            vector_data,
            vorticity_threshold_fraction,
        )
    threshold.UpdatePipeline()

    threshold_points = threshold.GetDataInformation().GetNumberOfPoints()
    if threshold_points == 0:
        print(
            f"Warning: {snapshot.name} has no points above the "
            f"{field} threshold; rendering a blank frame.",
            flush=True,
        )

    render_view = simple.GetActiveViewOrCreate("RenderView")

    # ResetSession can leave a stale representation registered in headless
    # pvbatch runs. Hide everything before showing this snapshot's threshold.
    simple.HideAll(render_view)
    render_proxies = []
    if show_domain_boundary:
        domain_outline = simple.Outline(
            registrationName="ChomboDataDomainOutline",
            Input=source,
        )
        domain_outline.UpdatePipeline()
        outline_display = simple.Show(
            domain_outline,
            render_view,
            "GeometryRepresentation",
        )
        outline_display.Representation = "Surface"
        # Some ParaView versions leave a new outline display associated with
        # "NONE", which ColorBy cannot resolve when switching to solid color.
        outline_display.ColorArrayName = ["POINTS", ""]
        simple.ColorBy(outline_display, None)
        outline_display.AmbientColor = [0.12, 0.12, 0.12]
        outline_display.DiffuseColor = [0.12, 0.12, 0.12]
        outline_display.LineWidth = 2.5
        render_proxies.append(domain_outline)

    threshold_display = simple.Show(
        threshold,
        render_view,
        "UnstructuredGridRepresentation",
    )
    threshold_display.Representation = "Surface"
    if threshold_points > 0:
        simple.ColorBy(threshold_display, ("POINTS", "Velocity", "Magnitude"))
        threshold_display.RescaleTransferFunctionToDataRange(True, False)
        threshold_display.SetScalarBarVisibility(render_view, True)

    render_view.CameraPosition = CAMERA_POSITION
    render_view.CameraFocalPoint = CAMERA_FOCAL_POINT
    render_view.CameraViewUp = CAMERA_VIEW_UP
    render_view.CameraParallelScale = CAMERA_PARALLEL_SCALE
    render_view.ViewSize = IMAGE_RESOLUTION
    if show_domain_boundary:
        # Preserve the oblique viewing direction but fit all visible Chombo
        # data, including its outline, inside the frame.
        simple.ResetCamera(render_view)

    # SaveScreenshot performs the render, so a separate Render call is unnecessary.
    simple.SaveScreenshot(
        str(png_path),
        render_view,
        ImageResolution=IMAGE_RESOLUTION,
        TransparentBackground=1,
    )

    if not png_path.is_file() or png_path.stat().st_size == 0:
        raise RuntimeError(f"ParaView did not create a valid PNG: {png_path}")

    # Explicitly unregister proxies so ParaView does not retain this frame's
    # representations and reader caches until the next session reset.
    simple.HideAll(render_view)
    for proxy in [
        *render_proxies,
        *field_proxies,
        *vector_proxies,
        cell_to_point,
        source,
    ]:
        simple.Delete(proxy)
    simple.RemoveViewsAndLayouts()

    return source_cells, source_points


def find_ffmpeg_command():
    if BRIDGES_FFMPEG_IMAGE.is_file():
        singularity = shutil.which("singularity")
        if singularity is None:
            raise RuntimeError(
                f"Found the Bridges-2 FFmpeg image at {BRIDGES_FFMPEG_IMAGE}, "
                "but singularity was not found."
            )
        return [
            singularity,
            "exec",
            "-B",
            "/ocean",
            str(BRIDGES_FFMPEG_IMAGE),
            "ffmpeg",
        ]

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            "ffmpeg was not found. PNG frames were saved, but the GIF could "
            "not be created. Install ffmpeg and rerun the script."
        )
    return [ffmpeg]


def build_gif(frames_folder, snapshots, gif_path):
    gif_snapshots = snapshots[1:]
    if not gif_snapshots:
        raise ValueError(
            "Cannot build a GIF after excluding the first PNG: "
            "at least two rendered frames are required."
        )

    print(f"Excluding first GIF frame: {snapshots[0].name}", flush=True)
    ffmpeg_command = find_ffmpeg_command()
    print(f"Using FFmpeg: {' '.join(ffmpeg_command)}", flush=True)

    staging_folder = gif_path.parent / "_gif_frames"
    shutil.rmtree(staging_folder, ignore_errors=True)
    staging_folder.mkdir(parents=True)

    try:
        for frame_index, snapshot in enumerate(gif_snapshots):
            source_png = frames_folder / f"flowTime_{snapshot_step(snapshot)}.png"
            staged_png = staging_folder / f"frame_{frame_index:05d}.png"
            staged_png.symlink_to(source_png.resolve())

        subprocess.run(
            ffmpeg_command
            + [
                "-y",
                "-framerate",
                str(FPS),
                "-i",
                str(staging_folder / "frame_%05d.png"),
                "-filter_complex",
                TRANSPARENT_GIF_FILTER,
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


def frame_manifest_row(
    frame_index,
    snapshot,
    png_path,
    source_cells,
    source_points,
    show_domain_boundary,
):
    return {
        "frame_index": frame_index,
        "snapshot_step": snapshot_step(snapshot),
        "snapshot_file": str(snapshot.resolve()),
        "source_cells": source_cells,
        "source_points": source_points,
        "domain_boundary": int(show_domain_boundary),
        "png_file": str(png_path.resolve()),
    }


def render_assigned_frames(
    args,
    snapshots,
    output_folder,
    frames_folder,
    old_manifest,
):
    assignments = assigned_snapshots(
        snapshots,
        args.worker_index,
        args.worker_count,
    )
    needs_render = any(
        not reusable_png(
            frames_folder / f"flowTime_{snapshot_step(snapshot)}.png"
        )
        for _, snapshot in assignments
    )
    simple = load_paraview() if needs_render else None
    rows = []

    print(
        f"Frame worker {args.worker_index + 1}/{args.worker_count}: "
        f"{len(assignments)} snapshots",
        flush=True,
    )
    for assignment_index, (frame_index, snapshot) in enumerate(assignments):
        step = snapshot_step(snapshot)
        png_path = frames_folder / f"flowTime_{step}.png"
        reuse = args.resume and reusable_png(png_path)
        print(
            f"[worker {args.worker_index + 1}, "
            f"{assignment_index + 1}/{len(assignments)}] "
            f"{'Reusing' if reuse else 'Rendering'} {snapshot.name}",
            flush=True,
        )
        if reuse:
            old_row = old_manifest.get(step, {})
            source_cells = old_row.get("source_cells", "")
            source_points = old_row.get("source_points", "")
        else:
            source_cells, source_points = render_snapshot(
                simple,
                snapshot.resolve(),
                png_path,
                args.field,
                args.vorticity_threshold_fraction,
                args.show_domain_boundary,
            )
        rows.append(
            frame_manifest_row(
                frame_index,
                snapshot,
                png_path,
                source_cells,
                source_points,
                args.show_domain_boundary,
            )
        )

    manifest_path = worker_manifest_path(output_folder, args.worker_index)
    write_manifest(manifest_path, rows)
    print(f"Frame worker manifest: {manifest_path}", flush=True)


def assemble_parallel_output(
    args,
    snapshots,
    output_folder,
    frames_folder,
    gif_path,
    manifest_path,
):
    rows_by_step = read_worker_rows(output_folder, args.worker_count)
    rows = []
    for frame_index, snapshot in enumerate(snapshots):
        step = snapshot_step(snapshot)
        png_path = frames_folder / f"flowTime_{step}.png"
        if not reusable_png(png_path):
            raise RuntimeError(f"Missing rendered PNG for snapshot step {step}: {png_path}")
        if step not in rows_by_step:
            raise RuntimeError(f"No frame-worker result for snapshot step {step}")
        row = rows_by_step[step]
        if int(row["frame_index"]) != frame_index:
            raise RuntimeError(
                f"Frame-worker index mismatch for snapshot step {step}: "
                f"expected {frame_index}, got {row['frame_index']}"
            )
        if Path(row["snapshot_file"]).resolve() != snapshot.resolve():
            raise RuntimeError(
                f"Frame-worker source mismatch for snapshot step {step}"
            )
        rows.append(row)

    write_manifest(manifest_path, rows)
    print("All parallel PNG frames rendered. Building GIF...", flush=True)
    build_gif(frames_folder, snapshots, gif_path)
    print(f"Done: {gif_path}", flush=True)
    print(f"PNG frames: {frames_folder}", flush=True)
    print(f"Manifest: {manifest_path}", flush=True)


def main():
    args = parse_args()

    try:
        snapshot_folder = find_snapshot_folder(args.output_folder)
        snapshots = discover_snapshots(snapshot_folder, args.stride)
        output_folder, frames_folder, gif_path, manifest_path = output_paths(
            snapshot_folder,
            args.field,
            args.output_dir,
        )
        if args.worker_index >= args.worker_count:
            raise ValueError(
                f"--worker-index {args.worker_index} must be less than "
                f"--worker-count {args.worker_count}"
            )
        if args.prepare_only:
            prepare_parallel_output(
                output_folder,
                frames_folder,
                args.resume,
                args.worker_count,
            )
            print(f"Prepared parallel output: {output_folder}", flush=True)
            return 0

        if args.frames_only:
            prepare_output_folders(output_folder, frames_folder, resume=True)
            worker_manifest_folder(output_folder).mkdir(parents=True, exist_ok=True)
            old_manifest = (
                existing_manifest_rows(manifest_path) if args.resume else {}
            )
            render_assigned_frames(
                args,
                snapshots,
                output_folder,
                frames_folder,
                old_manifest,
            )
            return 0

        if args.assemble_only:
            assemble_parallel_output(
                args,
                snapshots,
                output_folder,
                frames_folder,
                gif_path,
                manifest_path,
            )
            return 0

        old_manifest = existing_manifest_rows(manifest_path) if args.resume else {}
        prepare_output_folders(output_folder, frames_folder, args.resume)
        assignments = assigned_snapshots(snapshots, 0, 1)
        needs_render = any(
            not reusable_png(
                frames_folder / f"flowTime_{snapshot_step(snapshot)}.png"
            )
            for _, snapshot in assignments
        )
        simple = load_paraview() if needs_render else None

        print(f"Snapshot folder: {snapshot_folder}", flush=True)
        print(f"Field:           {args.field}", flush=True)
        print(f"Snapshots used:  {len(snapshots)}", flush=True)
        print(f"Stride:          {args.stride}", flush=True)
        print(f"Resume:          {args.resume}", flush=True)
        print(f"Domain boundary: {args.show_domain_boundary}", flush=True)
        if args.field == "vorticity":
            print(
                "Vorticity cutoff: "
                f"{args.vorticity_threshold_fraction:g} of max |vorticity|",
                flush=True,
            )
        print(f"PNG folder:      {frames_folder}", flush=True)
        print(f"Output GIF:      {gif_path}", flush=True)

        with manifest_path.open("w", newline="") as manifest_file:
            manifest_writer = csv.DictWriter(
                manifest_file,
                fieldnames=MANIFEST_COLUMNS,
            )
            manifest_writer.writeheader()

            for frame_index, snapshot in enumerate(snapshots):
                step = snapshot_step(snapshot)
                png_path = frames_folder / f"flowTime_{step}.png"
                reuse = args.resume and reusable_png(png_path)
                print(
                    f"[{frame_index + 1}/{len(snapshots)}] "
                    f"{'Reusing' if reuse else 'Rendering'} "
                    f"{snapshot.name}",
                    flush=True,
                )
                if reuse:
                    old_row = old_manifest.get(step, {})
                    source_cells = old_row.get("source_cells", "")
                    source_points = old_row.get("source_points", "")
                else:
                    source_cells, source_points = render_snapshot(
                        simple,
                        snapshot.resolve(),
                        png_path,
                        args.field,
                        args.vorticity_threshold_fraction,
                        args.show_domain_boundary,
                    )
                manifest_writer.writerow(
                    frame_manifest_row(
                        frame_index,
                        snapshot,
                        png_path,
                        source_cells,
                        source_points,
                        args.show_domain_boundary,
                    )
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
