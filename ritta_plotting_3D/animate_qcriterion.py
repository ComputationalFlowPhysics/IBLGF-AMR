#!/usr/bin/env pvpython

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path

from paraview.simple import *  # noqa: F401,F403
import paraview


# ----------------------------
# Easy-to-edit visualization settings
# ----------------------------
CONTOUR_ISO_VALUE = -1.4283038627771472e-4
OUTPUT_VECTOR_NAME = "Velocity"
Q_ARRAY_NAME = "Q Criterion"
MP4_FPS = 8
IMAGE_RESOLUTION = [1280, 720]
USE_TRACED_CAMERA = True
PROGRESS_UPDATE_SECONDS = 10
KEEP_FRAME_PNGS = False

# Camera copied from your traced snapshot script.
CAMERA_POSITION = [16.65225298909394, 0.7536788379147482, -4.296194431796246]
CAMERA_FOCAL_POINT = [3.0625000000000027, -1.0977953873560387e-15, 1.807254991429669e-15]
CAMERA_VIEW_UP = [-0.016372486611386468, 0.9923582196167854, 0.12229924628207733]
CAMERA_PARALLEL_SCALE = 9.711408782056534


paraview.simple._DisableFirstRenderCameraReset()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a 3D Q-criterion contour animation from an IBLGF-AMR output folder."
    )
    parser.add_argument(
        "run_folder",
        help="Path to the 3D run folder. Snapshots may be directly inside it or inside an output/ subfolder.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Use every Nth snapshot. Example: --stride 5",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="MP4 output path. Default: ritta_plotting_3D/outputs/<run_folder_name>_qcriterion.mp4",
    )
    return parser.parse_args()


def snapshot_step(path: Path) -> int:
    match = re.search(r"flowTime_(\d+)\.hdf5$", path.name)
    if not match:
        raise ValueError(f"Unexpected snapshot name: {path.name}")
    return int(match.group(1))


def resolve_snapshot_dir(run_folder: Path) -> Path:
    direct_snapshots = list(run_folder.glob("flowTime_*.hdf5"))
    output_dir = run_folder / "output"
    output_snapshots = list(output_dir.glob("flowTime_*.hdf5")) if output_dir.is_dir() else []

    if direct_snapshots:
        return run_folder
    if output_snapshots:
        return output_dir

    raise ValueError(
        "No flowTime_*.hdf5 snapshots found.\n"
        f"Checked:\n"
        f"  - {run_folder}\n"
        f"  - {output_dir}"
    )


def find_snapshots(run_folder: Path, stride: int) -> list[str]:
    if stride < 1:
        raise ValueError("Stride must be at least 1.")

    snapshot_dir = resolve_snapshot_dir(run_folder)
    snapshots = sorted(snapshot_dir.glob("flowTime_*.hdf5"), key=snapshot_step)
    snapshots = [path for path in snapshots if path.name != "flow_final.hdf5"]
    if not snapshots:
        raise ValueError(f"No usable flowTime_*.hdf5 snapshots found in {snapshot_dir}")

    return [str(path) for path in snapshots[::stride]]


def default_output_path(run_folder: Path) -> Path:
    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{run_folder.name}_qcriterion.mp4"


def progress_monitor(output_path: Path, total_frames: int, stop_event: threading.Event) -> None:
    start_time = time.time()
    while not stop_event.wait(PROGRESS_UPDATE_SECONDS):
        elapsed = int(time.time() - start_time)
        if output_path.exists():
            size_mb = output_path.stat().st_size / (1024 * 1024)
            print(
                f"[progress] Still rendering {total_frames} frame(s)... "
                f"elapsed={elapsed}s output_size={size_mb:.1f} MB",
                flush=True,
            )
        else:
            print(
                f"[progress] Still rendering {total_frames} frame(s)... "
                f"elapsed={elapsed}s output file not created yet",
                flush=True,
            )


def frame_dir_for_output(output_path: Path) -> Path:
    return output_path.with_suffix("") / "frames"


def build_mp4_from_frames(frame_dir: Path, output_path: Path) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        print("ffmpeg not found. Leaving PNG frames instead of MP4.", flush=True)
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg,
        "-y",
        "-framerate",
        str(MP4_FPS),
        "-i",
        str(frame_dir / "frame_%05d.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    print("Combining frames into MP4 with ffmpeg...", flush=True)
    subprocess.run(cmd, check=True)
    return True


def main() -> None:
    args = parse_args()
    run_folder = Path(args.run_folder).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve() if args.output else default_output_path(run_folder)

    snapshot_paths = find_snapshots(run_folder, args.stride)
    total_frames = len(snapshot_paths)
    snapshot_steps = [snapshot_step(Path(path)) for path in snapshot_paths]

    print(f"Run folder:   {run_folder}")
    print(f"Snapshots:    {total_frames}")
    print(f"Stride:       {args.stride}")
    print(f"Output movie: {output_path}")
    print("Opening snapshot series...", flush=True)

    source = OpenDataFile(snapshot_paths)
    if source is None:
        raise RuntimeError("ParaView failed to open the snapshot series.")
    print("Opened snapshot series.", flush=True)

    render_view = GetActiveViewOrCreate("RenderView")
    source_display = Show(source, render_view, "AMRRepresentation")
    source_display.Representation = "Outline"
    render_view.Update()

    # data -> select u_0,u_1,u_2 -> cell data to point data
    print("Selecting u_0, u_1, u_2 arrays...", flush=True)
    source.CellArrayStatus = ["u_0", "u_1", "u_2"]
    render_view.Update()

    print("Applying Cell Data to Point Data...", flush=True)
    cell_to_point = CellDatatoPointData(registrationName="CellDatatoPointData1", Input=source)
    cell_to_point_display = Show(cell_to_point, render_view, "AMRRepresentation")
    cell_to_point_display.Representation = "Outline"
    Hide(source, render_view)
    render_view.Update()

    print("Merging vector components...", flush=True)
    merged = MergeVectorComponents(registrationName="MergeVectorComponents1", Input=cell_to_point)
    merged.OutputVectorName = OUTPUT_VECTOR_NAME
    merged_display = Show(merged, render_view, "AMRRepresentation")
    merged_display.Representation = "Outline"
    Hide(cell_to_point, render_view)
    render_view.Update()

    print("Computing gradient and Q criterion...", flush=True)
    gradient = Gradient(registrationName="Gradient1", Input=merged)
    gradient.ComputeQCriterion = 1
    gradient.QCriterionArrayName = Q_ARRAY_NAME
    gradient_display = Show(gradient, render_view, "AMRRepresentation")
    gradient_display.Representation = "Outline"
    Hide(merged, render_view)
    render_view.Update()

    print("Applying contour filter...", flush=True)
    contour = Contour(registrationName="Contour1", Input=gradient)
    contour.ContourBy = ["POINTS", Q_ARRAY_NAME]
    contour.Isosurfaces = [CONTOUR_ISO_VALUE]
    contour_display = Show(contour, render_view, "GeometryRepresentation")
    contour_display.Representation = "Surface"
    contour_display.SetScalarBarVisibility(render_view, True)
    Hide(gradient, render_view)
    render_view.Update()

    GetColorTransferFunction(Q_ARRAY_NAME)
    GetOpacityTransferFunction(Q_ARRAY_NAME)
    GetTransferFunction2D(Q_ARRAY_NAME)

    animation_scene = GetAnimationScene()
    animation_scene.UpdateAnimationUsingDataTimeSteps()

    if USE_TRACED_CAMERA:
        render_view.CameraPosition = CAMERA_POSITION
        render_view.CameraFocalPoint = CAMERA_FOCAL_POINT
        render_view.CameraViewUp = CAMERA_VIEW_UP
        render_view.CameraParallelScale = CAMERA_PARALLEL_SCALE
    else:
        render_view.ResetCamera(False, 0.9)

    render_view.ViewSize = IMAGE_RESOLUTION
    print("Rendering first view...", flush=True)
    Render()
    print("First view rendered.", flush=True)

    frame_dir = frame_dir_for_output(output_path)
    frame_dir.mkdir(parents=True, exist_ok=True)

    source_times = list(source.TimestepValues) if hasattr(source, "TimestepValues") else []
    if len(source_times) != total_frames:
        source_times = [None] * total_frames

    print(f"Saving {total_frames} PNG frame(s) first...", flush=True)
    stop_event = threading.Event()
    monitor = threading.Thread(
        target=progress_monitor,
        args=(output_path, total_frames, stop_event),
        daemon=True,
    )
    monitor.start()
    try:
        for idx in range(total_frames):
            if source_times[idx] is not None:
                animation_scene.AnimationTime = source_times[idx]
            Render()
            frame_path = frame_dir / f"flowTime_{snapshot_steps[idx]}.png"
            print(
                f"[frame {idx + 1}/{total_frames}] {frame_path.name}",
                flush=True,
            )
            SaveScreenshot(str(frame_path), render_view, ImageResolution=IMAGE_RESOLUTION)
    finally:
        stop_event.set()
        monitor.join(timeout=1)

    print(f"Finished writing PNG frames to {frame_dir}", flush=True)

    if build_mp4_from_frames(frame_dir, output_path):
        print(f"Done: {output_path}", flush=True)
        if not KEEP_FRAME_PNGS:
            shutil.rmtree(frame_dir, ignore_errors=True)
            print("Deleted intermediate PNG frames.", flush=True)
    else:
        print(f"PNG frames kept at: {frame_dir}", flush=True)


if __name__ == "__main__":
    main()
