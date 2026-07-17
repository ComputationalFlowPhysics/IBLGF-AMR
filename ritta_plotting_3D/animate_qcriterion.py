#!/usr/bin/env pvpython

from __future__ import annotations

import argparse
import re
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


def main() -> None:
    args = parse_args()
    run_folder = Path(args.run_folder).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve() if args.output else default_output_path(run_folder)

    snapshot_paths = find_snapshots(run_folder, args.stride)

    print(f"Run folder:   {run_folder}")
    print(f"Snapshots:    {len(snapshot_paths)}")
    print(f"Stride:       {args.stride}")
    print(f"Output movie: {output_path}")

    source = OpenDataFile(snapshot_paths)
    if source is None:
        raise RuntimeError("ParaView failed to open the snapshot series.")

    render_view = GetActiveViewOrCreate("RenderView")
    source_display = Show(source, render_view, "AMRRepresentation")
    source_display.Representation = "Outline"
    render_view.Update()

    # data -> select u_0,u_1,u_2 -> cell data to point data
    source.CellArrayStatus = ["u_0", "u_1", "u_2"]
    render_view.Update()

    cell_to_point = CellDatatoPointData(registrationName="CellDatatoPointData1", Input=source)
    cell_to_point_display = Show(cell_to_point, render_view, "AMRRepresentation")
    cell_to_point_display.Representation = "Outline"
    Hide(source, render_view)
    render_view.Update()

    merged = MergeVectorComponents(registrationName="MergeVectorComponents1", Input=cell_to_point)
    merged.OutputVectorName = OUTPUT_VECTOR_NAME
    merged_display = Show(merged, render_view, "AMRRepresentation")
    merged_display.Representation = "Outline"
    Hide(cell_to_point, render_view)
    render_view.Update()

    gradient = Gradient(registrationName="Gradient1", Input=merged)
    gradient.ComputeQCriterion = 1
    gradient.QCriterionArrayName = Q_ARRAY_NAME
    gradient_display = Show(gradient, render_view, "AMRRepresentation")
    gradient_display.Representation = "Outline"
    Hide(merged, render_view)
    render_view.Update()

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
    Render()

    print("Saving MP4...")
    SaveAnimation(
        str(output_path),
        render_view,
        ImageResolution=IMAGE_RESOLUTION,
        FrameRate=MP4_FPS,
    )
    print(f"Done: {output_path}")


if __name__ == "__main__":
    main()
