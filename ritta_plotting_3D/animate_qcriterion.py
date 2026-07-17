#!/usr/bin/env pvpython

import re
import shutil
import subprocess
import sys
from pathlib import Path

from paraview.simple import *
import paraview

paraview.simple._DisableFirstRenderCameraReset()


if len(sys.argv) < 2:
    print("Usage:")
    print("  animate_qcriterion.py RUN_FOLDER [STRIDE] [OUTPUT_MP4]")
    sys.exit(1)


run_folder = Path(sys.argv[1]).expanduser().resolve()
stride = int(sys.argv[2]) if len(sys.argv) >= 3 else 1

if len(sys.argv) >= 4:
    output_mp4 = Path(sys.argv[3]).expanduser().resolve()
    output_folder = output_mp4.parent
else:
    output_folder = Path(__file__).resolve().parent / "outputs" / f"{run_folder.name}_qcriterion"
    output_mp4 = output_folder / f"{run_folder.name}_qcriterion.mp4"

frames_folder = output_folder / "frames"
ffmpeg_folder = output_folder / "_ffmpeg_frames"

output_folder.mkdir(parents=True, exist_ok=True)
frames_folder.mkdir(parents=True, exist_ok=True)
ffmpeg_folder.mkdir(parents=True, exist_ok=True)

if (run_folder / "flowTime_0.hdf5").exists():
    snapshot_folder = run_folder
elif (run_folder / "output").is_dir():
    snapshot_folder = run_folder / "output"
else:
    print("Could not find snapshots.")
    print(f"Checked {run_folder}")
    print(f"Checked {run_folder / 'output'}")
    sys.exit(1)

all_snapshots = list(snapshot_folder.glob("flowTime_*.hdf5"))
all_snapshots = [x for x in all_snapshots if x.name != "flow_final.hdf5"]

if not all_snapshots:
    print(f"No flowTime_*.hdf5 files found in {snapshot_folder}")
    sys.exit(1)

all_snapshots.sort(key=lambda x: int(re.search(r"flowTime_(\d+)\.hdf5$", x.name).group(1)))
snapshots = all_snapshots[::stride]

print(f"Run folder:     {run_folder}", flush=True)
print(f"Snapshot dir:   {snapshot_folder}", flush=True)
print(f"Snapshots used: {len(snapshots)}", flush=True)
print(f"Stride:         {stride}", flush=True)
print(f"Output folder:  {output_folder}", flush=True)
print(f"Output mp4:     {output_mp4}", flush=True)


# edit these directly if you want different settings
contour_value = -3.0
fps = 8
image_resolution = [1280, 720]
camera_position = [16.65225298909394, 0.7536788379147482, -4.296194431796246]
camera_focal_point = [3.0625000000000027, -1.0977953873560387e-15, 1.807254991429669e-15]
camera_view_up = [-0.016372486611386468, 0.9923582196167854, 0.12229924628207733]
camera_parallel_scale = 9.711408782056534


print("Opening snapshot series...", flush=True)
print("Rendering frames...", flush=True)
for i, snapshot in enumerate(snapshots):
    print(f"Opening snapshot {i + 1}/{len(snapshots)}: {snapshot.name}", flush=True)

    ResetSession()
    paraview.simple._DisableFirstRenderCameraReset()

    source = OpenDataFile(str(snapshot))
    if source is None:
        print(f"Could not open {snapshot}", flush=True)
        sys.exit(1)

    render_view = GetActiveViewOrCreate("RenderView")

    source.CellArrayStatus = ["u_0", "u_1", "u_2"]
    render_view.Update()

    cell_to_point = CellDatatoPointData(Input=source)
    render_view.Update()

    merged = MergeVectorComponents(Input=cell_to_point)
    merged.OutputVectorName = "Velocity"
    merged.XArray = "u_0"
    merged.YArray = "u_1"
    merged.ZArray = "u_2"
    render_view.Update()

    gradient = Gradient(Input=merged)
    gradient.ScalarArray = ["POINTS", "Velocity"]
    gradient.ComputeQCriterion = 1
    gradient.QCriterionArrayName = "Q Criterion"
    render_view.Update()

    contour = Contour(Input=gradient)
    contour.ContourBy = ["POINTS", "Q Criterion"]
    contour.Isosurfaces = [contour_value]
    contour_display = Show(contour, render_view, "GeometryRepresentation")
    contour_display.Representation = "Surface"
    render_view.Update()

    render_view.CameraPosition = camera_position
    render_view.CameraFocalPoint = camera_focal_point
    render_view.CameraViewUp = camera_view_up
    render_view.CameraParallelScale = camera_parallel_scale
    render_view.ViewSize = image_resolution

    Render()

    step = int(re.search(r"flowTime_(\d+)\.hdf5$", snapshot.name).group(1))
    png_name = f"flowTime_{step}.png"
    png_path = frames_folder / png_name

    print(f"[{i + 1}/{len(snapshots)}] {png_name}", flush=True)
    SaveScreenshot(str(png_path), render_view, ImageResolution=image_resolution)

    ffmpeg_png = ffmpeg_folder / f"frame_{i:05d}.png"
    shutil.copy2(png_path, ffmpeg_png)

print(f"Finished PNG frames: {frames_folder}", flush=True)

ffmpeg = shutil.which("ffmpeg")
if ffmpeg is None:
    print("ffmpeg not found. Leaving PNG frames instead of MP4.", flush=True)
    print(f"PNG frames kept at: {frames_folder}", flush=True)
    sys.exit(0)

print("Building MP4...", flush=True)
subprocess.run(
    [
        ffmpeg,
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(ffmpeg_folder / "frame_%05d.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(output_mp4),
    ],
    check=True,
)

shutil.rmtree(ffmpeg_folder, ignore_errors=True)

print(f"Done: {output_mp4}", flush=True)
print(f"PNG frames kept at: {frames_folder}", flush=True)
