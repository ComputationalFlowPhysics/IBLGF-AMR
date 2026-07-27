#!/usr/bin/env pvpython
"""Plot and animate leading-vortex circulation from 3D vortex-ring snapshots.

Usage:
    pvbatch plot_circulation.py OUTPUT_FOLDER [STRIDE] [--config CONFIG_FILE]

``OUTPUT_FOLDER`` may be the folder containing ``flowTime_*.hdf5`` files or
the run folder containing an ``output`` subfolder. The vortex ring is assumed
to travel along x with its symmetry axis at y = z = 0. Generated CSV, plot,
PNG frames, and GIF files are saved under ``ritta_plotting_3D/outputs``.
"""

import argparse
import csv
import re
import shutil
import subprocess
import sys
from pathlib import Path


SNAPSHOT_PATTERN = re.compile(r"flowTime_(\d+)\.hdf5$")
VORTICITY_COMPONENT = "edge_aux_2"
BRIDGES_FFMPEG_IMAGE = Path(
    "/opt/packages/ffmpeg/4.3.1/singularity-ffmpeg-4.3.1.sif"
)

# Analysis settings. These reproduce the ParaView trace and the paper's
# 2%-of-maximum closed-vorticity-contour definition. The tiny z offset avoids
# double-counting coincident cut faces when z=0 is a Chombo cell boundary.
SLICE_ORIGIN = [0.0, 0.0, 1.0e-6]
SLICE_NORMAL = [0.0, 0.0, 1.0]
CLIP_ORIGIN = [0.0, 0.0, 0.0]
CLIP_NORMAL = [0.0, 1.0, 0.0]
CLIP_INVERT = 0
VORTICITY_THRESHOLD_FRACTION = 0.02
PLOT_RESOLUTION = [1200, 800]
FRAME_RESOLUTION = [1280, 720]
CAMERA_MARGIN_FRACTION = 0.08
FPS = 8


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
        description=(
            "Calculate, plot, and animate leading-vortex circulation from 3D "
            "vortex-ring snapshots."
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
        help="analyze every STRIDE-th snapshot (default: 1)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help=(
            "simulation config used for physical time; normally discovered "
            "from the run folder"
        ),
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


def config_from_meta(run_folder):
    meta_path = run_folder / "meta.txt"
    if not meta_path.is_file():
        return None

    for line in meta_path.read_text().splitlines():
        key, separator, value = line.partition(":")
        if separator and key.strip() == "config":
            config_path = Path(value.strip())
            if not config_path.is_absolute():
                config_path = run_folder / config_path
            if config_path.is_file():
                return config_path.resolve()
    return None


def find_config_file(snapshot_folder, requested_config):
    if requested_config is not None:
        config_path = requested_config.expanduser().resolve()
        if not config_path.is_file():
            raise ValueError(f"Config file does not exist: {config_path}")
        return config_path

    nearby_folders = [snapshot_folder]
    if snapshot_folder.name == "output":
        nearby_folders.insert(0, snapshot_folder.parent)

    for folder in nearby_folders:
        meta_config = config_from_meta(folder)
        if meta_config is not None:
            return meta_config

    candidates = sorted(
        {
            path.resolve()
            for folder in nearby_folders
            for path in folder.glob("config*")
            if path.is_file()
        }
    )
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise ValueError(
            "No simulation config was found beside the snapshots. "
            "Pass it explicitly with --config."
        )
    raise ValueError(
        "Multiple simulation configs were found. Pass the one used for this "
        "run explicitly with --config."
    )


def config_text_without_comments(config_path):
    text = config_path.read_text()
    return re.sub(r"//.*?$|#.*?$", "", text, flags=re.MULTILINE)


def read_config_scalar(text, name, required=True):
    match = re.search(
        rf"\b{re.escape(name)}\s*=\s*"
        r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*;",
        text,
    )
    if match is None:
        if required:
            raise ValueError(f"Could not read {name} from the simulation config")
        return None
    return float(match.group(1))


def read_time_metadata(config_path):
    text = config_text_without_comments(config_path)
    if read_config_scalar(text, "dt", required=False) is not None:
        raise ValueError(
            "The simulation config sets an explicit dt, so physical time "
            "cannot be inferred from the project CFL convention."
        )

    cfl = read_config_scalar(text, "cfl")
    dx_base = read_config_scalar(text, "dx_base")
    levels_value = read_config_scalar(text, "nLevels")
    levels = int(levels_value)

    if levels_value != levels or levels < 0:
        raise ValueError("nLevels must be a nonnegative integer")
    if cfl <= 0.0 or dx_base <= 0.0:
        raise ValueError("cfl and dx_base must be positive")
    return cfl, dx_base, levels


def physical_time(step, cfl, dx_base, levels):
    return cfl * step * dx_base / (2**levels)


def output_paths(snapshot_folder):
    run_name = (
        snapshot_folder.parent.name
        if snapshot_folder.name == "output"
        else snapshot_folder.name
    )
    output_folder = (
        Path(__file__).resolve().parent / "outputs" / f"{run_name}_circulation"
    )
    frames_folder = output_folder / "frames"
    return (
        output_folder,
        frames_folder,
        output_folder / "leading_vortex_connectivity.gif",
        output_folder / "leading_vortex_circulation.csv",
        output_folder / "leading_vortex_circulation.png",
    )


def prepare_frames_folder(frames_folder):
    frames_folder.mkdir(parents=True, exist_ok=True)
    for old_frame in frames_folder.iterdir():
        if old_frame.is_file() and re.fullmatch(
            r"flowTime_\d+\.png", old_frame.name
        ):
            old_frame.unlink()


def load_paraview():
    try:
        from paraview import servermanager, simple
    except ImportError as error:
        raise RuntimeError(
            "ParaView's Python module is unavailable. Run this script with "
            "pvpython or pvbatch, not a regular Python interpreter."
        ) from error
    return simple, servermanager


def point_array_range(proxy, array_name):
    proxy.UpdatePipeline()
    array_info = proxy.GetPointDataInformation().GetArray(array_name)
    if array_info is None:
        raise RuntimeError(f"Pipeline output is missing point array {array_name}")
    return array_info.GetComponentRange(0)


def fetched_scalar(dataset, array_name):
    for data in (
        dataset.GetPointData(),
        dataset.GetCellData(),
        dataset.GetFieldData(),
    ):
        array = data.GetArray(array_name)
        if array is not None and array.GetNumberOfTuples() > 0:
            return float(array.GetComponent(0, 0))
    raise RuntimeError(f"Integrated output is missing array {array_name}")


def build_leading_region(simple, snapshot):
    source = simple.VisItChomboReader(
        registrationName=snapshot.name,
        FileName=[str(snapshot)],
    )
    source.CellArrayStatus = [VORTICITY_COMPONENT]
    source.UpdatePipeline()

    cell_data = source.GetCellDataInformation()
    if cell_data.GetArray(VORTICITY_COMPONENT) is None:
        raise RuntimeError(
            f"{snapshot} is missing required cell array {VORTICITY_COMPONENT}"
        )

    cell_to_point = simple.CellDatatoPointData(
        registrationName="CellDataToPointData",
        Input=source,
    )

    slice_filter = simple.Slice(
        registrationName="MeridionalSlice",
        Input=cell_to_point,
    )
    slice_filter.SliceType.Origin = SLICE_ORIGIN
    slice_filter.SliceType.Normal = SLICE_NORMAL

    clip = simple.Clip(
        registrationName="PositiveRadialHalfPlane",
        Input=slice_filter,
    )
    clip.ClipType.Origin = CLIP_ORIGIN
    clip.ClipType.Normal = CLIP_NORMAL
    clip.Invert = CLIP_INVERT

    omega_min, omega_max = point_array_range(clip, VORTICITY_COMPONENT)
    peak_vorticity = max(abs(omega_min), abs(omega_max))
    if peak_vorticity == 0.0:
        return None, 0.0, 0, 0

    normalized_vorticity = simple.PythonCalculator(
        registrationName="NormalizedNormalVorticity",
        Input=clip,
    )
    normalized_vorticity.Expression = (
        f"abs({VORTICITY_COMPONENT}) / max(abs({VORTICITY_COMPONENT}))"
    )
    normalized_vorticity.ArrayName = "normalized_normal_vorticity"

    threshold = simple.Threshold(
        registrationName="TwoPercentVorticityThreshold",
        Input=normalized_vorticity,
    )
    threshold.Scalars = ["POINTS", "normalized_normal_vorticity"]
    threshold.UpperThreshold = VORTICITY_THRESHOLD_FRACTION
    threshold.ThresholdMethod = "Above Upper Threshold"
    threshold.UpdatePipeline()
    threshold_cells = threshold.GetDataInformation().GetNumberOfCells()
    if threshold_cells == 0:
        return None, peak_vorticity, 0, 0

    merge_blocks = simple.MergeBlocks(
        registrationName="MergeThresholdBlocks",
        Input=threshold,
    )
    merge_blocks.OutputDataSetType = "Unstructured Grid"
    merge_blocks.MergePartitionsOnly = 0
    merge_blocks.MergePoints = 1
    merge_blocks.Tolerance = 0.0

    connectivity = simple.Connectivity(
        registrationName="LeadingVortexRegion",
        Input=merge_blocks,
    )
    connectivity.ExtractionMode = "Extract Largest Region"
    connectivity.UpdatePipeline()
    leading_cells = connectivity.GetDataInformation().GetNumberOfCells()
    if leading_cells == 0:
        return None, peak_vorticity, threshold_cells, 0
    return connectivity, peak_vorticity, threshold_cells, leading_cells


def padded_camera_bounds(bounds):
    x_min = min(0.0, bounds[0])
    x_max = max(0.0, bounds[1])
    y_min = min(0.0, bounds[2])
    y_max = max(0.0, bounds[3])

    x_span = max(x_max - x_min, 1.0)
    y_span = max(y_max - y_min, 1.0)
    x_margin = CAMERA_MARGIN_FRACTION * x_span
    y_margin = CAMERA_MARGIN_FRACTION * y_span
    return [
        x_min - x_margin,
        x_max + x_margin,
        y_min - y_margin,
        y_max + y_margin,
    ]


def determine_camera_bounds(simple, snapshots):
    for snapshot in reversed(snapshots):
        simple.ResetSession()
        connectivity, _, _, _ = build_leading_region(simple, snapshot.resolve())
        if connectivity is not None:
            bounds = connectivity.GetDataInformation().GetBounds()
            camera_bounds = padded_camera_bounds(bounds)
            simple.ResetSession()
            return camera_bounds, snapshot
    raise RuntimeError("No selected snapshot contains a leading-vortex region")


def render_connectivity(simple, connectivity, png_path, camera_bounds):
    render_view = simple.GetActiveViewOrCreate("RenderView")
    simple.HideAll(render_view)
    render_view.ViewSize = FRAME_RESOLUTION
    render_view.InteractionMode = "2D"
    render_view.CameraParallelProjection = 1
    render_view.OrientationAxesVisibility = 0
    render_view.UseColorPaletteForBackground = 0
    render_view.Background = [1.0, 1.0, 1.0]

    x_min, x_max, y_min, y_max = camera_bounds
    x_center = 0.5 * (x_min + x_max)
    y_center = 0.5 * (y_min + y_max)
    x_span = x_max - x_min
    y_span = y_max - y_min
    aspect_ratio = FRAME_RESOLUTION[0] / FRAME_RESOLUTION[1]
    parallel_scale = 0.5 * max(y_span, x_span / aspect_ratio)
    camera_distance = max(x_span, y_span, 1.0)

    render_view.CameraPosition = [
        x_center,
        y_center,
        camera_distance,
    ]
    render_view.CameraFocalPoint = [x_center, y_center, 0.0]
    render_view.CameraViewUp = [0.0, 1.0, 0.0]
    render_view.CameraParallelScale = parallel_scale

    if connectivity is not None:
        display = simple.Show(
            connectivity,
            render_view,
            "UnstructuredGridRepresentation",
        )
        display.Representation = "Surface"
        simple.ColorBy(display, ("POINTS", VORTICITY_COMPONENT))
        display.RescaleTransferFunctionToDataRange(True, False)
        display.SetScalarBarVisibility(render_view, True)
        color_map = simple.GetColorTransferFunction(VORTICITY_COMPONENT)
        scalar_bar = simple.GetScalarBar(color_map, render_view)
        scalar_bar.TitleColor = [0.0, 0.0, 0.0]
        scalar_bar.LabelColor = [0.0, 0.0, 0.0]

    simple.SaveScreenshot(
        str(png_path),
        render_view,
        ImageResolution=FRAME_RESOLUTION,
    )
    if not png_path.is_file() or png_path.stat().st_size == 0:
        raise RuntimeError(f"ParaView did not create a valid PNG: {png_path}")


def calculate_and_render(
    simple,
    servermanager,
    snapshot,
    png_path,
    camera_bounds,
):
    # Resetting between snapshots avoids reader cache and time-state errors
    # observed when Chombo files are loaded as one ParaView file series.
    simple.ResetSession()
    (
        connectivity,
        peak_vorticity,
        threshold_cells,
        leading_cells,
    ) = build_leading_region(simple, snapshot)

    circulation = 0.0
    if connectivity is not None:
        integrate = simple.IntegrateVariables(
            registrationName="LeadingVortexIntegral",
            Input=connectivity,
        )
        integrate.UpdatePipeline()
        integrated_data = servermanager.Fetch(integrate)
        circulation = abs(fetched_scalar(integrated_data, VORTICITY_COMPONENT))

    render_connectivity(simple, connectivity, png_path, camera_bounds)
    return circulation, peak_vorticity, threshold_cells, leading_cells


def write_plot(simple, csv_path, png_path):
    simple.ResetSession()

    reader = simple.CSVReader(
        registrationName="CirculationData",
        FileName=[str(csv_path)],
    )
    reader.UpdatePipeline()

    chart = simple.CreateView("XYChartView")
    chart.ViewSize = PLOT_RESOLUTION
    chart.ChartTitle = "Circulation of the leading vortex"
    chart.LeftAxisTitle = "Circulation"
    chart.BottomAxisTitle = "Time"

    display = simple.Show(reader, chart, "XYChartRepresentation")
    display.UseIndexForXAxis = 0
    display.XArrayName = "time"
    display.SeriesVisibility = ["circulation"]
    display.SeriesLabel = ["circulation", "Leading vortex circulation"]
    display.SeriesLineThickness = ["circulation", "3"]

    simple.Render(chart)
    simple.SaveScreenshot(
        str(png_path),
        chart,
        ImageResolution=PLOT_RESOLUTION,
    )
    if not png_path.is_file() or png_path.stat().st_size == 0:
        raise RuntimeError(f"ParaView did not create a valid plot: {png_path}")


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
            "ffmpeg was not found. PNG frames and the circulation plot were "
            "saved, but the GIF could not be created."
        )
    return [ffmpeg]


def build_gif(frames_folder, snapshots, gif_path):
    gif_snapshots = snapshots[1:]
    if not gif_snapshots:
        raise ValueError(
            "Cannot build a GIF after excluding the initial frame: "
            "at least two analyzed snapshots are required."
        )

    print(f"Excluding initial GIF frame: {snapshots[0].name}", flush=True)
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
        config_path = find_config_file(snapshot_folder, args.config)
        cfl, dx_base, levels = read_time_metadata(config_path)
        (
            output_folder,
            frames_folder,
            gif_path,
            csv_path,
            plot_path,
        ) = output_paths(snapshot_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
        prepare_frames_folder(frames_folder)
        simple, servermanager = load_paraview()
        camera_bounds, camera_snapshot = determine_camera_bounds(simple, snapshots)

        print(f"Snapshot folder: {snapshot_folder}", flush=True)
        print(f"Config:          {config_path}", flush=True)
        print(f"Snapshots used:  {len(snapshots)}", flush=True)
        print(f"Stride:          {args.stride}", flush=True)
        print(f"CFL:             {cfl:g}", flush=True)
        print(f"dx_base:         {dx_base:g}", flush=True)
        print(f"nLevels:         {levels}", flush=True)
        print(f"Camera reference: {camera_snapshot.name}", flush=True)
        print(f"PNG frames:      {frames_folder}", flush=True)
        print(f"Output GIF:      {gif_path}", flush=True)
        print(f"Output CSV:      {csv_path}", flush=True)
        print(f"Output plot:     {plot_path}", flush=True)

        with csv_path.open("w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(
                [
                    "frame_index",
                    "snapshot_step",
                    "time",
                    "circulation",
                    "peak_vorticity",
                    "threshold_cells",
                    "leading_region_cells",
                    "snapshot_file",
                    "png_file",
                ]
            )

            for frame_index, snapshot in enumerate(snapshots):
                step = snapshot_step(snapshot)
                time = physical_time(step, cfl, dx_base, levels)
                frame_path = frames_folder / f"flowTime_{step}.png"
                print(
                    f"[{frame_index + 1}/{len(snapshots)}] "
                    f"Analyzing {snapshot.name}",
                    flush=True,
                )
                (
                    circulation,
                    peak_vorticity,
                    threshold_cells,
                    leading_cells,
                ) = calculate_and_render(
                    simple,
                    servermanager,
                    snapshot.resolve(),
                    frame_path,
                    camera_bounds,
                )
                writer.writerow(
                    [
                        frame_index,
                        step,
                        f"{time:.15g}",
                        f"{circulation:.16g}",
                        f"{peak_vorticity:.16g}",
                        threshold_cells,
                        leading_cells,
                        str(snapshot.resolve()),
                        str(frame_path.resolve()),
                    ]
                )
                csv_file.flush()
                print(
                    f"    time={time:.8g}, circulation={circulation:.8g}",
                    flush=True,
                )

        write_plot(simple, csv_path, plot_path)
        build_gif(frames_folder, snapshots, gif_path)
        simple.ResetSession()
        print(f"Done: {plot_path}", flush=True)
        print(f"GIF:  {gif_path}", flush=True)
        print(f"PNGs: {frames_folder}", flush=True)
        print(f"Data: {csv_path}", flush=True)
        return 0
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        print(f"Error: {error}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
