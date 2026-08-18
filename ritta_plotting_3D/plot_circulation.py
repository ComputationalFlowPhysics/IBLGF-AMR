#!/usr/bin/env pvpython
"""Plot circulation and Lamb's center for a 3D vortex ring.

Usage:
    pvbatch plot_circulation.py OUTPUT_FOLDER [STRIDE] [--config CONFIG_FILE]
        [--vorticity-threshold-fraction FRACTION]
        [--center-threshold-fraction FRACTION] [--view-only] [--resume]

``OUTPUT_FOLDER`` may be the folder containing ``flowTime_*.hdf5`` files or
the run folder containing an ``output`` subfolder. The vortex ring is assumed
to travel along x with its symmetry axis at y = z = 0. The center calculation
uses the positive-y half of the z=0 meridional slice, so its y coordinate is
the radial center coordinate. Outputs are saved under ``ritta_plotting_3D/outputs``.
View-only mode creates just PNG frames and a GIF.
"""

import argparse
import csv
import math
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
# Lamb-center core cutoff as a fraction of the maximum absolute vorticity.
# This is independently configurable with --center-threshold-fraction.
CENTER_THRESHOLD_FRACTION = 0.4
R2_VORTICITY_ARRAY = "center_r2_vorticity"
X_R2_VORTICITY_ARRAY = "center_x_r2_vorticity"
CONTOUR_COLOR = [0.0, 0.0, 0.0]
CONTOUR_LINE_WIDTH = 3.0
CENTER_MARKER_COLOR = [0.1, 0.8, 0.1]
CENTER_MARKER_RADIUS = 0.06
PLOT_RESOLUTION = [1200, 800]
FRAME_RESOLUTION = [1280, 720]
CAMERA_MARGIN_FRACTION = 0.08
FPS = 8
CSV_FIELDNAMES = [
    "frame_index",
    "snapshot_step",
    "time",
    "circulation",
    "peak_vorticity",
    "vorticity_threshold_fraction",
    "threshold_cells",
    "leading_region_cells",
    "center_threshold_fraction",
    "center_threshold_cells",
    "center_region_cells",
    "center_x",
    "center_y",
    "snapshot_file",
    "png_file",
]
TRANSPARENT_GIF_FILTER = (
    "[0:v]split[gif][palette_source];"
    "[palette_source]palettegen=stats_mode=diff:reserve_transparent=1[palette];"
    "[gif][palette]paletteuse=dither=sierra2_4a:alpha_threshold=128"
)


def positive_integer(value):
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("stride must be an integer") from error
    if number < 1:
        raise argparse.ArgumentTypeError("stride must be at least 1")
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
            "Render a 2D vortex-ring slice and calculate leading-vortex "
            "circulation and Lamb-center coordinates."
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
        help="process every STRIDE-th snapshot (default: 1)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help=(
            "simulation config used for physical time; normally discovered "
            "from the run folder"
        ),
    )
    parser.add_argument(
        "--vorticity-threshold-fraction",
        type=threshold_fraction,
        default=VORTICITY_THRESHOLD_FRACTION,
        help=(
            "leading-vortex cutoff as a fraction of maximum absolute "
            f"vorticity (default: {VORTICITY_THRESHOLD_FRACTION:g})"
        ),
    )
    parser.add_argument(
        "--center-threshold-fraction",
        type=threshold_fraction,
        default=CENTER_THRESHOLD_FRACTION,
        help=(
            "vorticity-core cutoff as a fraction of maximum absolute "
            f"vorticity (default: {CENTER_THRESHOLD_FRACTION:g})"
        ),
    )
    parser.add_argument(
        "--view-only",
        action="store_true",
        help=(
            "generate only positive-half-slice PNG frames and a GIF; skip "
            "the contour, circulation, and center analysis"
        ),
    )
    parser.add_argument(
        "--data-only",
        action="store_true",
        help=(
            "calculate CSV data and time-series plots without rendering "
            "slice PNGs or a GIF"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="analysis output folder; defaults to outputs/<run-name>_<mode>",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "reuse completed CSV rows and nonempty PNG frames, then process "
            "only missing snapshots"
        ),
    )
    parser.add_argument(
        "--render-from-csv",
        action="store_true",
        help=(
            "render missing slice frames using an existing complete CSV; "
            "reuse its circulation and center values instead of integrating "
            "them again"
        ),
    )
    args = parser.parse_args()
    if args.view_only and args.data_only:
        parser.error("--view-only and --data-only cannot be used together")
    if args.render_from_csv and (args.view_only or args.data_only):
        parser.error(
            "--render-from-csv cannot be combined with --view-only or --data-only"
        )
    return args


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


def output_paths(snapshot_folder, view_only, output_dir=None):
    run_name = (
        snapshot_folder.parent.name
        if snapshot_folder.name == "output"
        else snapshot_folder.name
    )
    output_suffix = "slice_view" if view_only else "circulation"
    output_folder = (
        output_dir.expanduser().resolve()
        if output_dir is not None
        else Path(__file__).resolve().parent
        / "outputs"
        / f"{run_name}_{output_suffix}"
    )
    frames_folder = output_folder / "frames"
    gif_name = (
        "positive_half_slice.gif"
        if view_only
        else "leading_vortex_connectivity.gif"
    )
    return (
        output_folder,
        frames_folder,
        output_folder / gif_name,
        output_folder / "leading_vortex_circulation.csv",
        output_folder / "leading_vortex_circulation.png",
        output_folder / "leading_vortex_center_x.png",
        output_folder / "leading_vortex_center_y.png",
    )


def prepare_frames_folder(frames_folder, resume=False):
    frames_folder.mkdir(parents=True, exist_ok=True)
    if resume:
        return
    for old_frame in frames_folder.iterdir():
        if old_frame.is_file() and re.fullmatch(
            r"flowTime_\d+\.png", old_frame.name
        ):
            old_frame.unlink()


def write_csv_rows(csv_path, rows):
    temporary_path = csv_path.with_suffix(f"{csv_path.suffix}.tmp")
    with temporary_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(csv_path)


def load_resume_rows(
    csv_path,
    snapshots,
    frames_folder,
    cfl,
    dx_base,
    levels,
    vorticity_threshold_fraction,
    center_threshold_fraction,
    require_frame=True,
):
    if not csv_path.is_file():
        return {}

    with csv_path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        missing_columns = [
            name
            for name in CSV_FIELDNAMES
            if name != "vorticity_threshold_fraction"
            and name not in (reader.fieldnames or [])
        ]
        if missing_columns:
            raise ValueError(
                "Cannot resume because the existing CSV is missing columns: "
                + ", ".join(missing_columns)
            )
        rows_by_step = {}
        for line_number, row in enumerate(reader, start=2):
            try:
                step = int(row["snapshot_step"])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Cannot resume: invalid snapshot_step on CSV line {line_number}"
                ) from error
            if step in rows_by_step:
                raise ValueError(
                    f"Cannot resume: duplicate snapshot step {step} in {csv_path}"
                )
            rows_by_step[step] = row

    reusable_rows = {}
    for frame_index, snapshot in enumerate(snapshots):
        step = snapshot_step(snapshot)
        row = rows_by_step.get(step)
        if row is None:
            continue

        try:
            saved_index = int(row["frame_index"])
            saved_time = float(row["time"])
            saved_vorticity_fraction = float(
                row.get("vorticity_threshold_fraction")
                or VORTICITY_THRESHOLD_FRACTION
            )
            saved_fraction = float(row["center_threshold_fraction"])
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Cannot resume: invalid metadata for snapshot step {step}"
            ) from error

        expected_time = physical_time(step, cfl, dx_base, levels)
        if saved_index != frame_index or not math.isclose(
            saved_time,
            expected_time,
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        ):
            raise ValueError(
                "Cannot resume because the stride or time metadata differs "
                f"at snapshot step {step}."
            )
        if not math.isclose(
            saved_vorticity_fraction,
            vorticity_threshold_fraction,
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        ):
            raise ValueError(
                "Cannot resume because --vorticity-threshold-fraction differs "
                f"at snapshot step {step}."
            )
        if not math.isclose(
            saved_fraction,
            center_threshold_fraction,
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        ):
            raise ValueError(
                "Cannot resume because --center-threshold-fraction differs "
                f"at snapshot step {step}."
            )
        saved_snapshot = Path(row["snapshot_file"]).expanduser().resolve()
        if saved_snapshot != snapshot.resolve():
            raise ValueError(
                f"Cannot resume: source snapshot differs at step {step}."
            )

        frame_path = frames_folder / f"flowTime_{step}.png"
        if not require_frame or (
            frame_path.is_file() and frame_path.stat().st_size > 0
        ):
            reusable_rows[step] = row

    return reusable_rows


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


def lamb_center_from_moments(
    vorticity_integral,
    r2_vorticity_integral,
    x_r2_vorticity_integral,
):
    """Return (x_c, r_c) from Lamb's signed-vorticity moment equations."""
    values = (
        vorticity_integral,
        r2_vorticity_integral,
        x_r2_vorticity_integral,
    )
    if not all(math.isfinite(value) for value in values):
        return math.nan, math.nan
    if vorticity_integral == 0.0 or r2_vorticity_integral == 0.0:
        return math.nan, math.nan

    radial_center_squared = r2_vorticity_integral / vorticity_integral
    if radial_center_squared < 0.0:
        return math.nan, math.nan

    axial_center = x_r2_vorticity_integral / r2_vorticity_integral
    return axial_center, math.sqrt(radial_center_squared)


def extract_largest_region(simple, input_proxy, fraction, name):
    threshold = simple.Threshold(
        registrationName=f"{name}Threshold",
        Input=input_proxy,
    )
    threshold.Scalars = ["POINTS", "normalized_normal_vorticity"]
    threshold.UpperThreshold = fraction
    threshold.ThresholdMethod = "Above Upper Threshold"
    threshold.UpdatePipeline()
    threshold_cells = threshold.GetDataInformation().GetNumberOfCells()
    if threshold_cells == 0:
        return None, 0, 0

    merge_blocks = simple.MergeBlocks(
        registrationName=f"Merge{name}ThresholdBlocks",
        Input=threshold,
    )
    merge_blocks.OutputDataSetType = "Unstructured Grid"
    merge_blocks.MergePartitionsOnly = 0
    merge_blocks.MergePoints = 1
    merge_blocks.Tolerance = 0.0

    connectivity = simple.Connectivity(
        registrationName=f"{name}Region",
        Input=merge_blocks,
    )
    connectivity.ExtractionMode = "Extract Largest Region"
    connectivity.UpdatePipeline()
    region_cells = connectivity.GetDataInformation().GetNumberOfCells()
    if region_cells == 0:
        return None, threshold_cells, 0
    return connectivity, threshold_cells, region_cells


def build_leading_regions(
    simple,
    snapshot,
    vorticity_threshold_fraction=VORTICITY_THRESHOLD_FRACTION,
    center_threshold_fraction=CENTER_THRESHOLD_FRACTION,
    analyze=True,
    calculate_center=True,
):
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
    if not analyze or peak_vorticity == 0.0:
        return None, None, clip, None, peak_vorticity, 0, 0, 0, 0

    normalized_vorticity = simple.PythonCalculator(
        registrationName="NormalizedNormalVorticity",
        Input=clip,
    )
    normalized_vorticity.Expression = (
        f"abs({VORTICITY_COMPONENT}) / {peak_vorticity:.17g}"
    )
    normalized_vorticity.ArrayName = "normalized_normal_vorticity"

    circulation_region, threshold_cells, leading_cells = extract_largest_region(
        simple,
        normalized_vorticity,
        vorticity_threshold_fraction,
        "LeadingVortex",
    )
    center_region = None
    center_threshold_cells = 0
    center_region_cells = 0
    if calculate_center:
        (
            center_region,
            center_threshold_cells,
            center_region_cells,
        ) = extract_largest_region(
            simple, normalized_vorticity, center_threshold_fraction, "LambCenterCore"
        )
    return (
        circulation_region,
        center_region,
        clip,
        normalized_vorticity,
        peak_vorticity,
        threshold_cells,
        leading_cells,
        center_threshold_cells,
        center_region_cells,
    )


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
        _, _, clip, _, _, _, _, _, _ = build_leading_regions(
            simple,
            snapshot.resolve(),
            analyze=False,
        )
        if clip.GetDataInformation().GetNumberOfCells() > 0:
            bounds = clip.GetDataInformation().GetBounds()
            camera_bounds = padded_camera_bounds(bounds)
            simple.ResetSession()
            return camera_bounds, snapshot
    raise RuntimeError("No selected snapshot contains a nonempty clipped slice")


def render_slice(
    simple,
    clip,
    normalized_vorticity,
    connectivity,
    peak_vorticity,
    vorticity_threshold_fraction,
    center_x,
    center_y,
    png_path,
    camera_bounds,
):
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

    if clip is not None:
        display = simple.Show(
            clip,
            render_view,
        )
        display.Representation = "Surface"
        simple.ColorBy(display, ("POINTS", VORTICITY_COMPONENT))
        display.SetScalarBarVisibility(render_view, True)
        color_map = simple.GetColorTransferFunction(VORTICITY_COMPONENT)
        color_map.RescaleTransferFunction(-peak_vorticity, peak_vorticity)
        scalar_bar = simple.GetScalarBar(color_map, render_view)
        scalar_bar.TitleColor = [0.0, 0.0, 0.0]
        scalar_bar.LabelColor = [0.0, 0.0, 0.0]

    if connectivity is not None and normalized_vorticity is not None:
        contour_input = simple.MergeBlocks(
            registrationName="MergeContourInputBlocks",
            Input=normalized_vorticity,
        )
        contour_input.OutputDataSetType = "Unstructured Grid"
        contour_input.MergePartitionsOnly = 0
        contour_input.MergePoints = 1
        contour_input.Tolerance = 0.0

        contour = simple.Contour(
            registrationName="VorticityCutoffContour",
            Input=contour_input,
        )
        contour.ContourBy = ["POINTS", "normalized_normal_vorticity"]
        contour.Isosurfaces = [vorticity_threshold_fraction]

        leading_bounds = connectivity.GetDataInformation().GetBounds()
        leading_center = [
            0.5 * (leading_bounds[0] + leading_bounds[1]),
            0.5 * (leading_bounds[2] + leading_bounds[3]),
            SLICE_ORIGIN[2],
        ]
        leading_contour = simple.Connectivity(
            registrationName="LeadingVortexContour",
            Input=contour,
        )
        leading_contour.ExtractionMode = "Extract Closest Point Region"
        leading_contour.ClosestPoint = leading_center

        contour_display = simple.Show(
            leading_contour,
            render_view,
            "GeometryRepresentation",
        )
        contour_display.Representation = "Surface"
        # Establish a point-data association before disabling scalar coloring.
        simple.ColorBy(
            contour_display,
            ("POINTS", "normalized_normal_vorticity"),
        )
        simple.ColorBy(contour_display, None)
        contour_display.AmbientColor = CONTOUR_COLOR
        contour_display.DiffuseColor = CONTOUR_COLOR
        contour_display.LineWidth = CONTOUR_LINE_WIDTH

    if math.isfinite(center_x) and math.isfinite(center_y):
        center_marker = simple.Sphere(
            registrationName="LambCenterMarker",
        )
        center_marker.Center = [center_x, center_y, SLICE_ORIGIN[2]]
        center_marker.Radius = CENTER_MARKER_RADIUS
        center_marker.ThetaResolution = 24
        center_marker.PhiResolution = 24

        marker_display = simple.Show(
            center_marker,
            render_view,
            "GeometryRepresentation",
        )
        marker_display.Representation = "Surface"
        marker_display.AmbientColor = CENTER_MARKER_COLOR
        marker_display.DiffuseColor = CENTER_MARKER_COLOR

    simple.SaveScreenshot(
        str(png_path),
        render_view,
        ImageResolution=FRAME_RESOLUTION,
        TransparentBackground=1,
    )
    if not png_path.is_file() or png_path.stat().st_size == 0:
        raise RuntimeError(f"ParaView did not create a valid PNG: {png_path}")


def calculate_and_render(
    simple,
    servermanager,
    snapshot,
    png_path,
    camera_bounds,
    vorticity_threshold_fraction=VORTICITY_THRESHOLD_FRACTION,
    center_threshold_fraction=CENTER_THRESHOLD_FRACTION,
    analyze=True,
    render=True,
    saved_row=None,
):
    # Resetting between snapshots avoids reader cache and time-state errors
    # observed when Chombo files are loaded as one ParaView file series.
    simple.ResetSession()
    (
        circulation_region,
        center_region,
        clip,
        normalized_vorticity,
        peak_vorticity,
        threshold_cells,
        leading_cells,
        center_threshold_cells,
        center_region_cells,
    ) = build_leading_regions(
        simple,
        snapshot,
        vorticity_threshold_fraction=vorticity_threshold_fraction,
        center_threshold_fraction=center_threshold_fraction,
        analyze=analyze or saved_row is not None,
        calculate_center=saved_row is None,
    )

    circulation = float(saved_row["circulation"]) if saved_row else 0.0
    if saved_row is None and analyze and circulation_region is not None:
        integrate = simple.IntegrateVariables(
            registrationName="LeadingVortexIntegral",
            Input=circulation_region,
        )
        integrate.UpdatePipeline()
        integrated_data = servermanager.Fetch(integrate)
        circulation = abs(fetched_scalar(integrated_data, VORTICITY_COMPONENT))

    center_x = float(saved_row["center_x"]) if saved_row else math.nan
    center_y = float(saved_row["center_y"]) if saved_row else math.nan
    if saved_row is None and analyze and center_region is not None:
        r2_vorticity = simple.Calculator(
            registrationName="LambCenterR2Vorticity",
            Input=center_region,
        )
        r2_vorticity.Function = (
            f"coordsY * coordsY * {VORTICITY_COMPONENT}"
        )
        r2_vorticity.ResultArrayName = R2_VORTICITY_ARRAY

        x_r2_vorticity = simple.Calculator(
            registrationName="LambCenterXR2Vorticity",
            Input=r2_vorticity,
        )
        x_r2_vorticity.Function = f"coordsX * {R2_VORTICITY_ARRAY}"
        x_r2_vorticity.ResultArrayName = X_R2_VORTICITY_ARRAY

        center_integral = simple.IntegrateVariables(
            registrationName="LambCenterMoments",
            Input=x_r2_vorticity,
        )
        center_integral.UpdatePipeline()
        center_data = servermanager.Fetch(center_integral)
        center_x, center_y = lamb_center_from_moments(
            fetched_scalar(center_data, VORTICITY_COMPONENT),
            fetched_scalar(center_data, R2_VORTICITY_ARRAY),
            fetched_scalar(center_data, X_R2_VORTICITY_ARRAY),
        )

    if render:
        render_slice(
            simple,
            clip,
            normalized_vorticity,
            circulation_region,
            peak_vorticity,
            vorticity_threshold_fraction,
            center_x,
            center_y,
            png_path,
            camera_bounds,
        )
    return (
        circulation,
        peak_vorticity,
        threshold_cells,
        leading_cells,
        center_x,
        center_y,
        center_threshold_cells,
        center_region_cells,
    )


def write_time_series_plot(
    simple,
    csv_path,
    png_path,
    series_name,
    chart_title,
    y_axis_title,
    series_label,
):
    simple.ResetSession()

    reader = simple.CSVReader(
        registrationName=f"{series_name}Data",
        FileName=[str(csv_path)],
    )
    reader.UpdatePipeline()

    chart = simple.CreateView("XYChartView")
    chart.ViewSize = PLOT_RESOLUTION
    chart.ChartTitle = chart_title
    chart.LeftAxisTitle = y_axis_title
    chart.BottomAxisTitle = "Time"

    display = simple.Show(reader, chart, "XYChartRepresentation")
    display.UseIndexForXAxis = 0
    display.XArrayName = "time"
    display.SeriesVisibility = [series_name]
    display.SeriesLabel = [series_name, series_label]
    display.SeriesLineThickness = [series_name, "3"]

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
            circulation_plot_path,
            center_x_plot_path,
            center_y_plot_path,
        ) = output_paths(snapshot_folder, args.view_only, args.output_dir)
        output_folder.mkdir(parents=True, exist_ok=True)
        if not args.data_only:
            prepare_frames_folder(
                frames_folder,
                resume=args.resume or args.render_from_csv,
            )
        simple, servermanager = load_paraview()
        camera_bounds = None
        camera_snapshot = None
        if not args.data_only:
            camera_bounds, camera_snapshot = determine_camera_bounds(
                simple, snapshots
            )

        print(f"Snapshot folder: {snapshot_folder}", flush=True)
        print(f"Config:          {config_path}", flush=True)
        print(f"Snapshots used:  {len(snapshots)}", flush=True)
        print(f"Stride:          {args.stride}", flush=True)
        print(f"CFL:             {cfl:g}", flush=True)
        print(f"dx_base:         {dx_base:g}", flush=True)
        print(f"nLevels:         {levels}", flush=True)
        if camera_snapshot is not None:
            print(f"Camera reference: {camera_snapshot.name}", flush=True)
        print(f"View only:        {args.view_only}", flush=True)
        print(f"Data only:        {args.data_only}", flush=True)
        print(f"Resume:           {args.resume}", flush=True)
        print(f"Render from CSV:  {args.render_from_csv}", flush=True)
        if not args.data_only:
            print(f"PNG frames:      {frames_folder}", flush=True)
            print(f"Output GIF:      {gif_path}", flush=True)
        if not args.view_only:
            print(
                "Circulation threshold: "
                f"{args.vorticity_threshold_fraction:g} of max |vorticity|",
                flush=True,
            )
            print(
                "Center threshold: "
                f"{args.center_threshold_fraction:g} of max |vorticity|",
                flush=True,
            )
            print(f"Output CSV:      {csv_path}", flush=True)
            print(f"Circulation plot: {circulation_plot_path}", flush=True)
            print(f"Center x plot:    {center_x_plot_path}", flush=True)
            print(f"Center y plot:    {center_y_plot_path}", flush=True)

        reusable_rows = {}
        csv_file = None
        writer = None
        if not args.view_only:
            if args.resume or args.render_from_csv:
                reusable_rows = load_resume_rows(
                    csv_path,
                    snapshots,
                    frames_folder,
                    cfl,
                    dx_base,
                    levels,
                    args.vorticity_threshold_fraction,
                    args.center_threshold_fraction,
                    require_frame=not (
                        args.data_only or args.render_from_csv
                    ),
                )
            if args.render_from_csv and len(reusable_rows) != len(snapshots):
                raise ValueError(
                    "--render-from-csv requires a complete, compatible "
                    f"CSV for all {len(snapshots)} selected snapshots; found "
                    f"{len(reusable_rows)} reusable rows in {csv_path}."
                )
            write_csv_rows(
                csv_path,
                [
                    reusable_rows[snapshot_step(snapshot)]
                    for snapshot in snapshots
                    if snapshot_step(snapshot) in reusable_rows
                ],
            )
            csv_file = csv_path.open("a", newline="")
            writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDNAMES)
            print(
                f"Reusable completed frames: {len(reusable_rows)}",
                flush=True,
            )

        try:
            for frame_index, snapshot in enumerate(snapshots):
                step = snapshot_step(snapshot)
                time = physical_time(step, cfl, dx_base, levels)
                frame_path = frames_folder / f"flowTime_{step}.png"
                reusable_view = (
                    args.view_only
                    and frame_path.is_file()
                    and frame_path.stat().st_size > 0
                )
                saved_row = reusable_rows.get(step)
                reusable_analysis = not args.view_only and saved_row is not None
                reusable_frame = (
                    frame_path.is_file() and frame_path.stat().st_size > 0
                )
                can_skip = reusable_view or (
                    reusable_analysis
                    and (not args.render_from_csv or reusable_frame)
                )
                if (args.resume or args.render_from_csv) and can_skip:
                    print(
                        f"[{frame_index + 1}/{len(snapshots)}] "
                        f"Reusing {snapshot.name}",
                        flush=True,
                    )
                    continue
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
                    center_x,
                    center_y,
                    center_threshold_cells,
                    center_region_cells,
                ) = calculate_and_render(
                    simple,
                    servermanager,
                    snapshot.resolve(),
                    frame_path,
                    camera_bounds,
                    vorticity_threshold_fraction=(
                        args.vorticity_threshold_fraction
                    ),
                    center_threshold_fraction=args.center_threshold_fraction,
                    analyze=not args.view_only,
                    render=not args.data_only,
                    saved_row=saved_row if args.render_from_csv else None,
                )
                if writer is not None and saved_row is None:
                    row = {
                        "frame_index": frame_index,
                        "snapshot_step": step,
                        "time": f"{time:.15g}",
                        "circulation": f"{circulation:.16g}",
                        "peak_vorticity": f"{peak_vorticity:.16g}",
                        "vorticity_threshold_fraction": (
                            f"{args.vorticity_threshold_fraction:.16g}"
                        ),
                        "threshold_cells": threshold_cells,
                        "leading_region_cells": leading_cells,
                        "center_threshold_fraction": (
                            f"{args.center_threshold_fraction:.16g}"
                        ),
                        "center_threshold_cells": center_threshold_cells,
                        "center_region_cells": center_region_cells,
                        "center_x": f"{center_x:.16g}",
                        "center_y": f"{center_y:.16g}",
                        "snapshot_file": str(snapshot.resolve()),
                        "png_file": (
                            "" if args.data_only else str(frame_path.resolve())
                        ),
                    }
                    writer.writerow(row)
                    reusable_rows[step] = row
                    csv_file.flush()
                    print(
                        f"    time={time:.8g}, "
                        f"circulation={circulation:.8g}, "
                        f"center=({center_x:.8g}, {center_y:.8g})",
                        flush=True,
                    )
                else:
                    print(f"    time={time:.8g}", flush=True)
        finally:
            if csv_file is not None:
                csv_file.close()

        if not args.view_only:
            write_csv_rows(
                csv_path,
                [reusable_rows[snapshot_step(snapshot)] for snapshot in snapshots],
            )

        if not args.view_only:
            write_time_series_plot(
                simple,
                csv_path,
                circulation_plot_path,
                "circulation",
                "Circulation of the leading vortex",
                "Circulation",
                "Leading vortex circulation",
            )
            write_time_series_plot(
                simple,
                csv_path,
                center_x_plot_path,
                "center_x",
                "Axial center of the leading vortex",
                "Center x-coordinate",
                "Lamb center x",
            )
            write_time_series_plot(
                simple,
                csv_path,
                center_y_plot_path,
                "center_y",
                "Radial center of the leading vortex",
                "Center y-coordinate",
                "Lamb center y",
            )
        if not args.data_only:
            build_gif(frames_folder, snapshots, gif_path)
        simple.ResetSession()
        if not args.data_only:
            print(f"GIF:  {gif_path}", flush=True)
            print(f"PNGs: {frames_folder}", flush=True)
        if not args.view_only:
            print(f"Circulation plot: {circulation_plot_path}", flush=True)
            print(f"Center x plot:    {center_x_plot_path}", flush=True)
            print(f"Center y plot:    {center_y_plot_path}", flush=True)
            print(f"Data:             {csv_path}", flush=True)
        return 0
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        print(f"Error: {error}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
