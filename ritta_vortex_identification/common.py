"""Shared configuration, HDF5, coordinate, and output helpers."""

from __future__ import annotations

import math
import os
import re
import shlex
import sys
import tomllib
from pathlib import Path

import h5py
import numpy as np


FLOW_TIME_RE = re.compile(r"^flowTime_(\d+)\.hdf5$")
NUMBER_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


# Configuration and frame ordering

def load_config(path: str | Path) -> dict:
    """Load the workflow TOML and validate settings shared by all stages."""
    path = Path(path).expanduser().resolve()
    with path.open("rb") as handle:
        config = tomllib.load(handle)

    for section in ("input", "hmaxima", "region", "fit", "plot", "time_series"):
        if section not in config:
            raise ValueError(f"Missing [{section}] section in {path}")

    if config["input"].get("field_name") != "edge_aux":
        raise ValueError("[input] field_name must be exactly 'edge_aux' for this 2D workflow.")
    if int(config["hmaxima"].get("connectivity", 0)) != 8:
        raise ValueError("[hmaxima] connectivity must be 8.")

    plot_config = config["plot"]
    for axis in ("x", "y"):
        minimum_name = f"{axis}_axis_min"
        maximum_name = f"{axis}_axis_max"
        try:
            minimum = float(plot_config.get(minimum_name, math.nan))
            maximum = float(plot_config.get(maximum_name, math.nan))
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"[plot] {minimum_name} and {maximum_name} must be numbers or nan."
            ) from error
        if math.isinf(minimum) or math.isinf(maximum):
            raise ValueError(
                f"[plot] {minimum_name} and {maximum_name} must be finite or nan."
            )
        if math.isfinite(minimum) and math.isfinite(maximum) and minimum >= maximum:
            raise ValueError(f"[plot] {minimum_name} must be smaller than {maximum_name}.")

    config["_path"] = path
    return config


def require_positive(config: dict, section: str, name: str) -> float:
    """Return a required finite positive configuration value."""
    value = float(config[section].get(name, math.nan))
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"[{section}] {name} must be replaced with a finite value greater than zero.")
    return value


def require_nonnegative(config: dict, section: str, name: str) -> float:
    value = float(config[section].get(name, math.nan))
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"[{section}] {name} must be replaced with a finite non-negative value.")
    return value


def discover_frames(run_folder: str | Path, config: dict) -> list[Path]:
    """Find output frames and order them by the integer in flowTime_<n>.hdf5."""
    run_folder = Path(run_folder).expanduser().resolve()
    output_folder = run_folder / "output"
    if not output_folder.is_dir():
        raise FileNotFoundError(f"Output folder does not exist: {output_folder}")

    pattern = str(config["input"].get("hdf5_glob", "flowTime_*.hdf5"))
    frames = [path for path in output_folder.glob(pattern) if path.is_file()]
    if not frames:
        raise FileNotFoundError(f"No HDF5 frames matched {output_folder / pattern}")

    invalid = [path.name for path in frames if FLOW_TIME_RE.fullmatch(path.name) is None]
    if invalid:
        raise ValueError(
            "Every selected frame must be named flowTime_<integer>.hdf5; invalid names: "
            + ", ".join(sorted(invalid))
        )
    return sorted(frames, key=lambda path: int(FLOW_TIME_RE.fullmatch(path.name).group(1)))


def frame_step(path: str | Path) -> int:
    match = FLOW_TIME_RE.fullmatch(Path(path).name)
    if match is None:
        raise ValueError(f"Cannot obtain a timestep from {Path(path).name}")
    return int(match.group(1))


# Simulation metadata and physical time

def _finite_override(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _strip_cpp_comments(text: str) -> str:
    return re.sub(r"//.*", "", text)


def _read_scalar(text: str, name: str) -> float | None:
    match = re.search(rf"\b{re.escape(name)}\s*=\s*({NUMBER_RE})\s*;", text)
    return float(match.group(1)) if match else None


def _read_vector(text: str, name: str) -> tuple[float, ...] | None:
    match = re.search(rf"\b{re.escape(name)}\s*=\s*\(([^)]*)\)\s*;", text)
    if match is None:
        return None
    try:
        return tuple(float(item.strip()) for item in match.group(1).split(","))
    except ValueError:
        return None


def _find_simulation_config(run_folder: Path, input_config: dict) -> Path | None:
    configured = str(input_config.get("simulation_config", "")).strip()
    if configured:
        path = Path(configured).expanduser()
        if not path.is_absolute():
            path = run_folder / path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Configured simulation file does not exist: {path}")
        return path

    pattern = str(input_config.get("simulation_config_glob", "config*"))
    candidates = {
        path.resolve(): path
        for folder in (run_folder / "output", run_folder)
        if folder.is_dir()
        for path in folder.glob(pattern)
        if path.is_file()
    }
    paths = sorted(candidates.values(), key=lambda path: (len(path.parts), path.name))
    if len(paths) > 1:
        names = ", ".join(str(path) for path in paths)
        raise ValueError(f"Multiple simulation configs were found; set [input] simulation_config: {names}")
    return paths[0] if paths else None


def simulation_metadata(run_folder: str | Path, config: dict) -> dict:
    """Read time constants and the physical coordinate origin independently."""
    run_folder = Path(run_folder).expanduser().resolve()
    input_config = config["input"]
    simulation_path = _find_simulation_config(run_folder, input_config)
    text = ""
    if simulation_path is not None:
        text = _strip_cpp_comments(simulation_path.read_text())

    cfl = _finite_override(input_config.get("cfl"))
    dx_base = _finite_override(input_config.get("dx_base"))
    level_override = input_config.get("num_amr_levels", -1)
    try:
        num_amr_levels = int(level_override)
    except (TypeError, ValueError):
        num_amr_levels = -1

    cfl = cfl if cfl is not None else _read_scalar(text, "cfl")
    dx_base = dx_base if dx_base is not None else _read_scalar(text, "dx_base")
    if num_amr_levels < 0:
        parsed_levels = _read_scalar(text, "nLevels")
        num_amr_levels = int(parsed_levels) if parsed_levels is not None else -1

    origin_x = _finite_override(input_config.get("origin_x_index"))
    origin_y = _finite_override(input_config.get("origin_y_index"))
    if origin_x is None or origin_y is None:
        block_match = re.search(r"\bblock\s*\{([^{}]*)\}", text, re.DOTALL)
        block_text = block_match.group(1) if block_match else ""
        base = _read_vector(block_text, "base")
        extent = _read_vector(block_text, "extent")
        if base is None or extent is None:
            base = _read_vector(text, "bd_base")
            extent = _read_vector(text, "bd_extent")
        if base is not None and extent is not None and len(base) >= 2 and len(extent) >= 2:
            origin_x = base[0] + 0.5 * extent[0]
            origin_y = base[1] + 0.5 * extent[1]

    if origin_x is None or origin_y is None:
        raise ValueError(
            "Physical coordinate origin was not found. Set origin_x_index and origin_y_index "
            "in [input], or provide a simulation config containing domain.block base/extent."
        )

    return {
        "cfl": cfl,
        "dx_base": dx_base,
        "num_amr_levels": num_amr_levels,
        "origin_index": (origin_x, origin_y),
        "source": str(simulation_path) if simulation_path is not None else "TOML overrides",
    }


def simulation_time(path: Path, frame_index: int, config: dict, metadata: dict) -> float:
    """Apply t = CFL * dx_base * filename step / 2**nLevels."""
    cfl = metadata.get("cfl")
    dx_base = metadata.get("dx_base")
    levels = int(metadata.get("num_amr_levels", -1))
    if cfl is not None and dx_base is not None and cfl > 0.0 and dx_base > 0.0 and levels >= 0:
        return float(cfl) * float(dx_base) * frame_step(path) / (2 ** levels)

    spacing = _finite_override(config["input"].get("fallback_time_spacing"))
    if spacing is None or spacing <= 0.0:
        raise ValueError(
            "Simulation time constants are incomplete and [input] fallback_time_spacing is not set."
        )
    return frame_index * spacing


# Small helpers for decoding Chombo HDF5 metadata

def _decode(value):
    if isinstance(value, bytes):
        return value.decode()
    array = np.asarray(value)
    if array.size == 1:
        item = array.reshape(-1)[0]
        return item.decode() if isinstance(item, bytes) else item.item() if hasattr(item, "item") else item
    return value


def _box_bounds(record, dims: int = 2) -> tuple[np.ndarray, np.ndarray]:
    names = getattr(record.dtype, "names", None)
    values = [int(record[name]) for name in names] if names else [int(v) for v in np.asarray(record).reshape(-1)]
    if len(values) < 2 * dims:
        raise ValueError("HDF5 box record does not contain enough bounds for 2D data.")
    return np.asarray(values[:dims]), np.asarray(values[dims:2 * dims])


def _chunk_boundaries(offsets: np.ndarray, boxes, data_size: int, components: int) -> np.ndarray:
    sizes = []
    for box in boxes:
        lower, upper = _box_bounds(box)
        sizes.append(int(np.prod(upper - lower + 1)) * components)

    # Different writers store offsets as boundaries or as chunk lengths.
    candidates = []
    if len(offsets) == len(sizes) + 1:
        candidates.append(offsets)
        if offsets[0] == 0:
            candidates.append(np.cumsum(offsets))
    if len(offsets) == len(sizes):
        candidates.append(np.concatenate(([0], np.cumsum(offsets))))

    for candidate in candidates:
        candidate = np.asarray(candidate, dtype=int)
        if (
            len(candidate) == len(sizes) + 1
            and candidate[0] == 0
            and candidate[-1] == data_size
            and np.array_equal(np.diff(candidate), sizes)
        ):
            return candidate
    raise ValueError("Could not interpret HDF5 chunk offsets.")


def _component_index(handle: h5py.File, field_name: str) -> int:
    fields = {}
    for name, value in handle.attrs.items():
        decoded = _decode(value)
        if not isinstance(decoded, str):
            continue
        match = re.search(r"(\d+)$", name)
        if match:
            fields[decoded] = int(match.group(1))
    if field_name not in fields:
        available = ", ".join(sorted(fields)) or "none"
        raise KeyError(f"Field '{field_name}' is absent. Available fields: {available}")
    return fields[field_name]


# Load the original AMR tiles before choosing a plotting or integration view.

def _load_vorticity_tiles(path: Path, config: dict, metadata: dict) -> list[dict]:
    """Read each original AMR chunk once and retain its physical geometry."""
    path = Path(path)
    tiles = []
    with h5py.File(path, "r") as handle:
        dims = int(_decode(handle["Chombo_global"].attrs["SpaceDim"]))
        if dims != 2:
            raise ValueError(f"{path.name} is {dims}D; this workflow accepts only 2D output.")
        levels = int(_decode(handle.attrs["num_levels"]))
        components = int(_decode(handle.attrs["num_components"]))
        # Component numbers vary between executables, so look up edge_aux by name.
        component = _component_index(handle, config["input"]["field_name"])

        for level in range(levels):
            group = handle[f"level_{level}"]
            hdf_dx = float(_decode(group.attrs["dx"]))
            dx = (
                float(metadata["dx_base"]) / (2 ** level)
                if metadata.get("dx_base") is not None
                else hdf_dx
            )
            origin = np.asarray(metadata["origin_index"], dtype=float) * (2 ** level)
            boxes = group["boxes"]
            data = group["data:datatype=0"]
            if "data_attributes" in group and "offsets" in group["data_attributes"]:
                offsets = np.asarray(group["data_attributes/offsets"], dtype=int)
            else:
                offsets = np.asarray(group["offsets"], dtype=int)
            boundaries = _chunk_boundaries(offsets, boxes, len(data), components)

            for chunk_index, box in enumerate(boxes):
                lower, upper = _box_bounds(box)
                counts = upper - lower + 1
                raw = np.asarray(data[boundaries[chunk_index]:boundaries[chunk_index + 1]])
                # Chombo flattens each CArrayBox in Fortran order.
                chunk = raw.reshape(int(counts[0]), int(counts[1]), components, order="F")
                values = np.asarray(chunk[..., component], dtype=float).T
                tiles.append({
                    "level": level,
                    "dx": dx,
                    "values": values,
                    "bounds": (
                        (lower[0] - origin[0]) * dx,
                        (upper[0] + 1 - origin[0]) * dx,
                        (lower[1] - origin[1]) * dx,
                        (upper[1] + 1 - origin[1]) * dx,
                    ),
                })

    if not tiles:
        raise ValueError(f"No vorticity cells were found in {path.name}")
    return tiles


def _rasterize_vorticity(tiles: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Create the plotting/fitting raster; finer values overwrite coarse coverage."""
    finest_dx = min(tile["dx"] for tile in tiles)
    x_min = min(tile["bounds"][0] for tile in tiles)
    x_max = max(tile["bounds"][1] for tile in tiles)
    y_min = min(tile["bounds"][2] for tile in tiles)
    y_max = max(tile["bounds"][3] for tile in tiles)
    nx = int(round((x_max - x_min) / finest_dx))
    ny = int(round((y_max - y_min) / finest_dx))
    omega = np.full((ny, nx), np.nan, dtype=float)

    # Write coarse levels first so finer data replaces covered coarse values.
    for tile in sorted(tiles, key=lambda item: item["level"]):
        ratio = int(round(tile["dx"] / finest_dx))
        if ratio < 1 or not math.isclose(tile["dx"], ratio * finest_dx, rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError("AMR level spacings are not integer multiples of the finest spacing.")
        values = np.repeat(np.repeat(tile["values"], ratio, axis=0), ratio, axis=1)
        bx0, _, by0, _ = tile["bounds"]
        ix0 = int(round((bx0 - x_min) / finest_dx))
        iy0 = int(round((by0 - y_min) / finest_dx))
        omega[iy0:iy0 + values.shape[0], ix0:ix0 + values.shape[1]] = values

    x = x_min + (np.arange(nx) + 0.5) * finest_dx
    y = y_min + (np.arange(ny) + 0.5) * finest_dx
    return x, y, omega, finest_dx


def _visible_amr_cells(tiles: list[dict]) -> dict[str, np.ndarray]:
    """Return original cells and dx^2 areas after removing finer-level coverage."""
    cell_x = []
    cell_y = []
    cell_omega = []
    cell_area = []
    for tile in tiles:
        x0, _, y0, _ = tile["bounds"]
        rows, columns = tile["values"].shape
        x = x0 + (np.arange(columns) + 0.5) * tile["dx"]
        y = y0 + (np.arange(rows) + 0.5) * tile["dx"]
        x_grid, y_grid = np.meshgrid(x, y)
        visible = np.ones(tile["values"].shape, dtype=bool)
        # Integrals keep native cells but remove every coarse cell covered by a finer box.
        for finer in tiles:
            if finer["level"] <= tile["level"]:
                continue
            fx0, fx1, fy0, fy1 = finer["bounds"]
            covered_x = (x >= fx0) & (x < fx1)
            covered_y = (y >= fy0) & (y < fy1)
            visible &= ~np.outer(covered_y, covered_x)
        cell_x.append(x_grid[visible])
        cell_y.append(y_grid[visible])
        cell_omega.append(tile["values"][visible])
        cell_area.append(np.full(np.count_nonzero(visible), tile["dx"] ** 2))
    return {
        "x": np.concatenate(cell_x),
        "y": np.concatenate(cell_y),
        "vorticity": np.concatenate(cell_omega),
        "area": np.concatenate(cell_area),
    }


def load_vorticity_frame(
    path: str | Path,
    frame_index: int,
    config: dict,
    metadata: dict,
    include_cells: bool = False,
) -> dict:
    """Load edge_aux as a physical raster and optionally expose visible original cells."""
    path = Path(path)
    tiles = _load_vorticity_tiles(path, config, metadata)
    x, y, omega, finest_dx = _rasterize_vorticity(tiles)
    frame = {
        "source_filename": path.name,
        "source_path": str(path.resolve()),
        "step": frame_step(path),
        "time": simulation_time(path, frame_index, config, metadata),
        "dx": finest_dx,
        "x": x,
        "y": y,
        "vorticity": omega,
    }
    if include_cells:
        frame["cells"] = _visible_amr_cells(tiles)
    return frame


# Output paths and small saved-file helpers

def result_folder(run_folder: str | Path) -> Path:
    # run_all.py uses the override for disposable intermediate stage files.
    override = os.environ.get("RITTA_VORTEX_RESULT_FOLDER", "").strip()
    folder = (
        Path(override).expanduser().resolve()
        if override
        else Path(__file__).resolve().parent / "outputs" / Path(run_folder).expanduser().resolve().name
    )
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def stage_command(script_name: str, run_folder: str | Path, config_file: str | Path) -> str:
    """Return an exact command for a missing prerequisite stage."""
    script = Path(__file__).with_name(script_name)
    return " ".join(
        shlex.quote(str(item))
        for item in (
            sys.executable,
            script,
            Path(run_folder).expanduser().resolve(),
            Path(config_file).expanduser().resolve(),
        )
    )


def write_string_dataset(handle: h5py.File, name: str, values: list[str]) -> None:
    handle.create_dataset(name, data=np.asarray(values, dtype=h5py.string_dtype("utf-8")))


def read_frame_order(handle: h5py.File) -> list[str]:
    return [item.decode() if isinstance(item, bytes) else str(item) for item in handle["frame_order"][:]]
