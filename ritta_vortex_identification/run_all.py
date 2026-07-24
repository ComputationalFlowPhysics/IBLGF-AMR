"""Run Stages 1-4 headlessly for one run or a folder of runs."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from common import discover_frames, load_config


SCRIPT_FOLDER = Path(__file__).resolve().parent
STAGES = (
    "01_find_hmaxima.py",
    "02_make_regions.py",
    "03_fit_vortices.py",
    "04_positive_vortex_metrics.py",
)
SAVED_RESULTS = (
    "hmaxima.h5",
    "regions.h5",
    "fits.h5",
    "positive_vortex_metrics.csv",
)


def output_has_frames(output_folder: Path, pattern: str) -> bool:
    return output_folder.is_dir() and any(path.is_file() for path in output_folder.glob(pattern))


def one_run(folder: str | Path, pattern: str) -> Path:
    """Accept either a run folder or its canonical output folder."""
    folder = Path(folder).expanduser().resolve()
    if output_has_frames(folder / "output", pattern):
        return folder
    if folder.name == "output" and output_has_frames(folder, pattern):
        return folder.parent
    raise FileNotFoundError(
        f"Expected either a run folder containing output/{pattern}, or that output folder itself: {folder}"
    )


def batch_runs(folder: str | Path, pattern: str) -> list[Path]:
    """Find every run below a parent by locating folders named output."""
    folder = Path(folder).expanduser().resolve()
    if not folder.is_dir():
        raise FileNotFoundError(f"Batch folder does not exist: {folder}")
    candidates = []
    if folder.name == "output" and output_has_frames(folder, pattern):
        candidates.append(folder.parent)
    if output_has_frames(folder / "output", pattern):
        candidates.append(folder)
    candidates.extend(
        output_folder.parent
        for output_folder in folder.rglob("output")
        if output_has_frames(output_folder, pattern)
    )
    runs = sorted(set(candidates))
    if not runs:
        raise FileNotFoundError(f"No run folders containing output/{pattern} were found under {folder}")
    return runs


def dataset_label(run_folder: Path, search_root: Path, batch: bool) -> str:
    """Use a relative path so nested batch runs get distinct readable labels."""
    if not batch:
        return run_folder.name
    try:
        relative = run_folder.relative_to(search_root)
    except ValueError:
        return run_folder.name
    return run_folder.name if str(relative) == "." else relative.as_posix()


def filename_key(label: str) -> str:
    """Turn a dataset label into a safe result-folder name."""
    key = re.sub(r"[^A-Za-z0-9._-]+", "__", label).strip("._-")
    return key or "dataset"


def run_stages(run_folder: Path, config_file: Path, destination: Path, stride: int) -> None:
    """Run Stages 1-4 in order and keep their HDF5 files and final CSV."""
    with tempfile.TemporaryDirectory(prefix=f"ritta-vortex-{run_folder.name}-") as temporary:
        temporary_folder = Path(temporary)
        environment = os.environ.copy()
        # Stage files are created together before the completed set is copied out.
        environment["RITTA_VORTEX_RESULT_FOLDER"] = str(temporary_folder)
        environment["MPLCONFIGDIR"] = str(temporary_folder / ".matplotlib")
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        # Subprocesses preserve the exact numbered-script behavior and dependency order.
        for stage in STAGES:
            command = [
                sys.executable,
                str(SCRIPT_FOLDER / stage),
                str(run_folder),
                str(config_file),
                "--no-preview",
            ]
            if stage == "01_find_hmaxima.py":
                command.extend(("--stride", str(stride)))
            print(f"\n[{run_folder.name}] {stage}", flush=True)
            subprocess.run(command, cwd=SCRIPT_FOLDER, env=environment, check=True)

        missing = [name for name in SAVED_RESULTS if not (temporary_folder / name).is_file()]
        if missing:
            raise FileNotFoundError("Pipeline did not create: " + ", ".join(missing))
        destination.mkdir(parents=True, exist_ok=True)
        for name in SAVED_RESULTS:
            shutil.copy2(temporary_folder / name, destination / name)


def write_manifest(path: Path, datasets: list[dict]) -> None:
    """Write the small file where the user edits combined-plot legend names."""
    lines = [
        "# Edit each name to control the legend labels in the combined plots.",
        "# CSV paths are relative to this file unless they are absolute.",
        "",
    ]
    for dataset in datasets:
        lines.extend((
            "[[dataset]]",
            f"name = {json.dumps(dataset['name'])}",
            f"csv = {json.dumps(dataset['csv'])}",
            f"run_folder = {json.dumps(dataset['run_folder'])}",
            "",
        ))
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run vortex-identification Stages 1-4 without previews and keep HDF5 and CSV results."
    )
    parser.add_argument("input_folder", type=Path, help="One run/output folder, or a parent folder with --batch.")
    parser.add_argument("config_file", type=Path)
    parser.add_argument("--batch", action="store_true", help="Recursively process every run containing output HDF5 files.")
    parser.add_argument("--stride", type=int, default=1, help="Process every Nth sorted HDF5 frame.")
    parser.add_argument("--dataset-name", help="Legend name for single-run mode; batch names are edited in datasets.toml.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=SCRIPT_FOLDER / "outputs" / "pipeline_results",
        help="Folder for final CSV files and datasets.toml.",
    )
    args = parser.parse_args()
    if args.stride < 1:
        parser.error("--stride must be a positive integer.")

    config_file = args.config_file.expanduser().resolve()
    config = load_config(config_file)
    pattern = str(config["input"].get("hdf5_glob", "flowTime_*.hdf5"))
    search_root = args.input_folder.expanduser().resolve()
    # Batch mode recursively discovers runs; single mode normalizes one supplied path.
    runs = batch_runs(search_root, pattern) if args.batch else [one_run(search_root, pattern)]
    if args.batch and args.dataset_name:
        parser.error("--dataset-name is only available for a single run.")

    results_folder = args.results_dir.expanduser().resolve()
    results_folder.mkdir(parents=True, exist_ok=True)
    datasets = []
    used_keys = set()
    failures = []
    for index, run_folder in enumerate(runs, start=1):
        label = args.dataset_name or dataset_label(run_folder, search_root, args.batch)
        key = filename_key(dataset_label(run_folder, search_root, args.batch))
        if key in used_keys:
            raise ValueError(f"Two run folders produce the same output name: {key}")
        used_keys.add(key)
        destination = results_folder / key
        print(f"\n=== Run {index}/{len(runs)}: {run_folder} ===", flush=True)
        try:
            discover_frames(run_folder, config)
            run_stages(run_folder, config_file, destination, args.stride)
        except (OSError, ValueError, subprocess.CalledProcessError) as error:
            failures.append((run_folder, error))
            print(f"FAILED: {run_folder}\n{error}", file=sys.stderr, flush=True)
            if not args.batch:
                return 1
            # A bad batch member does not discard CSV files from successful runs.
            continue
        datasets.append({
            "name": label,
            "csv": (Path(key) / "positive_vortex_metrics.csv").as_posix(),
            "run_folder": str(run_folder),
        })
        print(f"Saved HDF5 and CSV results: {destination}", flush=True)

    manifest = results_folder / "datasets.toml"
    write_manifest(manifest, datasets)
    print(f"\nSaved editable dataset list: {manifest}")
    if failures:
        print(f"{len(failures)} of {len(runs)} runs failed; successful results were kept.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
