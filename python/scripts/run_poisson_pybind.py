from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path
import shutil
import subprocess
import sys


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def _git_commit(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _have_mpi_launcher() -> list[str]:
    if shutil.which("mpiexec"):
        return ["mpiexec", "-np"]
    if shutil.which("mpirun"):
        return ["mpirun", "-n"]
    raise RuntimeError("Neither `mpiexec` nor `mpirun` was found in PATH.")


def _in_mpi_job() -> bool:
    markers = (
        "OMPI_COMM_WORLD_SIZE",
        "PMI_SIZE",
        "MV2_COMM_WORLD_SIZE",
    )
    return any(name in os.environ for name in markers)


def _build_pythonpath(repo_root: Path) -> str:
    entries = [str(repo_root / "python")]
    existing = os.environ.get("PYTHONPATH")
    if existing:
        entries.append(existing)
    return os.pathsep.join(entries)


def _write_metadata(
    run_dir: Path,
    script_path: Path,
    config_name: str,
    mpi_ranks: int,
    build_dir: Path | None,
) -> None:
    repo_root = _repo_root()
    command = [sys.executable, str(script_path), config_name, "--worker"]
    if build_dir is not None:
        command.extend(["--build-dir", str(build_dir)])

    meta = "\n".join(
        [
            "test_name: poisson_pybind",
            f"script: {script_path}",
            f"config: {config_name}",
            f"mpi_ranks: {mpi_ranks}",
            f"git_commit: {_git_commit(repo_root)}",
            f"timestamp: {_timestamp()}",
            f"run_dir: {run_dir}",
            f"command: {' '.join(_have_mpi_launcher() + [str(mpi_ranks)] + command)}",
            "",
        ]
    )
    (run_dir / "meta.txt").write_text(meta)


def _run_worker(config_path: str) -> int:
    from iblgf import poisson

    result = poisson.run(config_path)
    print(f"config: {result.config_path}")
    print(f"measured_linf_error: {result.measured_linf_error}")
    print(f"expected_linf_error: {result.expected_linf_error}")
    print(f"difference: {result.difference}")
    return 0


def _run_controller(config_path: Path, mpi_ranks: int, build_dir: str | None) -> int:
    repo_root = _repo_root()
    run_dir = repo_root / "runs" / "poisson" / _timestamp()
    run_dir.mkdir(parents=True, exist_ok=False)

    staged_config = run_dir / config_path.name
    shutil.copy2(config_path, staged_config)

    script_path = Path(__file__).resolve()
    build_dir_path = Path(build_dir).resolve() if build_dir else None
    _write_metadata(run_dir, script_path, staged_config.name, mpi_ranks, build_dir_path)

    env = os.environ.copy()
    env["PYTHONPATH"] = _build_pythonpath(repo_root)
    if build_dir_path is not None:
        env["IBLGF_PYBUILD_DIR"] = str(build_dir_path)

    launcher = _have_mpi_launcher()
    worker_cmd = launcher + [str(mpi_ranks), sys.executable, str(script_path), "--worker", f"./{staged_config.name}"]

    print("==> Running test 'poisson_pybind'")
    print(f"    Run dir:  {run_dir}")
    print(f"    Config:   {staged_config.name}")
    print(f"    MPI:      {mpi_ranks}")
    print("    Logs:     stdout.log / stderr.log")
    print("    Expect:   output files created inside the run dir.")

    with (run_dir / "stdout.log").open("w") as stdout, (run_dir / "stderr.log").open("w") as stderr:
        completed = subprocess.run(
            worker_cmd,
            cwd=run_dir,
            env=env,
            stdout=stdout,
            stderr=stderr,
            check=False,
        )

    if completed.returncode != 0:
        print("==> Run failed.")
        print(f"    Inspect:  {run_dir / 'stderr.log'}")
        return completed.returncode

    print("==> Done.")
    print(f"    Outputs are in: {run_dir}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the Poisson pybind prototype in a staged runs/ directory."
    )
    parser.add_argument("config_path", help="Path to the Poisson config file to run.")
    parser.add_argument(
        "-n",
        "--np",
        type=int,
        default=2,
        help="MPI ranks to use when launching the pybind worker.",
    )
    parser.add_argument(
        "--build-dir",
        help="Optional build directory that contains the iblgf_bindings extension.",
    )
    parser.add_argument(
        "--worker",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    if args.worker:
        return _run_worker(args.config_path)

    if _in_mpi_job():
        raise SystemExit(
            "Run `run_poisson_pybind.py` directly, not under `mpiexec`. "
            "The script launches MPI itself so it can stage runs/, logs, and metadata."
        )

    config_path = Path(args.config_path).resolve()
    if not config_path.is_file():
        raise SystemExit(f"Config file not found: {config_path}")

    return _run_controller(config_path, args.np, args.build_dir)


if __name__ == "__main__":
    raise SystemExit(main())
