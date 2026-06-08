#!/usr/bin/env bash

set -euo pipefail

script_dir() {
  cd "$(dirname "${BASH_SOURCE[0]}")" && pwd
}

repo_root() {
  script_dir
}

cpu_count() {
  if command -v nproc >/dev/null 2>&1; then
    nproc
  elif command -v sysctl >/dev/null 2>&1; then
    sysctl -n hw.ncpu
  else
    echo 4
  fi
}

usage() {
  cat <<'EOF'
Usage:
  ./iblgf-pybind.sh poisson <config_path> [-n MPI_RANKS] [-- <extra script args>]
  ./iblgf-pybind.sh ns-amr-2d <config_path> [-n MPI_RANKS] [-- <extra script args>]

Examples:
  ./iblgf-pybind.sh poisson ./tests/poisson/configFile_1ring -n 8
  ./iblgf-pybind.sh poisson ./tests/poisson/configFile_1ring -n 8 -- --vortex-override R=0.2
  ./iblgf-pybind.sh ns-amr-2d ./tests/ns_amr_lgf2D/configs/configFile_0 -- --simulation-override Re=250.0

Environment overrides:
  IBLGF_PYBUILD_DIR   Build directory for the pybind extension (default: <repo>/build-pybind)
  IBLGF_BUILD_JOBS    Parallel build jobs (default: detected CPU count)
EOF
}

die() {
  echo "Error: $*" >&2
  exit 1
}

solver="${1:-}"
config_path="${2:-}"

[[ -n "$solver" && -n "$config_path" ]] || {
  usage
  exit 1
}

shift 2

mpi_ranks="${IBLGF_MPI_RANKS:-2}"
extra_args=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -n|--np)
      [[ $# -ge 2 ]] || die "Missing value for $1"
      mpi_ranks="$2"
      shift 2
      ;;
    --)
      shift
      extra_args=("$@")
      break
      ;;
    *)
      die "Unknown option: $1"
      ;;
  esac
done

case "$solver" in
  poisson)
    script_path="$(repo_root)/python/scripts/simple_run_poisson_pybind.py"
    ;;
  ns-amr-2d)
    script_path="$(repo_root)/python/scripts/simple_run_ns_amr_2d_pybind.py"
    ;;
  *)
    die "Unknown solver '$solver'. Expected 'poisson' or 'ns-amr-2d'."
    ;;
esac

command -v python3 >/dev/null 2>&1 || die "python3 was not found"
command -v cmake >/dev/null 2>&1 || die "cmake was not found"
command -v mpiexec >/dev/null 2>&1 || die "mpiexec was not found"

if ! pybind11_cmake_dir="$(python3 -m pybind11 --cmakedir 2>/dev/null)"; then
  die "pybind11 is not available in this environment."
fi

build_dir="${IBLGF_PYBUILD_DIR:-$(repo_root)/build-pybind}"
build_jobs="${IBLGF_BUILD_JOBS:-$(cpu_count)}"

needs_configure=0
if [[ ! -f "$build_dir/Makefile" || ! -f "$build_dir/CMakeCache.txt" ]]; then
  needs_configure=1
elif ! grep -q '^IBLGF_BUILD_PYTHON:BOOL=ON$' "$build_dir/CMakeCache.txt"; then
  needs_configure=1
fi

if [[ "$needs_configure" -eq 1 ]]; then
  echo "==> Configuring pybind build in $build_dir"
  cmake -S "$(repo_root)" -B "$build_dir" \
    -DIBLGF_BUILD_PYTHON=ON \
    -Dpybind11_DIR="$pybind11_cmake_dir"
fi

echo "==> Building iblgf_bindings (-j $build_jobs)"
cmake --build "$build_dir" -j "$build_jobs" --target iblgf_bindings

cd "$(repo_root)"
export PYTHONPATH="$(repo_root)/python${PYTHONPATH:+:$PYTHONPATH}"
export IBLGF_PYBUILD_DIR="$build_dir"

echo "==> Running $solver with $mpi_ranks MPI rank(s)"
mpiexec -n "$mpi_ranks" python3 "$script_path" "$config_path" "${extra_args[@]}"
