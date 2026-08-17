#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_script="${script_dir}/plot_circulation.py"

# Override with: export PARAVIEW_BIN_DIR=/path/to/ParaView/bin
paraview_bin_dir="${PARAVIEW_BIN_DIR:-/ocean/projects/mch250004p/mchoi10/apps/ParaView-5.13.3-osmesa-MPI-Linux-Python3.10-x86_64/bin}"
pvbatch="${paraview_bin_dir}/pvbatch"
mpi_ranks="${PARAVIEW_MPI_RANKS:-1}"

usage() {
  cat <<EOF
Usage:
  $(basename "$0") RUN_FOLDER [STRIDE] [--vorticity-threshold-fraction FRACTION] [--center-threshold-fraction FRACTION] [--config CONFIG_FILE] [--data-only] [--output-dir FOLDER] [--resume]

Examples:
  $(basename "$0") runs/ns_amr_lgf/res_sweep/amr_1 1
  $(basename "$0") runs/ns_amr_lgf/formation/tau_5p0 1 --vorticity-threshold-fraction 0.02
  $(basename "$0") runs/ns_amr_lgf/res_sweep/amr_1 1 --center-threshold-fraction 0.4
  $(basename "$0") runs/ns_amr_lgf/res_sweep/amr_2 1 --resume

Outputs:
  circulation CSV and plot, x-center plot, y-center plot, slice frames, and GIF

Environment:
  PARAVIEW_MPI_RANKS=N  run pvbatch with N MPI ranks (default: 1)
  PARAVIEW_MPIEXEC=PATH override ParaView's bundled mpiexec
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

if [[ ! -x "${pvbatch}" ]]; then
  echo "Error: pvbatch not found or not executable at:"
  echo "  ${pvbatch}"
  echo "Set PARAVIEW_BIN_DIR to the correct ParaView bin directory."
  exit 1
fi
if [[ ! "$mpi_ranks" =~ ^[1-9][0-9]*$ ]]; then
  echo "Error: PARAVIEW_MPI_RANKS must be a positive integer." >&2
  exit 1
fi

echo "Using pvbatch: ${pvbatch}"
echo "Arguments:     $*"
echo "MPI ranks:     ${mpi_ranks}"
echo

if [[ "$mpi_ranks" -eq 1 ]]; then
  exec env -u PYTHONHOME -u PYTHONPATH "${pvbatch}" "${python_script}" "$@"
fi

mpi_exec="${PARAVIEW_MPIEXEC:-${paraview_bin_dir}/mpiexec}"
if [[ ! -x "$mpi_exec" ]]; then
  echo "Error: ParaView's MPI launcher was not found or is not executable:" >&2
  echo "  $mpi_exec" >&2
  echo "Set PARAVIEW_MPIEXEC to the mpiexec bundled with this ParaView build." >&2
  exit 1
fi

echo "Using mpiexec: ${mpi_exec}"
exec env -u PYTHONHOME -u PYTHONPATH \
  "$mpi_exec" -np "$mpi_ranks" "$pvbatch" "$python_script" "$@"
