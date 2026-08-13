#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_script="${script_dir}/plot_circulation.py"

# Override with: export PARAVIEW_BIN_DIR=/path/to/ParaView/bin
paraview_bin_dir="${PARAVIEW_BIN_DIR:-/ocean/projects/mch250004p/mchoi10/apps/ParaView-5.13.3-osmesa-MPI-Linux-Python3.10-x86_64/bin}"
pvbatch="${paraview_bin_dir}/pvbatch"

usage() {
  cat <<EOF
Usage:
  $(basename "$0") RUN_FOLDER [STRIDE] [--vorticity-threshold-fraction FRACTION] [--center-threshold-fraction FRACTION] [--config CONFIG_FILE] [--resume]

Examples:
  $(basename "$0") runs/ns_amr_lgf/res_sweep/amr_1 1
  $(basename "$0") runs/ns_amr_lgf/formation/tau_5p0 1 --vorticity-threshold-fraction 0.02
  $(basename "$0") runs/ns_amr_lgf/res_sweep/amr_1 1 --center-threshold-fraction 0.4
  $(basename "$0") runs/ns_amr_lgf/res_sweep/amr_2 1 --resume

Outputs:
  circulation CSV and plot, x-center plot, y-center plot, slice frames, and GIF
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

echo "Using pvbatch: ${pvbatch}"
echo "Arguments:     $*"
echo

exec env -u PYTHONHOME -u PYTHONPATH "${pvbatch}" "${python_script}" "$@"
