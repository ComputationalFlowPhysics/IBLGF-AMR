#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="${SCRIPT_DIR}/plot_3d.py"

# Override this with:
#   export PARAVIEW_BIN_DIR=/path/to/ParaView-.../bin
PARAVIEW_BIN_DIR="${PARAVIEW_BIN_DIR:-/ocean/projects/mch250004p/mchoi10/apps/ParaView-5.13.3-osmesa-MPI-Linux-Python3.10-x86_64/bin}"
PVBATCH="${PARAVIEW_BIN_DIR}/pvbatch"

usage() {
  cat <<EOF
Usage:
  $(basename "$0") RUN_FOLDER [STRIDE] [--field q-criterion|vorticity] [--vorticity-threshold-fraction FRACTION] [--show-domain-boundary] [--output-dir FOLDER] [--resume]

Examples:
  $(basename "$0") /ocean/projects/mch250004p/mchoi10/IBLGF-AMR/runs/ns_amr_lgf/2026-07-16_20-27-21
  $(basename "$0") /path/to/run 5 --field q-criterion
  $(basename "$0") /path/to/run 5 --field vorticity --vorticity-threshold-fraction 0.02 --show-domain-boundary
  $(basename "$0") /path/to/run 5 --field vorticity --resume

Optional environment override:
  export PARAVIEW_BIN_DIR=/your/paraview/bin
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

if [[ ! -x "${PVBATCH}" ]]; then
  echo "Error: pvbatch not found or not executable at:"
  echo "  ${PVBATCH}"
  echo
  echo "Set the correct ParaView bin folder, for example:"
  echo "  export PARAVIEW_BIN_DIR=/ocean/projects/mch250004p/mchoi10/apps/ParaView-5.13.3-osmesa-MPI-Linux-Python3.10-x86_64/bin"
  exit 1
fi

echo "Using pvbatch: ${PVBATCH}"
echo "Arguments:     $*"
echo

"${PVBATCH}" "${PY_SCRIPT}" "$@"
