#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="${SCRIPT_DIR}/animate_qcriterion.py"

# Override this with:
#   export PARAVIEW_BIN_DIR=/path/to/ParaView-.../bin
PARAVIEW_BIN_DIR="${PARAVIEW_BIN_DIR:-/ocean/projects/mch250004p/mchoi10/apps/ParaView-5.13.3-osmesa-MPI-Linux-Python3.10-x86_64/bin}"
PVBATCH="${PARAVIEW_BIN_DIR}/pvbatch"

usage() {
  cat <<EOF
Usage:
  $(basename "$0") RUN_FOLDER [STRIDE] [OUTPUT_MP4]

Examples:
  $(basename "$0") /ocean/projects/mch250004p/mchoi10/IBLGF-AMR/runs/ns_amr_lgf/2026-07-16_20-27-21
  $(basename "$0") /ocean/projects/mch250004p/mchoi10/IBLGF-AMR/runs/ns_amr_lgf/2026-07-16_20-27-21 5
  $(basename "$0") /ocean/projects/mch250004p/mchoi10/IBLGF-AMR/runs/ns_amr_lgf/2026-07-16_20-27-21 5 /ocean/projects/mch250004p/mchoi10/IBLGF-AMR/ritta_plotting_3D/outputs/ring_check.mp4

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

RUN_FOLDER="$1"
STRIDE="${2:-1}"
OUTPUT_MP4="${3:-}"

if [[ ! -x "${PVBATCH}" ]]; then
  echo "Error: pvbatch not found or not executable at:"
  echo "  ${PVBATCH}"
  echo
  echo "Set the correct ParaView bin folder, for example:"
  echo "  export PARAVIEW_BIN_DIR=/ocean/projects/mch250004p/mchoi10/apps/ParaView-5.13.3-osmesa-MPI-Linux-Python3.10-x86_64/bin"
  exit 1
fi

CMD=("${PVBATCH}" "${PY_SCRIPT}" "${RUN_FOLDER}" "${STRIDE}")

if [[ -n "${OUTPUT_MP4}" ]]; then
  CMD+=("${OUTPUT_MP4}")
fi

echo "Using pvbatch: ${PVBATCH}"
echo "Run folder:     ${RUN_FOLDER}"
echo "Stride:         ${STRIDE}"
if [[ -n "${OUTPUT_MP4}" ]]; then
  echo "Output MP4:     ${OUTPUT_MP4}"
fi
echo

"${CMD[@]}"
