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
  $(basename "$0") RUN_FOLDER [STRIDE] [--field q-criterion|vorticity] [--vorticity-threshold-fraction FRACTION] [--show-domain-boundary] [--output-dir FOLDER] [--resume] [--workers N]

Examples:
  $(basename "$0") /ocean/projects/mch250004p/mchoi10/IBLGF-AMR/runs/ns_amr_lgf/2026-07-16_20-27-21
  $(basename "$0") /path/to/run 5 --field q-criterion
  $(basename "$0") /path/to/run 5 --field vorticity --vorticity-threshold-fraction 0.02 --show-domain-boundary
  $(basename "$0") /path/to/run 5 --field vorticity --resume --workers 8

Optional environment override:
  export PARAVIEW_BIN_DIR=/your/paraview/bin
  export PARAVIEW_FRAME_WORKERS=8
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

frame_workers="${PARAVIEW_FRAME_WORKERS:-1}"
forward_args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --workers)
      if [[ $# -lt 2 ]]; then
        echo "Error: --workers requires a positive integer." >&2
        exit 2
      fi
      frame_workers="$2"
      shift 2
      ;;
    --workers=*)
      frame_workers="${1#*=}"
      shift
      ;;
    *)
      forward_args+=("$1")
      shift
      ;;
  esac
done

if [[ ! "$frame_workers" =~ ^[1-9][0-9]*$ ]]; then
  echo "Error: --workers must be a positive integer; got $frame_workers" >&2
  exit 2
fi
if [[ "$frame_workers" -gt 1 ]]; then
  export OMP_NUM_THREADS=1
  export OPENBLAS_NUM_THREADS=1
  export MKL_NUM_THREADS=1
  export VTK_SMP_MAX_THREADS=1
fi

echo "Using pvbatch: ${PVBATCH}"
echo "Arguments:     ${forward_args[*]}"
echo "Frame workers: ${frame_workers}"
echo

if [[ "$frame_workers" -eq 1 ]]; then
  exec "${PVBATCH}" "${PY_SCRIPT}" "${forward_args[@]}"
fi

prepare_args=("${forward_args[@]}" --prepare-only --worker-count "$frame_workers")
"${PVBATCH}" "${PY_SCRIPT}" "${prepare_args[@]}"

worker_pids=()
worker_indices=()
terminate_workers() {
  local pid
  for pid in "${worker_pids[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap terminate_workers INT TERM

launch_worker() {
  local worker_index="$1"
  local worker_args=(
    "${forward_args[@]}"
    --resume
    --frames-only
    --worker-index "$worker_index"
    --worker-count "$frame_workers"
  )

  if [[ -n "${SLURM_JOB_ID:-}" ]] && command -v srun >/dev/null 2>&1; then
    srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task=1 --cpu-bind=cores \
      "${PVBATCH}" "${PY_SCRIPT}" "${worker_args[@]}" &
  else
    MV2_ENABLE_AFFINITY=0 \
      "${PVBATCH}" "${PY_SCRIPT}" "${worker_args[@]}" &
  fi
  worker_pids+=("$!")
  worker_indices+=("$worker_index")
}

for ((worker_index = 0; worker_index < frame_workers; worker_index++)); do
  launch_worker "$worker_index"
done

failed=0
for array_index in "${!worker_pids[@]}"; do
  if ! wait "${worker_pids[$array_index]}"; then
    echo "Frame worker ${worker_indices[$array_index]} failed." >&2
    failed=1
  fi
done
trap - INT TERM

if [[ "$failed" -ne 0 ]]; then
  echo "One or more frame workers failed; the partial PNG set can be resumed." >&2
  exit 1
fi

"${PVBATCH}" "${PY_SCRIPT}" \
  "${forward_args[@]}" \
  --resume \
  --assemble-only \
  --worker-count "$frame_workers"
