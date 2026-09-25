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
  $(basename "$0") RUN_FOLDER [STRIDE] [--vorticity-threshold-fraction FRACTION] [--center-threshold-fraction FRACTION] [--config CONFIG_FILE] [--data-only] [--output-dir FOLDER] [--resume] [--render-from-csv] [--workers N]

Examples:
  $(basename "$0") runs/ns_amr_lgf/res_sweep/amr_1 1
  $(basename "$0") runs/ns_amr_lgf/formation/tau_5p0 1 --vorticity-threshold-fraction 0.02
  $(basename "$0") runs/ns_amr_lgf/res_sweep/amr_1 1 --center-threshold-fraction 0.4
  $(basename "$0") runs/ns_amr_lgf/res_sweep/amr_2 1 --resume
  $(basename "$0") runs/ns_amr_lgf/formation/tau_5p0 1 --data-only --workers 64
  $(basename "$0") runs/ns_amr_lgf/formation/tau_5p0 1 --render-from-csv --output-dir ritta_plotting_3D/outputs/formation_tau_5p0_circulation

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

default_workers="${PARAVIEW_FRAME_WORKERS:-}"
if [[ -z "$default_workers" ]]; then
  if [[ "${SLURM_NTASKS:-1}" =~ ^[1-9][0-9]*$ ]] && [[ "${SLURM_NTASKS:-1}" -gt 1 ]]; then
    default_workers="$SLURM_NTASKS"
  elif [[ "${SLURM_CPUS_PER_TASK:-1}" =~ ^[1-9][0-9]*$ ]] && [[ "${SLURM_CPUS_PER_TASK:-1}" -gt 1 ]]; then
    default_workers="$SLURM_CPUS_PER_TASK"
  else
    default_workers=1
  fi
fi

frame_workers="$default_workers"
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

if [[ ${#forward_args[@]} -lt 1 ]]; then
  usage >&2
  exit 2
fi
if [[ ! "$frame_workers" =~ ^[1-9][0-9]*$ ]]; then
  echo "Error: --workers must be a positive integer; got $frame_workers" >&2
  exit 2
fi

input_folder="${forward_args[0]}"
requested_stride=1
if [[ "${forward_args[1]:-}" =~ ^[1-9][0-9]*$ ]]; then
  requested_stride="${forward_args[1]}"
fi
snapshot_folder="$input_folder"
if [[ -d "$input_folder/output" ]]; then
  snapshot_folder="$input_folder/output"
fi
shopt -s nullglob
snapshot_files=("$snapshot_folder"/flowTime_*.hdf5)
selected_snapshot_count=$(((${#snapshot_files[@]} + requested_stride - 1) / requested_stride))
if [[ "$selected_snapshot_count" -gt 0 && "$frame_workers" -gt "$selected_snapshot_count" ]]; then
  echo "Reducing frame workers from $frame_workers to $selected_snapshot_count selected snapshots."
  frame_workers="$selected_snapshot_count"
fi

if [[ "$frame_workers" -gt 1 ]]; then
  data_only=0
  for argument in "${forward_args[@]}"; do
    [[ "$argument" == "--data-only" ]] && data_only=1
  done
  if [[ "$data_only" -ne 1 ]]; then
    echo "Error: multiple circulation workers currently require --data-only." >&2
    exit 2
  fi
  export OMP_NUM_THREADS=1
  export OPENBLAS_NUM_THREADS=1
  export MKL_NUM_THREADS=1
  export VTK_SMP_MAX_THREADS=1
fi

if [[ ! -x "${pvbatch}" ]]; then
  echo "Error: pvbatch not found or not executable at:"
  echo "  ${pvbatch}"
  echo "Set PARAVIEW_BIN_DIR to the correct ParaView bin directory."
  exit 1
fi
echo "Using pvbatch: ${pvbatch}"
echo "Arguments:     ${forward_args[*]}"
echo "Frame workers: ${frame_workers}"
echo

if [[ "$frame_workers" -eq 1 ]]; then
  exec env -u PYTHONHOME -u PYTHONPATH \
    "${pvbatch}" "${python_script}" "${forward_args[@]}"
fi

env -u PYTHONHOME -u PYTHONPATH \
  "${pvbatch}" "${python_script}" \
  "${forward_args[@]}" \
  --prepare-only \
  --worker-count "$frame_workers"

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
    --worker-only
    --worker-index "$worker_index"
    --worker-count "$frame_workers"
  )

  if [[ -n "${SLURM_JOB_ID:-}" ]] && command -v srun >/dev/null 2>&1; then
    srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task=1 --cpu-bind=cores \
      env -u PYTHONHOME -u PYTHONPATH \
      "${pvbatch}" "${python_script}" "${worker_args[@]}" &
  else
    MV2_ENABLE_AFFINITY=0 env -u PYTHONHOME -u PYTHONPATH \
      "${pvbatch}" "${python_script}" "${worker_args[@]}" &
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
    echo "Circulation worker ${worker_indices[$array_index]} failed." >&2
    failed=1
  fi
done
trap - INT TERM

if [[ "$failed" -ne 0 ]]; then
  echo "One or more circulation workers failed; rerun with --resume." >&2
  exit 1
fi

env -u PYTHONHOME -u PYTHONPATH \
  "${pvbatch}" "${python_script}" \
  "${forward_args[@]}" \
  --assemble-only \
  --worker-count "$frame_workers"
