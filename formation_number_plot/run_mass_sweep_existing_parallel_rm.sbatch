#!/bin/bash
# Run an existing mass-config sweep in parallel on one full RM node.
# Default layout: 4 concurrent jobs x 32 MPI ranks = 128 cores total.

#SBATCH -A mch250004p
#SBATCH -p RM
#SBATCH -N 1
#SBATCH --ntasks-per-node=128
#SBATCH -t 48:00:00
#SBATCH -J massfreq4
#SBATCH -o formation_number_plot/slurm-massfreq4-%j.out
#SBATCH -e formation_number_plot/slurm-massfreq4-%j.err

set -euo pipefail

repo_root="/ocean/projects/mch250004p/mchoi10/IBLGF-AMR"
configs_dir="${CONFIGS_DIR:-$repo_root/formation_number_plot/mass_configs}"
config_glob="${CONFIG_GLOB:-config_freq*.cfg}"
test_name="${TEST_NAME:-ns_amr_lgf2D}"
ranks_per_run="${RANKS_PER_RUN:-32}"
max_parallel="${MAX_PARALLEL:-4}"
progress_interval="${PROGRESS_INTERVAL:-20}"

cd "$repo_root"

if [[ ! -x ./iblgf_remote.sh ]]; then
  echo "Missing runner: $repo_root/iblgf_remote.sh" >&2
  exit 1
fi

if [[ ! "$ranks_per_run" =~ ^[1-9][0-9]*$ ]]; then
  echo "RANKS_PER_RUN must be a positive integer: $ranks_per_run" >&2
  exit 1
fi
if [[ ! "$max_parallel" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_PARALLEL must be a positive integer: $max_parallel" >&2
  exit 1
fi
if [[ ! "$progress_interval" =~ ^[1-9][0-9]*$ ]]; then
  echo "PROGRESS_INTERVAL must be a positive integer: $progress_interval" >&2
  exit 1
fi

total_ranks=$((ranks_per_run * max_parallel))
if (( total_ranks > 128 )); then
  echo "Requested $total_ranks total ranks, which exceeds one RM node." >&2
  exit 1
fi

mkdir -p formation_number_plot/mass_logs_remote

shopt -s nullglob
unsorted_configs=("$configs_dir"/$config_glob)
shopt -u nullglob

configs=()
if [[ "${#unsorted_configs[@]}" -gt 0 ]]; then
  while IFS= read -r config; do
    configs+=("$config")
  done < <(printf '%s\n' "${unsorted_configs[@]}" | sort -V)
fi

if [[ "${#configs[@]}" -eq 0 ]]; then
  echo "No configs matched '$config_glob' in $configs_dir" >&2
  exit 1
fi

echo "Parallel RM mass sweep"
echo "  Job:           ${SLURM_JOB_ID:-none}"
echo "  Host:          $(hostname)"
echo "  Configs:       $configs_dir/$config_glob"
echo "  Count:         ${#configs[@]}"
echo "  Test:          $test_name"
echo "  Ranks/run:     $ranks_per_run"
echo "  Parallel runs: $max_parallel"
echo "  Total ranks:   $total_ranks / 128"
echo

PIDS=()
declare -A PID_TO_NAME

launch_one() {
  local config="$1"
  local config_name config_stem stdout_log stderr_log
  config_name="$(basename "$config")"
  config_stem="${config_name%.cfg}"
  stdout_log="formation_number_plot/mass_logs_remote/${config_stem}.stdout.log"
  stderr_log="formation_number_plot/mass_logs_remote/${config_stem}.stderr.log"

  echo "Launching: $config_name"

  (
    export IBLGF_LAUNCHER=srun
    export IBLGF_MPI_ARGS="--exclusive"
    export PROGRESS_INTERVAL
    env -u SLURM_MPI_TYPE ./iblgf_remote.sh run-test "$test_name" "$config" -n "$ranks_per_run"
  ) >"$stdout_log" 2>"$stderr_log" &

  local pid=$!
  PIDS+=("$pid")
  PID_TO_NAME["$pid"]="$config_name"
}

wait_for_slot() {
  while (( ${#PIDS[@]} >= max_parallel )); do
    local new_pids=()
    local pid status name
    for pid in "${PIDS[@]}"; do
      if kill -0 "$pid" 2>/dev/null; then
        new_pids+=("$pid")
        continue
      fi

      if wait "$pid"; then
        echo "Completed: ${PID_TO_NAME[$pid]}"
      else
        name="${PID_TO_NAME[$pid]}"
        echo "Run failed: $name" >&2
        echo "  stdout: formation_number_plot/mass_logs_remote/${name%.cfg}.stdout.log" >&2
        echo "  stderr: formation_number_plot/mass_logs_remote/${name%.cfg}.stderr.log" >&2
        return 1
      fi
      unset 'PID_TO_NAME[$pid]'
    done
    PIDS=("${new_pids[@]}")
    sleep 5
  done
}

for config in "${configs[@]}"; do
  wait_for_slot
  launch_one "$config"
done

for pid in "${PIDS[@]}"; do
  if wait "$pid"; then
    echo "Completed: ${PID_TO_NAME[$pid]}"
  else
    name="${PID_TO_NAME[$pid]}"
    echo "Run failed: $name" >&2
    echo "  stdout: formation_number_plot/mass_logs_remote/${name%.cfg}.stdout.log" >&2
    echo "  stderr: formation_number_plot/mass_logs_remote/${name%.cfg}.stderr.log" >&2
    exit 1
  fi
done

echo
echo "All ${#configs[@]} configs completed."
echo "Logs: formation_number_plot/mass_logs_remote"
