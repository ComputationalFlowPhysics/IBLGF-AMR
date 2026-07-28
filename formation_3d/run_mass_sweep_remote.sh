#!/usr/bin/env bash
# Generate 3D mass-sweep configs and run them through the remote/HPC launcher.
# Existing configs can still be run without regeneration via `--existing`.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"

test_name="${TEST_NAME:-ns_amr_lgf}"
configs_dir="${CONFIGS_DIR:-$script_dir/mass_configs}"
logs_dir="${LOGS_DIR:-$script_dir/mass_logs_remote}"
runner="${IBLGF_RUNNER:-$repo_root/iblgf_remote.sh}"
generation_mode=1
sample_config=""
generator_script=""
progress_interval="${PROGRESS_INTERVAL:-20}"

sweep_log() {
  printf '[mass-sweep] %s\n' "$*"
}

case_log() {
  local case_name="$1"
  shift
  printf '[%s:%s] %s\n' "$test_name" "$case_name" "$*"
}

case_error() {
  case_log "$@" >&2
}

prefix_case_lines() {
  local case_name="$1"
  local line
  while IFS= read -r line || [[ -n "$line" ]]; do
    case_log "$case_name" "$line"
  done
}

cfg_int_value() {
  local cfg="$1"
  local key="$2"
  sed -n "s/^[[:space:]]*${key}[[:space:]]*=[[:space:]]*\\([0-9][0-9]*\\)[[:space:]]*;.*/\\1/p" "$cfg" | head -n 1
}

count_output_frames() {
  local run_dir="$1"
  local output_dir="$run_dir/output"
  local count=0

  [[ -d "$output_dir" ]] || {
    echo 0
    return 0
  }

  count=$(find "$output_dir" -maxdepth 1 -type f \( -name 'flowTime_*.hdf5' -o -name 'flow_*.hdf5' \) \
    ! -name 'flow_final.hdf5.hdf5' ! -name 'flow_final.hdf5' | wc -l | tr -d ' ')

  if [[ -f "$output_dir/flow_final.hdf5.hdf5" || -f "$output_dir/flow_final.hdf5" ]]; then
    count=$((count + 1))
  fi

  echo "$count"
}

format_elapsed() {
  local total="$1"
  printf '%02d:%02d:%02d' $((total / 3600)) $(((total % 3600) / 60)) $((total % 60))
}

monitor_run_progress() {
  local runner_pid="$1"
  local cfg="$2"
  local sweep_stdout_log="$3"
  local case_name="$4"

  local total_steps output_every approx_frames
  total_steps="$(cfg_int_value "$cfg" nBaseLevelTimeSteps || true)"
  output_every="$(cfg_int_value "$cfg" output_frequency || true)"
  approx_frames=0
  if [[ "${total_steps:-}" =~ ^[0-9]+$ ]] && [[ "${output_every:-}" =~ ^[1-9][0-9]*$ ]]; then
    approx_frames=$((2 + total_steps / output_every))
  fi

  local run_dir="" sim_stdout="" last_step="" last_frames="-1" start_ts elapsed frames step
  start_ts="$(date +%s)"

  while kill -0 "$runner_pid" 2>/dev/null; do
    if [[ -z "$run_dir" && -f "$sweep_stdout_log" ]]; then
      run_dir="$(sed -n 's/^    Run dir:  //p' "$sweep_stdout_log" | tail -n 1)"
      if [[ -n "$run_dir" ]]; then
        case_log "$case_name" "Run dir: $run_dir"
        sim_stdout="$run_dir/stdout.log"
      fi
    fi

    step=""
    frames=0
    if [[ -n "$sim_stdout" && -f "$sim_stdout" ]]; then
      step="$(sed -n 's/^T = [^,]*, n = \([0-9][0-9]*\).*/\1/p' "$sim_stdout" | tail -n 1)"
    fi
    if [[ -n "$run_dir" ]]; then
      frames="$(count_output_frames "$run_dir")"
    fi

    if [[ "$step" != "$last_step" || "$frames" != "$last_frames" ]]; then
      elapsed=$(( $(date +%s) - start_ts ))
      if [[ -n "$step" && "${total_steps:-}" =~ ^[0-9]+$ && "$total_steps" -gt 0 ]]; then
        if [[ "$approx_frames" -gt 0 ]]; then
          case_log "$case_name" "Progress: step $step/$total_steps, frame $frames/$approx_frames, elapsed $(format_elapsed "$elapsed")"
        else
          case_log "$case_name" "Progress: step $step/$total_steps, elapsed $(format_elapsed "$elapsed")"
        fi
      elif [[ -n "$run_dir" ]]; then
        if [[ "$approx_frames" -gt 0 ]]; then
          case_log "$case_name" "Progress: frame $frames/$approx_frames, elapsed $(format_elapsed "$elapsed")"
        else
          case_log "$case_name" "Progress: frame $frames, elapsed $(format_elapsed "$elapsed")"
        fi
      fi
      last_step="$step"
      last_frames="$frames"
    fi

    sleep "$progress_interval"
  done
}

usage() {
  cat <<EOF
Usage:
  $0 <sample_config> [mpi_ranks] [generator|freq|tau]
  $0 --existing [mpi_ranks] [config_glob]

Examples:
  $0 formation_3d/config3D_test 32 freq
  $0 formation_3d/config3D_test 32 tau
  $0 --existing 32 'config_freq*.cfg'

Environment overrides:
  IBLGF_RUNS_ROOT=/scratch/\$USER/iblgf-runs
  IBLGF_BUILD_DIR=/scratch/\$USER/iblgf-build
  IBLGF_LIB_ROOT=/path/to/iblgf-lib
  IBLGF_RUNNER=/path/to/IBLGF-AMR/iblgf_remote.sh
  CONFIGS_DIR=/path/to/configs
  CONFIG_GLOB='config_freq*.cfg'
  LOGS_DIR=/path/to/sweep-logs
  PROGRESS_INTERVAL=20
  TEST_NAME=ns_amr_lgf
  GENERATOR_SCRIPT=generate_mass_configs.py

This 3D version targets the ns_amr_lgf test. The remote launcher prefers the
MPI installation's mpiexec/mpirun, including inside a Slurm allocation. It
falls back to srun if necessary. This script does not request an allocation.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ "${1:-}" == "--existing" ]]; then
  generation_mode=0
  mpi_ranks="${2:-${IBLGF_MPI_RANKS:-8}}"
  config_pattern="${3:-${CONFIG_GLOB:-config_*.cfg}}"
else
  sample_config="${1:-}"
  mpi_ranks="${2:-${IBLGF_MPI_RANKS:-8}}"
  generator_arg="${3:-${GENERATOR_SCRIPT:-generate_mass_configs.py}}"

  if [[ -z "$sample_config" ]]; then
    usage >&2
    exit 2
  fi

  case "$generator_arg" in
    tau)
      generator_script="generate_mass_configs.py"
      config_pattern="config_tau*.cfg"
      ;;
    freq|2)
      generator_script="generate_mass_configs_2.py"
      config_pattern="config_freq*.cfg"
      ;;
    *.py)
      generator_script="$generator_arg"
      generator_base="$(basename "$generator_arg")"
      if [[ "$generator_base" == "generate_mass_configs_2.py" ]]; then
        config_pattern="config_freq*.cfg"
      else
        config_pattern="${CONFIG_GLOB:-config_*.cfg}"
      fi
      ;;
    *)
      generator_script="$generator_arg"
      config_pattern="${CONFIG_GLOB:-config_*.cfg}"
      ;;
  esac

  if [[ "$generator_script" != /* ]]; then
    generator_script="$script_dir/$generator_script"
  fi
fi

if [[ ! "$mpi_ranks" =~ ^[1-9][0-9]*$ ]]; then
  echo "MPI rank count must be a positive integer: $mpi_ranks" >&2
  usage >&2
  exit 2
fi
if [[ ! "$progress_interval" =~ ^[1-9][0-9]*$ ]]; then
  echo "PROGRESS_INTERVAL must be a positive integer: $progress_interval" >&2
  exit 2
fi

if [[ "$generation_mode" -eq 1 ]]; then
  if [[ ! -f "$sample_config" ]]; then
    echo "Sample config not found: $sample_config" >&2
    exit 2
  fi
  if [[ ! -f "$generator_script" ]]; then
    echo "Generator script not found: $generator_script" >&2
    exit 2
  fi
  if ! command -v python3 >/dev/null 2>&1; then
    echo "python3 is required to generate sweep configs." >&2
    exit 2
  fi

  echo "Generating remote 3D sweep configs"
  echo "  Sample:    $sample_config"
  echo "  Generator: $generator_script"
  echo "  Output:    $configs_dir"
  python3 "$generator_script" "$sample_config" --output-dir "$configs_dir"
elif [[ ! -d "$configs_dir" ]]; then
  echo "Config directory not found: $configs_dir" >&2
  exit 2
fi

if [[ ! -x "$runner" ]]; then
  echo "Remote IBLGF runner is not executable: $runner" >&2
  echo "Expected: $repo_root/iblgf_remote.sh" >&2
  exit 2
fi

if command -v srun >/dev/null 2>&1 && [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "Warning: Slurm is available, but no allocation is active." >&2
  echo "Obtain an allocation with salloc/sbatch before a large 3D sweep." >&2
fi

mkdir -p "$logs_dir"

shopt -s nullglob
unsorted_configs=("$configs_dir"/$config_pattern)
shopt -u nullglob

configs=()
if [[ "${#unsorted_configs[@]}" -gt 0 ]]; then
  while IFS= read -r config; do
    configs+=("$config")
  done < <(printf '%s\n' "${unsorted_configs[@]}" | sort -V)
fi

if [[ "${#configs[@]}" -eq 0 ]]; then
  echo "No configs matched '$config_pattern' in $configs_dir" >&2
  exit 1
fi

echo "Remote IBLGF 3D mass sweep"
echo "  Runner:  $runner"
echo "  Test:    $test_name"
echo "  Configs: $configs_dir/$config_pattern"
echo "  Count:   ${#configs[@]}"
echo "  MPI:     $mpi_ranks rank(s)"
echo "  Logs:    $logs_dir"
if [[ -n "${IBLGF_RUNS_ROOT:-}" ]]; then
  echo "  Outputs: $IBLGF_RUNS_ROOT"
else
  echo "  Outputs: $repo_root/runs"
fi

total="${#configs[@]}"
for idx in "${!configs[@]}"; do
  config="${configs[$idx]}"
  config_name="$(basename "$config")"
  config_stem="${config_name%.cfg}"
  stdout_log="$logs_dir/${config_stem}.stdout.log"
  stderr_log="$logs_dir/${config_stem}.stderr.log"

  case_log "$config_stem" "Running $((idx + 1))/$total: $config_name"
  "$runner" run-test "$test_name" "$config" -n "$mpi_ranks" \
    > "$stdout_log" \
    2> "$stderr_log" &
  runner_pid=$!

  monitor_run_progress "$runner_pid" "$config" "$stdout_log" "$config_stem"

  if ! wait "$runner_pid"; then
    case_error "$config_stem" "Run failed for $config_name"
    case_error "$config_stem" "Sweep stdout log: $stdout_log"
    case_error "$config_stem" "Sweep stderr log: $stderr_log"
    case_error "$config_stem" "Last lines from sweep stdout:"
    tail -n 40 "$stdout_log" | prefix_case_lines "$config_stem" >&2 || true
    if [[ -s "$stderr_log" ]]; then
      case_error "$config_stem" "Last lines from sweep stderr:"
      tail -n 40 "$stderr_log" | prefix_case_lines "$config_stem" >&2 || true
    fi

    run_dir="$(sed -n 's/^    Run dir:  //p' "$stdout_log" | tail -n 1)"
    if [[ -n "$run_dir" && -d "$run_dir" ]]; then
      case_error "$config_stem" "Simulation run directory: $run_dir"
      if [[ -s "$run_dir/stderr.log" ]]; then
        case_error "$config_stem" "Last lines from simulation stderr:"
        tail -n 40 "$run_dir/stderr.log" | prefix_case_lines "$config_stem" >&2 || true
      fi
    fi
    exit 1
  fi

  run_dir="$(sed -n 's/^    Run dir:  //p' "$stdout_log" | tail -n 1)"
  case_log "$config_stem" "Completed $((idx + 1))/$total: $config_name"
  if [[ -n "$run_dir" ]]; then
    case_log "$config_stem" "Output dir: $run_dir"
  fi
done

sweep_log "All $total configs completed."
if [[ "$generation_mode" -eq 1 ]]; then
  sweep_log "Configs written to: $configs_dir"
fi
sweep_log "Sweep logs: $logs_dir"
