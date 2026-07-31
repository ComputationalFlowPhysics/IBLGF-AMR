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
resumable_mode=0
sample_config=""
generator_script=""
progress_interval="${PROGRESS_INTERVAL:-20}"
state_dir="${SWEEP_STATE_DIR:-}"

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

write_state_file() {
  local destination="$1"
  local value="$2"
  local temporary="${destination}.tmp.$$"
  printf '%s\n' "$value" > "$temporary"
  mv -f -- "$temporary" "$destination"
}

state_event() {
  local case_name="$1"
  local event="$2"
  local detail="${3:-}"
  printf '%s\t%s\t%s\t%s\t%s\n' \
    "$(date -Iseconds 2>/dev/null || date)" \
    "${SLURM_JOB_ID:-manual}" \
    "$case_name" \
    "$event" \
    "$detail" \
    >> "$state_dir/events.tsv"
}

record_run_dir_from_log() {
  local sweep_stdout_log="$1"
  local run_dir_file="$2"
  local discovered=""
  if [[ -f "$sweep_stdout_log" ]]; then
    discovered="$(sed -n 's/^    Run dir:  //p' "$sweep_stdout_log" | tail -n 1)"
  fi
  if [[ -n "$discovered" && -d "$discovered" ]]; then
    write_state_file "$run_dir_file" "$discovered"
    printf '%s\n' "$discovered"
  fi
}

recover_run_dir() {
  local case_name="$1"
  local run_dir_file="$state_dir/run_dirs/${case_name}.txt"
  local run_dir=""
  if [[ -f "$run_dir_file" ]]; then
    run_dir="$(head -n 1 "$run_dir_file")"
    if [[ -n "$run_dir" ]]; then
      printf '%s\n' "$run_dir"
      return 0
    fi
  fi

  local attempt_log
  attempt_log="$(find "$logs_dir/$case_name" -maxdepth 1 -type f -name 'attempt_*.stdout.log' \
    2>/dev/null | sort -V | tail -n 1)"
  if [[ -n "$attempt_log" ]]; then
    record_run_dir_from_log "$attempt_log" "$run_dir_file"
  fi
}

checkpoint_directory_ready() {
  local checkpoint_dir="$1"
  [[ -f "$checkpoint_dir/restart_field.hdf5" &&
     -f "$checkpoint_dir/restart_info" &&
     -f "$checkpoint_dir/tree_info.bin" ]]
}

checkpoint_ready() {
  checkpoint_directory_ready "$1/restart"
}

checkpoint_available() {
  local run_dir="$1"
  checkpoint_directory_ready "$run_dir/restart" ||
    checkpoint_directory_ready "$run_dir/restart/backup" ||
    checkpoint_directory_ready "$run_dir/restart/backup_tmp"
}

restore_checkpoint_backup() {
  local run_dir="$1"
  local case_name="$2"
  local backup_dir restart_file

  for backup_dir in "$run_dir/restart/backup" "$run_dir/restart/backup_tmp"; do
    if checkpoint_directory_ready "$backup_dir"; then
      case_log "$case_name" "Restoring complete checkpoint backup: $backup_dir"
      for restart_file in restart_field.hdf5 restart_info tree_info.bin; do
        cp -a "$backup_dir/$restart_file" "$run_dir/restart/$restart_file"
      done
      state_event "$case_name" "checkpoint-restored" "$backup_dir"
      return 0
    fi
  done
  return 1
}

run_has_final_output() {
  local run_dir="$1"
  [[ -f "$run_dir/output/flow_final.hdf5" ||
     -f "$run_dir/output/flow_final.hdf5.hdf5" ]]
}

enable_restart_in_run_config() {
  local run_config="$1"
  local temporary="${run_config}.restart.$$"

  if ! sed -E \
    -e 's/^([[:space:]]*use_restart[[:space:]]*=[[:space:]]*)(true|false);/\1true;/' \
    -e 's/^([[:space:]]*write_restart[[:space:]]*=[[:space:]]*)(true|false);/\1true;/' \
    "$run_config" > "$temporary"; then
    rm -f -- "$temporary"
    return 1
  fi
  mv -f -- "$temporary" "$run_config"

  grep -Eq '^[[:space:]]*use_restart[[:space:]]*=[[:space:]]*true;' "$run_config" || {
    echo "Could not enable use_restart in $run_config" >&2
    return 1
  }
  grep -Eq '^[[:space:]]*write_restart[[:space:]]*=[[:space:]]*true;' "$run_config" || {
    echo "Could not enable write_restart in $run_config" >&2
    return 1
  }
}

next_attempt_number() {
  local case_name="$1"
  local attempt_file="$state_dir/attempts/${case_name}.count"
  local attempt=0
  if [[ -f "$attempt_file" ]]; then
    attempt="$(head -n 1 "$attempt_file")"
  fi
  [[ "$attempt" =~ ^[0-9]+$ ]] || attempt=0
  attempt=$((attempt + 1))
  write_state_file "$attempt_file" "$attempt"
  printf '%s\n' "$attempt"
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
  local run_dir_file="${5:-}"

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
        if [[ -n "$run_dir_file" && -d "$run_dir" ]]; then
          write_state_file "$run_dir_file" "$run_dir"
        fi
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
  $0 --resumable [mpi_ranks] [config_glob]

Examples:
  $0 formation_3d/config3D_test 32 freq
  $0 formation_3d/config3D_test 32 tau
  $0 --existing 32 '*.cfg'
  SWEEP_STATE_DIR=/path/to/state $0 --resumable 32 '*.cfg'

Environment overrides:
  IBLGF_RUNS_ROOT=/scratch/\$USER/iblgf-runs
  IBLGF_BUILD_DIR=/scratch/\$USER/iblgf-build
  IBLGF_LIB_ROOT=/path/to/iblgf-lib
  IBLGF_RUNNER=/path/to/IBLGF-AMR/iblgf_remote.sh
  CONFIGS_DIR=/path/to/configs
  CONFIG_GLOB='*.cfg'
  LOGS_DIR=/path/to/sweep-logs
  SWEEP_STATE_DIR=/path/to/persistent-sweep-state
  PROGRESS_INTERVAL=20
  TEST_NAME=ns_amr_lgf
  GENERATOR_SCRIPT=generate_mass_configs.py

This 3D version targets the ns_amr_lgf test. The remote launcher prefers the
MPI installation's mpiexec/mpirun, including inside a Slurm allocation. It
falls back to srun if necessary. This script does not request an allocation.
Resumable mode keeps a manifest, per-attempt logs, run-directory pointers,
and completion markers under SWEEP_STATE_DIR.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ "${1:-}" == "--resumable" ]]; then
  generation_mode=0
  resumable_mode=1
  mpi_ranks="${2:-${IBLGF_MPI_RANKS:-8}}"
  config_pattern="${3:-${CONFIG_GLOB:-*.cfg}}"
elif [[ "${1:-}" == "--existing" ]]; then
  generation_mode=0
  mpi_ranks="${2:-${IBLGF_MPI_RANKS:-8}}"
  config_pattern="${3:-${CONFIG_GLOB:-*.cfg}}"
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

initialize_resumable_state() {
  if [[ -z "$state_dir" ]]; then
    state_dir="$logs_dir/sweep_state"
  fi
  mkdir -p \
    "$state_dir/attempts" \
    "$state_dir/completed" \
    "$state_dir/run_dirs" \
    "$logs_dir"

  if command -v flock >/dev/null 2>&1; then
    exec 9> "$state_dir/sweep.lock"
    if ! flock -n 9; then
      echo "Another sweep process is already using $state_dir" >&2
      exit 2
    fi
  else
    echo "Warning: flock is unavailable; concurrent submissions cannot be detected." >&2
  fi

  local manifest_temp="$state_dir/manifest.tsv.tmp.$$"
  printf 'index\tconfig\tchecksum\tbytes\tpath\n' > "$manifest_temp"
  local index config case_name checksum bytes ignored
  declare -A seen_cases=()
  for index in "${!configs[@]}"; do
    config="${configs[$index]}"
    case_name="$(basename "${config%.cfg}")"
    if [[ -n "${seen_cases[$case_name]:-}" ]]; then
      echo "Duplicate config stem in sweep: $case_name" >&2
      rm -f -- "$manifest_temp"
      exit 2
    fi
    seen_cases[$case_name]=1
    read -r checksum bytes ignored < <(cksum "$config")
    printf '%d\t%s\t%s\t%s\t%s\n' \
      "$((index + 1))" "$case_name" "$checksum" "$bytes" "$config" \
      >> "$manifest_temp"
  done

  if [[ -f "$state_dir/manifest.tsv" ]]; then
    if ! cmp -s "$manifest_temp" "$state_dir/manifest.tsv"; then
      echo "The config list differs from the existing resumable sweep manifest:" >&2
      echo "  $state_dir/manifest.tsv" >&2
      echo "Use the original folder/glob, or submit with a new SWEEP_NAME." >&2
      rm -f -- "$manifest_temp"
      exit 2
    fi
    rm -f -- "$manifest_temp"
  else
    mv -- "$manifest_temp" "$state_dir/manifest.tsv"
    state_event "-" "manifest-created" "${#configs[@]} configs"
  fi
}

write_resumable_summary() {
  local summary_temp="$state_dir/summary.tsv.tmp.$$"
  printf 'index\tconfig\tstatus\tattempts\trun_dir\n' > "$summary_temp"
  local index config case_name status attempts run_dir
  for index in "${!configs[@]}"; do
    config="${configs[$index]}"
    case_name="$(basename "${config%.cfg}")"
    status="pending"
    attempts="0"
    run_dir=""
    if [[ -f "$state_dir/attempts/${case_name}.count" ]]; then
      attempts="$(head -n 1 "$state_dir/attempts/${case_name}.count")"
    fi
    if [[ -f "$state_dir/run_dirs/${case_name}.txt" ]]; then
      run_dir="$(head -n 1 "$state_dir/run_dirs/${case_name}.txt")"
      status="incomplete"
      if [[ -n "$run_dir" ]] && checkpoint_available "$run_dir"; then
        status="resume-ready"
      fi
    fi
    if [[ -f "$state_dir/completed/${case_name}.done" ]]; then
      status="completed"
    fi
    printf '%d\t%s\t%s\t%s\t%s\n' \
      "$((index + 1))" "$case_name" "$status" "$attempts" "$run_dir" \
      >> "$summary_temp"
  done
  mv -f -- "$summary_temp" "$state_dir/summary.tsv"
}

mark_case_complete() {
  local case_name="$1"
  local run_dir="$2"
  write_state_file \
    "$state_dir/completed/${case_name}.done" \
    "$(date -Iseconds 2>/dev/null || date)\t$run_dir"
  touch "$run_dir/.iblgf_sweep_complete"
  state_event "$case_name" "completed" "$run_dir"
}

run_resumable_case() {
  local config="$1"
  local position="$2"
  local total_count="$3"
  local config_name case_name run_dir run_config run_mode attempt attempt_tag job_tag
  local case_log_dir stdout_log stderr_log run_dir_file run_status runner_pid

  config_name="$(basename "$config")"
  case_name="${config_name%.cfg}"
  run_dir_file="$state_dir/run_dirs/${case_name}.txt"

  if [[ -f "$state_dir/completed/${case_name}.done" ]]; then
    case_log "$case_name" "Skipping completed case $position/$total_count"
    return 0
  fi

  run_dir="$(recover_run_dir "$case_name" || true)"
  if [[ -n "$run_dir" ]]; then
    if [[ ! -d "$run_dir" ]]; then
      case_error "$case_name" "Recorded run directory does not exist: $run_dir"
      return 2
    fi
    if [[ -f "$run_dir/.iblgf_sweep_complete" ]] || run_has_final_output "$run_dir"; then
      case_log "$case_name" "Found final output; marking case completed: $run_dir"
      mark_case_complete "$case_name" "$run_dir"
      return 0
    fi
    if ! checkpoint_ready "$run_dir"; then
      if ! restore_checkpoint_backup "$run_dir" "$case_name"; then
        case_error "$case_name" "The interrupted run has no complete restart checkpoint: $run_dir/restart"
        case_error "$case_name" "Refusing to start a duplicate run. Inspect the run or choose a new SWEEP_NAME."
        state_event "$case_name" "checkpoint-missing" "$run_dir"
        return 2
      fi
    fi
    run_config="$run_dir/$config_name"
    if [[ ! -f "$run_config" ]]; then
      case_error "$case_name" "Copied run config not found: $run_config"
      return 2
    fi
    enable_restart_in_run_config "$run_config" || return 2
    run_mode="resume"
  else
    if ! grep -Eq '^[[:space:]]*write_restart[[:space:]]*=[[:space:]]*true;' "$config"; then
      case_error "$case_name" "Fresh resumable runs require write_restart=true in $config"
      return 2
    fi
    if ! grep -Eq '^[[:space:]]*use_restart[[:space:]]*=[[:space:]]*false;' "$config"; then
      case_error "$case_name" "Fresh runs require use_restart=false in $config"
      return 2
    fi
    run_config="$config"
    run_mode="fresh"
  fi

  attempt="$(next_attempt_number "$case_name")"
  attempt_tag="$(printf '%03d' "$attempt")"
  job_tag="${SLURM_JOB_ID:-manual}"
  case_log_dir="$logs_dir/$case_name"
  mkdir -p "$case_log_dir"
  stdout_log="$case_log_dir/attempt_${attempt_tag}_${job_tag}.stdout.log"
  stderr_log="$case_log_dir/attempt_${attempt_tag}_${job_tag}.stderr.log"

  if [[ "$run_mode" == "resume" ]]; then
    local saved_file
    for saved_file in stdout.log stderr.log meta.txt; do
      if [[ -f "$run_dir/$saved_file" ]]; then
        cp -a \
          "$run_dir/$saved_file" \
          "$run_dir/${saved_file%.*}.before-resume-${attempt_tag}.${saved_file##*.}"
      fi
    done
  fi

  case_log "$case_name" "${run_mode^} attempt $attempt for case $position/$total_count: $config_name"
  state_event "$case_name" "$run_mode-started" "attempt=$attempt run_dir=${run_dir:-pending}"

  if [[ "$run_mode" == "resume" ]]; then
    "$runner" run-test "$test_name" "$run_config" -n "$mpi_ranks" --resume "$run_dir" \
      > "$stdout_log" \
      2> "$stderr_log" &
  else
    "$runner" run-test "$test_name" "$run_config" -n "$mpi_ranks" \
      > "$stdout_log" \
      2> "$stderr_log" &
  fi
  runner_pid=$!

  monitor_run_progress "$runner_pid" "$config" "$stdout_log" "$case_name" "$run_dir_file"

  run_status=0
  wait "$runner_pid" || run_status=$?
  if [[ -z "$run_dir" ]]; then
    run_dir="$(record_run_dir_from_log "$stdout_log" "$run_dir_file" || true)"
  fi

  if [[ "$run_status" -ne 0 ]]; then
    case_error "$case_name" "Attempt $attempt exited with status $run_status"
    case_error "$case_name" "Stdout: $stdout_log"
    case_error "$case_name" "Stderr: $stderr_log"
    if [[ -n "$run_dir" ]]; then
      case_error "$case_name" "Run directory retained for the next resume: $run_dir"
    fi
    state_event "$case_name" "attempt-exited" "status=$run_status run_dir=${run_dir:-unknown}"
    return "$run_status"
  fi

  if [[ -z "$run_dir" || ! -d "$run_dir" ]]; then
    case_error "$case_name" "Successful runner did not report a valid run directory."
    state_event "$case_name" "run-dir-missing" "attempt=$attempt"
    return 2
  fi

  mark_case_complete "$case_name" "$run_dir"
  case_log "$case_name" "Completed case $position/$total_count: $run_dir"
}

finalize_resumable_state() {
  local status=$?
  trap - EXIT
  write_resumable_summary || true
  if [[ "$status" -ne 0 ]]; then
    sweep_log "Sweep stopped with status $status. Submit the same sweep again to continue."
    sweep_log "State summary: $state_dir/summary.tsv"
  fi
  exit "$status"
}

if [[ "$resumable_mode" -eq 1 ]]; then
  initialize_resumable_state
  write_resumable_summary
  trap finalize_resumable_state EXIT
fi

echo "Remote IBLGF 3D mass sweep"
echo "  Runner:  $runner"
echo "  Test:    $test_name"
echo "  Configs: $configs_dir/$config_pattern"
echo "  Count:   ${#configs[@]}"
echo "  MPI:     $mpi_ranks rank(s)"
echo "  Logs:    $logs_dir"
if [[ "$resumable_mode" -eq 1 ]]; then
  echo "  State:   $state_dir"
  echo "  Summary: $state_dir/summary.tsv"
fi
if [[ -n "${IBLGF_RUNS_ROOT:-}" ]]; then
  echo "  Outputs: $IBLGF_RUNS_ROOT"
else
  echo "  Outputs: $repo_root/runs"
fi

total="${#configs[@]}"
if [[ "$resumable_mode" -eq 1 ]]; then
  for idx in "${!configs[@]}"; do
    config="${configs[$idx]}"
    run_resumable_case "$config" "$((idx + 1))" "$total"
    write_resumable_summary
  done
else
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
fi

sweep_log "All $total configs completed."
if [[ "$generation_mode" -eq 1 ]]; then
  sweep_log "Configs written to: $configs_dir"
fi
sweep_log "Sweep logs: $logs_dir"
if [[ "$resumable_mode" -eq 1 ]]; then
  write_resumable_summary
  sweep_log "Sweep state: $state_dir"
fi
