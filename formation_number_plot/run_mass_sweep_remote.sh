#!/usr/bin/env bash
# Generate mass-sweep configs and run them through the remote/HPC launcher.
# Existing configs can still be run without regeneration via `--existing`.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"

test_name="${TEST_NAME:-ns_amr_lgf2D}"
configs_dir="${CONFIGS_DIR:-$script_dir/mass_configs}"
logs_dir="${LOGS_DIR:-$script_dir/mass_logs_remote}"
runner="${IBLGF_RUNNER:-$repo_root/iblgf_remote.sh}"
generation_mode=1
sample_config=""
generator_script=""

usage() {
  cat <<EOF
Usage:
  $0 <sample_config> [mpi_ranks] [generator|freq|tau]
  $0 --existing [mpi_ranks] [config_glob]

Examples:
  $0 formation_number_plot/config2D_new_FM_test 40 freq
  $0 ../tests/ns_amr_lgf2D/configs/configFile_0 32 tau
  $0 --existing 40 'config_freq*.cfg'

Environment overrides:
  IBLGF_RUNS_ROOT=/scratch/\$USER/iblgf-runs
  IBLGF_BUILD_DIR=/scratch/\$USER/iblgf-build
  IBLGF_LIB_ROOT=/path/to/iblgf-lib
  IBLGF_RUNNER=/path/to/IBLGF-AMR/iblgf_remote.sh
  CONFIGS_DIR=/path/to/configs
  CONFIG_GLOB='config_freq*.cfg'
  LOGS_DIR=/path/to/sweep-logs
  TEST_NAME=ns_amr_lgf2D
  GENERATOR_SCRIPT=generate_mass_configs.py

The remote launcher prefers the MPI installation's mpiexec/mpirun, including
inside a Slurm allocation. It falls back to srun if necessary. This script
does not request an allocation.
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

  echo "Generating remote sweep configs"
  echo "  Sample:    $sample_config"
  echo "  Generator: $generator_script"
  echo "  Output:    $configs_dir"
  python3 "$generator_script" \
    "$sample_config" \
    --output-dir "$configs_dir"
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
  echo "Obtain an allocation with salloc/sbatch before a large sweep." >&2
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

echo "Remote IBLGF mass sweep"
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

  echo "Running $((idx + 1))/$total: $config_name"
  if ! "$runner" run-test "$test_name" "$config" -n "$mpi_ranks" \
    > "$stdout_log" \
    2> "$stderr_log"; then
    echo "Run failed for $config_name" >&2
    echo "Sweep stdout log: $stdout_log" >&2
    echo "Sweep stderr log: $stderr_log" >&2

    echo >&2
    echo "Last lines from sweep stdout:" >&2
    tail -n 40 "$stdout_log" >&2 || true
    if [[ -s "$stderr_log" ]]; then
      echo >&2
      echo "Last lines from sweep stderr:" >&2
      tail -n 40 "$stderr_log" >&2 || true
    fi

    # The simulation itself writes logs inside its timestamped run directory.
    run_dir="$(sed -n 's/^    Run dir:  //p' "$stdout_log" | tail -n 1)"
    if [[ -n "$run_dir" && -d "$run_dir" ]]; then
      echo >&2
      echo "Simulation run directory: $run_dir" >&2
      if [[ -s "$run_dir/stderr.log" ]]; then
        echo "Last lines from simulation stderr:" >&2
        tail -n 40 "$run_dir/stderr.log" >&2 || true
      fi
    fi
    exit 1
  fi
done

echo "All $total configs completed."
if [[ "$generation_mode" -eq 1 ]]; then
  echo "Configs written to: $configs_dir"
fi
echo "Sweep logs: $logs_dir"
