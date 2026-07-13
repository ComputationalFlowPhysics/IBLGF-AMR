#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"

sample_config="${1:-}"
mpi_ranks="${2:-8}"
generator_arg="${3:-${GENERATOR_SCRIPT:-generate_mass_configs.py}}"
test_name="${TEST_NAME:-ns_amr_lgf2D}"
configs_dir="${CONFIGS_DIR:-$script_dir/mass_configs}"
logs_dir="${LOGS_DIR:-$script_dir/mass_logs}"

if [[ -z "$sample_config" ]]; then
  echo "Usage: $0 <sample_config> [mpi_ranks] [generator|freq|tau]" >&2
  echo "Example: $0 ../tests/ns_amr_lgf2D/configs/configFile_0 8" >&2
  echo "Frequency sweep: $0 formation_number_plot/config2D_new_train_test 8 freq" >&2
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

if [[ ! -f "$generator_script" ]]; then
  echo "Generator script not found: $generator_script" >&2
  exit 2
fi

python3 "$generator_script" \
  "$sample_config" \
  --output-dir "$configs_dir"

mkdir -p "$logs_dir"

cd "$repo_root"

shopt -s nullglob
configs=("$configs_dir"/$config_pattern)
shopt -u nullglob
if [[ "${#configs[@]}" -gt 0 ]]; then
  mapfile -t configs < <(printf '%s\n' "${configs[@]}" | sort -V)
fi

if [[ "${#configs[@]}" -eq 0 ]]; then
  echo "No generated configs found in $configs_dir" >&2
  exit 1
fi

total="${#configs[@]}"
for idx in "${!configs[@]}"; do
  config="${configs[$idx]}"
  config_name="$(basename "$config")"
  config_stem="${config_name%.cfg}"
  stdout_log="$logs_dir/${config_stem}.stdout.log"
  stderr_log="$logs_dir/${config_stem}.stderr.log"

  echo "Running $((idx + 1))/$total: $config_name"
  if ! ./iblgf.sh run-test "$test_name" "$config" -n "$mpi_ranks" \
    > "$stdout_log" \
    2> "$stderr_log"; then
    echo "Run failed for $config_name" >&2
    echo "Sweep stdout log: $stdout_log" >&2
    echo "Sweep stderr log: $stderr_log" >&2
    echo "" >&2
    echo "Last lines from sweep stdout:" >&2
    tail -n 40 "$stdout_log" >&2 || true
    if [[ -s "$stderr_log" ]]; then
      echo "" >&2
      echo "Last lines from sweep stderr:" >&2
      tail -n 40 "$stderr_log" >&2 || true
    fi
    exit 1
  fi
done

echo "Configs written to: $configs_dir"
echo "Logs written to:    $logs_dir"
