#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
analysis_script="${script_dir}/run_circulation_analysis.sh"
comparison_script="${script_dir}/plot_resolution_comparison.py"

usage() {
  cat <<EOF
Usage:
  $(basename "$0") SWEEP_FOLDER [STRIDE] [CENTER_THRESHOLD_FRACTION]

Example:
  $(basename "$0") runs/ns_amr_lgf/res_sweep 1 0.4

Existing leading_vortex_circulation.csv files are reused. ParaView analysis is
run only for cases whose CSV is missing.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
if [[ $# -lt 1 || $# -gt 3 ]]; then
  usage >&2
  exit 1
fi

sweep_folder="$(cd "$1" && pwd)"
stride="${2:-1}"
center_threshold_fraction="${3:-0.4}"

if [[ ! "$stride" =~ ^[1-9][0-9]*$ ]]; then
  echo "Error: stride must be a positive integer." >&2
  exit 1
fi

shopt -s nullglob
run_count=0
for run_folder in "$sweep_folder"/*; do
  [[ -d "$run_folder/output" ]] || continue
  snapshots=("$run_folder"/output/flowTime_*.hdf5)
  [[ ${#snapshots[@]} -gt 0 ]] || continue
  run_count=$((run_count + 1))

  run_name="$(basename "$run_folder")"
  csv_path="${script_dir}/outputs/${run_name}_circulation/leading_vortex_circulation.csv"
  if [[ -s "$csv_path" ]] && grep -Fq "$run_folder/output/" "$csv_path"; then
    echo "[$run_name] Reusing $csv_path"
    continue
  fi
  if [[ -s "$csv_path" ]]; then
    echo "[$run_name] Existing CSV belongs to another run; recalculating it."
  fi

  config_args=()
  exact_config="$run_folder/${run_name}.cfg"
  if [[ -f "$exact_config" ]]; then
    config_args=(--config "$exact_config")
  else
    configs=("$run_folder"/*.cfg)
    if [[ ${#configs[@]} -eq 1 ]]; then
      config_args=(--config "${configs[0]}")
    elif [[ ${#configs[@]} -gt 1 ]]; then
      echo "Error: multiple .cfg files found in $run_folder" >&2
      echo "Run the individual analysis with an explicit --config first." >&2
      exit 1
    fi
  fi

  echo "[$run_name] Analysis CSV is missing; running ParaView analysis."
  "$analysis_script" \
    "$run_folder" \
    "$stride" \
    --center-threshold-fraction "$center_threshold_fraction" \
    "${config_args[@]}"
done

if [[ $run_count -eq 0 ]]; then
  echo "Error: no child runs with output/flowTime_*.hdf5 were found in $sweep_folder" >&2
  exit 1
fi

if ! python -c 'import matplotlib' 2>/dev/null; then
  # The same environment used by the standalone analysis already provides Matplotlib.
  source "${script_dir}/../ritta_vortex_identification/open_viewer_venv.sh"
fi

python "$comparison_script" "$sweep_folder"
