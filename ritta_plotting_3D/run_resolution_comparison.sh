#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
analysis_script="${script_dir}/run_circulation_analysis.sh"
comparison_script="${script_dir}/plot_resolution_comparison.py"

usage() {
  cat <<EOF
Usage:
  $(basename "$0") SWEEP_FOLDER [STRIDE] [CENTER_THRESHOLD_FRACTION] [LEGEND_BY] [VORTICITY_THRESHOLD_FRACTION]

Examples:
  $(basename "$0") runs/ns_amr_lgf/res_sweep 1 0.4
  $(basename "$0") runs/ns_amr_lgf/formation 1 0.4 tau 0.02

Existing leading_vortex_circulation.csv files are reused. ParaView analysis is
run only for cases whose CSV is missing. LEGEND_BY is resolution (default) or
tau. VORTICITY_THRESHOLD_FRACTION defaults to the paper's 0.02 cutoff;
CENTER_THRESHOLD_FRACTION independently controls the Lamb-center calculation.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
if [[ $# -lt 1 || $# -gt 5 ]]; then
  usage >&2
  exit 1
fi

sweep_folder="$(cd "$1" && pwd)"
stride="${2:-1}"
center_threshold_fraction="${3:-0.4}"
legend_by="${4:-resolution}"
vorticity_threshold_fraction="${5:-0.02}"

if [[ ! "$stride" =~ ^[1-9][0-9]*$ ]]; then
  echo "Error: stride must be a positive integer." >&2
  exit 1
fi
if [[ "$legend_by" != "resolution" && "$legend_by" != "tau" ]]; then
  echo "Error: LEGEND_BY must be resolution or tau." >&2
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
  csv_matches_threshold=0
  if [[ -s "$csv_path" ]]; then
    if python -c '
import csv
import math
import sys

with open(sys.argv[1], newline="") as csv_file:
    rows = list(csv.DictReader(csv_file))
requested = float(sys.argv[2])
values = [
    float(row.get("vorticity_threshold_fraction") or 0.02)
    for row in rows
]
matches = rows and all(
    math.isclose(value, requested, rel_tol=1e-12, abs_tol=1e-12)
    for value in values
)
raise SystemExit(0 if matches else 1)
' "$csv_path" "$vorticity_threshold_fraction"; then
      csv_matches_threshold=1
    fi
  fi
  if [[ -s "$csv_path" ]] && [[ $csv_matches_threshold -eq 1 ]] && grep -Fq "$run_folder/output/" "$csv_path"; then
    echo "[$run_name] Reusing $csv_path"
    continue
  fi
  if [[ -s "$csv_path" ]]; then
    if ! grep -Fq "$run_folder/output/" "$csv_path"; then
      echo "[$run_name] Existing CSV belongs to another run; recalculating it."
    else
      echo "[$run_name] Existing CSV uses a different vorticity threshold; recalculating it."
    fi
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

  echo "[$run_name] Running ParaView circulation analysis."
  "$analysis_script" \
    "$run_folder" \
    "$stride" \
    --vorticity-threshold-fraction "$vorticity_threshold_fraction" \
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

python "$comparison_script" "$sweep_folder" --legend-by "$legend_by"
