#!/usr/bin/env bash

set -euo pipefail

test_root="$(mktemp -d /tmp/iblgf-resumable-sweep.XXXXXX)"
trap 'rm -rf -- "$test_root"' EXIT

configs_dir="$test_root/configs"
runs_dir="$test_root/runs"
logs_dir="$test_root/logs"
state_dir="$test_root/state"
call_log="$test_root/calls.log"
fake_runner="$test_root/fake_runner.sh"
mkdir -p "$configs_dir" "$runs_dir"

for case_name in config_a config_b config_c; do
  printf '%s\n' \
    'simulation_parameters' \
    '{' \
    '    nBaseLevelTimeSteps=100;' \
    '    output_frequency=10;' \
    '    write_restart=true;' \
    '    use_restart=false;' \
    '}' \
    > "$configs_dir/${case_name}.cfg"
done

cat > "$fake_runner" <<'RUNNER'
#!/usr/bin/env bash
set -euo pipefail

config_file="$3"
config_name="$(basename "$config_file")"
case_name="${config_name%.cfg}"
run_dir="$FAKE_RUNS_DIR/$case_name"
mode="fresh"

shift 3
while [[ $# -gt 0 ]]; do
  if [[ "$1" == "--resume" ]]; then
    mode="resume"
    run_dir="$2"
    shift 2
  else
    shift
  fi
done

mkdir -p "$run_dir/output" "$run_dir/restart"
if [[ "$mode" == "fresh" ]]; then
  cp "$config_file" "$run_dir/$config_name"
fi
printf '%s %s\n' "$mode" "$case_name" >> "$FAKE_CALL_LOG"
printf '    Run dir:  %s\n' "$run_dir"
printf 'T = 1.0, n = 10\n' > "$run_dir/stdout.log"
: > "$run_dir/stderr.log"
printf 'mpi_ranks: 4\n' > "$run_dir/meta.txt"

if [[ "$mode" == "resume" ]]; then
  grep -Eq '^[[:space:]]*use_restart[[:space:]]*=[[:space:]]*true;' "$config_file"
fi

if [[ "$case_name" == "config_b" && "$mode" == "fresh" ]]; then
  mkdir -p "$run_dir/restart/backup"
  : > "$run_dir/restart/backup/restart_field.hdf5"
  : > "$run_dir/restart/backup/restart_info"
  : > "$run_dir/restart/backup/tree_info.bin"
  exit 143
fi

if [[ "$mode" == "resume" ]]; then
  test -f "$run_dir/restart/restart_field.hdf5"
  test -f "$run_dir/restart/restart_info"
  test -f "$run_dir/restart/tree_info.bin"
fi

: > "$run_dir/output/flow_final.hdf5"
RUNNER
chmod +x "$fake_runner"

run_sweep() {
  IBLGF_RUNNER="$fake_runner" \
  CONFIGS_DIR="$configs_dir" \
  LOGS_DIR="$logs_dir" \
  SWEEP_STATE_DIR="$state_dir" \
  PROGRESS_INTERVAL=1 \
  FAKE_RUNS_DIR="$runs_dir" \
  FAKE_CALL_LOG="$call_log" \
    "$1" --resumable 4 'config_*.cfg'
}

sweep_script="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/run_mass_sweep_remote.sh"

first_status=0
run_sweep "$sweep_script" || first_status=$?
if [[ "$first_status" -ne 143 ]]; then
  echo "Expected the first sweep to stop with status 143; got $first_status" >&2
  exit 1
fi

grep -F $'config_a\tcompleted' "$state_dir/summary.tsv" >/dev/null
grep -F $'config_b\tresume-ready' "$state_dir/summary.tsv" >/dev/null
grep -F $'config_c\tpending' "$state_dir/summary.tsv" >/dev/null

run_sweep "$sweep_script"

expected_calls="$test_root/expected_calls.log"
printf '%s\n' \
  'fresh config_a' \
  'fresh config_b' \
  'resume config_b' \
  'fresh config_c' \
  > "$expected_calls"
cmp "$expected_calls" "$call_log"

completed_count="$(awk -F '\t' 'NR > 1 && $3 == "completed" {count++} END {print count + 0}' "$state_dir/summary.tsv")"
if [[ "$completed_count" -ne 3 ]]; then
  echo "Expected all three cases to be completed; got $completed_count" >&2
  exit 1
fi

calls_before="$(wc -l < "$call_log" | tr -d ' ')"
run_sweep "$sweep_script"
calls_after="$(wc -l < "$call_log" | tr -d ' ')"
if [[ "$calls_after" -ne "$calls_before" ]]; then
  echo "A completed sweep unexpectedly launched another simulation." >&2
  exit 1
fi

printf '\n// changed after the sweep started\n' >> "$configs_dir/config_c.cfg"
mismatch_status=0
run_sweep "$sweep_script" || mismatch_status=$?
if [[ "$mismatch_status" -ne 2 ]]; then
  echo "Expected a changed config manifest to be rejected; got $mismatch_status" >&2
  exit 1
fi

echo "Resumable mass-sweep test passed."
