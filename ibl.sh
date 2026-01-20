#!/usr/bin/env bash
# ibl.sh — friendly wrapper for configure/build/test/run
# Put this file in the root of the IBLGF-AMR repo (next to CMakeLists.txt).

set -euo pipefail

die() {
  echo "Error: $*" >&2
  exit 1
}

have() {
  command -v "$1" >/dev/null 2>&1
}

cpu_count() {
  if have nproc; then
    nproc
  elif have sysctl; then
    sysctl -n hw.ncpu
  else
    echo 4
  fi
}

script_dir() {
  cd "$(dirname "${BASH_SOURCE[0]}")" && pwd
}

repo_root() {
  script_dir
}

build_dir_default() {
  echo "$(repo_root)/build"
}

ensure_repo_root() {
  local root
  root="$(repo_root)"
  if [[ ! -f "$root/CMakeLists.txt" ]]; then
    die "CMakeLists.txt not found in $root"
  fi
}

print_usage() {
  cat <<'USAGE'
Usage:
  ./ibl.sh help
  ./ibl.sh configure
  ./ibl.sh build [-j N]
  ./ibl.sh test  [-j N]
  ./ibl.sh clean

Run an existing built executable with a config:
  ./ibl.sh run <exe-or-target> <config> [-n MPI_RANKS] [-- <extra args>]

Run a named test (staged run dir + logs + metadata):
  ./ibl.sh run-test <test_name> <config_name_or_path> [-n MPI_RANKS]

USAGE
}

# -----------------------------
# Run / test helpers
# -----------------------------

timestamp() {
  # Produces a filesystem-safe timestamp like:
  # 2026-01-19_21-15-03
  date +"%Y-%m-%d_%H-%M-%S"
}

git_commit() {
  # Record which git commit produced the outputs (for reproducibility)
  if have git && git -C "$(repo_root)" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git -C "$(repo_root)" rev-parse --short HEAD
  else
    echo "unknown"
  fi
}

runs_root() {
  # Central place where all test outputs go
  # This keeps outputs out of build/ and src/
  echo "$(repo_root)/runs"
}

find_test_executable() {
  # Find the executable for a test in the canonical layout:
  # build/tests/<test_name>/<test_name>.x

  local test_name="$1"
  local candidate

  candidate="$(build_dir_default)/tests/${test_name}/${test_name}.x"
  if [[ -x "$candidate" ]]; then
    echo "$candidate"
    return 0
  fi

  # Fallback: search anywhere under build/ for <test_name>.x
  find "$(build_dir_default)" -type f -perm -111 -name "${test_name}.x" 2>/dev/null | head -n 1
}

find_test_config() {
  # Resolve a config argument into an actual file path.
  # If the user passes an existing file path, we use it directly.
  # Otherwise we look in: tests/<test_name>/configs/<config_name>

  local test_name="$1"
  local config_arg="$2"

  # If user gave a real file path, accept it.
  if [[ -f "$config_arg" ]]; then
    echo "$config_arg"
    return 0
  fi

  local candidate
  candidate="$(repo_root)/tests/${test_name}/configs/${config_arg}"
  if [[ -f "$candidate" ]]; then
    echo "$candidate"
    return 0
  fi

  return 1
}


do_configure() {
  ensure_repo_root
  local root build_dir
  root="$(repo_root)"
  build_dir="$(build_dir_default)"
  mkdir -p "$build_dir"

  echo "==> Configuring"
  if have ninja; then
    cmake -S "$root" -B "$build_dir" -G Ninja
  else
    cmake -S "$root" -B "$build_dir"
  fi
}

do_build() {
  ensure_repo_root
  local build_dir jobs
  build_dir="$(build_dir_default)"
  jobs="$(cpu_count)"

  while [[ $# -gt 0 ]]; do
    case "$1" in
      -j) shift; jobs="$1"; shift ;;
      *) die "Unknown option: $1" ;;
    esac
  done

  [[ -d "$build_dir" ]] || do_configure

  echo "==> Building (-j $jobs)"
  cmake --build "$build_dir" -j "$jobs"
}

do_test() {
  do_build "$@"
  echo "==> Running tests"
  ctest --test-dir "$(build_dir_default)" --output-on-failure
}

find_executable_in_build() {
  find "$(build_dir_default)" -type f -perm -111 \
    \( -name "$1" -o -name "$1.x" \) 2>/dev/null | head -n 1
}

do_run() {
  local exe config mpi
  mpi=1

  exe="$1"
  config="$2"
  shift 2

  while [[ $# -gt 0 ]]; do
    case "$1" in
      -n) shift; mpi="$1"; shift ;;
      --) shift; break ;;
      *) die "Unknown option: $1" ;;
    esac
  done

  local exe_path
  if [[ -x "$exe" ]]; then
    exe_path="$exe"
  else
    exe_path="$(find_executable_in_build "$exe")"
  fi

  [[ -n "$exe_path" ]] || die "Executable not found"
  [[ -f "$config" ]] || die "Config not found"

  echo "==> Running $exe_path with $config (-n $mpi)"

  if [[ "$mpi" -gt 1 ]] && have mpirun; then
    mpirun -n "$mpi" "$exe_path" "$config"
  else
    "$exe_path" "$config"
  fi
}

do_run_test() {
  ensure_repo_root

  [[ $# -ge 2 ]] || die "Usage: ./ibl.sh run-test <test_name> <config_name_or_path> [-n MPI_RANKS]"

  local test_name="$1"
  local config_arg="$2"
  shift 2

  local mpi=2
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -n)
        shift
        mpi="${1:-}"
        [[ -n "$mpi" ]] || die "Missing value after -n"
        shift
        ;;
      *)
        die "Unknown option for run-test: $1"
        ;;
    esac
  done

  local config_path
  config_path="$(find_test_config "$test_name" "$config_arg" || true)"
  [[ -n "$config_path" ]] || die "Config not found. Tried: '$config_arg' and tests/$test_name/configs/$config_arg"

  # Ensure build directory exists; if not, configure first.
  [[ -d "$(build_dir_default)" ]] || do_configure

  # Build to ensure the test executable exists and is up to date.
  do_build

  local exe
  exe="$(find_test_executable "$test_name" || true)"
  [[ -n "$exe" ]] || die "Could not find executable for test '$test_name'. Expected: build/tests/$test_name/$test_name.x"

  # Create a clean run directory where outputs will go.
  local run_dir
  run_dir="$(runs_root)/${test_name}/$(timestamp)"
  mkdir -p "$run_dir"

  # Copy config into run directory so run is self-contained.
  local cfg_name
  cfg_name="$(basename "$config_path")"
  cp "$config_path" "$run_dir/$cfg_name"

  # Record metadata for reproducibility.
  {
    echo "test_name: $test_name"
    echo "exe: $exe"
    echo "config: $cfg_name"
    echo "mpi_ranks: $mpi"
    echo "git_commit: $(git_commit)"
    echo "timestamp: $(date -Iseconds 2>/dev/null || date)"
    echo "run_dir: $run_dir"
    echo "command: mpiexec -np $mpi $exe ./$cfg_name"
  } > "$run_dir/meta.txt"

  echo "==> Running test '$test_name'"
  echo "    Run dir:  $run_dir"
  echo "    Exe:      $exe"
  echo "    Config:   $cfg_name"
  echo "    MPI:      $mpi"
  echo "    Logs:     stdout.log / stderr.log"
  echo "    Expect:   output files created inside the run dir."

  # Run inside run_dir so output files land there.
  (
    cd "$run_dir"

    if [[ "$mpi" -gt 1 ]]; then
      if have mpiexec; then
        mpiexec -np "$mpi" "$exe" "./$cfg_name" > stdout.log 2> stderr.log
      elif have mpirun; then
        mpirun -n "$mpi" "$exe" "./$cfg_name" > stdout.log 2> stderr.log
      else
        die "Neither mpiexec nor mpirun found in PATH."
      fi
    else
      "$exe" "./$cfg_name" > stdout.log 2> stderr.log
    fi
  )

  echo "==> Done."
  echo "    Outputs are in: $run_dir"
}

do_clean() {
  echo "==> Cleaning build/"
  rm -rf "$(build_dir_default)"
}

cmd="${1:-help}"
shift || true

case "$cmd" in
  help) print_usage ;;
  configure) do_configure ;;
  build) do_build "$@" ;;
  test) do_test "$@" ;;
  run) do_run "$@" ;;
  run-test) do_run_test "$@" ;;
  clean) do_clean ;;
  *) echo "Unknown command: $cmd"; print_usage; exit 1 ;;
esac
