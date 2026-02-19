#!/usr/bin/env bash
# iblgf.sh — friendly wrapper for configure/build/test/run
# Put this file in the root of the IBLGF-AMR repo (next to CMakeLists.txt).

set -euo pipefail

USE_GPU=0

build_dir() {
  if [[ "$USE_GPU" -eq 1 ]]; then
    echo "$(repo_root)/build-gpu"
  else
    echo "$(repo_root)/build"
  fi
}

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

# *added*
default_build_jobs() {
  # Env override wins, otherwise use cpu_count()
  echo "${IBLGF_BUILD_JOBS:-$(cpu_count)}"
}

default_test_jobs() {
  # How many tests ctest runs in parallel (NOT MPI ranks)
  echo "${IBLGF_TEST_JOBS:-1}"
}

default_mpi_ranks() {
  # Default MPI ranks for run/run-test
  echo "${IBLGF_MPI_RANKS:-2}"
}

need_nvcc_if_gpu() {
  if [[ "$USE_GPU" -eq 1 ]]; then
    if ! command -v nvcc >/dev/null 2>&1; then
      die "GPU build requested (--gpu) but 'nvcc' was not found. Use the GPU docker image / a CUDA-enabled machine, or set CUDAToolkit_ROOT."
    fi
  fi
}
# *end of added*

script_dir() {
  cd "$(dirname "${BASH_SOURCE[0]}")" && pwd
}

repo_root() {
  script_dir
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
  ./iblgf.sh help
  ./iblgf.sh configure
  ./iblgf.sh build [-j N]
  ./iblgf.sh test  [-j N]
  ./iblgf.sh clean

Env overrides:
  IBLGF_BUILD_JOBS=12   default build parallelism
  IBLGF_TEST_JOBS=6     default ctest parallelism
  IBLGF_MPI_RANKS=8     default MPI ranks for run/run-test

Run an existing built executable with a config:
  ./iblgf.sh run <exe-or-target> <config> [-n MPI_RANKS] [-- <extra args>]

Run a named test (staged run dir + logs + metadata):
  ./iblgf.sh run-test <test_name> <config_name_or_path> [-n MPI_RANKS]

USAGE
}

time_cmd() {
  # prints "real_seconds" to stdout
  # uses /usr/bin/time -p and parses "real <sec>"
  local out
  out="$(/usr/bin/time -p "$@" 2>&1 >/dev/null)"
  echo "$out" | awk '/^real /{print $2}'
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

  candidate="$(build_dir)/tests/${test_name}/${test_name}.x"
  if [[ -x "$candidate" ]]; then
    echo "$candidate"
    return 0
  fi

  # Fallback: search anywhere under build/ for <test_name>.x
  find "$(build_dir)" -type f -perm -111 -name "${test_name}.x" 2>/dev/null | head -n 1
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

# *added*
latest_run_dir() {
  local test_name="$1"
  local base
  base="$(runs_root)/${test_name}"
  [[ -d "$base" ]] || return 1

  # Pick newest directory by modification time
  ls -1dt "$base"/*/ 2>/dev/null | head -n 1
}
# *end of added*

do_configure() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --gpu|-g) USE_GPU=1; shift ;;
      *) die "Unknown option for configure: $1" ;;
    esac
  done

  need_nvcc_if_gpu

  ensure_repo_root
  local root build_dir
  root="$(repo_root)"
  build_dir="$(build_dir)"
  mkdir -p "$build_dir"

  echo "==> Configuring (USE_GPU=$USE_GPU)"
  local cmake_args=()
  if [[ "$USE_GPU" -eq 1 ]]; then
    cmake_args+=(-DUSE_GPU=True)
  fi
  
  # Allow overriding MPI ranks for tests via environment variable
  if [[ -n "${IBLGF_MPI_NP:-}" ]]; then
    cmake_args+=(-DIBLGF_MPI_NP="$IBLGF_MPI_NP")
  fi

  cmake -S "$root" -B "$build_dir" -G "Unix Makefiles" "${cmake_args[@]}"
}

do_build() {
  ensure_repo_root
  local build_dir jobs
  jobs="$(default_build_jobs)"

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --gpu|-g) USE_GPU=1; shift ;;
      -j) shift; jobs="$1"; shift ;;
      *) die "Unknown option: $1" ;;
    esac
  done

  build_dir="$(build_dir)"

  [[ -d "$build_dir" ]] || do_configure

  echo "==> Building (-j $jobs)"
  cmake --build "$build_dir" -j "$jobs"
}

do_test() {
  ensure_repo_root

  local build_jobs test_jobs
  build_jobs="$(default_build_jobs)"
  test_jobs="$(default_test_jobs)"

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --gpu|-g) USE_GPU=1; shift ;;
      -j)
        shift
        test_jobs="$1"
        shift
        ;;
      *)
        die "Unknown option for test: $1"
        ;;
    esac
  done

  if [[ "$USE_GPU" -eq 1 ]]; then
    do_build --gpu -j "$build_jobs"
  else
    do_build -j "$build_jobs"
  fi

  echo "==> Running tests (ctest -j $test_jobs)"
  ctest --test-dir "$(build_dir)" -j "$test_jobs" --output-on-failure
}

find_executable_in_build() {
  find "$(build_dir)" -type f -perm -111 \
    \( -name "$1" -o -name "$1.x" \) 2>/dev/null | head -n 1
}

do_run() {
  local exe config mpi
  mpi="${IBLGF_MPI_RANKS:-1}"

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

  [[ $# -ge 2 ]] || die "Usage: ./iblgf.sh run-test <test_name> <config_name_or_path> [-n MPI_RANKS]"

  local test_name="$1"
  local config_arg="$2"
  shift 2

  local mpi
  mpi="$(default_mpi_ranks)"

  local resume=0
  local resume_dir=""

  local bench=0

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --gpu|-g) USE_GPU=1; shift ;;
      -n)
        shift
        mpi="${1:-}"
        [[ -n "$mpi" ]] || die "Missing value after -n"
        shift
        ;;
      --resume)
        resume=1
        shift
        
        if [[ $# -gt 0 && "$1" != -* && "$1" != --* ]]; then
          resume_dir="$1"
          shift
        fi
        ;;
      --bench)
        bench=1
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
  [[ -d "$(build_dir)" ]] || do_configure

  # Build to ensure the test executable exists and is up to date.
  do_build

  local exe
  exe="$(find_test_executable "$test_name" || true)"
  [[ -n "$exe" ]] || die "Could not find executable for test '$test_name'. Expected: build/tests/$test_name/$test_name.x"

  # Create run directory:
  local run_dir
  local tmp_dir=""

  if [[ "$bench" -eq 1 ]]; then
    tmp_dir="$(mktemp -d)"
    run_dir="$tmp_dir"
  else
    if [[ "$resume" -eq 1 ]]; then
      if [[ -n "$resume_dir" ]]; then
        run_dir="$resume_dir"
      else
        run_dir="$(latest_run_dir "$test_name" || true)"
      fi
      [[ -n "$run_dir" ]] || die "No previous run directory found to resume for '$test_name' under $(runs_root)/$test_name/"
      [[ -d "$run_dir" ]] || die "Resume directory not found: $run_dir"
    else
      run_dir="$(runs_root)/${test_name}/$(timestamp)"
      mkdir -p "$run_dir"
    fi
  fi

  # Config handling:
  local cfg_name
  cfg_name="$(basename "$config_path")"

  if [[ "$resume" -eq 1 ]]; then
    # Prefer the config already in the run dir if it exists; otherwise copy it in.
    if [[ ! -f "$run_dir/$cfg_name" ]]; then
      cp "$config_path" "$run_dir/$cfg_name"
    fi
  else
    cp "$config_path" "$run_dir/$cfg_name"
  fi

  # Record metadata for reproducibility.
  if [[ "$bench" -eq 0 ]]; then
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
  fi

  echo "==> Running test '$test_name'"
  echo "    Run dir:  $run_dir"
  echo "    Exe:      $exe"
  echo "    Config:   $cfg_name"
  echo "    MPI:      $mpi"
  echo "    Logs:     stdout.log / stderr.log"
  echo "    Expect:   output files created inside the run dir."

  # Run inside run_dir so output files land there.
  if [[ "$bench" -eq 1 ]]; then
    # Benchmark mode: no stdout/stderr logs, just timing
    (
      cd "$run_dir"

      local real_s=""
      if [[ "$mpi" -gt 1 ]]; then
        if have mpiexec; then
          real_s="$(time_cmd mpiexec -np "$mpi" "$exe" "./$cfg_name")"
        elif have mpirun; then
          real_s="$(time_cmd mpirun -n "$mpi" "$exe" "./$cfg_name")"
        else
          die "Neither mpiexec nor mpirun found in PATH."
        fi
      else
        real_s="$(time_cmd "$exe" "./$cfg_name")"
      fi

      echo "BENCH $test_name real_s=$real_s"
    )

    # cleanup temp directory
    rm -rf "$run_dir"
    return 0
  fi

  # Normal mode (current behavior)
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
  rm -rf "$(build_dir)"
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
