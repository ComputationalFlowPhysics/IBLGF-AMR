#!/usr/bin/env bash
# iblgf_remote.sh — configure, build, and run IBLGF on a remote Linux/HPC host.
#
# Expected default layout:
#   <personal-directory>/IBLGF-AMR    (this repository)
#   <personal-directory>/iblgf-lib    (locally installed dependencies)
#
# The dependency root, build directory, and output directory can all be
# overridden; see `./iblgf_remote.sh help`.

set -euo pipefail

USE_GPU=0
ENVIRONMENT_READY=0

die() {
  echo "Error: $*" >&2
  exit 1
}

warn() {
  echo "Warning: $*" >&2
}

have() {
  command -v "$1" >/dev/null 2>&1
}

script_dir() {
  cd "$(dirname "${BASH_SOURCE[0]}")" && pwd
}

repo_root() {
  script_dir
}

dependency_root() {
  if [[ -n "${IBLGF_LIB_ROOT:-}" ]]; then
    cd "$IBLGF_LIB_ROOT" 2>/dev/null && pwd
  else
    local candidate
    candidate="$(dirname "$(repo_root)")/iblgf-lib"
    cd "$candidate" 2>/dev/null && pwd
  fi
}

build_dir() {
  if [[ -n "${IBLGF_BUILD_DIR:-}" ]]; then
    if [[ "$IBLGF_BUILD_DIR" = /* ]]; then
      echo "$IBLGF_BUILD_DIR"
    else
      echo "$(repo_root)/$IBLGF_BUILD_DIR"
    fi
  elif [[ "$USE_GPU" -eq 1 ]]; then
    echo "$(repo_root)/build-remote-gpu"
  else
    echo "$(repo_root)/build-remote"
  fi
}

runs_root() {
  if [[ -n "${IBLGF_RUNS_ROOT:-}" ]]; then
    if [[ "$IBLGF_RUNS_ROOT" = /* ]]; then
      echo "$IBLGF_RUNS_ROOT"
    else
      echo "$(repo_root)/$IBLGF_RUNS_ROOT"
    fi
  else
    echo "$(repo_root)/runs"
  fi
}

ensure_repo_root() {
  [[ -f "$(repo_root)/CMakeLists.txt" ]] ||
    die "CMakeLists.txt not found beside this script."
}

cpu_count() {
  if [[ -n "${SLURM_TASKS_PER_NODE:-}" ]]; then
    # Common forms are "40", "40(x2)", and "40,32". Compilation happens on
    # the current node, so use its allocated tasks rather than all node cores.
    local tasks_per_node cpus_per_task
    tasks_per_node="${SLURM_TASKS_PER_NODE%%,*}"
    tasks_per_node="${tasks_per_node%%(*}"
    cpus_per_task="${SLURM_CPUS_PER_TASK:-1}"
    if [[ "$tasks_per_node" =~ ^[1-9][0-9]*$ ]] &&
       [[ "$cpus_per_task" =~ ^[1-9][0-9]*$ ]]; then
      echo $((tasks_per_node * cpus_per_task))
      return 0
    fi
  fi

  if [[ -n "${SLURM_NTASKS:-}" && "${SLURM_JOB_NUM_NODES:-1}" -eq 1 ]]; then
    echo "$SLURM_NTASKS"
  elif have nproc; then
    nproc
  elif have getconf; then
    getconf _NPROCESSORS_ONLN
  else
    echo 4
  fi
}

default_build_jobs() {
  echo "${IBLGF_BUILD_JOBS:-$(cpu_count)}"
}

default_test_jobs() {
  echo "${IBLGF_TEST_JOBS:-1}"
}

default_mpi_ranks() {
  echo "${IBLGF_MPI_RANKS:-2}"
}

require_positive_integer() {
  local label="$1"
  local value="$2"
  [[ "$value" =~ ^[1-9][0-9]*$ ]] ||
    die "$label must be a positive integer; got '$value'."
}

prepend_path() {
  local var_name="$1"
  local directory="$2"
  [[ -d "$directory" ]] || return 0

  local current="${!var_name:-}"
  case ":$current:" in
    *":$directory:"*) return 0 ;;
  esac

  if [[ -n "$current" ]]; then
    printf -v "$var_name" '%s:%s' "$directory" "$current"
  else
    printf -v "$var_name" '%s' "$directory"
  fi
  export "$var_name"
}

load_modules() {
  [[ -n "${IBLGF_MODULES:-}" ]] || return 0

  # On many clusters `module` is initialized only for login shells.
  if ! type module >/dev/null 2>&1; then
    if [[ -r /etc/profile.d/modules.sh ]]; then
      # shellcheck disable=SC1091
      source /etc/profile.d/modules.sh
    elif [[ -r /usr/share/Modules/init/bash ]]; then
      # shellcheck disable=SC1091
      source /usr/share/Modules/init/bash
    fi
  fi

  type module >/dev/null 2>&1 ||
    die "IBLGF_MODULES is set, but the environment-modules command is unavailable."

  local module_name
  for module_name in $IBLGF_MODULES; do
    echo "==> Loading module: $module_name"
    module load "$module_name"
  done
}

setup_environment() {
  [[ "$ENVIRONMENT_READY" -eq 0 ]] || return 0

  load_modules

  local lib_root setup_script
  lib_root="$(dependency_root || true)"
  [[ -n "$lib_root" ]] || {
    die "Dependency directory not found. Expected ../iblgf-lib; set IBLGF_LIB_ROOT to its absolute path."
  }

  setup_script="${IBLGF_SETUP_SCRIPT:-$lib_root/setup_iblgf.sh}"
  if [[ "$setup_script" != "none" ]]; then
    [[ -f "$setup_script" ]] ||
      die "Environment setup script not found: $setup_script (or set IBLGF_SETUP_SCRIPT=none)."
    echo "==> Loading dependency environment: $setup_script"
    # Some third-party setup scripts are not written for nounset mode.
    set +u
    # shellcheck disable=SC1090
    source "$setup_script"
    set -u
  fi

  # These defaults support the iblgf-lib layout while preserving anything the
  # setup script or cluster module system already supplied.
  prepend_path PATH "$lib_root/cmake/bin"
  prepend_path PATH "$lib_root/boost/bin"
  prepend_path PATH "$lib_root/petsc/bin"

  local prefix
  for prefix in \
    "$lib_root" \
    "$lib_root/boost" \
    "$lib_root/petsc" \
    "$lib_root/xtensor-stack" \
    "$lib_root/xsimd-7.4.9" \
    "$lib_root/xtensor-0.23.10" \
    "$lib_root/xtensor-blas-0.19.2" \
    "$lib_root/xtl-0.7.5"; do
    prepend_path CMAKE_PREFIX_PATH "$prefix"
  done

  for prefix in "$lib_root" "$lib_root/boost" "$lib_root/petsc" "$lib_root/xtensor-stack"; do
    prepend_path LD_LIBRARY_PATH "$prefix/lib"
    prepend_path LD_LIBRARY_PATH "$prefix/lib64"
    prepend_path PKG_CONFIG_PATH "$prefix/lib/pkgconfig"
    prepend_path PKG_CONFIG_PATH "$prefix/lib64/pkgconfig"
  done

  if [[ -d "$lib_root/boost" ]]; then
    export BOOST_ROOT="${BOOST_ROOT:-$lib_root/boost}"
    export Boost_ROOT="${Boost_ROOT:-$lib_root/boost}"
  fi
  if [[ -d "$lib_root/petsc" ]]; then
    export PETSC_DIR="${PETSC_DIR:-$lib_root/petsc}"
  fi

  ENVIRONMENT_READY=1
}

select_c_compiler() {
  if [[ -n "${IBLGF_CC:-}" ]]; then
    echo "$IBLGF_CC"
  elif [[ -n "${CC:-}" ]]; then
    echo "$CC"
  elif have mpicc; then
    command -v mpicc
  elif have cc; then
    command -v cc
  else
    return 1
  fi
}

select_cxx_compiler() {
  if [[ -n "${IBLGF_CXX:-}" ]]; then
    echo "$IBLGF_CXX"
  elif [[ -n "${CXX:-}" ]]; then
    echo "$CXX"
  elif have mpicxx; then
    command -v mpicxx
  elif have mpic++; then
    command -v mpic++
  elif have c++; then
    command -v c++
  else
    return 1
  fi
}

need_nvcc_if_gpu() {
  if [[ "$USE_GPU" -eq 1 ]] && ! have nvcc; then
    die "GPU build requested, but nvcc is unavailable. Load the cluster CUDA module first."
  fi
}

print_usage() {
  cat <<'USAGE'
Usage:
  ./iblgf_remote.sh doctor
  ./iblgf_remote.sh configure [--fresh] [--gpu] [--cmake-arg ARG]
  ./iblgf_remote.sh build [-j N] [--gpu]
  ./iblgf_remote.sh test [-j N] [--gpu]
  ./iblgf_remote.sh run <exe-or-target> <config> [-n MPI_RANKS] [-- extra-args]
  ./iblgf_remote.sh run-test <test_name> <config_name_or_path>
                          [-n MPI_RANKS] [--resume [RUN_DIR]] [--bench]
  ./iblgf_remote.sh clean

Default directory layout:
  ../IBLGF-AMR       repository containing this script
  ../iblgf-lib       Boost, CMake, PETSc, and xtensor installations
  build-remote/      separate CMake cache for the remote machine
  runs/              timestamped simulation output

Useful environment overrides:
  IBLGF_LIB_ROOT=/path/to/iblgf-lib
  IBLGF_SETUP_SCRIPT=/path/to/setup_iblgf.sh  (use "none" to skip)
  IBLGF_BUILD_DIR=/scratch/$USER/iblgf-build
  IBLGF_RUNS_ROOT=/scratch/$USER/iblgf-runs
  IBLGF_MODULES="gcc openmpi hdf5 fftw openblas"
  IBLGF_CC=mpicc
  IBLGF_CXX=mpicxx
  IBLGF_BUILD_JOBS=8
  IBLGF_TEST_JOBS=1
  IBLGF_MPI_RANKS=8
  IBLGF_MPI_NP=8               CTest MPI rank count at configure time
  IBLGF_LAUNCHER=auto          auto, mpiexec, mpirun, or srun
  IBLGF_MPI_ARGS="..."         extra arguments placed before the executable

Launcher selection:
  auto prefers the MPI installation's mpiexec, then mpirun. It only falls
  back to srun inside a Slurm allocation. To force Slurm's direct MPI launch,
  set IBLGF_LAUNCHER=srun and, when required, IBLGF_MPI_ARGS='--mpi=pmix'.
  This script does not request a cluster allocation.

Examples:
  ./iblgf_remote.sh doctor
  ./iblgf_remote.sh configure --fresh
  ./iblgf_remote.sh build -j 8
  ./iblgf_remote.sh test -j 1
  ./iblgf_remote.sh run-test ns_amr_lgf2D configFile_0 -n 8

  IBLGF_RUNS_ROOT=/scratch/$USER/iblgf-runs \
    ./iblgf_remote.sh run-test ns_amr_lgf2D path/to/config.cfg -n 12
USAGE
}

doctor_command_version() {
  local label="$1"
  local command_name="$2"
  if have "$command_name"; then
    local command_path version
    command_path="$(command -v "$command_name")"
    version="$($command_name --version 2>&1 | head -n 1 || true)"
    printf '  %-12s %s\n' "$label:" "$command_path"
    [[ -n "$version" ]] && printf '  %-12s %s\n' "" "$version"
  else
    printf '  %-12s MISSING\n' "$label:"
  fi
}

do_doctor() {
  ensure_repo_root
  setup_environment

  local lib_root cc cxx launcher errors
  lib_root="$(dependency_root)"
  cc="$(select_c_compiler || true)"
  cxx="$(select_cxx_compiler || true)"
  launcher="$(select_launcher || true)"
  errors=0

  echo "IBLGF remote environment"
  echo "  repository:  $(repo_root)"
  echo "  dependencies: $lib_root"
  echo "  build:       $(build_dir)"
  echo "  runs:        $(runs_root)"
  echo "  launcher:    ${launcher:-MISSING}"
  echo "  C compiler:  ${cc:-MISSING}"
  echo "  C++ compiler:${cxx:+ }${cxx:-MISSING}"
  echo
  echo "Commands"
  doctor_command_version "cmake" cmake
  doctor_command_version "ctest" ctest
  doctor_command_version "mpicc" mpicc
  doctor_command_version "mpicxx" mpicxx
  doctor_command_version "mpiexec" mpiexec
  doctor_command_version "srun" srun

  have cmake || errors=$((errors + 1))
  [[ -n "$cc" ]] || errors=$((errors + 1))
  [[ -n "$cxx" ]] || errors=$((errors + 1))
  [[ -n "$launcher" ]] || errors=$((errors + 1))

  echo
  if have h5pcc; then
    echo "Parallel HDF5"
    h5pcc -showconfig 2>/dev/null | awk '/Parallel HDF5:/{print "  " $0}' || true
  elif have h5cc; then
    warn "h5cc exists but h5pcc does not; verify that HDF5 was built with MPI support."
  else
    warn "No h5pcc/h5cc wrapper found; CMake may still find HDF5 through CMAKE_PREFIX_PATH."
  fi

  if [[ -z "${SLURM_JOB_ID:-}" ]] && have srun; then
    warn "srun is installed, but no Slurm allocation is active. Use salloc/sbatch before a large simulation."
  fi

  if [[ "$errors" -gt 0 ]]; then
    die "$errors required tool(s) are missing. Load the appropriate modules and run doctor again."
  fi

  echo
  echo "Environment looks ready for CMake configuration."
}

timestamp() {
  date +"%Y-%m-%d_%H-%M-%S"
}

iso_timestamp() {
  date -Iseconds 2>/dev/null || date
}

git_commit() {
  if have git && git -C "$(repo_root)" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git -C "$(repo_root)" rev-parse --short HEAD
  else
    echo unknown
  fi
}

do_configure() {
  ensure_repo_root
  setup_environment

  local fresh=0
  local -a extra_cmake_args=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --fresh) fresh=1; shift ;;
      --gpu|-g) USE_GPU=1; shift ;;
      --cmake-arg)
        shift
        [[ $# -gt 0 ]] || die "Missing value after --cmake-arg"
        extra_cmake_args+=("$1")
        shift
        ;;
      *) die "Unknown configure option: $1" ;;
    esac
  done

  need_nvcc_if_gpu

  local root build cc cxx
  root="$(repo_root)"
  build="$(build_dir)"
  cc="$(select_c_compiler || true)"
  cxx="$(select_cxx_compiler || true)"
  [[ -n "$cc" ]] || die "No C compiler found. Load an MPI/compiler module or set IBLGF_CC."
  [[ -n "$cxx" ]] || die "No C++ compiler found. Load an MPI/compiler module or set IBLGF_CXX."

  if [[ "$fresh" -eq 1 && -d "$build" ]]; then
    echo "==> Removing remote CMake cache: $build"
    rm -rf "$build"
  fi
  mkdir -p "$build"

  local -a cmake_args=(
    -S "$root"
    -B "$build"
    -G "Unix Makefiles"
    -DCMAKE_C_COMPILER="$cc"
    -DCMAKE_CXX_COMPILER="$cxx"
    -DCMAKE_BUILD_TYPE="${IBLGF_BUILD_TYPE:-Release}"
  )

  [[ "$USE_GPU" -eq 0 ]] || cmake_args+=(-DUSE_GPU=ON)
  [[ -z "${CMAKE_PREFIX_PATH:-}" ]] ||
    cmake_args+=(-DCMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH//:/;}")
  [[ -z "${BOOST_ROOT:-}" ]] || cmake_args+=(-DBOOST_ROOT="$BOOST_ROOT")
  [[ -z "${IBLGF_MPI_NP:-}" ]] || cmake_args+=(-DIBLGF_MPI_NP="$IBLGF_MPI_NP")
  [[ -z "${IBLGF_TOOLCHAIN_FILE:-}" ]] ||
    cmake_args+=(-DCMAKE_TOOLCHAIN_FILE="$IBLGF_TOOLCHAIN_FILE")
  cmake_args+=("${extra_cmake_args[@]}")

  echo "==> Configuring remote build"
  echo "    Build dir: $build"
  echo "    C:         $cc"
  echo "    C++:       $cxx"
  cmake "${cmake_args[@]}"
}

parse_build_options() {
  BUILD_JOBS="$(default_build_jobs)"
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --gpu|-g) USE_GPU=1; shift ;;
      -j)
        shift
        [[ $# -gt 0 ]] || die "Missing value after -j"
        BUILD_JOBS="$1"
        shift
        ;;
      *) die "Unknown build option: $1" ;;
    esac
  done
}

do_build() {
  ensure_repo_root
  setup_environment
  parse_build_options "$@"
  require_positive_integer "Build job count" "$BUILD_JOBS"

  local build
  build="$(build_dir)"
  if [[ ! -f "$build/CMakeCache.txt" ]]; then
    if [[ "$USE_GPU" -eq 1 ]]; then
      do_configure --gpu
    else
      do_configure
    fi
  fi

  echo "==> Building remote tree (-j $BUILD_JOBS)"
  cmake --build "$build" -j "$BUILD_JOBS"
}

do_test() {
  ensure_repo_root
  setup_environment

  local test_jobs build_jobs
  test_jobs="$(default_test_jobs)"
  build_jobs="$(default_build_jobs)"
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --gpu|-g) USE_GPU=1; shift ;;
      -j)
        shift
        [[ $# -gt 0 ]] || die "Missing value after -j"
        test_jobs="$1"
        shift
        ;;
      *) die "Unknown test option: $1" ;;
    esac
  done
  require_positive_integer "CTest job count" "$test_jobs"

  if [[ "$USE_GPU" -eq 1 ]]; then
    do_build --gpu -j "$build_jobs"
  else
    do_build -j "$build_jobs"
  fi

  echo "==> Running CTest (-j $test_jobs)"
  ctest --test-dir "$(build_dir)" -j "$test_jobs" --output-on-failure
}

select_launcher() {
  local requested="${IBLGF_LAUNCHER:-auto}"
  case "$requested" in
    auto)
      # Prefer the launcher installed with the MPI library used to link the
      # executable. A plain `srun` can otherwise start N singleton processes
      # when Slurm's PMI/PMIx plugin does not match that MPI installation.
      if have mpiexec; then
        echo mpiexec
      elif have mpirun; then
        echo mpirun
      elif [[ -n "${SLURM_JOB_ID:-}" ]] && have srun; then
        echo srun
      else
        return 1
      fi
      ;;
    srun|mpiexec|mpirun)
      have "$requested" || die "Requested MPI launcher is unavailable: $requested"
      echo "$requested"
      ;;
    *) die "IBLGF_LAUNCHER must be auto, srun, mpiexec, or mpirun." ;;
  esac
}

MPI_COMMAND=()
build_run_command() {
  local ranks="$1"
  shift
  MPI_COMMAND=()
  require_positive_integer "MPI rank count" "$ranks"

  if [[ "$ranks" -le 1 ]]; then
    MPI_COMMAND=("$@")
    return 0
  fi

  local launcher
  launcher="$(select_launcher || true)"
  [[ -n "$launcher" ]] || die "No MPI launcher found (mpiexec, mpirun, or srun)."

  case "$launcher" in
    srun) MPI_COMMAND=(srun --ntasks "$ranks") ;;
    mpiexec) MPI_COMMAND=(mpiexec -np "$ranks") ;;
    mpirun) MPI_COMMAND=(mpirun -n "$ranks") ;;
  esac

  if [[ -n "${IBLGF_MPI_ARGS:-}" ]]; then
    local -a extra_mpi_args
    # This variable is intentionally split into launcher arguments.
    # shellcheck disable=SC2206
    extra_mpi_args=($IBLGF_MPI_ARGS)
    MPI_COMMAND+=("${extra_mpi_args[@]}")
  fi
  MPI_COMMAND+=("$@")
}

find_executable_in_build() {
  find "$(build_dir)" -type f -perm -111 \
    \( -name "$1" -o -name "$1.x" \) 2>/dev/null | head -n 1
}

find_test_executable() {
  local test_name="$1"
  local candidate
  candidate="$(build_dir)/tests/$test_name/$test_name.x"
  if [[ -x "$candidate" ]]; then
    echo "$candidate"
  else
    find "$(build_dir)" -type f -perm -111 -name "$test_name.x" 2>/dev/null | head -n 1
  fi
}

find_test_config() {
  local test_name="$1"
  local config_arg="$2"
  if [[ -f "$config_arg" ]]; then
    cd "$(dirname "$config_arg")" && echo "$(pwd)/$(basename "$config_arg")"
    return 0
  fi

  local candidate
  candidate="$(repo_root)/tests/$test_name/configs/$config_arg"
  [[ -f "$candidate" ]] || return 1
  echo "$candidate"
}

latest_run_dir() {
  local test_name="$1"
  local base
  base="$(runs_root)/$test_name"
  [[ -d "$base" ]] || return 1
  ls -1dt "$base"/*/ 2>/dev/null | head -n 1
}

do_run() {
  setup_environment
  [[ $# -ge 2 ]] || die "Usage: ./iblgf_remote.sh run <exe-or-target> <config> [-n RANKS] [-- extra-args]"

  local exe="$1"
  local config="$2"
  local mpi
  local -a extra_args=()
  mpi="${IBLGF_MPI_RANKS:-1}"
  shift 2

  while [[ $# -gt 0 ]]; do
    case "$1" in
      -n)
        shift
        [[ $# -gt 0 ]] || die "Missing value after -n"
        mpi="$1"
        shift
        ;;
      --)
        shift
        extra_args=("$@")
        break
        ;;
      *) die "Unknown run option: $1" ;;
    esac
  done

  local exe_path
  if [[ -x "$exe" ]]; then
    exe_path="$exe"
  else
    exe_path="$(find_executable_in_build "$exe" || true)"
  fi
  [[ -n "$exe_path" ]] || die "Executable not found: $exe"
  [[ -f "$config" ]] || die "Config not found: $config"

  build_run_command "$mpi" "$exe_path" "$config" "${extra_args[@]}"
  echo "==> Running with $mpi rank(s): ${MPI_COMMAND[*]}"
  "${MPI_COMMAND[@]}"
}

time_run_command() {
  local timing_file="$1"
  shift
  /usr/bin/time -p -o "$timing_file" "$@"
  awk '/^real /{print $2; exit}' "$timing_file"
}

do_run_test() {
  ensure_repo_root
  setup_environment
  [[ $# -ge 2 ]] ||
    die "Usage: ./iblgf_remote.sh run-test <test_name> <config> [-n RANKS] [--resume [RUN_DIR]] [--bench]"

  local test_name="$1"
  local config_arg="$2"
  local mpi resume resume_dir bench
  test_name="$1"
  config_arg="$2"
  mpi="$(default_mpi_ranks)"
  resume=0
  resume_dir=""
  bench=0
  shift 2

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --gpu|-g) USE_GPU=1; shift ;;
      -n)
        shift
        [[ $# -gt 0 ]] || die "Missing value after -n"
        mpi="$1"
        shift
        ;;
      --resume)
        resume=1
        shift
        if [[ $# -gt 0 && "$1" != -* ]]; then
          resume_dir="$1"
          shift
        fi
        ;;
      --bench) bench=1; shift ;;
      *) die "Unknown run-test option: $1" ;;
    esac
  done

  local config_path
  config_path="$(find_test_config "$test_name" "$config_arg" || true)"
  [[ -n "$config_path" ]] ||
    die "Config not found: $config_arg (also checked tests/$test_name/configs)."

  local build_jobs
  build_jobs="$(default_build_jobs)"
  if [[ "$USE_GPU" -eq 1 ]]; then
    do_build --gpu -j "$build_jobs"
  else
    do_build -j "$build_jobs"
  fi

  local exe
  exe="$(find_test_executable "$test_name" || true)"
  [[ -n "$exe" ]] ||
    die "Test executable not found; expected $(build_dir)/tests/$test_name/$test_name.x"

  local run_dir tmp_dir
  tmp_dir=""
  if [[ "$bench" -eq 1 ]]; then
    tmp_dir="$(mktemp -d)"
    run_dir="$tmp_dir"
  elif [[ "$resume" -eq 1 ]]; then
    if [[ -n "$resume_dir" ]]; then
      run_dir="$resume_dir"
    else
      run_dir="$(latest_run_dir "$test_name" || true)"
    fi
    [[ -n "$run_dir" && -d "$run_dir" ]] ||
      die "No prior run directory found for $test_name."
  else
    run_dir="$(runs_root)/$test_name/$(timestamp)"
    mkdir -p "$run_dir"
  fi

  local cfg_name
  cfg_name="$(basename "$config_path")"
  if [[ "$resume" -eq 0 || ! -f "$run_dir/$cfg_name" ]]; then
    cp "$config_path" "$run_dir/$cfg_name"
  fi

  build_run_command "$mpi" "$exe" "./$cfg_name"
  if [[ "$bench" -eq 0 ]]; then
    {
      echo "test_name: $test_name"
      echo "exe: $exe"
      echo "config: $cfg_name"
      echo "mpi_ranks: $mpi"
      echo "launcher: ${MPI_COMMAND[0]}"
      echo "git_commit: $(git_commit)"
      echo "timestamp: $(iso_timestamp)"
      echo "host: $(hostname)"
      echo "run_dir: $run_dir"
      echo "command: ${MPI_COMMAND[*]}"
    } > "$run_dir/meta.txt"
  fi

  if [[ "$bench" -eq 1 ]]; then
    echo "==> Running benchmark '$test_name' with $mpi rank(s)" >&2
    local timing_file real_seconds
    timing_file="$run_dir/time.txt"
    (
      cd "$run_dir"
      real_seconds="$(time_run_command "$timing_file" "${MPI_COMMAND[@]}")"
      echo "$real_seconds"
    )
    rm -rf "$tmp_dir"
    return 0
  fi

  echo "==> Running test '$test_name'"
  echo "    Host:     $(hostname)"
  echo "    Run dir:  $run_dir"
  echo "    Exe:      $exe"
  echo "    Config:   $cfg_name"
  echo "    MPI:      $mpi (${MPI_COMMAND[0]})"
  echo "    Logs:     stdout.log / stderr.log"

  (
    cd "$run_dir"
    "${MPI_COMMAND[@]}" > stdout.log 2> stderr.log
  )

  echo "==> Done. Outputs are in: $run_dir"
}

do_clean() {
  local build
  build="$(build_dir)"
  [[ -d "$build" ]] || {
    echo "Nothing to clean: $build does not exist."
    return 0
  }
  echo "==> Removing remote build directory: $build"
  rm -rf "$build"
}

cmd="${1:-help}"
shift || true

case "$cmd" in
  help|-h|--help) print_usage ;;
  doctor) do_doctor "$@" ;;
  configure) do_configure "$@" ;;
  build) do_build "$@" ;;
  test) do_test "$@" ;;
  run) do_run "$@" ;;
  run-test) do_run_test "$@" ;;
  clean) do_clean ;;
  *) echo "Unknown command: $cmd" >&2; print_usage; exit 1 ;;
esac
