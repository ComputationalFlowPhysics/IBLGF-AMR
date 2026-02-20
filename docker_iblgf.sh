#!/usr/bin/env bash
# docker_iblgf.sh — enter IBLGF Docker env with optional CPU limit and Python
set -euo pipefail

CPU_BASE_IMAGE="ccardina/my-app:cpu"

GPU_BASE_IMAGE="ccardina/my-app:gpu"

USE_GPU=0

# Where the repo lives
CONTAINER_ROOT="/workspace2"
CONTAINER_REPO_DIR="$CONTAINER_ROOT/IBLGF-AMR"

usage() {
  cat <<EOF
Usage:
  ./docker_iblgf.sh
  ./docker_iblgf.sh -c N
  ./docker_iblgf.sh -g
  ./docker_iblgf.sh -g -c N

Options:
  -c N    Limit Docker container to N CPU cores
  -g      Use GPU image 

Examples:
  ./docker_iblgf.sh
  ./docker_iblgf.sh -c 4
  ./docker_iblgf.sh -g
  ./docker_iblgf.sh -g -c 4
EOF
}

CPUS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --)
      shift
      break   # <-- stop option parsing here
      ;;
    -c|--cpus)
      shift
      CPUS="${1:-}"
      [[ -n "$CPUS" ]] || { echo "Missing value for -c"; exit 1; }
      # basic numeric validation
      [[ "$CPUS" =~ ^[0-9]+$ ]] || { echo "CPU count must be an integer"; exit 1; }
      [[ "$CPUS" -ge 1 ]] || { echo "CPU count must be >= 1"; exit 1; }
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    -g|--gpu)
      USE_GPU=1
      shift
      ;;
    *)
      echo "Unknown option: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ "$USE_GPU" -eq 1 ]]; then
  BASE_IMAGE="$GPU_BASE_IMAGE"
else
  BASE_IMAGE="$CPU_BASE_IMAGE"
fi

echo "==> Base image: $BASE_IMAGE"

# Pull base image if missing
if ! docker image inspect "$BASE_IMAGE" >/dev/null 2>&1; then
  echo "==> Pulling base image..."
  docker pull "$BASE_IMAGE"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Case A: script sits in repo root and repo has CMakeLists.txt
if [[ -f "$SCRIPT_DIR/CMakeLists.txt" ]]; then
  HOST_REPO_DIR="$SCRIPT_DIR"
# Case B: script is in a subdir of the repo 
  SEARCH_DIR="$SCRIPT_DIR"
  HOST_REPO_DIR=""
  while [[ "$SEARCH_DIR" != "/" ]]; do
    if [[ -f "$SEARCH_DIR/CMakeLists.txt" ]]; then
      HOST_REPO_DIR="$SEARCH_DIR"
      break
    fi
    SEARCH_DIR="$(dirname "$SEARCH_DIR")"
  done
  [[ -n "$HOST_REPO_DIR" ]] || { echo "ERROR: Could not locate repo root (CMakeLists.txt)"; exit 1; }
fi

echo "==> Host repo dir: $HOST_REPO_DIR"
echo "==> Container repo dir: $CONTAINER_REPO_DIR"

INTERACTIVE=1
if [[ $# -gt 0 ]]; then
  INTERACTIVE=0
fi

DOCKER_ARGS=(docker run --rm)

DOCKER_ARGS+=( --user "$(id -u):$(id -g)" )

if [[ "$INTERACTIVE" -eq 1 ]]; then
  DOCKER_ARGS+=( -it )
fi

# Mount repo and set working directory
DOCKER_ARGS+=(
  -v "$HOST_REPO_DIR:$CONTAINER_REPO_DIR"
  -w "$CONTAINER_REPO_DIR"
)

DOCKER_ARGS+=( -e LD_LIBRARY_PATH="/usr/local/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}" )
#check if env var CI is set to true or if GITHUB_ACTIONS is set to non-empty string, and if so, set OMPI_MCA_rmaps_base_oversubscribe=1
if [[ "${CI:-false}" == "true" ]]; then
  echo "==> Detected CI environment, setting OMPI_MCA_rmaps_base_oversubscribe=1"
  DOCKER_ARGS+=( -e OMPI_MCA_rmaps_base_oversubscribe=1 )
fi


if [[ "$USE_GPU" -eq 1 ]]; then
  DOCKER_ARGS+=(--gpus all)
fi

if [[ -n "$CPUS" ]]; then
  echo "==> Limiting container to $CPUS CPU(s)"
  DOCKER_ARGS+=(--cpuset-cpus="0-$((CPUS-1))")
fi

DOCKER_ARGS+=("$BASE_IMAGE")

if [[ $# -gt 0 ]]; then
  # Non-interactive
  DOCKER_ARGS+=("$@")
else
  # Interactive shell
  DOCKER_ARGS+=("bash")
fi

exec "${DOCKER_ARGS[@]}"
