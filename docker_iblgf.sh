#!/usr/bin/env bash
# docker_iblgf.sh — enter IBLGF Docker env with optional CPU limit and Python
set -euo pipefail

CPU_BASE_IMAGE="ccardina/my-app:cpu"
CPU_PY_IMAGE="ccardina/my-app:cpu-python"

GPU_BASE_IMAGE="ccardina/my-app:gpu"
GPU_PY_IMAGE="ccardina/my-app:gpu-python"

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
  PY_IMAGE="$GPU_PY_IMAGE"
else
  BASE_IMAGE="$CPU_BASE_IMAGE"
  PY_IMAGE="$CPU_PY_IMAGE"
fi

echo "==> Base image: $BASE_IMAGE"

# Pull base image if missing
if ! docker image inspect "$BASE_IMAGE" >/dev/null 2>&1; then
  echo "==> Pulling base image..."
  docker pull "$BASE_IMAGE"
fi

# Build Python-extended image if missing
if ! docker image inspect "$PY_IMAGE" >/dev/null 2>&1; then
  echo "==> Building Python-enabled image: $PY_IMAGE"

  docker build -t "$PY_IMAGE" - <<EOF
FROM ${BASE_IMAGE}

USER root
RUN mkdir -p /var/lib/apt/lists/partial

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
 && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir --upgrade pip \
 && python3 -m pip install --no-cache-dir numpy scipy matplotlib

USER 1000:1000
WORKDIR /workspace2
EOF
fi

# ---- Determine repo root robustly (works whether script is run inside or outside repo) ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Case A: script sits in repo root and repo has CMakeLists.txt
if [[ -f "$SCRIPT_DIR/CMakeLists.txt" ]]; then
  HOST_REPO_DIR="$SCRIPT_DIR"
# Case B: script is in a subdir of the repo (e.g., scripts/), search upward
else
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

DOCKER_ARGS=(
  docker run -it --rm
  # Mount repo to a stable path inside container
  -v "$HOST_REPO_DIR:$CONTAINER_REPO_DIR"
  # Always start inside the repo
  -w "$CONTAINER_REPO_DIR"
)

if [[ "$USE_GPU" -eq 1 ]]; then
  DOCKER_ARGS+=(--gpus all)
fi

if [[ -n "$CPUS" ]]; then
  echo "==> Limiting container to $CPUS CPU(s)"
  DOCKER_ARGS+=(--cpuset-cpus="0-$((CPUS-1))")
fi

DOCKER_ARGS+=("$PY_IMAGE")

exec "${DOCKER_ARGS[@]}"
