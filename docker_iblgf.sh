#!/usr/bin/env bash
# docker_iblgf.sh — enter IBLGF Docker env with optional CPU limit and Python

set -euo pipefail

BASE_IMAGE="ccardina/my-app:cpu"
PY_IMAGE="ccardina/my-app:cpu-python"
WORKDIR="/workspace2"

usage() {
  cat <<EOF
Usage:
  ./docker_iblgf.sh
  ./docker_iblgf.sh -c N

Options:
  -c N    Limit Docker container to N CPU cores

Examples:
  ./docker_iblgf.sh
  ./docker_iblgf.sh -c 4
EOF
}

CPUS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -c|--cpus)
      shift
      CPUS="${1:-}"
      [[ -n "$CPUS" ]] || { echo "Missing value for -c"; exit 1; }
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      usage
      exit 1
      ;;
  esac
done

echo "==> Base image: $BASE_IMAGE"

# Pull base image if missing
if ! docker image inspect "$BASE_IMAGE" >/dev/null 2>&1; then
  echo "==> Pulling base image..."
  docker pull "$BASE_IMAGE"
fi

# Build Python-extended image if missing
if ! docker image inspect "$PY_IMAGE" >/dev/null 2>&1; then
  echo "==> Building Python-enabled image: $PY_IMAGE"

  docker build -t "$PY_IMAGE" - <<'EOF'
FROM ccardina/my-app:cpu

# Temporarily become root to install packages
USER root

# Some minimal images don't have apt lists dirs created
RUN mkdir -p /var/lib/apt/lists/partial

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
 && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir --upgrade pip \
 && python3 -m pip install --no-cache-dir numpy scipy matplotlib

# Switch back to whatever the base image expects (often 1000:1000)
# If the base image uses a different user, runtime can still override with --user.
USER 1000:1000

WORKDIR /workspace2
EOF
else
  echo "==> Python-enabled image already exists"
fi

DOCKER_ARGS=(
  docker run -it --rm
  -v "$(pwd):$WORKDIR"
  -w "$WORKDIR"
)

if [[ -n "$CPUS" ]]; then
  echo "==> Limiting container to $CPUS CPU(s)"
  DOCKER_ARGS+=(--cpuset-cpus="0-$((CPUS-1))")
fi

DOCKER_ARGS+=("$PY_IMAGE")

exec "${DOCKER_ARGS[@]}"
