#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="iblgf-amrex"
HOST_BUILD_DIR="${ROOT_DIR}/build-docker-amrex"
CONTAINER_BUILD_DIR="/workspace/build-docker-amrex"

REBUILD=false
if [[ "${1:-}" == "--rebuild" ]]; then
  REBUILD=true
  shift
fi

CTEST_ARGS_STR=""
TEST_FILTER="${1:-}"
if [[ -n "${TEST_FILTER}" ]]; then
  CTEST_ARGS_STR="-R ${TEST_FILTER}"
fi

if [[ "${REBUILD}" == true || ! -d "${HOST_BUILD_DIR}" ]]; then
  echo "Build directory not found: ${HOST_BUILD_DIR}"
  echo "Configuring AMReX build inside Docker..."
  docker run --rm \
    -v "${ROOT_DIR}:/workspace" \
    -w /workspace \
    "${IMAGE}" \
    /bin/bash -lc "cmake -S . -B build-docker-amrex -DIBLGF_USE_AMREX=ON && cmake --build build-docker-amrex -j"
fi

docker run --rm \
  -v "${ROOT_DIR}:/workspace" \
  -w /workspace \
  -e OMPI_ALLOW_RUN_AS_ROOT=1 \
  -e OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
  "${IMAGE}" \
  /bin/bash -lc "ctest --test-dir '${CONTAINER_BUILD_DIR}' --output-on-failure -V ${CTEST_ARGS_STR}"
