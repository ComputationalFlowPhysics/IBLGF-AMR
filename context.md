# IBLGF-AMR GPU Context (2026-04-06)

## Goal
Create a new GPU-resident library/test target `ns_amr_lgf_gpu_new` using the logic of `ns_amr_lgf`, with data always on GPU and CPU used only for halo exchange, and wire GPU-only paths for interpolation/add_source_correction/IFHERK.

## Key Changes (recent)
- Added GPU-only operators helpers:
  - `zero_boundary_device`, `set_constant_field_device`, `add_body_force_device` in:
    - `/Users/carolinecardinale/Desktop/iblgf-docker/IBLGF-AMR/include/iblgf/operators/operators_GPU.hpp`
    - `/Users/carolinecardinale/Desktop/iblgf-docker/IBLGF-AMR/src/operators/operators_GPU.cu`
- IFHERK updated to use GPU path for `test_type` fill, clean boundary, and add_body_force (no CPU fallbacks):
  - `/Users/carolinecardinale/Desktop/iblgf-docker/IBLGF-AMR/include/iblgf/solver/time_integration/ifherk.hpp`
- GPU interpolation helper `add_source_correction_device` wired in GPU-resident path, and CPU fallback Dim variable conflict fixed:
  - `/Users/carolinecardinale/Desktop/iblgf-docker/IBLGF-AMR/include/iblgf/interpolation/cell_center_nli_intrp.hpp`
    - Renamed CPU `Dim` to `dim_cpu` to avoid redeclaration.
    - Moved `max_1D_child_n` and `max_relative_pos` earlier in class for CUDA visibility.

## New Target
- `ns_amr_lgf_gpu_new` target in tests:
  - `/Users/carolinecardinale/Desktop/iblgf-docker/IBLGF-AMR/tests/ns_amr_lgf_gpu_new/CMakeLists.txt`
  - Source: `/Users/carolinecardinale/Desktop/iblgf-docker/IBLGF-AMR/tests/ns_amr_lgf_gpu_new/ns_amr_lgf_gpu_new.cu`
- CMake adds target when `USE_GPU=ON` and defines `IBLGF_GPU_RESIDENT`.

## Docker Build (compile-only)
Built the GPU compile container from the repo Dockerfile:
- Dockerfile: `/Users/carolinecardinale/Desktop/iblgf-docker/IBLGF-AMR/.github/Dockerfile`
- Image tag used: `iblgf-gpu-build`
- Build command:
  ```bash
  docker build --platform=linux/amd64 -f .github/Dockerfile \
    --build-arg BASE_IMAGE=nvidia/cuda:12.1.1-devel-ubuntu22.04 \
    --build-arg TARGET=gpu -t iblgf-gpu-build .
  ```

## Compile Command in Container
Use a fresh build dir to avoid host cache conflicts:
```bash
docker run --rm --platform=linux/amd64 --user "$(id -u):$(id -g)" \
  -v "$PWD:/workspace2/IBLGF-AMR" -w /workspace2/IBLGF-AMR \
  iblgf-gpu-build /bin/bash -lc \
  "cmake -S . -B build-gpu-docker -DUSE_GPU=ON && \
   cmake --build build-gpu-docker --target ns_amr_lgf_gpu_new.x -j"
```
- Note: the container prints a warning about missing NVIDIA Driver; this is expected for compile-only.

## Build Status
- `ns_amr_lgf_gpu_new.x` successfully compiles in the Docker container after the fixes above.
- Typical warnings: CUDA arch policy CMP0104, NVCC warnings in headers (sign conversion, unreachable loops, etc.).

## Remaining TODOs / Follow-ups
- Continue porting GPU-only data paths for Poisson/FMM (ensure all buffers stay GPU-resident).
- Wire IF-HERK scratch fields to be GPU-resident end-to-end.
- Ensure interpolation/prolong/restrict kernels are fully GPU-resident across all call sites.
- Consider cleaning CUDA_ARCHITECTURES warnings by setting CMAKE_CUDA_ARCHITECTURES.
