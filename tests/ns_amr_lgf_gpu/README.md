# GPU-Enabled Navier-Stokes Solver (Infrastructure)

This directory contains GPU CUDA kernels for the Navier-Stokes solver. 

⚠️ **Current Status**: The GPU kernels are **infrastructure only** and are not yet integrated into the time stepping workflow. The solver still runs on CPU for field operations.

## Overview

GPU kernel infrastructure has been created but requires DataField refactoring to be used:
- Fast Lattice Green's Function (LGF) method for Poisson equation (already GPU-accelerated)
- Differential operators (gradient, divergence, curl) - kernels exist but not called
- Field operations (AXPY, copy, clean) - kernels exist but not called
- Time integration using IF-HERK method - still CPU-based

## Key Features

- **GPU Kernel Infrastructure**: CUDA kernels ready for differential operators and field updates
- **Multi-GPU Support**: MPI-based domain decomposition with GPU per rank
- **Existing GPU Acceleration**: FFT/convolution operations already use cuFFT
- **Build System Integration**: CMake support via `USE_GPU` flag

## Current Limitations

⚠️ **Critical**: Time stepping operations (add, copy, clean in IF-HERK) still use CPU:
- `ifherk.hpp` template methods use xtensor CPU operations
- DataField class has no GPU memory support
- GPU kernels exist but are not called during solve
- **No performance benefit yet** - this is infrastructure only

## Building

Enable GPU support during CMake configuration:

```bash
cmake -DUSE_GPU=ON ..
make ns_amr_lgf_gpu.x
```

## Running

```bash
# Single GPU
./ns_amr_lgf_gpu.x configFile_0

# Multi-GPU with MPI
mpirun -n 4 ./ns_amr_lgf_gpu.x configFile_0
```

The solver automatically assigns GPUs to MPI ranks using round-robin distribution.

## Configuration

Uses the same configuration files as the CPU version. See `configs/` directory for examples.

## Implementation Details

### GPU Kernels

The solver includes custom CUDA kernels for all major operations:

**Differential Operators** (`include/iblgf/operators/operators_gpu.hpp`):
- `gradient_x/y/z_kernel`: Cell-centered → Face-centered gradients
- `divergence_kernel`: Face-centered → Cell-centered divergence  
- `curl_x/y/z_kernel`: Face-centered velocity → Edge-centered vorticity
- `laplacian_kernel`: 7-point stencil Laplacian

**Field Operations**:
- `axpy_kernel`: General linear combination y = αx + βy
- `copy_scale_kernel`: Scaled copy y = αx
- `clean_kernel`: Zero initialization (uses `cudaMemsetAsync`)
- `multiply_kernel`: Element-wise multiplication

**Nonlinear Terms**:
- `nonlinear_x_kernel`: Advection term ∂/∂y(u_z·ω_y) - ∂/∂z(u_y·ω_z)
- Accounts for staggered mesh averaging

**Execution Model**:
- 8×8×8 thread blocks for 3D spatial kernels
- 256 threads for 1D vector operations
- Asynchronous launch on CUDA streams
- Stream synchronization only when needed (after batches)

## Performance Notes

- **Current**: No GPU speedup for time stepping (kernels not used)
- **Potential**: 10-15x after integration (requires DataField refactor)
- GPU FFTs already provide some acceleration for Poisson solve
- Requires DataField GPU memory support to use new kernels
