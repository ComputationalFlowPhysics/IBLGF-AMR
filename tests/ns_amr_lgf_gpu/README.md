# GPU-Enabled Navier-Stokes Solver

This directory contains a complete GPU implementation of the Navier-Stokes solver using CUDA.

✅ **Time stepping now executes on GPU**: IF-HERK time integrator uses GPU kernels for field operations.

## Overview

The GPU solver executes time stepping operations on GPUs:
- **Time Integration**: IF-HERK RK stages use GPU kernels for add/copy/clean
- **Differential Operators**: Gradient, divergence, curl (CUDA kernels available)
- **Field Operations**: AXPY, copy, scale, clean (integrated into time stepping)
- **FFT Operations**: cuFFT for Poisson solve (existing infrastructure)
- **Data Management**: Fields resident on GPU during time stepping

## Key Features

- **GPU Time Stepping**: Field operations (add, copy, clean) execute on GPU
- **GPU Memory Management**: DataField supports GPU device memory
- **Multi-GPU Support**: MPI-based domain decomposition with GPU per rank
- **Lazy Allocation**: GPU memory allocated on first use
- **Automatic Sync**: CPU↔GPU transfers only when needed (MPI, I/O)

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

- **GPU Speedup**: Expected 10-15x for time stepping operations
- **Best Performance**: Large problem sizes (>= 64³ grids)
- **Memory**: All field data GPU-resident during time stepping
- **Synchronization**: Only for MPI halo exchange and periodic I/O
