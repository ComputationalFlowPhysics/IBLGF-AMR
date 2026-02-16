# GPU-Enabled Navier-Stokes Solver

This directory contains a comprehensive GPU-accelerated implementation of the Navier-Stokes solver using CUDA.

## Overview

The GPU solver executes **all** computational operations on GPUs, not just FFTs:
- **Differential Operators**: Gradient, divergence, curl, Laplacian (CUDA kernels)
- **Field Operations**: AXPY, copy, scale, clean (CUDA kernels)
- **Nonlinear Terms**: Advection operators ∇×(u × ω) (CUDA kernels)
- **FFT Operations**: Forward/inverse transforms (cuFFT library)
- **LGF Convolutions**: Batched spectral convolution (cuFFT + custom kernels)
- **Time Integration**: IF-HERK 3-stage Runge-Kutta (orchestrated on GPU)

## Key Features

- **End-to-End GPU Execution**: Data stays on GPU throughout time stepping
- **Multi-GPU Support**: MPI-based domain decomposition with GPU per rank
- **Custom CUDA Kernels**: Optimized differential operators for staggered grids
- **Batched Operations**: FFTs and convolutions processed in batches
- **Minimal Host-Device Transfers**: Only for I/O and MPI halo exchange

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

- GPU implementation is most effective for large problem sizes (>= 64^3 grids)
- Multiple GPUs provide best scaling for multi-level AMR problems
- Ensure sufficient GPU memory for domain decomposition and AMR hierarchy
