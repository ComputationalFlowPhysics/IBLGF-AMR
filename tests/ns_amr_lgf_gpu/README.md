# GPU-Enabled Navier-Stokes Solver

This directory contains a GPU-accelerated implementation of the Navier-Stokes solver using CUDA.

## Overview

The GPU solver uses the same algorithms as the CPU version but executes entirely on GPUs:
- Fast Lattice Green's Function (LGF) method for Poisson equation
- Adaptive Mesh Refinement (AMR) on octree structures
- Mimetic finite volume discretization
- Time integration using IF-HERK method

## Key Features

- **GPU Acceleration**: Uses CUDA for FFT operations (cuFFT) and convolution operations
- **Multi-GPU Support**: Automatically distributes work across GPUs based on MPI ranks
- **Same Algorithms**: Maintains algorithmic consistency with CPU version via `IBLGF_COMPILE_CUDA` flag

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

- **CUDA Streams**: Asynchronous execution for overlapping computation and communication
- **Batched FFTs**: Processes multiple FFTs in batches for better GPU utilization
- **Unified Memory**: Minimizes host-device data transfers
- **Device Selection**: Automatically assigns `rank % deviceCount` to balance GPU load

## Performance Notes

- GPU implementation is most effective for large problem sizes (>= 64^3 grids)
- Multiple GPUs provide best scaling for multi-level AMR problems
- Ensure sufficient GPU memory for domain decomposition and AMR hierarchy
