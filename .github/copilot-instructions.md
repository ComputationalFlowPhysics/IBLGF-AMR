# IBLGF-AMR AI Coding Agent Instructions

## Project Overview

IBLGF (Immersed Boundary Lattice Green's Function) is a C++17 HPC flow solver for incompressible Navier-Stokes equations on unbounded domains using mimetic finite volume, lattice Green's functions, and adaptive mesh refinement (AMR).

**Key References:**
- Liska & Colonius (2016) - Fast lattice Green's function method
- Dorschner et al. (2020) - Multi-resolution lattice Green's function

## Core Architecture

### Template-Based Design Pattern

The codebase uses **CRTP (Curiously Recurring Template Pattern)** extensively. Each simulation is defined by:

1. **Setup Classes** (`setups/setup_*.hpp`) - Define problem-specific behavior:
   - `SetupBase<Setup, SetupTraits>` - Base template providing default fields
   - Derived setups: `setup_helmholtz.hpp`, `setup_linear.hpp`, `setup_Newton.hpp`
   - User setup inherits via CRTP: `struct NS_AMR_LGF : public SetupBase<NS_AMR_LGF, parameters>`

2. **Domain → Simulation → Setup hierarchy**:
   ```
   Domain<Dim, DataBlock> → Simulation<Domain> → Setup<Domain>
   ```

3. **Field Registration Macro** (`REGISTER_FIELDS`) defines compile-time data fields:
   - Cell-centered: `source_tmp`, `d_i`, `cell_aux`
   - Face-centered: `u_i`, `q_i`, `g_i` (velocity/flux fields)
   - Edge-centered: `stream_f` (2D/3D vorticity)

### Major Components

- **`include/iblgf/domain/`** - Octree AMR, data blocks, field storage
  - `Domain` manages octree, decomposition, iteration
  - `DataBlock` wraps fields on octree blocks with buffer regions
  - `octree/tree.hpp` - Adaptive octree with p4est-like structure

- **`include/iblgf/lgf/`** - Lattice Green's functions (LGF)
  - `lgf_ge.hpp` - Generates/caches kernel lookup tables
  - Used by Poisson solver for unbounded domains

- **`include/iblgf/solver/`** - Time integration & linear solvers
  - `poisson/` - LGF-based Poisson solver
  - `time_integration/` - IFHERK (implicit-explicit RK schemes)
  - `linsys/` - Linear system solvers (Schur complement, LGF application)

- **`include/iblgf/operators/`** - Stencil operators
  - `Operator` struct contains static methods: `levelDivergence`, `laplace`, `domainClean`
  - Operates on fields via template tags

- **`include/iblgf/fmm/`** - Fast Multipole Method for long-range interactions

- **`include/iblgf/dictionary/`** - Configuration file parser
  - Custom format: nested dictionaries with key=value pairs
  - Read via `Dictionary::get_or<T>(key, default_val)`

## Development Workflows

### Building & Testing (Use Helper Scripts)

**Primary workflow** - Use `./iblgf.sh` wrapper (NOT manual cmake):

```bash
./iblgf.sh build           # Configure and build (-j auto-detects cores)
./iblgf.sh build -j 8      # Build with 8 cores
./iblgf.sh test            # Build and run all tests
./iblgf.sh test -j 4       # Run up to 4 tests in parallel
```

**Running individual tests** with staged output directories:
```bash
./iblgf.sh run-test ns_amr_lgf configFile_0              # Default 2 MPI ranks
./iblgf.sh run-test ns_amr_lgf configFile_0 -n 4         # 4 MPI ranks
./iblgf.sh run-test ns_amr_lgf configFile_0 --resume     # Resume latest run
```

Outputs go to timestamped `runs/<test_name>/<timestamp>/` with metadata, logs, and HDF5 files.

**Environment variables** for defaults:
- `IBLGF_BUILD_JOBS=N` - Build parallelism
- `IBLGF_TEST_JOBS=N` - CTest parallelism
- `IBLGF_MPI_RANKS=N` - Default MPI ranks

### Docker Development Environment

**Preferred workflow** for reproducible builds:

```bash
./docker_iblgf.sh           # Launch interactive container
./docker_iblgf.sh -c 4      # Limit to 4 CPU cores
```

- Base image: `ccardina/my-app:cpu` (pre-built with all dependencies)
- Mounts repo at `/workspace2`
- Includes Python + numpy/scipy/matplotlib for post-processing

Inside Docker, use `./iblgf.sh` normally.

### Test Structure Convention

Each test under `tests/<test_name>/`:
- `<test_name>.cpp` - Main executable with `boost::mpi::environment`
- `<test_name>.hpp` - Problem-specific setup class
- `configs/` - Dictionary config files (e.g., `configFile_0`)
- `CMakeLists.txt` - Builds `<test_name>.x` executable

**Active tests** (uncommented in `tests/CMakeLists.txt`):
- `ns_amr_lgf`, `ns_amr_lgf2D` - Navier-Stokes with AMR
- `poisson`, `Poisson2D` - Poisson solver validation
- `operators` - Operator unit tests

Most tests are commented out; enable by uncommenting `add_subdirectory(...)`.

## Critical Conventions

### Configuration Files (Dictionary Format)

Example from `tests/ns_amr_lgf/configs/configFile_0`:
```
simulation_parameters
{
    nLevels=0;
    cfl = 0.35;
    Re = 1000.0;
    refinement_factor=0.125;
    adapt_frequency=10;
    output_frequency=4;
    
    output { directory=output; }
    
    restart
    {
        load_directory=restart;
        save_directory=restart;
    }
}

domain_parameters { ... }
```

Access in code: `simulation_.dictionary()->template get_or<float_type>("cfl", 0.2)`

### MPI and Parallelism

- **Always initialize MPI** via `boost::mpi::environment env(argc, argv)`
- Domain decomposition uses MPI for ghost cell communication
- Tests typically run with 2-16 MPI ranks (configurable via `-n` flag)
- CMake auto-detects `IBLGF_MPI_NP` from host CPUs

### Field Access Patterns

Fields accessed via **tag-based dispatch**:
```cpp
// Clean a field across all blocks
domain::Operator::domainClean<source_tmp_type>(domain_);

// Apply Laplacian: Source field → Dest field
domain::Operator::laplace<Source, Dest>(block, dx_level);

// Iterate over nodes in a block
for (auto& n : block) {
    n(dest_tag, field_idx) = n(source_tag, field_idx) + correction;
}
```

### Restart/Checkpointing

Enable in config file:
```
write_restart=true;
use_restart=true;
restart_write_frequency=20;
```

Writes `restart_field.hdf5` and `tree_info.bin`. Resume via `./iblgf.sh run-test <test> <config> --resume`.

## Common Pitfalls

1. **Don't manually run `cmake`** - Use `./iblgf.sh build` to ensure proper configuration
2. **Template compilation errors** - Check Setup→SetupTraits parameter pack matches
3. **Field buffer regions** - Operators expect ghost cells populated via `domain->update_ghosts()`
4. **Config file syntax** - No commas between entries, use `=` not `:`, nest with `{ }`
5. **HDF5 output paths** - Relative to run directory, not build directory

## External Dependencies

Required (install via package manager with MPI support):
- C++17 compiler (gcc ≥7, intel ≥2018)
- MPI (OpenMPI/MPICH)
- CMake ≥3.12
- Boost (system, filesystem, serialization, mpi)
- FFTW, HDF5, BLAS
- xtensor, xtensor-blas, xtl, xsimd

Optional:
- PETSc, SLEPc (for eigenvalue/stability tests)
- MKL, ScaLAPACK (for direct solvers)
- OpenMP (enable via `USE_OMP` in CMake)

## Adding New Tests

1. Create `tests/<new_test>/` directory
2. Add `<new_test>.cpp` with MPI initialization and Dictionary parsing
3. Create `<new_test>.hpp` with Setup class inheriting from `SetupBase`
4. Add `CMakeLists.txt` following pattern from `tests/ns_amr_lgf/`
5. Create `configs/configFile_0` with simulation parameters
6. Uncomment `add_subdirectory(<new_test>)` in `tests/CMakeLists.txt`
7. Build and run: `./iblgf.sh run-test <new_test> configFile_0`

## Key Files for Understanding

- `setups/setup_base.hpp` - Core setup architecture, field definitions
- `include/iblgf/domain/domain.hpp` - Domain management, AMR iteration
- `include/iblgf/operators/operators.hpp` - All stencil operations
- `tests/ns_amr_lgf/ns_amr_lgf.hpp` - Complete working example
- `iblgf.sh` - Build/test/run automation
- `docker_iblgf.sh` - Docker environment setup
