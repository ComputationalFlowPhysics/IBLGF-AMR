# Convolution Unit Tests

This directory contains comprehensive unit tests for the IBLGF convolution classes used in FFT-based operations for the Lattice Green's Function solver.

## Components Tested

### 1. `dfft_r2c` (Real-to-Complex FFT)
- **Purpose**: Performs forward FFT transformations from real to complex domains
- **Dimensions**: Supports both 2D and 3D transforms
- **Key Features**:
  - Whole domain transforms (`execute_whole()`)
  - Stage-wise transforms (`execute()`) for efficient partial domain processing
  - Input/output buffer management
  - Data copying with size validation

### 2. `dfft_c2r` (Complex-to-Real FFT)
- **Purpose**: Performs inverse FFT transformations from complex to real domains
- **Dimensions**: Supports both 2D and 3D transforms
- **Key Features**:
  - Multi-stage inverse transforms
  - Efficient memory layout for partial domain reconstruction
  - Normalization handling

### 3. `Convolution<Dim>` (FFT-based Convolution)
- **Purpose**: High-performance convolution using FFT transforms
- **Template Parameter**: `Dim` - spatial dimension (2 or 3)
- **Key Features**:
  - SIMD-accelerated complex arithmetic
  - Forward/backward transform management
  - Kernel caching and reuse
  - Accumulation of multiple convolutions before inverse transform

## Test Coverage

### Constructor and Initialization Tests
- Validate proper memory allocation for different dimensions
- Test both 2D and 3D configurations
- Verify buffer sizes match expected values

### Data Management Tests
- `copy_input()` with valid data sizes
- Error handling for invalid input sizes
- Input/output accessor validation
- Data persistence across operations

### Transform Tests
- Forward transforms (R2C)
- Inverse transforms (C2R)
- Stage-wise vs. whole domain execution
- Round-trip transforms (FFT → IFFT should recover input)

### Convolution-Specific Tests
- SIMD complex multiplication with accumulation
- Helper functions (`helper_next_pow_2`, `helper_all_prod`)
- Field cleaning and state reset
- Forward/backward operation counting

### Edge Cases
- Small dimensions (2×2)
- Asymmetric dimensions (16×8×4)
- Zero initialization
- Delta function transforms

### Analytical Solution Validation
- **Cosine Wave FFT**: Validates that FFT of cos(2πkx/N) produces peaks at wavenumber ±k
- **Parseval's Theorem**: Verifies energy conservation between time and frequency domains
- **Gaussian Transform**: Tests that Gaussian function produces Gaussian-like spectrum
- **Linearity Property**: Confirms FFT(af + bg) = a·FFT(f) + b·FFT(g)
- **Convolution Theorem**: Validates that convolution in space equals multiplication in frequency domain

These tests ensure the FFT implementation produces mathematically correct results against known analytical solutions.

## Building and Running the Tests

### Build the tests:
```bash
./iblgf.sh build           # Configure and build all tests
./iblgf.sh build -j 8      # Build with 8 parallel jobs
```

### Run all project tests:
```bash
./iblgf.sh test            # Run entire test suite
./iblgf.sh test -j 4       # Run up to 4 tests in parallel
```

### Run only the convolution tests:
```bash
cd build
ctest -R convolution_test -V    # Run with verbose output
```

### Direct execution:
```bash
./build/tests/convolution/convolution_test.x         # Run directly
mpirun -n 4 ./build/tests/convolution/convolution_test.x  # Run with 4 MPI ranks
```

### Quick rebuild and test:
```bash
./iblgf.sh build && ./iblgf.sh test
```

## Test Framework

- **Framework**: Google Test (GTest)
- **Test Fixtures**: Separate fixtures for 2D and 3D cases
- **Assertions**: Uses `EXPECT_*` for non-fatal checks, `ASSERT_*` for fatal checks
- **Floating-point comparison**: Uses `EXPECT_NEAR` with appropriate tolerances (1e-10)

## Key Test Patterns

### 1. Delta Function Test
Tests that a delta function (impulse) in real space produces a constant spectrum in frequency space:
```cpp
input[0] = 1.0;  // Delta at origin
// After FFT, all frequency components should be approximately equal
```

### 2. Round-Trip Transform
Verifies that FFT followed by IFFT recovers the original signal (with proper normalization):
```cpp
original → FFT → IFFT → recovered
EXPECT_NEAR(original[i], recovered[i], tolerance)
```

### 3. SIMD Arithmetic Verification
Tests vectorized complex multiplication against known analytical results:
```cpp
(a + bi) × (c + di) = (ac - bd) + (ad + bc)i
```

## Dependencies

- FFTW3: Fast Fourier Transform library
- xsimd: SIMD vectorization library
- Boost: Aligned memory allocators
- GTest: Testing framework

## Notes

- All tests are designed to run without MPI (single-process unit tests)
- Tests use small problem sizes (8×8, 16×16) for fast execution
- FFTW plans are created with `FFTW_PATIENT` for optimal performance
- Memory is aligned for SIMD operations (32-byte alignment)

## Future Enhancements

Potential areas for additional testing:
1. Performance benchmarks for different problem sizes
2. MPI-parallel convolution tests
3. Memory leak detection with valgrind
4. Numerical accuracy tests against analytical solutions

## GPU Tests

GPU-accelerated convolution tests are available when CUDA support is enabled.

### Prerequisites

- CUDA toolkit installed
- CUDA-capable GPU (compute capability 3.5+)
- CMake configured with `-DUSE_GPU=ON`

### Building GPU Tests

```bash
# Configure with GPU support
cd build
cmake .. -DUSE_GPU=ON
./iblgf.sh build
```

### Running GPU Tests

```bash
# Run all tests (includes GPU if enabled)
./iblgf.sh test

# Run only GPU convolution tests
cd build
ctest -R convolution_gpu_test -V

# Direct execution
./build/tests/convolution/convolution_gpu_test.x
```

### GPU Test Coverage

The GPU tests (`convolution_gpu_test.cu`) validate:

- **dfft_r2c_gpu** class for forward FFT transforms (R2C operations)
  - Constructor and memory allocation
  - Input/output accessor functions
  - Execute transforms (whole domain)
  - Cosine wave transform validation
  
- **dfft_c2r_gpu** class for inverse FFT transforms (C2R operations)
  - Constructor and memory allocation
  - Input/output accessor functions
  - Execute inverse transforms
  - Round-trip transform validation (R2C → C2R)
  - Constant/DC component handling

- **Convolution_GPU<Dim>** class for FFT-based convolution
  - Constructor with various dimensions
  - Data transfers between host and device memory
  - Memory management across multiple transforms
  - Edge cases: zero input, minimal dimensions, asymmetric sizes
  
- **Analytical solutions** (GPU implementations)
  - Cosine wave FFT peaks at correct wavenumber
  - Gaussian transform with DC dominance
  - Parseval's theorem (energy conservation)
  - Linearity property validation

- **Device properties**: GPU capability and memory verification

Tests automatically skip if no CUDA-capable GPU is detected.

### GPU vs CPU Tests

| Feature | CPU Tests (`convolution_test.cpp`) | GPU Tests (`convolution_gpu_test.cu`) |
|---------|-----------------------------------|--------------------------------------|
| FFT Library | FFTW3 | cuFFT |
| Memory | Host (CPU RAM) | Device (GPU VRAM) |
| Classes | `dfft_r2c`, `dfft_c2r`, `Convolution<Dim>` | `dfft_r2c_gpu`, `Convolution_GPU<Dim>` |
| Dimensions | 2D and 3D | 3D only |
| SIMD | xsimd (CPU vectors) | CUDA kernels (GPU threads) |

Both test suites validate correctness and robustness of FFT-based convolution operations in their respective compute environments.

