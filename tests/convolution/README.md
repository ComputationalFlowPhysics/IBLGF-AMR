# Convolution Unit Tests

This directory contains comprehensive unit tests for the IBLGF convolution classes used in FFT-based operations for the Lattice Green's Function solver, including both CPU and GPU implementations.

## Components Tested

### CPU Implementation

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

### GPU Implementation (CUDA)

### 4. `dfft_r2c_gpu` (GPU Real-to-Complex FFT)
- **Purpose**: GPU-accelerated forward FFT using cuFFT
- **Key Features**:
  - CUDA stream-based asynchronous execution
  - Pinned host memory for efficient HtoD transfers
  - cuFFT plan management
  - Template instantiation for `std::vector<double>`

### 5. `dfft_r2c_gpu_batch` (Batched GPU R2C FFT)
- **Purpose**: Process multiple FFTs in parallel on GPU
- **Key Features**:
  - Batch processing for multiple fields
  - GPU-direct memory copy using `cudaMemcpy3D`
  - Separate streams for computation and data transfer
  - Per-batch f0 pointer management for LGF kernels
  - Element-wise complex multiplication kernels
  - Batch accumulation on device

### 6. `dfft_c2r_gpu` (GPU Complex-to-Real FFT)
- **Purpose**: GPU-accelerated inverse FFT using cuFFT
- **Key Features**:
  - Device-only execution (`execute_device()`)
  - Explicit host-device data transfer control
  - Asynchronous copy with stream synchronization
  - Direct access to device pointers for chaining operations

### 7. `Convolution_GPU<Dim>` (GPU FFT-based Convolution)
- **Purpose**: Complete GPU-accelerated convolution workflow
- **Key Features**:
  - Batch accumulation before backward transform
  - Unified memory for LGF pointer arrays
  - Custom CUDA kernels (`prod_complex_add_ptr`, `sum_batches`, `scale_complex`)
  - Automatic batch flushing at capacity
  - Device-side scaling for normalization

## Test Coverage

### CPU Tests (convolution_test.cpp)

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

### GPU Tests (convolution_gpu_test.cpp)

### Device Availability Tests
- Check CUDA device presence before running GPU tests
- Automatically skip tests if no GPU available
- Clean device reset after each test fixture

### dfft_r2c_gpu Tests
- Constructor initialization with 3D dimensions
- Input/output buffer size validation
- `copy_input()` from `std::vector<double>` with layout verification
- `execute_whole()` and `execute()` transform correctness
- Delta function transform (should yield constant spectrum)

### dfft_r2c_gpu_batch Tests
- Batch size configuration and memory allocation
- Stream accessor validation (computation + transfer streams)
- F0 pointer/size management for LGF kernel references
- Input/output device buffer validity

### dfft_c2r_gpu Tests
- Constructor and buffer initialization
- Device pointer accessors (`output_cu_ptr()`)
- `execute_device()` for device-only transforms
- `copy_output_to_host()` with stream synchronization
- Round-trip transform accuracy (forward + backward)

### Convolution_GPU<3> Tests
- Helper functions (`helper_next_pow_2`, `helper_all_prod`)
- `fft_backward_field_clean()` counter reset
- `dft_r2c()` and `dft_r2c_size()` correctness
- Batch management (current/max batch size tracking)
- Empty batch flush safety

### CUDA Kernel Tests
- `scale_complex` kernel with arbitrary scale factors
- `sum_batches` kernel for multi-batch accumulation
- Verification of complex arithmetic on device

### Integration Tests
- End-to-end convolution workflow with simple data
- Spectrum computation and size validation

### Error Handling Tests
- Large dimension construction (memory stress test)
- Zero batch size handling
- Performance counter tracking (`fft_count_`, `number_fwrd_executed`)

### GPU FFT Tests (convolution_fft_gpu_test.cpp)

### dfft_r2c_gpu Basic Tests
- Constructor with 3D dimensions
- Input/output buffer size validation
- `copy_input()` with valid data
- `execute_whole()` and `execute()` transforms
- Device pointer validity (`output_cu()`)
- Delta function transform (constant spectrum)
- Constant input transform (DC component dominance)

### dfft_r2c_gpu_batch Tests
- Batch configuration and memory sizing
- Stream access (computation and transfer streams)
- Device pointer validation (input_cu, output_cu, result_cu)
- F0 pointer management vectors

### dfft_c2r_gpu Tests
- Constructor and buffer initialization
- Input/output size validation
- `execute()` and `execute_device()` transforms
- Device pointer access (`output_cu_ptr()`)
- `copy_output_to_host()` with synchronization
- Stream accessor

### Round-Trip GPU FFT Tests
- Forward and backward transform execution
- Round-trip accuracy (input → R2C → C2R → recovered)
- Delta function preservation through round-trip
- Scaling factor validation

### Analytical Solution GPU Tests
- **Cosine Wave FFT**: GPU validation of wavenumber peaks
- **Parseval's Theorem**: Energy conservation in GPU transforms
- **Gaussian Transform**: GPU Gaussian function produces Gaussian spectrum
- **Linearity Property**: GPU FFT(αf + βg) = α·FFT(f) + β·FFT(g)

### GPU Edge Cases and Performance
- Large transform sizes (64³)
- Asymmetric dimensions (16×8×4)
- Zero input handling
- Multiple sequential transforms

### GPU LGF Tests (convolution_lgf_gpu_test.cpp)

### GPU Point Source Tests
- Delta at origin on GPU
- Delta at arbitrary location on GPU
- Round-trip GPU convolution with sinusoidal source

### GPU Accuracy Tests
- GPU convolution vs. direct LGF evaluation at origin
- GPU accuracy with offset point source
- L2 and L∞ error metrics

### GPU Superposition Tests
- Linear superposition with multiple sources
- Uniform source field on GPU
- Zero source field handling
- Scaling property validation (α·f → α·result)

### GPU Batch Processing Tests
- Multiple forward transforms before backward (batch accumulation)
- Batch counter tracking
- Field cleaning and reset

### GPU Performance Tests
- Field cleaning resets GPU counters and batch state
- Larger problem sizes (32³)
- Memory management validation



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

## LGF Convolution Tests

The `convolution_lgf_test.cpp` tests the FFT-based convolution applied to the **Lattice Green's Function (LGF)** kernel for solving the Poisson equation. These tests validate the complete PDE solving pipeline.

### Test Coverage (3D and 2D)

#### 3D Tests
- **PointSourceAtOrigin**: Delta function at (0,0,0) produces LGF kernel
- **PointSourceNotAtOrigin**: Delta function at arbitrary location
- **RoundTripConvolution**: Forward convolution + backward transform
- **AccuracyPointSourceAtOrigin**: Compare with direct LGF evaluation
- **AccuracyOffsetPointSource**: Validate shifted kernel accuracy
- **LinearSuperposition**: Multiple point sources with different magnitudes
- **UniformSource**: Test with constant source field
- **ZeroSource**: Zero source field handling
- **ScalingProperty**: Verify α·f produces α·(result) relationship

#### 2D Tests (4 tests)
- **PointSourceAtOrigin2D**: 2D delta function at origin
- **RoundTripConvolution2D**: Forward/backward transforms in 2D
- **AccuracyPointSource2D**: 2D LGF accuracy validation
- **LinearSuperposition2D**: 2D superposition with multiple sources

### Physical Significance

These tests validate that:
1. **LGF kernel is correctly applied** via FFT convolution
2. **Poisson equation approximation** works: ∇²u = -f in unbounded domain
3. **Superposition principle** holds (linearity)
4. **Scaling** produces proportional results
5. **Numerical accuracy** matches analytical LGF values
6. **2D and 3D domains** work consistently

The tests use point sources (delta functions) where the analytical solution is known: u(r) = -1/(4πr) for 3D and u(r) = -ln(r)/(2π) for 2D.

## Building and Running the Tests

### Prerequisites for GPU Tests

GPU tests require:
- NVIDIA GPU with CUDA Compute Capability ≥ 3.5
- CUDA Toolkit (≥ 11.0 recommended)
- cuFFT library (included with CUDA Toolkit)

Enable GPU support during build:
```bash
cmake -DUSE_GPU=ON ..     # Enable GPU support in CMake configuration
```

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
ctest -R convolution -V              # Run all convolution tests (CPU + GPU + LGF)
ctest -R convolution_test -V         # Run basic CPU FFT tests
ctest -R convolution_fft_test -V     # Run CPU FFT analytical tests
ctest -R convolution_lgf_test -V     # Run CPU LGF convolution tests
ctest -R convolution_gpu_test -V     # Run GPU basic tests (requires USE_GPU=ON)
ctest -R convolution_fft_gpu_test -V # Run GPU FFT analytical tests (requires USE_GPU=ON)
ctest -R convolution_lgf_gpu_test -V # Run GPU LGF tests (requires USE_GPU=ON)
```

### Direct execution:
```bash
# CPU tests
./build/tests/convolution/convolution_test.x              # Basic CPU FFT tests
./build/tests/convolution/convolution_fft_test.x          # CPU FFT analytical tests
./build/tests/convolution/convolution_lgf_test.x          # CPU LGF convolution tests

# GPU tests (requires USE_GPU=ON)
./build/tests/convolution/convolution_gpu_test.x          # Basic GPU tests
./build/tests/convolution/convolution_fft_gpu_test.x      # GPU FFT analytical tests
./build/tests/convolution/convolution_lgf_gpu_test.x      # GPU LGF tests

# MPI execution (for LGF tests if needed)
mpirun -n 4 ./build/tests/convolution/convolution_lgf_test.x
```

### Quick rebuild and test:
```bash
./iblgf.sh build && ./iblgf.sh test
```

### GPU-specific notes:
- GPU tests automatically skip if no CUDA device is detected
- Tests perform `cudaDeviceReset()` after each fixture to ensure clean state
- If GPU tests fail, check `nvidia-smi` to verify GPU availability
- Use `CUDA_VISIBLE_DEVICES=0` to select specific GPU


## Test Framework

- **Framework**: Google Test (GTest)
- **Test Fixtures**: Separate fixtures for 2D/3D and CPU/GPU cases
- **Assertions**: Uses `EXPECT_*` for non-fatal checks, `ASSERT_*` for fatal checks
- **Floating-point comparison**: Uses `EXPECT_NEAR` with appropriate tolerances (1e-10)
- **GPU test skipping**: Uses `GTEST_SKIP()` when CUDA device unavailable
- **Resource cleanup**: Automatic cleanup via test fixture destructors and `cudaDeviceReset()`

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

### 4. GPU Device Check (GPU tests only)
Ensures CUDA device is available before running GPU tests:
```cpp
if (!cuda_device_available()) {
    GTEST_SKIP() << "CUDA device not available, skipping GPU tests";
}
```

### 5. Kernel Correctness Validation (GPU tests only)
Tests custom CUDA kernels with known inputs/outputs:
```cpp
// Launch kernel
kernel<<<blocks, threads>>>(d_in, d_out, size);
cudaDeviceSynchronize();
// Copy back and compare with expected results
EXPECT_NEAR(h_result[i], expected[i], 1e-10);
```

## Dependencies

### CPU Tests
- FFTW3: Fast Fourier Transform library
- xsimd: SIMD vectorization library
- Boost: Aligned memory allocators
- GTest: Testing framework

### GPU Tests (additional)
- CUDA Toolkit (≥ 11.0)
- cuFFT: GPU-accelerated FFT library
- CUDA Runtime API

## Notes

- CPU tests are designed to run without MPI (single-process unit tests)
- GPU tests automatically skip if `USE_GPU=OFF` or no CUDA device detected
- Tests use small problem sizes (8×8, 16×16) for fast execution
- FFTW plans are created with `FFTW_PATIENT` for optimal performance
- GPU tests use `cudaDeviceReset()` to ensure clean state between fixtures
- Memory is aligned for SIMD operations (32-byte alignment)
- GPU batched operations use unified memory for pointer arrays

## Future Enhancements

Potential areas for additional testing:
1. Performance benchmarks for different problem sizes (CPU vs GPU)
2. Multi-GPU support tests
3. Memory bandwidth utilization metrics
4. Overlap efficiency tests (compute + transfer streams)
5. Large-scale stress tests (GPU memory limits)
6. Accuracy comparison between CPU and GPU implementations

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
  
- **GPU LGF Convolution** (5 new tests)
  - Point source at origin (LGF-like behavior)
  - Offset point source convolution
  - Multiple point sources with superposition
  - Round-trip R2C → C2R with distributed source
  - Uniform source field handling
  
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

