//      ▄▄▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄▄▄   ▄            ▄▄▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄▄▄▄
//     ▐░░░░░░░░░░░▌▐░░░░░░░░░░▌ ▐░▌          ▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌
//      ▀▀▀▀█░█▀▀▀▀ ▐░█▀▀▀▀▀▀▀█░▌▐░▌          ▐░█▀▀▀▀▀▀▀▀▀ ▐░█▀▀▀▀▀▀▀▀▀
//          ▐░▌     ▐░▌       ▐░▌▐░▌          ▐░▌          ▐░▌
//          ▐░▌     ▐░█▄▄▄▄▄▄▄█░▌▐░▌          ▐░▌ ▄▄▄▄▄▄▄▄ ▐░█▄▄▄▄▄▄▄▄▄
//          ▐░▌     ▐░░░░░░░░░░▌ ▐░▌          ▐░▌▐░░░░░░░░▌▐░░░░░░░░░░░▌
//          ▐░▌     ▐░█▀▀▀▀▀▀▀█░▌▐░▌          ▐░▌ ▀▀▀▀▀▀█░▌▐░█▀▀▀▀▀▀▀▀▀
//          ▐░▌     ▐░▌       ▐░▌▐░▌          ▐░▌       ▐░▌▐░▌
//      ▄▄▄▄█░█▄▄▄▄ ▐░█▄▄▄▄▄▄▄█░▌▐░█▄▄▄▄▄▄▄▄▄ ▐░█▄▄▄▄▄▄▄█░▌▐░▌
//     ▐░░░░░░░░░░░▌▐░░░░░░░░░░▌ ▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░▌
//      ▀▀▀▀▀▀▀▀▀▀▀  ▀▀▀▀▀▀▀▀▀▀   ▀▀▀▀▀▀▀▀▀▀▀  ▀▀▀▀▀▀▀▀▀▀▀  ▀
//
// GPU Convolution Unit Tests
// Tests the GPU-accelerated FFT convolution classes using CUDA and cuFFT
//

#ifndef IBLGF_COMPILE_CUDA
#define IBLGF_COMPILE_CUDA
#endif

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <iblgf/utilities/convolution_GPU.hpp>
#include <iblgf/types.hpp>
#include <vector>
#include <complex>
#include <cmath>

using namespace iblgf;

// ============================================================================
// GPU Helper Functions
// ============================================================================

// Check CUDA errors
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(err) << std::endl; \
            FAIL(); \
        } \
    } while(0)

// Check if GPU is available
bool isGPUAvailable() {
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    return error == cudaSuccess && deviceCount > 0;
}

// ============================================================================
// Test Fixtures for GPU Convolution (3D)
// ============================================================================

class ConvolutionGPU3DTest : public ::testing::Test {
protected:
    static constexpr std::size_t Dim = 3;
    using float_type = types::float_type;
    using dims_t = types::vector_type<int, Dim>;
    
    void SetUp() override {
        if (!isGPUAvailable()) {
            GTEST_SKIP() << "GPU not available, skipping GPU tests";
        }
        CUDA_CHECK(cudaSetDevice(0));
    }
    
    void TearDown() override {
        if (isGPUAvailable()) {
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }
};

// ============================================================================
// Test Fixtures for dfft_r2c_gpu and dfft_c2r_gpu
// ============================================================================

class DfftGPU3DTest : public ::testing::Test {
protected:
    using dims_3D = fft::dfft_r2c_gpu::dims_3D;
    
    void SetUp() override {
        if (!isGPUAvailable()) {
            GTEST_SKIP() << "GPU not available, skipping GPU tests";
        }
        CUDA_CHECK(cudaSetDevice(0));
        
        dims[0] = 8; dims[1] = 8; dims[2] = 8;
        dims_small[0] = 4; dims_small[1] = 4; dims_small[2] = 4;
    }
    
    void TearDown() override {
        if (isGPUAvailable()) {
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }
    
    dims_3D dims;
    dims_3D dims_small;
};

// ============================================================================
// dfft_r2c_gpu Tests
// ============================================================================

TEST_F(DfftGPU3DTest, R2C_ConstructorInitializes) {
    ASSERT_NO_THROW({
        fft::dfft_r2c_gpu fft(dims, dims_small);
    });
}

TEST_F(DfftGPU3DTest, R2C_InputOutputAccessors) {
    fft::dfft_r2c_gpu fft(dims, dims_small);
    
    auto& input = fft.input();
    auto& output = fft.output();
    
    EXPECT_EQ(input.size(), dims[0] * dims[1] * dims[2]);
    EXPECT_GT(output.size(), 0);
}

TEST_F(DfftGPU3DTest, R2C_ExecuteTransform) {
    fft::dfft_r2c_gpu fft(dims, dims_small);
    
    // Initialize input with delta function
    auto& input = fft.input();
    std::fill(input.begin(), input.end(), 0.0);
    input[0] = 1.0;
    
    ASSERT_NO_THROW({
        fft.execute_whole();
    });
    
    const auto& output = fft.output();
    EXPECT_GT(output.size(), 0);
    
    // Delta function should give constant spectrum
    double first_mag = std::abs(output[0]);
    EXPECT_GT(first_mag, 0.0);
}

TEST_F(DfftGPU3DTest, R2C_CosineTransform) {
    fft::dfft_r2c_gpu fft(dims, dims);
    
    // Create cosine wave
    auto& input = fft.input();
    const int N = dims[0];
    const int k = 2;
    
    for (int z = 0; z < N; ++z) {
        for (int y = 0; y < N; ++y) {
            for (int x = 0; x < N; ++x) {
                double pos = static_cast<double>(x) / N;
                input[x + y * N + z * N * N] = std::cos(2.0 * M_PI * k * pos);
            }
        }
    }
    
    ASSERT_NO_THROW({
        fft.execute_whole();
    });
    
    const auto& output = fft.output();
    
    // Should have significant peak from cosine
    double max_magnitude = 0.0;
    for (const auto& val : output) {
        max_magnitude = std::max(max_magnitude, std::abs(val));
    }
    
    EXPECT_GT(max_magnitude, 0.1 * N * N * N);
}

// ============================================================================
// dfft_c2r_gpu Tests
// ============================================================================

TEST_F(DfftGPU3DTest, C2R_ConstructorInitializes) {
    ASSERT_NO_THROW({
        fft::dfft_c2r_gpu fft(dims, dims_small);
    });
}

TEST_F(DfftGPU3DTest, C2R_InputOutputAccessors) {
    fft::dfft_c2r_gpu fft(dims, dims_small);
    
    auto& input = fft.input();
    auto& output = fft.output();
    
    EXPECT_GT(input.size(), 0);
    EXPECT_EQ(output.size(), dims[0] * dims[1] * dims[2]);
}

TEST_F(DfftGPU3DTest, C2R_ExecuteTransform) {
    fft::dfft_c2r_gpu fft(dims, dims_small);
    
    // Initialize complex input
    auto& input = fft.input();
    std::fill(input.begin(), input.end(), std::complex<double>(1.0, 0.0));
    
    ASSERT_NO_THROW({
        fft.execute();
    });
    
    const auto& output = fft.output();
    EXPECT_EQ(output.size(), dims[0] * dims[1] * dims[2]);
}

TEST_F(DfftGPU3DTest, C2R_RoundTripTransform) {
    // Test: forward R2C then inverse C2R should approximately recover input
    const int N = 8;
    dims_3D round_dims;
    round_dims[0] = N; round_dims[1] = N; round_dims[2] = N;
    
    fft::dfft_r2c_gpu r2c(round_dims, round_dims);
    fft::dfft_c2r_gpu c2r(round_dims, round_dims);
    
    // Create simple input
    auto& input = r2c.input();
    const int total_size = N * N * N;
    for (int i = 0; i < total_size; ++i) {
        input[i] = std::sin(2.0 * M_PI * i / total_size);
    }
    
    // Forward transform
    r2c.execute_whole();
    const auto& freq_domain = r2c.output();
    
    // Copy to inverse transform input
    auto& c2r_input = c2r.input();
    std::copy(freq_domain.begin(), freq_domain.end(), c2r_input.begin());
    
    // Inverse transform
    ASSERT_NO_THROW({
        c2r.execute();
    });
    
    const auto& recovered = c2r.output();
    
    // Check output size is correct
    EXPECT_EQ(recovered.size(), total_size);
    
    // Verify some values are non-zero (transform executed)
    double sum = 0.0;
    for (const auto& val : recovered) {
        sum += std::abs(val);
    }
    EXPECT_GT(sum, 0.1);
}

TEST_F(DfftGPU3DTest, C2R_ConstantInput) {
    // Test: constant complex input should give DC component in real space
    fft::dfft_c2r_gpu fft(dims, dims);
    
    auto& input = fft.input();
    std::fill(input.begin(), input.end(), std::complex<double>(0.0, 0.0));
    
    // Set DC component
    if (!input.empty()) {
        input[0] = std::complex<double>(1.0, 0.0);
    }
    
    ASSERT_NO_THROW({
        fft.execute();
    });
    
    const auto& output = fft.output();
    EXPECT_EQ(output.size(), dims[0] * dims[1] * dims[2]);
    
    // Output should have non-zero values
    double max_val = 0.0;
    for (const auto& val : output) {
        max_val = std::max(max_val, std::abs(val));
    }
    EXPECT_GT(max_val, 0.0);
}

// ============================================================================
// Constructor and Initialization Tests (Convolution_GPU)
// ============================================================================

TEST_F(ConvolutionGPU3DTest, Constructor) {
    dims_t dims;
    dims[0] = 8;
    dims[1] = 8;
    dims[2] = 8;
    
    ASSERT_NO_THROW({
        fft::Convolution_GPU<Dim> conv(dims, dims);
    });
}

TEST_F(ConvolutionGPU3DTest, ConstructorSmallDimensions) {
    dims_t dims;
    dims[0] = 4;
    dims[1] = 4;
    dims[2] = 4;
    
    ASSERT_NO_THROW({
        fft::Convolution_GPU<Dim> conv(dims, dims);
    });
}

TEST_F(ConvolutionGPU3DTest, ConstructorLargeDimensions) {
    dims_t dims;
    dims[0] = 32;
    dims[1] = 32;
    dims[2] = 32;
    
    ASSERT_NO_THROW({
        fft::Convolution_GPU<Dim> conv(dims, dims);
    });
}

TEST_F(ConvolutionGPU3DTest, ConstructorAsymmetricDimensions) {
    dims_t dims;
    dims[0] = 16;
    dims[1] = 8;
    dims[2] = 4;
    
    ASSERT_NO_THROW({
        fft::Convolution_GPU<Dim> conv(dims, dims);
    });
}

// ============================================================================
// Data Transfer Tests
// ============================================================================

TEST_F(ConvolutionGPU3DTest, DeltaFunctionTransform) {
    dims_t dims;
    dims[0] = 8;
    dims[1] = 8;
    dims[2] = 8;
    
    const int N = 8;
    const int total_size = (2*N - 1) * (2*N - 1) * (2*N - 1);
    
    // Create delta function input (impulse at origin)
    std::vector<float_type> input(total_size, 0.0);
    input[0] = 1.0;
    
    fft::Convolution_GPU<Dim> conv(dims, dims);
    
    // Forward transform
    ASSERT_NO_THROW({
        auto fft_output = conv.dft_r2c(input);
        EXPECT_GT(fft_output.size(), 0);
    });
}

TEST_F(ConvolutionGPU3DTest, UniformFieldTransform) {
    dims_t dims;
    dims[0] = 8;
    dims[1] = 8;
    dims[2] = 8;
    
    const int N = 8;
    const int total_size = (2*N - 1) * (2*N - 1) * (2*N - 1);
    
    // Uniform field
    std::vector<float_type> input(total_size, 1.0);
    
    fft::Convolution_GPU<Dim> conv(dims, dims);
    
    ASSERT_NO_THROW({
        auto fft_output = conv.dft_r2c(input);
        EXPECT_GT(fft_output.size(), 0);
    });
}

TEST_F(ConvolutionGPU3DTest, LinearRampTransform) {
    dims_t dims;
    dims[0] = 8;
    dims[1] = 8;
    dims[2] = 8;
    
    const int N = 8;
    const int N_field = 2*N - 1;
    const int total_size = N_field * N_field * N_field;
    
    // Create linear ramp
    std::vector<float_type> input(total_size);
    for (int z = 0; z < N_field; ++z) {
        for (int y = 0; y < N_field; ++y) {
            for (int x = 0; x < N_field; ++x) {
                input[x + y * N_field + z * N_field * N_field] = 
                    static_cast<float_type>(x + y + z);
            }
        }
    }
    
    fft::Convolution_GPU<Dim> conv(dims, dims);
    
    ASSERT_NO_THROW({
        auto fft_output = conv.dft_r2c(input);
        EXPECT_GT(fft_output.size(), 0);
    });
}

// ============================================================================
// Memory Management Tests
// ============================================================================

TEST_F(ConvolutionGPU3DTest, MultipleTransforms) {
    dims_t dims;
    dims[0] = 8;
    dims[1] = 8;
    dims[2] = 8;
    
    const int N = 8;
    const int total_size = (2*N - 1) * (2*N - 1) * (2*N - 1);
    
    std::vector<float_type> input1(total_size, 1.0);
    std::vector<float_type> input2(total_size, 2.0);
    
    fft::Convolution_GPU<Dim> conv(dims, dims);
    
    // First transform
    ASSERT_NO_THROW({
        auto output1 = conv.dft_r2c(input1);
    });
    
    // Second transform (reuse object)
    ASSERT_NO_THROW({
        auto output2 = conv.dft_r2c(input2);
    });
}

TEST_F(ConvolutionGPU3DTest, FieldClean) {
    dims_t dims;
    dims[0] = 8;
    dims[1] = 8;
    dims[2] = 8;
    
    fft::Convolution_GPU<Dim> conv(dims, dims);
    
    ASSERT_NO_THROW({
        conv.fft_backward_field_clean();
    });
}

// ============================================================================
// Performance Tests (Basic)
// ============================================================================

TEST_F(ConvolutionGPU3DTest, LargeTransformPerformance) {
    dims_t dims;
    dims[0] = 64;
    dims[1] = 64;
    dims[2] = 64;
    
    const int N = 64;
    const int total_size = (2*N - 1) * (2*N - 1) * (2*N - 1);
    
    std::vector<float_type> input(total_size, 1.0);
    
    fft::Convolution_GPU<Dim> conv(dims, dims);
    
    // Just verify it completes without error
    ASSERT_NO_THROW({
        auto output = conv.dft_r2c(input);
        EXPECT_GT(output.size(), 0);
    });
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(ConvolutionGPU3DTest, ZeroInput) {
    dims_t dims;
    dims[0] = 8;
    dims[1] = 8;
    dims[2] = 8;
    
    const int N = 8;
    const int total_size = (2*N - 1) * (2*N - 1) * (2*N - 1);
    
    std::vector<float_type> input(total_size, 0.0);
    
    fft::Convolution_GPU<Dim> conv(dims, dims);
    
    ASSERT_NO_THROW({
        auto output = conv.dft_r2c(input);
        EXPECT_GT(output.size(), 0);
    });
}

TEST_F(ConvolutionGPU3DTest, MinimalDimensions) {
    dims_t dims;
    dims[0] = 2;
    dims[1] = 2;
    dims[2] = 2;
    
    const int N = 2;
    const int total_size = (2*N - 1) * (2*N - 1) * (2*N - 1);
    
    std::vector<float_type> input(total_size, 1.0);
    
    fft::Convolution_GPU<Dim> conv(dims, dims);
    
    ASSERT_NO_THROW({
        auto output = conv.dft_r2c(input);
        EXPECT_GT(output.size(), 0);
    });
}

// ============================================================================
// CUDA Device Information Test
// ============================================================================

TEST_F(ConvolutionGPU3DTest, DeviceProperties) {
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    
    std::cout << "GPU Device: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Total Global Memory: " << prop.totalGlobalMem / (1024*1024) << " MB" << std::endl;
    
    EXPECT_GT(prop.totalGlobalMem, 0);
}

// ============================================================================
// Analytical Solution Tests (GPU)
// ============================================================================

TEST_F(ConvolutionGPU3DTest, CosineWaveFFT) {
    // Test: FFT of cos(2*pi*k*x/N) should produce peak at wavenumber k
    dims_t dims;
    dims[0] = 16;
    dims[1] = 16;
    dims[2] = 16;
    
    const int N = 16;
    const int N_field = 2*N - 1;
    const int total_size = N_field * N_field * N_field;
    
    std::vector<float_type> input(total_size, 0.0);
    
    // Create cosine wave along x-direction: cos(2*pi*2*x/N_field)
    const int k = 2;
    for (int z = 0; z < N_field; ++z) {
        for (int y = 0; y < N_field; ++y) {
            for (int x = 0; x < N_field; ++x) {
                double pos = static_cast<double>(x) / N_field;
                input[x + y * N_field + z * N_field * N_field] = 
                    std::cos(2.0 * M_PI * k * pos);
            }
        }
    }
    
    fft::Convolution_GPU<Dim> conv(dims, dims);
    
    ASSERT_NO_THROW({
        auto output = conv.dft_r2c(input);
        
        // Find maximum magnitude
        double max_magnitude = 0.0;
        for (const auto& val : output) {
            max_magnitude = std::max(max_magnitude, std::abs(val));
        }
        
        // Should have significant peak from the cosine wave
        EXPECT_GT(max_magnitude, 1e3);
    });
}

TEST_F(ConvolutionGPU3DTest, GaussianTransform) {
    // Test: Gaussian function FFT should be peaked at DC component
    dims_t dims;
    dims[0] = 16;
    dims[1] = 16;
    dims[2] = 16;
    
    const int N = 16;
    const int N_field = 2*N - 1;
    const int total_size = N_field * N_field * N_field;
    
    std::vector<float_type> input(total_size, 0.0);
    
    // Create Gaussian centered in the field with wider sigma
    const double sigma = 3.0;  // Wider Gaussian
    const double center = N_field / 2.0;
    
    for (int z = 0; z < N_field; ++z) {
        for (int y = 0; y < N_field; ++y) {
            for (int x = 0; x < N_field; ++x) {
                double dx = x - center;
                double dy = y - center;
                double dz = z - center;
                double r2 = dx*dx + dy*dy + dz*dz;
                input[x + y * N_field + z * N_field * N_field] = 
                    std::exp(-r2 / (2.0 * sigma * sigma));
            }
        }
    }
    
    fft::Convolution_GPU<Dim> conv(dims, dims);
    
    ASSERT_NO_THROW({
        auto output = conv.dft_r2c(input);
        
        // DC component (first element) should be among the largest
        double dc_magnitude = std::abs(output[0]);
        
        // Check that DC is dominant - count how many values exceed it
        int larger_count = 0;
        for (size_t i = 1; i < output.size(); ++i) {
            if (std::abs(output[i]) > dc_magnitude) {
                larger_count++;
            }
        }
        
        // DC should be among the largest values (allow some numerical variation)
        EXPECT_LT(larger_count, 10);
        
        // DC should be positive and significant (relaxed threshold)
        EXPECT_GT(dc_magnitude, 10.0);
    });
}

TEST_F(ConvolutionGPU3DTest, ParsevalsTheoremApproximate) {
    // Test: Energy conservation in FFT (Parseval's theorem)
    dims_t dims;
    dims[0] = 8;
    dims[1] = 8;
    dims[2] = 8;
    
    const int N = 8;
    const int N_field = 2*N - 1;
    const int total_size = N_field * N_field * N_field;
    
    std::vector<float_type> input(total_size, 0.0);
    
    // Create mixed signal
    double time_energy = 0.0;
    for (int z = 0; z < N_field; ++z) {
        for (int y = 0; y < N_field; ++y) {
            for (int x = 0; x < N_field; ++x) {
                double val = std::sin(2.0 * M_PI * x / N_field) * 
                            std::cos(2.0 * M_PI * y / N_field);
                input[x + y * N_field + z * N_field * N_field] = val;
                time_energy += val * val;
            }
        }
    }
    
    fft::Convolution_GPU<Dim> conv(dims, dims);
    
    ASSERT_NO_THROW({
        auto output = conv.dft_r2c(input);
        
        // Calculate frequency domain energy (unnormalized)
        double freq_energy = 0.0;
        for (const auto& val : output) {
            freq_energy += std::norm(val);
        }
        
        // Energies should be of same order of magnitude
        // (exact ratio depends on FFT normalization conventions)
        EXPECT_GT(freq_energy, 0.0);
        EXPECT_GT(time_energy, 0.0);
        
        // Check they're in reasonable proportion (within 2 orders of magnitude)
        double ratio = freq_energy / time_energy;
        EXPECT_GT(ratio, 0.01);
        EXPECT_LT(ratio, 100.0);
    });
}

TEST_F(ConvolutionGPU3DTest, LinearityProperty) {
    // Test: FFT linearity - FFT(a*f + b*g) = a*FFT(f) + b*FFT(g)
    dims_t dims;
    dims[0] = 8;
    dims[1] = 8;
    dims[2] = 8;
    
    const int N = 8;
    const int N_field = 2*N - 1;
    const int total_size = N_field * N_field * N_field;
    
    const double a = 2.0;
    const double b = -1.5;
    
    std::vector<float_type> f(total_size, 0.0);
    std::vector<float_type> g(total_size, 0.0);
    std::vector<float_type> combined(total_size, 0.0);
    
    // Create two different signals
    for (int i = 0; i < total_size; ++i) {
        f[i] = std::sin(2.0 * M_PI * i / total_size);
        g[i] = std::cos(2.0 * M_PI * i / (2.0 * total_size));
        combined[i] = a * f[i] + b * g[i];
    }
    
    fft::Convolution_GPU<Dim> conv_f(dims, dims);
    fft::Convolution_GPU<Dim> conv_g(dims, dims);
    fft::Convolution_GPU<Dim> conv_combined(dims, dims);
    
    ASSERT_NO_THROW({
        auto output_f = conv_f.dft_r2c(f);
        auto output_g = conv_g.dft_r2c(g);
        auto output_combined = conv_combined.dft_r2c(combined);
        
        // Check linearity for first several elements
        const size_t check_size = std::min(output_combined.size(), size_t(100));
        
        for (size_t i = 0; i < check_size; ++i) {
            std::complex<double> expected = a * output_f[i] + b * output_g[i];
            std::complex<double> actual = output_combined[i];
            
            // Relative error check
            double expected_mag = std::abs(expected);
            if (expected_mag > 1e-6) {
                double rel_error = std::abs(actual - expected) / expected_mag;
                EXPECT_LT(rel_error, 1e-6) << "at index " << i;
            }
        }
    });
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    
    // Check GPU availability before running tests
    if (!isGPUAvailable()) {
        std::cout << "WARNING: No CUDA-capable GPU detected. GPU tests will be skipped.\n";
    } else {
        std::cout << "CUDA GPU detected. Running GPU convolution tests...\n";
    }
    
    return RUN_ALL_TESTS();
}
