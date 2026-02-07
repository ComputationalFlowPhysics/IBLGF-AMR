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
// GPU FFT Unit Tests
// Tests GPU-accelerated FFT operations for real-to-complex and complex-to-real transforms
//
#ifndef IBLGF_COMPILE_CUDA
#define IBLGF_COMPILE_CUDA
#endif
#include <gtest/gtest.h>
#include <iblgf/utilities/convolution_GPU.hpp>
#include <iblgf/types.hpp>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <cuda_runtime.h>

using namespace iblgf::fft;
using namespace iblgf;

// ============================================================================
// Helper Functions
// ============================================================================

bool cuda_device_available() {
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    return error == cudaSuccess && deviceCount > 0;
}

// ============================================================================
// dfft_r2c_gpu Tests (Real-to-Complex GPU FFT)
// ============================================================================

class DfftR2CGpu3DTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!cuda_device_available()) {
            GTEST_SKIP() << "CUDA device not available, skipping GPU tests";
        }
        
        dims_3d[0] = 8; dims_3d[1] = 8; dims_3d[2] = 8;
        dims_small_3d[0] = 4; dims_small_3d[1] = 4; dims_small_3d[2] = 4;
        fft_3d = std::make_unique<dfft_r2c_gpu>(dims_3d, dims_small_3d);
    }

    void TearDown() override {
        fft_3d.reset();
        cudaDeviceReset();
    }

    dfft_r2c_gpu::dims_3D dims_3d;
    dfft_r2c_gpu::dims_3D dims_small_3d;
    std::unique_ptr<dfft_r2c_gpu> fft_3d;
};

TEST_F(DfftR2CGpu3DTest, ConstructorInitializesCorrectly) {
    EXPECT_NO_THROW({
        dfft_r2c_gpu fft(dims_3d, dims_small_3d);
    });
}

TEST_F(DfftR2CGpu3DTest, InputOutputAccessors) {
    auto& input = fft_3d->input();
    auto& output = fft_3d->output();
    
    EXPECT_EQ(input.size(), dims_3d[0] * dims_3d[1] * dims_3d[2]);
    EXPECT_EQ(output.size(), dims_3d[1] * dims_3d[2] * ((dims_3d[0] / 2) + 1));
}

TEST_F(DfftR2CGpu3DTest, CopyInputValidSize) {
    std::vector<double> data(dims_3d[0] * dims_3d[1] * dims_3d[2], 1.0);
    
    EXPECT_NO_THROW({
        fft_3d->copy_input(data, dims_3d);
    });
    
    const auto& input = fft_3d->input();
    for (size_t i = 0; i < input.size(); ++i) {
        EXPECT_DOUBLE_EQ(input[i], 1.0);
    }
}

TEST_F(DfftR2CGpu3DTest, ExecuteWholeTransform) {
    auto& input = fft_3d->input();
    
    std::fill(input.begin(), input.end(), 0.0);
    input[0] = 1.0;
    
    EXPECT_NO_THROW({
        fft_3d->execute_whole();
    });
    
    // Copy output from device to host
    auto& output = fft_3d->output();
    cudaMemcpy(output.data(), fft_3d->output_cu(), output.size() * sizeof(std::complex<double>), cudaMemcpyDeviceToHost);
    
    EXPECT_GT(std::abs(output[0]), 0.0);
}

TEST_F(DfftR2CGpu3DTest, ExecuteStageWiseTransform) {
    auto& input = fft_3d->input();
    
    std::iota(input.begin(), input.end(), 0.0);
    
    EXPECT_NO_THROW({
        fft_3d->execute();
    });
    
    const auto& output = fft_3d->output();
    EXPECT_GT(output.size(), 0);
}

TEST_F(DfftR2CGpu3DTest, OutputCuPointer) {
    auto* output_cu = fft_3d->output_cu();
    EXPECT_NE(output_cu, nullptr) << "GPU device pointer should be valid";
}

TEST_F(DfftR2CGpu3DTest, DeltaFunctionTransform) {
    // Delta function should give constant spectrum
    auto& input = fft_3d->input();
    std::fill(input.begin(), input.end(), 0.0);
    input[0] = 1.0;
    
    fft_3d->execute_whole();
    
    // Copy output from device to host
    auto& output = fft_3d->output();
    cudaMemcpy(output.data(), fft_3d->output_cu(), output.size() * sizeof(std::complex<double>), cudaMemcpyDeviceToHost);
    
    // First few coefficients should be approximately 1.0 (delta transform)
    EXPECT_NEAR(output[0].real(), 1.0, 0.1);
    EXPECT_NEAR(output[0].imag(), 0.0, 0.1);
}

TEST_F(DfftR2CGpu3DTest, ConstantInputTransform) {
    // Constant input should have energy in DC component
    auto& input = fft_3d->input();
    std::fill(input.begin(), input.end(), 5.0);
    
    fft_3d->execute_whole();
    
    // Copy output from device to host
    auto& output = fft_3d->output();
    cudaMemcpy(output.data(), fft_3d->output_cu(), output.size() * sizeof(std::complex<double>), cudaMemcpyDeviceToHost);
    
    // DC component (first) should be largest
    double dc_magnitude = std::abs(output[0]);
    EXPECT_GT(dc_magnitude, 0.0);
    
    // Other components should be much smaller
    double other_magnitude = std::abs(output[10]);
    EXPECT_LT(other_magnitude, dc_magnitude * 0.1);
}

// ============================================================================
// dfft_r2c_gpu_batch Tests (Batched GPU R2C FFT)
// ============================================================================

class DfftR2CGpuBatch3DTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!cuda_device_available()) {
            GTEST_SKIP() << "CUDA device not available, skipping GPU tests";
        }
        
        dims_3d[0] = 8; dims_3d[1] = 8; dims_3d[2] = 8;
        dims_small_3d[0] = 4; dims_small_3d[1] = 4; dims_small_3d[2] = 4;
        batch_size = 4;
        fft_batch = std::make_unique<dfft_r2c_gpu_batch>(dims_3d, dims_small_3d, batch_size);
    }

    void TearDown() override {
        fft_batch.reset();
        cudaDeviceReset();
    }

    dfft_r2c_gpu_batch::dims_3D dims_3d;
    dfft_r2c_gpu_batch::dims_3D dims_small_3d;
    int batch_size;
    std::unique_ptr<dfft_r2c_gpu_batch> fft_batch;
};

TEST_F(DfftR2CGpuBatch3DTest, ConstructorInitializesCorrectly) {
    EXPECT_NO_THROW({
        dfft_r2c_gpu_batch fft(dims_3d, dims_small_3d, batch_size);
    });
}

TEST_F(DfftR2CGpuBatch3DTest, BatchSizeConfiguration) {
    EXPECT_EQ(fft_batch->input_size(), 
              static_cast<size_t>(dims_3d[0]) * dims_3d[1] * dims_3d[2] * batch_size);
}

TEST_F(DfftR2CGpuBatch3DTest, StreamAccess) {
    EXPECT_NO_THROW({
        auto& stream = fft_batch->stream();
        auto& transfer_stream = fft_batch->transfer_stream();
    });
}

TEST_F(DfftR2CGpuBatch3DTest, DevicePointerValidity) {
    auto* input_cu = fft_batch->input_cu();
    auto* output_cu = fft_batch->output_cu();
    auto* result_cu = fft_batch->result_cu();
    
    EXPECT_NE(input_cu, nullptr);
    EXPECT_NE(output_cu, nullptr);
    EXPECT_NE(result_cu, nullptr);
}

TEST_F(DfftR2CGpuBatch3DTest, F0ManagementVectors) {
    auto& f0_ptrs = fft_batch->f0_ptrs();
    auto& f0_sizes = fft_batch->f0_sizes();
    
    EXPECT_EQ(f0_ptrs.size(), 0);
    EXPECT_EQ(f0_sizes.size(), 0);
    
    // Add dummy entries
    cufftDoubleComplex* dummy = nullptr;
    cudaMalloc(&dummy, 64 * sizeof(cufftDoubleComplex));
    
    f0_ptrs.push_back(dummy);
    f0_sizes.push_back(64);
    
    EXPECT_EQ(f0_ptrs.size(), 1);
    EXPECT_EQ(f0_sizes.size(), 1);
    
    cudaFree(dummy);
}

// ============================================================================
// dfft_c2r_gpu Tests (Complex-to-Real GPU FFT)
// ============================================================================

class DfftC2RGpu3DTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!cuda_device_available()) {
            GTEST_SKIP() << "CUDA device not available, skipping GPU tests";
        }
        
        dims_3d[0] = 8; dims_3d[1] = 8; dims_3d[2] = 8;
        dims_small_3d[0] = 4; dims_small_3d[1] = 4; dims_small_3d[2] = 4;
        fft_3d = std::make_unique<dfft_c2r_gpu>(dims_3d, dims_small_3d);
    }

    void TearDown() override {
        fft_3d.reset();
        cudaDeviceReset();
    }

    dfft_c2r_gpu::dims_3D dims_3d;
    dfft_c2r_gpu::dims_3D dims_small_3d;
    std::unique_ptr<dfft_c2r_gpu> fft_3d;
};

TEST_F(DfftC2RGpu3DTest, ConstructorInitializesCorrectly) {
    EXPECT_NO_THROW({
        dfft_c2r_gpu fft(dims_3d, dims_small_3d);
    });
}

TEST_F(DfftC2RGpu3DTest, InputOutputAccessors) {
    auto& input = fft_3d->input();
    auto& output = fft_3d->output();
    
    EXPECT_EQ(input.size(), dims_3d[1] * dims_3d[2] * ((dims_3d[0] / 2) + 1));
    EXPECT_EQ(output.size(), dims_3d[0] * dims_3d[1] * dims_3d[2]);
}

TEST_F(DfftC2RGpu3DTest, ExecuteTransform) {
    auto& input = fft_3d->input();
    
    std::fill(input.begin(), input.end(), std::complex<double>(1.0, 0.0));
    
    EXPECT_NO_THROW({
        fft_3d->execute();
    });
    
    const auto& output = fft_3d->output();
    EXPECT_GT(output.size(), 0);
}

TEST_F(DfftC2RGpu3DTest, ExecuteDeviceTransform) {
    auto& input = fft_3d->input();
    std::fill(input.begin(), input.end(), std::complex<double>(1.0, 0.0));
    
    EXPECT_NO_THROW({
        fft_3d->execute_device();
    });
}

TEST_F(DfftC2RGpu3DTest, OutputCuPointerAccess) {
    float_type* output_ptr = nullptr;
    EXPECT_NO_THROW({
        output_ptr = fft_3d->output_cu_ptr();
    });
    EXPECT_NE(output_ptr, nullptr);
}

TEST_F(DfftC2RGpu3DTest, CopyOutputToHost) {
    auto& input = fft_3d->input();
    std::fill(input.begin(), input.end(), std::complex<double>(2.0, 0.0));
    
    fft_3d->execute_device();
    
    EXPECT_NO_THROW({
        fft_3d->copy_output_to_host();
    });
    
    cudaStreamSynchronize(fft_3d->stream());
    
    const auto& output = fft_3d->output();
    EXPECT_GT(output.size(), 0);
}

TEST_F(DfftC2RGpu3DTest, StreamAccess) {
    EXPECT_NO_THROW({
        auto& stream = fft_3d->stream();
    });
}

// ============================================================================
// Round-trip GPU FFT Tests
// ============================================================================

class RoundTripGpuFFT3DTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!cuda_device_available()) {
            GTEST_SKIP() << "CUDA device not available, skipping GPU tests";
        }
        
        dims_3d[0] = 8; dims_3d[1] = 8; dims_3d[2] = 8;
        dims_small_3d[0] = 4; dims_small_3d[1] = 4; dims_small_3d[2] = 4;
    }

    void TearDown() override {
        cudaDeviceReset();
    }

    dfft_r2c_gpu::dims_3D dims_3d;
    dfft_r2c_gpu::dims_3D dims_small_3d;
};

TEST_F(RoundTripGpuFFT3DTest, ForwardAndBackwardTransformsExecute) {
    dfft_r2c_gpu r2c(dims_3d, dims_small_3d);
    dfft_c2r_gpu c2r(dims_3d, dims_small_3d);
    
    auto& input = r2c.input();
    std::fill(input.begin(), input.end(), 1.0);
    
    EXPECT_NO_THROW(r2c.execute_whole());
    
    // Copy output from device to host
    auto& freq_domain_ref = r2c.output();
    cudaMemcpy(freq_domain_ref.data(), r2c.output_cu(), freq_domain_ref.size() * sizeof(std::complex<double>), cudaMemcpyDeviceToHost);
    const auto& freq_domain = freq_domain_ref;
    EXPECT_GT(freq_domain.size(), 0);
    
    auto& inverse_input = c2r.input();
    std::copy(freq_domain.begin(), freq_domain.end(), inverse_input.begin());
    
    EXPECT_NO_THROW(c2r.execute());
    
    const auto& output = c2r.output();
    EXPECT_GT(output.size(), 0);
}

TEST_F(RoundTripGpuFFT3DTest, RoundTripAccuracy) {
    // Test round-trip: input -> R2C -> C2R -> should recover (scaled) input
    dfft_r2c_gpu r2c(dims_3d, dims_small_3d);
    dfft_c2r_gpu c2r(dims_3d, dims_small_3d);
    
    auto& input = r2c.input();
    
    // Create known input
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = std::sin(2.0 * M_PI * i / input.size());
    }
    
    // Store original
    std::vector<double> original(input.begin(), input.end());
    
    // Forward transform
    r2c.execute_whole();
    
    // Copy output from device to host
    auto& freq_domain_ref = r2c.output();
    cudaMemcpy(freq_domain_ref.data(), r2c.output_cu(), freq_domain_ref.size() * sizeof(std::complex<double>), cudaMemcpyDeviceToHost);
    const auto& freq_domain = freq_domain_ref;
    
    // Backward transform
    auto& inverse_input = c2r.input();
    std::copy(freq_domain.begin(), freq_domain.end(), inverse_input.begin());
    c2r.execute();
    const auto& recovered = c2r.output();
    
    // Check recovery (with scaling)
    size_t total_size = dims_3d[0] * dims_3d[1] * dims_3d[2];
    double scale = 1.0 / total_size;
    
    double max_error = 0.0;
    for (size_t i = 0; i < std::min(original.size(), recovered.size()); ++i) {
        double error = std::abs(recovered[i] * scale - original[i]);
        max_error = std::max(max_error, error);
    }
    
    EXPECT_LT(max_error, 1e-6) << "Round-trip GPU FFT error should be small";
}

TEST_F(RoundTripGpuFFT3DTest, DeltaFunctionRoundTrip) {
    // Delta function should remain delta after round trip
    dfft_r2c_gpu r2c(dims_3d, dims_small_3d);
    dfft_c2r_gpu c2r(dims_3d, dims_small_3d);
    
    auto& input = r2c.input();
    std::fill(input.begin(), input.end(), 0.0);
    input[0] = 1.0;
    
    r2c.execute_whole();
    
    // Copy output from device to host
    auto& freq_domain_ref = r2c.output();
    cudaMemcpy(freq_domain_ref.data(), r2c.output_cu(), freq_domain_ref.size() * sizeof(std::complex<double>), cudaMemcpyDeviceToHost);
    const auto& freq_domain = freq_domain_ref;
    
    auto& inverse_input = c2r.input();
    std::copy(freq_domain.begin(), freq_domain.end(), inverse_input.begin());
    c2r.execute();
    const auto& recovered = c2r.output();
    
    size_t total_size = dims_3d[0] * dims_3d[1] * dims_3d[2];
    double scale = 1.0 / total_size;
    
    EXPECT_NEAR(recovered[0] * scale, 1.0, 1e-10);
    
    // Other values should be near zero
    double sum_others = 0.0;
    for (size_t i = 1; i < recovered.size(); ++i) {
        sum_others += std::abs(recovered[i]);
    }
    EXPECT_LT(sum_others * scale, 1e-8);
}

// ============================================================================
// Analytical Solution GPU Tests
// ============================================================================

TEST(AnalyticalSolutionsGpu, CosineWaveFFT3D) {
    if (!cuda_device_available()) {
        GTEST_SKIP() << "CUDA device not available";
    }
    
    const int N = 16;
    dfft_r2c_gpu::dims_3D dims;
    dims[0] = N; dims[1] = N; dims[2] = N;
    
    dfft_r2c_gpu fft(dims, dims);
    auto& input = fft.input();
    
    // Cosine wave with frequency k
    const int k = 2;
    for (int iz = 0; iz < N; ++iz) {
        for (int iy = 0; iy < N; ++iy) {
            for (int ix = 0; ix < N; ++ix) {
                size_t idx = ix + iy * N + iz * N * N;
                input[idx] = std::cos(2.0 * M_PI * k * ix / N);
            }
        }
    }
    
    fft.execute_whole();
    
    // Copy output from device to host
    auto& output = fft.output();
    cudaMemcpy(output.data(), fft.output_cu(), output.size() * sizeof(std::complex<double>), cudaMemcpyDeviceToHost);
    
    // Should have peaks at wavenumber ±k
    // DC should be near zero for a pure cosine
    double dc_magnitude = std::abs(output[0]);
    double max_magnitude = 0.0;
    for (const auto& value : output) {
        max_magnitude = std::max(max_magnitude, std::abs(value));
    }
    EXPECT_GT(max_magnitude, 1e-10);
    EXPECT_LT(dc_magnitude, max_magnitude);
}

TEST(AnalyticalSolutionsGpu, ParsevalsTheoremGpu) {
    if (!cuda_device_available()) {
        GTEST_SKIP() << "CUDA device not available";
    }
    
    const int N = 16;
    dfft_r2c_gpu::dims_3D dims;
    dims[0] = N; dims[1] = N; dims[2] = N;
    
    dfft_r2c_gpu fft(dims, dims);
    auto& input = fft.input();
    
    // Random-like signal
    double energy_time = 0.0;
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = std::sin(i * 0.5) + std::cos(i * 0.3);
        energy_time += input[i] * input[i];
    }
    
    fft.execute_whole();
    
    // Copy output from device to host
    auto& output = fft.output();
    cudaMemcpy(output.data(), fft.output_cu(), output.size() * sizeof(std::complex<double>), cudaMemcpyDeviceToHost);
    
    // Energy in frequency domain (account for Hermitian symmetry of R2C output)
    double energy_freq = 0.0;
    const size_t nx = static_cast<size_t>(N);
    const size_t ny = static_cast<size_t>(N);
    const size_t nz = static_cast<size_t>(N);
    const size_t nx_half = nx / 2;
    for (size_t iz = 0; iz < nz; ++iz) {
        for (size_t iy = 0; iy < ny; ++iy) {
            for (size_t ix = 0; ix <= nx_half; ++ix) {
                const size_t idx = ix + iy * (nx_half + 1) + iz * ny * (nx_half + 1);
                const double weight = (ix == 0 || ix == nx_half) ? 1.0 : 2.0;
                energy_freq += weight * std::norm(output[idx]);
            }
        }
    }
    
    // Parseval's theorem (with scaling)
    size_t total_size = N * N * N;
    double ratio = energy_freq / (energy_time * total_size);
    
    EXPECT_NEAR(ratio, 1.0, 0.1) << "Parseval's theorem should hold for GPU FFT";
    
}

TEST(AnalyticalSolutionsGpu, GaussianTransformGpu) {
    if (!cuda_device_available()) {
        GTEST_SKIP() << "CUDA device not available";
    }
    
    const int N = 32;
    dfft_r2c_gpu::dims_3D dims;
    dims[0] = N; dims[1] = N; dims[2] = N;
    
    dfft_r2c_gpu fft(dims, dims);
    auto& input = fft.input();
    
    // Gaussian function
    const double sigma = N / 8.0;
    const int center = N / 2;
    
    for (int iz = 0; iz < N; ++iz) {
        for (int iy = 0; iy < N; ++iy) {
            for (int ix = 0; ix < N; ++ix) {
                double dx = ix - center;
                double dy = iy - center;
                double dz = iz - center;
                double r2 = dx*dx + dy*dy + dz*dz;
                size_t idx = ix + iy * N + iz * N * N;
                input[idx] = std::exp(-r2 / (2.0 * sigma * sigma));
            }
        }
    }
    
    fft.execute_whole();
    
    // Copy output from device to host
    auto& output = fft.output();
    cudaMemcpy(output.data(), fft.output_cu(), output.size() * sizeof(std::complex<double>), cudaMemcpyDeviceToHost);
    
    // Gaussian transform should also be Gaussian-like
    // DC component should be largest
    double dc_magnitude = std::abs(output[0]);
    EXPECT_GT(dc_magnitude, 0.0);
    
    // Spectrum should decay with frequency
    size_t mid_idx = output.size() / 2;
    double mid_magnitude = std::abs(output[mid_idx]);
    EXPECT_LT(mid_magnitude, dc_magnitude);
    
}

TEST(AnalyticalSolutionsGpu, LinearityPropertyGpu) {
    if (!cuda_device_available()) {
        GTEST_SKIP() << "CUDA device not available";
    }
    
    const int N = 16;
    dfft_r2c_gpu::dims_3D dims;
    dims[0] = N; dims[1] = N; dims[2] = N;
    
    // Create three FFT objects
    dfft_r2c_gpu fft1(dims, dims);
    dfft_r2c_gpu fft2(dims, dims);
    dfft_r2c_gpu fft_combined(dims, dims);
    
    const double alpha = 2.5;
    const double beta = -1.3;
    
    // Signal 1
    auto& input1 = fft1.input();
    for (size_t i = 0; i < input1.size(); ++i) {
        input1[i] = std::sin(i * 0.1);
    }
    
    // Signal 2
    auto& input2 = fft2.input();
    for (size_t i = 0; i < input2.size(); ++i) {
        input2[i] = std::cos(i * 0.2);
    }
    
    // Combined signal
    auto& input_combined = fft_combined.input();
    for (size_t i = 0; i < input_combined.size(); ++i) {
        input_combined[i] = alpha * input1[i] + beta * input2[i];
    }
    
    // Transform all three
    fft1.execute_whole();
    fft2.execute_whole();
    fft_combined.execute_whole();
    
    // Copy outputs from device to host
    auto& output1 = fft1.output();
    auto& output2 = fft2.output();
    auto& output_combined = fft_combined.output();
    cudaMemcpy(output1.data(), fft1.output_cu(), output1.size() * sizeof(std::complex<double>), cudaMemcpyDeviceToHost);
    cudaMemcpy(output2.data(), fft2.output_cu(), output2.size() * sizeof(std::complex<double>), cudaMemcpyDeviceToHost);
    cudaMemcpy(output_combined.data(), fft_combined.output_cu(), output_combined.size() * sizeof(std::complex<double>), cudaMemcpyDeviceToHost);
    
    // Verify linearity: FFT(alpha*f1 + beta*f2) = alpha*FFT(f1) + beta*FFT(f2)
    double max_error = 0.0;
    for (size_t i = 0; i < output_combined.size(); ++i) {
        std::complex<double> expected = alpha * output1[i] + beta * output2[i];
        double error = std::abs(output_combined[i] - expected);
        max_error = std::max(max_error, error);
    }
    
    EXPECT_LT(max_error, 1e-8) << "GPU FFT linearity property should hold";
    
}

// ============================================================================
// GPU Performance and Edge Cases
// ============================================================================

TEST(GpuFFTEdgeCases, LargeTransformSize) {
    if (!cuda_device_available()) {
        GTEST_SKIP() << "CUDA device not available";
    }
    
    const int N = 64;
    dfft_r2c_gpu::dims_3D dims;
    dims[0] = N; dims[1] = N; dims[2] = N;
    
    EXPECT_NO_THROW({
        dfft_r2c_gpu fft(dims, dims);
        auto& input = fft.input();
        std::fill(input.begin(), input.end(), 1.0);
        fft.execute_whole();
    });
    
}

TEST(GpuFFTEdgeCases, AsymmetricDimensions) {
    if (!cuda_device_available()) {
        GTEST_SKIP() << "CUDA device not available";
    }
    
    dfft_r2c_gpu::dims_3D dims;
    dims[0] = 16; dims[1] = 8; dims[2] = 4;
    
    EXPECT_NO_THROW({
        dfft_r2c_gpu fft(dims, dims);
        auto& input = fft.input();
        std::fill(input.begin(), input.end(), 2.0);
        fft.execute_whole();
    });
    
}

TEST(GpuFFTEdgeCases, ZeroInput) {
    if (!cuda_device_available()) {
        GTEST_SKIP() << "CUDA device not available";
    }
    
    dfft_r2c_gpu::dims_3D dims;
    dims[0] = 8; dims[1] = 8; dims[2] = 8;
    
    dfft_r2c_gpu fft(dims, dims);
    auto& input = fft.input();
    std::fill(input.begin(), input.end(), 0.0);
    
    EXPECT_NO_THROW({
        fft.execute_whole();
    });
    
    const auto& output = fft.output();
    
    // All output should be near zero
    double max_magnitude = 0.0;
    for (const auto& val : output) {
        max_magnitude = std::max(max_magnitude, std::abs(val));
    }
    
    EXPECT_LT(max_magnitude, 1e-10) << "Zero input should produce zero output";
    
}

TEST(GpuFFTPerformance, MultipleSequentialTransforms) {
    if (!cuda_device_available()) {
        GTEST_SKIP() << "CUDA device not available";
    }
    
    dfft_r2c_gpu::dims_3D dims;
    dims[0] = 16; dims[1] = 16; dims[2] = 16;
    
    dfft_r2c_gpu fft(dims, dims);
    auto& input = fft.input();
    
    // Execute multiple transforms
    for (int iter = 0; iter < 10; ++iter) {
        for (size_t i = 0; i < input.size(); ++i) {
            input[i] = std::sin(i * 0.1 * iter);
        }
        
        EXPECT_NO_THROW({
            fft.execute_whole();
        });
    }
    
}
