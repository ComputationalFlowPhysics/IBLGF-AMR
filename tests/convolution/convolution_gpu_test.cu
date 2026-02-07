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

// Check if CUDA device is available
bool cuda_device_available() {
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    return error == cudaSuccess && deviceCount > 0;
}

static void expect_complex_near(const cuDoubleComplex& actual, const cuDoubleComplex& expected, double tol)
{
    EXPECT_NEAR(cuCreal(actual), cuCreal(expected), tol);
    EXPECT_NEAR(cuCimag(actual), cuCimag(expected), tol);
}

// ============================================================================
// Kernel Tests (prod_complex_add_ptr)
// ============================================================================

TEST(KernelProdComplexAddPtr, MultipliesAndZerosBySize) {
    if (!cuda_device_available()) {
        GTEST_SKIP() << "CUDA device not available";
    }

    const int batch_size = 2;
    const size_t output_size_per_batch = 4;
    const size_t total_size = static_cast<size_t>(batch_size) * output_size_per_batch;

    // Host buffers
    std::vector<cuDoubleComplex> h_f0_batch0(output_size_per_batch);
    std::vector<cuDoubleComplex> h_f0_batch1(output_size_per_batch);
    std::vector<cuDoubleComplex> h_output(total_size);
    std::vector<cuDoubleComplex> h_result(total_size);

    // Initialize f0 values
    h_f0_batch0[0] = make_cuDoubleComplex(1.0, 0.0);
    h_f0_batch0[1] = make_cuDoubleComplex(2.0, -1.0);
    h_f0_batch0[2] = make_cuDoubleComplex(-1.0, 0.5);
    h_f0_batch0[3] = make_cuDoubleComplex(0.0, 3.0);

    h_f0_batch1[0] = make_cuDoubleComplex(0.5, -0.5);
    h_f0_batch1[1] = make_cuDoubleComplex(4.0, 1.0);
    h_f0_batch1[2] = make_cuDoubleComplex(-2.0, 2.0);
    h_f0_batch1[3] = make_cuDoubleComplex(1.0, 1.0);

    // Output values (batched, contiguous)
    for (size_t i = 0; i < total_size; ++i) {
        h_output[i] = make_cuDoubleComplex(static_cast<double>(i + 1), -static_cast<double>(i));
    }

    // f0 sizes: batch0 has 3 valid elements, batch1 has 1
    std::vector<size_t> h_f0_sizes = {3, 1};

    // Device allocations
    cuDoubleComplex* d_f0_batch0 = nullptr;
    cuDoubleComplex* d_f0_batch1 = nullptr;
    cuDoubleComplex* d_output = nullptr;
    cuDoubleComplex* d_result = nullptr;
    size_t* d_f0_sizes = nullptr;
    cuDoubleComplex** d_f0_ptrs = nullptr;

    ASSERT_EQ(cudaMalloc(&d_f0_batch0, output_size_per_batch * sizeof(cuDoubleComplex)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_f0_batch1, output_size_per_batch * sizeof(cuDoubleComplex)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_output, total_size * sizeof(cuDoubleComplex)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_result, total_size * sizeof(cuDoubleComplex)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_f0_sizes, batch_size * sizeof(size_t)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_f0_ptrs, batch_size * sizeof(cuDoubleComplex*)), cudaSuccess);

    ASSERT_EQ(cudaMemcpy(d_f0_batch0, h_f0_batch0.data(), output_size_per_batch * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_f0_batch1, h_f0_batch1.data(), output_size_per_batch * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_output, h_output.data(), total_size * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_f0_sizes, h_f0_sizes.data(), batch_size * sizeof(size_t), cudaMemcpyHostToDevice), cudaSuccess);

    std::vector<cuDoubleComplex*> h_f0_ptrs = {d_f0_batch0, d_f0_batch1};
    ASSERT_EQ(cudaMemcpy(d_f0_ptrs, h_f0_ptrs.data(), batch_size * sizeof(cuDoubleComplex*), cudaMemcpyHostToDevice), cudaSuccess);

    // Launch kernel
    int block_size = 128;
    int grid_size = static_cast<int>((total_size + block_size - 1) / block_size);
    size_t shared_mem = batch_size * sizeof(size_t);
    prod_complex_add_ptr<<<grid_size, block_size, shared_mem>>>(
        d_f0_ptrs,
        d_output,
        d_result,
        d_f0_sizes,
        batch_size,
        output_size_per_batch);

    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(h_result.data(), d_result, total_size * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost), cudaSuccess);

    // Expected results
    // Batch 0: idx 0..2 multiply, idx 3 zero
    expect_complex_near(h_result[0], cuCmul(h_f0_batch0[0], h_output[0]), 1e-12);
    expect_complex_near(h_result[1], cuCmul(h_f0_batch0[1], h_output[1]), 1e-12);
    expect_complex_near(h_result[2], cuCmul(h_f0_batch0[2], h_output[2]), 1e-12);
    expect_complex_near(h_result[3], make_cuDoubleComplex(0.0, 0.0), 1e-12);

    // Batch 1: idx 0 multiply, idx 1..3 zero
    expect_complex_near(h_result[4], cuCmul(h_f0_batch1[0], h_output[4]), 1e-12);
    expect_complex_near(h_result[5], make_cuDoubleComplex(0.0, 0.0), 1e-12);
    expect_complex_near(h_result[6], make_cuDoubleComplex(0.0, 0.0), 1e-12);
    expect_complex_near(h_result[7], make_cuDoubleComplex(0.0, 0.0), 1e-12);

    cudaFree(d_f0_ptrs);
    cudaFree(d_f0_sizes);
    cudaFree(d_result);
    cudaFree(d_output);
    cudaFree(d_f0_batch1);
    cudaFree(d_f0_batch0);
}

// ============================================================================
// dfft_r2c_gpu Tests
// ============================================================================

class DfftR2cGpuTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!cuda_device_available()) {
            GTEST_SKIP() << "CUDA device not available, skipping GPU tests";
        }
        
        dims_padded[0] = 16; dims_padded[1] = 16; dims_padded[2] = 16;
        dims_non_zero[0] = 8; dims_non_zero[1] = 8; dims_non_zero[2] = 8;
        
        fft_gpu = std::make_unique<dfft_r2c_gpu>(dims_padded, dims_non_zero);
    }

    void TearDown() override {
        fft_gpu.reset();
        cudaDeviceReset();
    }

    types::vector_type<int, 3> dims_padded;
    types::vector_type<int, 3> dims_non_zero;
    std::unique_ptr<dfft_r2c_gpu> fft_gpu;
};

TEST_F(DfftR2cGpuTest, ConstructorInitializesCorrectly) {
    EXPECT_NO_THROW({
        dfft_r2c_gpu gpu(dims_padded, dims_non_zero);
    });
}

TEST_F(DfftR2cGpuTest, InputOutputAccessors) {
    auto& input = fft_gpu->input();
    auto& output = fft_gpu->output();
    
    EXPECT_GT(input.size(), 0);
    EXPECT_GT(output.size(), 0);
    
    // Input should have size of padded dimensions
    size_t expected_input_size = dims_padded[0] * dims_padded[1] * dims_padded[2];
    EXPECT_EQ(input.size(), expected_input_size);
}

TEST_F(DfftR2cGpuTest, CopyInputFromVector) {
    size_t data_size = dims_padded[0] * dims_padded[1] * dims_padded[2];
    std::vector<double> test_data(data_size);
    
    // Fill with simple pattern
    for (size_t i = 0; i < data_size; ++i) {
        test_data[i] = static_cast<double>(i % 100) / 10.0;
    }
    
    EXPECT_NO_THROW({
        fft_gpu->copy_input(test_data, dims_padded);
    });
    
    // Verify data was copied to input buffer
    auto& input = fft_gpu->input();
    for (size_t idx = 0; idx < test_data.size(); ++idx) {
        EXPECT_NEAR(input[idx], test_data[idx], 1e-10);
    }
}

TEST_F(DfftR2cGpuTest, ExecuteWholeTransform) {
    // Create simple test data - delta function
    auto& input = fft_gpu->input();
    std::fill(input.begin(), input.end(), 0.0);
    input[0] = 1.0; // Delta at origin
    
    EXPECT_NO_THROW({
        fft_gpu->execute_whole();
    });
    
    // Copy output from device to host
    auto& output = fft_gpu->output();
    cudaMemcpy(output.data(), fft_gpu->output_cu(), output.size() * sizeof(std::complex<double>), cudaMemcpyDeviceToHost);
    
    EXPECT_GT(output.size(), 0);
    
    // DFT of delta function should be constant
    double expected_real = 1.0;
    for (size_t i = 0; i < std::min(size_t(10), output.size()); ++i) {
        EXPECT_NEAR(std::abs(output[i].real()), expected_real, 0.1);
    }
}

TEST_F(DfftR2cGpuTest, ExecuteTransform) {
    // Create constant input
    auto& input = fft_gpu->input();
    std::fill(input.begin(), input.end(), 2.0);
    
    EXPECT_NO_THROW({
        fft_gpu->execute();
    });
    
    // Copy output from device to host
    auto& output = fft_gpu->output();
    cudaMemcpy(output.data(), fft_gpu->output_cu(), output.size() * sizeof(std::complex<double>), cudaMemcpyDeviceToHost);
    
    EXPECT_GT(output.size(), 0);
}

// ============================================================================
// dfft_r2c_gpu_batch Tests
// ============================================================================

class DfftR2cGpuBatchTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!cuda_device_available()) {
            GTEST_SKIP() << "CUDA device not available, skipping GPU tests";
        }
        
        dims_padded[0] = 16; dims_padded[1] = 16; dims_padded[2] = 16;
        dims_non_zero[0] = 8; dims_non_zero[1] = 8; dims_non_zero[2] = 8;
        batch_size = 4;
        
        fft_batch = std::make_unique<dfft_r2c_gpu_batch>(dims_padded, dims_non_zero, batch_size);
    }

    void TearDown() override {
        fft_batch.reset();
        cudaDeviceReset();
    }

    types::vector_type<int, 3> dims_padded;
    types::vector_type<int, 3> dims_non_zero;
    int batch_size;
    std::unique_ptr<dfft_r2c_gpu_batch> fft_batch;
};

TEST_F(DfftR2cGpuBatchTest, ConstructorInitializesCorrectly) {
    EXPECT_NO_THROW({
        dfft_r2c_gpu_batch gpu_batch(dims_padded, dims_non_zero, batch_size);
    });
}

TEST_F(DfftR2cGpuBatchTest, BatchSizeConfiguration) {
    // Verify batch configuration
    EXPECT_EQ(fft_batch->input_size(), 
              static_cast<size_t>(dims_padded[0]) * dims_padded[1] * dims_padded[2] * batch_size);
}

TEST_F(DfftR2cGpuBatchTest, StreamAccessors) {
    EXPECT_NO_THROW({
        auto& stream = fft_batch->stream();
        auto& transfer_stream = fft_batch->transfer_stream();
    });
}

TEST_F(DfftR2cGpuBatchTest, F0PointersAndSizes) {
    auto& f0_ptrs = fft_batch->f0_ptrs();
    auto& f0_sizes = fft_batch->f0_sizes();
    
    EXPECT_EQ(f0_ptrs.size(), 0); // Initially empty
    EXPECT_EQ(f0_sizes.size(), 0);
    
    // Test adding entries
    cufftDoubleComplex* dummy_ptr = nullptr;
    cudaMalloc(&dummy_ptr, 100 * sizeof(cufftDoubleComplex));
    
    f0_ptrs.push_back(dummy_ptr);
    f0_sizes.push_back(100);
    
    EXPECT_EQ(f0_ptrs.size(), 1);
    EXPECT_EQ(f0_sizes.size(), 1);
    EXPECT_EQ(f0_sizes[0], 100);
    
    cudaFree(dummy_ptr);
}

TEST_F(DfftR2cGpuBatchTest, InputOutputBuffers) {
    auto& input = fft_batch->input();
    auto& output = fft_batch->output();
    auto* input_cu = fft_batch->input_cu();
    auto* output_cu = fft_batch->output_cu();
    
    EXPECT_NE(input, nullptr);
    EXPECT_GT(output.size(), 0);
    EXPECT_NE(input_cu, nullptr);
    EXPECT_NE(output_cu, nullptr);
}

// ============================================================================
// dfft_c2r_gpu Tests
// ============================================================================

class DfftC2rGpuTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!cuda_device_available()) {
            GTEST_SKIP() << "CUDA device not available, skipping GPU tests";
        }
        
        dims_padded[0] = 16; dims_padded[1] = 16; dims_padded[2] = 16;
        dims_small[0] = 8; dims_small[1] = 8; dims_small[2] = 8;
        
        ifft_gpu = std::make_unique<dfft_c2r_gpu>(dims_padded, dims_small);
    }

    void TearDown() override {
        ifft_gpu.reset();
        cudaDeviceReset();
    }

    types::vector_type<int, 3> dims_padded;
    types::vector_type<int, 3> dims_small;
    std::unique_ptr<dfft_c2r_gpu> ifft_gpu;
};

TEST_F(DfftC2rGpuTest, ConstructorInitializesCorrectly) {
    EXPECT_NO_THROW({
        dfft_c2r_gpu gpu(dims_padded, dims_small);
    });
}

TEST_F(DfftC2rGpuTest, InputOutputAccessors) {
    auto& input = ifft_gpu->input();
    auto& output = ifft_gpu->output();
    
    EXPECT_GT(input.size(), 0);
    EXPECT_GT(output.size(), 0);
}

TEST_F(DfftC2rGpuTest, StreamAccessor) {
    EXPECT_NO_THROW({
        auto& stream = ifft_gpu->stream();
    });
}

TEST_F(DfftC2rGpuTest, OutputCuPointer) {
    float_type* output_ptr = nullptr;
    EXPECT_NO_THROW({
        output_ptr = ifft_gpu->output_cu_ptr();
    });
    EXPECT_NE(output_ptr, nullptr);
}

TEST_F(DfftC2rGpuTest, ExecuteDeviceTransform) {
    // Initialize input with simple complex values
    auto& input = ifft_gpu->input();
    std::fill(input.begin(), input.end(), std::complex<double>(1.0, 0.0));
    
    EXPECT_NO_THROW({
        ifft_gpu->execute_device();
    });
}

TEST_F(DfftC2rGpuTest, CopyOutputToHost) {
    auto& input = ifft_gpu->input();
    std::fill(input.begin(), input.end(), std::complex<double>(1.0, 0.0));
    
    ifft_gpu->execute_device();
    
    EXPECT_NO_THROW({
        ifft_gpu->copy_output_to_host();
    });
    
    // Sync to ensure copy completes
    cudaStreamSynchronize(ifft_gpu->stream());
    
    auto& output = ifft_gpu->output();
    EXPECT_GT(output.size(), 0);
}

TEST_F(DfftC2rGpuTest, RoundTripTransform) {
    // Test forward + backward should give approximately original data
    dfft_r2c_gpu fft_forward(dims_padded, dims_small);
    
    // Create simple input
    auto& fwd_input = fft_forward.input();
    std::fill(fwd_input.begin(), fwd_input.end(), 0.0);
    
    // Set a few values
    fwd_input[0] = 1.0;
    fwd_input[1] = 2.0;
    fwd_input[dims_padded[0]] = 3.0;
    
    // Forward transform
    fft_forward.execute();
    
    // Copy forward output from device to host
    auto& fwd_output = fft_forward.output();
    cudaMemcpy(fwd_output.data(), fft_forward.output_cu(), fwd_output.size() * sizeof(std::complex<double>), cudaMemcpyDeviceToHost);
    
    // Copy output to backward input
    auto& input = ifft_gpu->input();
    std::copy(fwd_output.begin(), fwd_output.end(), input.begin());
    
    // Backward transform (handles HtoD and DtoH)
    ifft_gpu->execute();
    
    auto& output = ifft_gpu->output();
    
    // Check first few values (with scaling factor)
    size_t total_size = dims_padded[0] * dims_padded[1] * dims_padded[2];
    double scale = 1.0 / total_size;
    
    EXPECT_NEAR(output[0] * scale, 1.0, 1e-6);
    EXPECT_NEAR(output[1] * scale, 2.0, 1e-6);
    EXPECT_NEAR(output[dims_padded[0]] * scale, 3.0, 1e-6);
}

// ============================================================================
// Convolution_GPU Tests (3D)
// ============================================================================

class ConvolutionGpu3DTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!cuda_device_available()) {
            GTEST_SKIP() << "CUDA device not available, skipping GPU tests";
        }
        
        dims0[0] = 8; dims0[1] = 8; dims0[2] = 8;
        dims1[0] = 4; dims1[1] = 4; dims1[2] = 4;
        
        conv_gpu = std::make_unique<Convolution_GPU<3>>(dims0, dims1);
    }

    void TearDown() override {
        conv_gpu.reset();
        cudaDeviceReset();
    }

    types::vector_type<int, 3> dims0;
    types::vector_type<int, 3> dims1;
    std::unique_ptr<Convolution_GPU<3>> conv_gpu;
};

TEST_F(ConvolutionGpu3DTest, ConstructorInitializesCorrectly) {
    EXPECT_NO_THROW({
        Convolution_GPU<3> gpu(dims0, dims1);
    });
}

TEST_F(ConvolutionGpu3DTest, HelperNextPow2) {
    types::vector_type<int, 3> test_dims;
    test_dims[0] = 7; test_dims[1] = 9; test_dims[2] = 5;
    auto result = conv_gpu->helper_next_pow_2(test_dims);
    
    // Current implementation just copies
    EXPECT_EQ(result[0], 7);
    EXPECT_EQ(result[1], 9);
    EXPECT_EQ(result[2], 5);
}

TEST_F(ConvolutionGpu3DTest, HelperAllProd) {
    types::vector_type<int, 3> test_dims;
    test_dims[0] = 2; test_dims[1] = 3; test_dims[2] = 4;
    auto result = conv_gpu->helper_all_prod(test_dims);
    
    EXPECT_EQ(result, 24);
}

TEST_F(ConvolutionGpu3DTest, FftBackwardFieldClean) {
    conv_gpu->number_fwrd_executed = 5;
    
    EXPECT_NO_THROW({
        conv_gpu->fft_backward_field_clean();
    });
    
    EXPECT_EQ(conv_gpu->number_fwrd_executed, 0);
}

TEST_F(ConvolutionGpu3DTest, DftR2cSize) {
    size_t size = conv_gpu->dft_r2c_size();
    EXPECT_GT(size, 0);
}

TEST_F(ConvolutionGpu3DTest, OutputAccessor) {
    auto& output = conv_gpu->output();
    EXPECT_GT(output.size(), 0);
}

TEST_F(ConvolutionGpu3DTest, BatchSizeManagement) {
    EXPECT_EQ(conv_gpu->current_batch_size_, 0);
    EXPECT_EQ(conv_gpu->max_batch_size_, 10);
}

TEST_F(ConvolutionGpu3DTest, FlushEmptyBatch) {
    // Flushing empty batch should not crash
    EXPECT_NO_THROW({
        conv_gpu->flush_batch();
    });
}

// ============================================================================
// CUDA Kernel Tests
// ============================================================================

class CudaKernelTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!cuda_device_available()) {
            GTEST_SKIP() << "CUDA device not available, skipping GPU tests";
        }
    }

    void TearDown() override {
        cudaDeviceReset();
    }
};

TEST_F(CudaKernelTest, ScaleComplexKernel) {
    const size_t size = 128;
    const double scale_factor = 2.5;
    
    // Allocate device memory
    cuDoubleComplex* d_data = nullptr;
    cudaMalloc(&d_data, size * sizeof(cuDoubleComplex));
    
    // Initialize with host data
    std::vector<cuDoubleComplex> h_data(size);
    for (size_t i = 0; i < size; ++i) {
        h_data[i] = make_cuDoubleComplex(i * 1.0, i * 0.5);
    }
    cudaMemcpy(d_data, h_data.data(), size * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    
    // Launch kernel
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    scale_complex<<<numBlocks, blockSize>>>(d_data, size, scale_factor);
    cudaDeviceSynchronize();
    
    // Copy back and verify
    std::vector<cuDoubleComplex> h_result(size);
    cudaMemcpy(h_result.data(), d_data, size * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
    
    for (size_t i = 0; i < size; ++i) {
        EXPECT_NEAR(h_result[i].x, h_data[i].x * scale_factor, 1e-10);
        EXPECT_NEAR(h_result[i].y, h_data[i].y * scale_factor, 1e-10);
    }
    
    cudaFree(d_data);
}

TEST_F(CudaKernelTest, SumBatchesKernel) {
    const size_t size = 64;
    const int batch_size = 4;
    
    // Allocate device memory
    cuDoubleComplex* d_input = nullptr;
    cuDoubleComplex* d_output = nullptr;
    cudaMalloc(&d_input, size * batch_size * sizeof(cuDoubleComplex));
    cudaMalloc(&d_output, size * sizeof(cuDoubleComplex));
    
    // Initialize input with known values
    std::vector<cuDoubleComplex> h_input(size * batch_size);
    for (int b = 0; b < batch_size; ++b) {
        for (size_t i = 0; i < size; ++i) {
            h_input[i + b * size] = make_cuDoubleComplex(b + 1.0, b * 0.5);
        }
    }
    
    // Initialize output to zero
    std::vector<cuDoubleComplex> h_output(size, make_cuDoubleComplex(0.0, 0.0));
    
    cudaMemcpy(d_input, h_input.data(), size * batch_size * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, h_output.data(), size * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    
    // Launch kernel
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    sum_batches<<<numBlocks, blockSize>>>(d_input, d_output, batch_size, size);
    cudaDeviceSynchronize();
    
    // Copy back and verify
    std::vector<cuDoubleComplex> h_result(size);
    cudaMemcpy(h_result.data(), d_output, size * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
    
    // Expected sum: (1+2+3+4) = 10 for real, (0+0.5+1.0+1.5) = 3.0 for imag
    double expected_real = 10.0;
    double expected_imag = 3.0;
    
    for (size_t i = 0; i < size; ++i) {
        EXPECT_NEAR(h_result[i].x, expected_real, 1e-10);
        EXPECT_NEAR(h_result[i].y, expected_imag, 1e-10);
    }
    
    cudaFree(d_input);
    cudaFree(d_output);
}

// ============================================================================
// Integration Tests
// ============================================================================

TEST_F(ConvolutionGpu3DTest, EndToEndConvolution) {
    // Test complete convolution workflow with simple data
    types::vector_type<int, 3> padded_dims;
    padded_dims[0] = dims0[0] + dims1[0] - 1;
    padded_dims[1] = dims0[1] + dims1[1] - 1;
    padded_dims[2] = dims0[2] + dims1[2] - 1;
    size_t data_size = padded_dims[0] * padded_dims[1] * padded_dims[2];
    std::vector<double> test_data(data_size, 1.0);
    
    // Perform forward transform
    auto* spectrum = conv_gpu->dft_r2c(test_data);
    EXPECT_NE(spectrum, nullptr);
    
    // Verify spectrum size
    size_t spectrum_size = conv_gpu->dft_r2c_size();
    EXPECT_GT(spectrum_size, 0);
}

// ============================================================================
// Error Handling Tests
// ============================================================================

class GpuErrorHandlingTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!cuda_device_available()) {
            GTEST_SKIP() << "CUDA device not available, skipping GPU tests";
        }
    }
};

TEST_F(GpuErrorHandlingTest, LargeDimensionsConstruction) {
    types::vector_type<int, 3> large_dims;
    large_dims[0] = 256; large_dims[1] = 256; large_dims[2] = 256;
    types::vector_type<int, 3> small_dims;
    small_dims[0] = 128; small_dims[1] = 128; small_dims[2] = 128;
    
    // Should succeed but may take significant memory
    EXPECT_NO_THROW({
        dfft_r2c_gpu gpu(large_dims, small_dims);
    });
}

TEST_F(GpuErrorHandlingTest, ZeroBatchSize) {
    types::vector_type<int, 3> dims;
    dims[0] = 8; dims[1] = 8; dims[2] = 8;
    
    // Zero batch size should still construct
    EXPECT_NO_THROW({
        dfft_r2c_gpu_batch gpu_batch(dims, dims, 0);
    });
}

// ============================================================================
// Performance Metrics Tests
// ============================================================================

TEST_F(ConvolutionGpu3DTest, BatchExecutionCounter) {
    // Verify fft_count_ and number_fwrd_executed tracking
    EXPECT_EQ(conv_gpu->fft_count_, 0);
    EXPECT_EQ(conv_gpu->number_fwrd_executed, 0);
    
    // After clean, counters should reset
    conv_gpu->number_fwrd_executed = 10;
    conv_gpu->fft_backward_field_clean();
    EXPECT_EQ(conv_gpu->number_fwrd_executed, 0);
}
