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
// GPU Convolution with Lattice Green's Function (LGF) Unit Tests
// Tests GPU-accelerated FFT-based convolution applied to LGF kernels for the Poisson equation
//
#ifndef IBLGF_COMPILE_CUDA
#define IBLGF_COMPILE_CUDA
#endif
#include <gtest/gtest.h>
#include <iblgf/utilities/convolution_GPU.hpp>
#include <iblgf/lgf/lgf_gl.hpp>
#include <iblgf/domain/dataFields/blockDescriptor.hpp>
#include <iblgf/types.hpp>
#include <cmath>
#include <numeric>
#include <cuda_runtime.h>

using namespace iblgf;
using namespace iblgf::fft;

// ============================================================================
// Helper Functions
// ============================================================================

bool cuda_device_available() {
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    return error == cudaSuccess && deviceCount > 0;
}

// ============================================================================
// Test Fixtures for LGF GPU Convolution
// ============================================================================

class LGFConvolutionGpu3DTest : public ::testing::Test {
protected:
    static constexpr std::size_t Dim = 3;
    using float_type = types::float_type;
    using dims_t = types::vector_type<int, Dim>;
    using blk_d_t = domain::BlockDescriptor<int, Dim>;
    using coordinate_t = blk_d_t::coordinate_type;
    using lgf_t = lgf::LGF_GL<Dim>;
    
    void SetUp() override {
        if (!cuda_device_available()) {
            GTEST_SKIP() << "CUDA device not available, skipping GPU tests";
        }
        
        N = 16;
        dims[0] = N; dims[1] = N; dims[2] = N;
        
        // Create block descriptors
        coordinate_t lgf_base(-N + 1);
        coordinate_t base(0);
        coordinate_t extent(2*N - 1);
        
        kernel_block = std::make_unique<blk_d_t>(lgf_base, extent);
        source_block = std::make_unique<blk_d_t>(base, N);
        
        // Initialize GPU convolution object
        conv_gpu = std::make_unique<Convolution_GPU<Dim>>(dims, dims);
    }
    
    void TearDown() override {
        conv_gpu.reset();
        cudaDeviceReset();
    }
    
    int N;
    dims_t dims;
    std::unique_ptr<blk_d_t> kernel_block;
    std::unique_ptr<blk_d_t> source_block;
    std::unique_ptr<Convolution_GPU<Dim>> conv_gpu;
    lgf_t lgf;
};

// ============================================================================
// Delta Function (Point Source) GPU Tests
// ============================================================================

TEST_F(LGFConvolutionGpu3DTest, PointSourceAtOriginGpu) {
    // Test: Delta function at origin should produce LGF kernel using GPU
    domain::DataField<float_type, Dim> source_field;
    source_field.initialize(*source_block);
    
    // Initialize to zero
    for(int k = 0; k < N; ++k) {
        for(int j = 0; j < N; ++j) {
            for(int i = 0; i < N; ++i) {
                source_field.get_real_local(i, j, k) = 0.0;
            }
        }
    }
    
    // Delta function at origin
    source_field.get_real_local(0, 0, 0) = 1.0;
    
    // Apply GPU convolution with LGF kernel
    conv_gpu->fft_backward_field_clean();
    int level_diff = 0;
    
    ASSERT_NO_THROW(conv_gpu->apply_forward_add(*kernel_block, &lgf, level_diff, source_field));
    
    // Verify GPU convolution was applied
    EXPECT_GT(conv_gpu->number_fwrd_executed, 0);
}

TEST_F(LGFConvolutionGpu3DTest, PointSourceNotAtOriginGpu) {
    // Test: Delta function at arbitrary location on GPU
    domain::DataField<float_type, Dim> source_field;
    source_field.initialize(*source_block);
    
    // Initialize to zero
    for(int k = 0; k < N; ++k) {
        for(int j = 0; j < N; ++j) {
            for(int i = 0; i < N; ++i) {
                source_field.get_real_local(i, j, k) = 0.0;
            }
        }
    }
    
    // Delta function at arbitrary point
    int cx = N / 3, cy = N / 4, cz = N / 2;
    source_field.get_real_local(cx, cy, cz) = 1.0;
    
    // Apply GPU convolution
    conv_gpu->fft_backward_field_clean();
    int level_diff = 0;
    
    ASSERT_NO_THROW(conv_gpu->apply_forward_add(*kernel_block, &lgf, level_diff, source_field));
    
    EXPECT_GT(conv_gpu->number_fwrd_executed, 0);
}

TEST_F(LGFConvolutionGpu3DTest, RoundTripConvolutionGpu) {
    // Test: Forward GPU convolution followed by backward transform
    domain::DataField<float_type, Dim> source_field, result_field;
    source_field.initialize(*source_block);
    result_field.initialize(*source_block);
    
    // Create simple source field
    for(int k = 0; k < N; ++k) {
        for(int j = 0; j < N; ++j) {
            for(int i = 0; i < N; ++i) {
                source_field.get_real_local(i, j, k) = 
                    std::sin(2.0 * M_PI * i / N) * std::cos(2.0 * M_PI * j / N);
                result_field.get_real_local(i, j, k) = 0.0;
            }
        }
    }
    
    // Apply forward and backward transforms on GPU
    conv_gpu->fft_backward_field_clean();
    int level_diff = 0;
    
    conv_gpu->apply_forward_add(*kernel_block, &lgf, level_diff, source_field);
    conv_gpu->apply_backward(*source_block, result_field, 1.0);
    
    // Verify result is non-trivial
    double sum = 0.0;
    double max_val = 0.0;
    for(int k = 0; k < N; ++k) {
        for(int j = 0; j < N; ++j) {
            for(int i = 0; i < N; ++i) {
                float_type val = result_field.get_real_local(i, j, k);
                sum += std::abs(val);
                max_val = std::max(max_val, std::abs(val));
            }
        }
    }
    
    EXPECT_GT(sum, 0.0) << "GPU result field should have non-zero values";
    EXPECT_GT(max_val, 0.0) << "GPU result field should have non-zero maximum";
}

// ============================================================================
// GPU Accuracy Tests (Comparison with LGF)
// ============================================================================

TEST_F(LGFConvolutionGpu3DTest, AccuracyPointSourceAtOriginGpu) {
    // Test: Compare GPU convolution result with direct LGF evaluation
    domain::DataField<float_type, Dim> source_field, result_field;
    source_field.initialize(*source_block);
    result_field.initialize(*source_block);
    
    // Initialize to zero
    for(int k = 0; k < N; ++k) {
        for(int j = 0; j < N; ++j) {
            for(int i = 0; i < N; ++i) {
                source_field.get_real_local(i, j, k) = 0.0;
                result_field.get_real_local(i, j, k) = 0.0;
            }
        }
    }
    
    // Delta function at origin
    source_field.get_real_local(0, 0, 0) = 1.0;
    
    // Apply GPU convolution
    conv_gpu->fft_backward_field_clean();
    int level_diff = 0;
    conv_gpu->apply_forward_add(*kernel_block, &lgf, level_diff, source_field);
    conv_gpu->apply_backward(*source_block, result_field, 1.0);
    
    // Compare with direct LGF evaluation
    double l2_error = 0.0;
    double linf_error = 0.0;
    
    for(int k = 0; k < N; ++k) {
        for(int j = 0; j < N; ++j) {
            for(int i = 0; i < N; ++i) {
                dims_t coord;
                coord[0] = i; coord[1] = j; coord[2] = k;
                
                float_type numerical = result_field.get_real_local(i, j, k);
                float_type lgf_val = lgf.get(coord);
                float_type error = std::abs(numerical - lgf_val);
                
                l2_error += error * error;
                linf_error = std::max(linf_error, error);
            }
        }
    }
    
    l2_error = std::sqrt(l2_error / (N * N * N));
    
    // Verify GPU error is reasonable
    EXPECT_LT(l2_error, 1e-5) << "GPU L2 error should be small for LGF accuracy";
    EXPECT_LT(linf_error, 1e-4) << "GPU L-infinity error should be small";
}

TEST_F(LGFConvolutionGpu3DTest, AccuracyOffsetPointSourceGpu) {
    // Test: GPU accuracy with delta function at non-origin location
    domain::DataField<float_type, Dim> source_field, result_field;
    source_field.initialize(*source_block);
    result_field.initialize(*source_block);
    
    // Initialize to zero
    for(int k = 0; k < N; ++k) {
        for(int j = 0; j < N; ++j) {
            for(int i = 0; i < N; ++i) {
                source_field.get_real_local(i, j, k) = 0.0;
                result_field.get_real_local(i, j, k) = 0.0;
            }
        }
    }
    
    // Delta function at offset location
    int cx = N / 3, cy = N / 4, cz = N / 2;
    source_field.get_real_local(cx, cy, cz) = 1.0;
    
    // Apply GPU convolution
    conv_gpu->fft_backward_field_clean();
    int level_diff = 0;
    conv_gpu->apply_forward_add(*kernel_block, &lgf, level_diff, source_field);
    conv_gpu->apply_backward(*source_block, result_field, 1.0);
    
    // Compare with shifted LGF evaluation
    double l2_error = 0.0;
    int count = 0;
    
    for(int k = 0; k < N; ++k) {
        for(int j = 0; j < N; ++j) {
            for(int i = 0; i < N; ++i) {
                dims_t coord;
                coord[0] = i - cx;
                coord[1] = j - cy;
                coord[2] = k - cz;
                
                float_type numerical = result_field.get_real_local(i, j, k);
                float_type lgf_val = lgf.get(coord);
                float_type error = std::abs(numerical - lgf_val);
                
                l2_error += error * error;
                count++;
            }
        }
    }
    
    l2_error = std::sqrt(l2_error / count);
    
    EXPECT_LT(l2_error, 1e-5) << "GPU L2 error with offset source should be small";
}

// ============================================================================
// GPU Superposition Tests
// ============================================================================

TEST_F(LGFConvolutionGpu3DTest, LinearSuperpositionGpu) {
    // Test: GPU convolution with multiple point sources (linearity)
    domain::DataField<float_type, Dim> source_field, result_field;
    source_field.initialize(*source_block);
    result_field.initialize(*source_block);
    
    // Initialize to zero
    for(int k = 0; k < N; ++k) {
        for(int j = 0; j < N; ++j) {
            for(int i = 0; i < N; ++i) {
                source_field.get_real_local(i, j, k) = 0.0;
                result_field.get_real_local(i, j, k) = 0.0;
            }
        }
    }
    
    // Two point sources with different magnitudes
    float_type mag1 = 2.5, mag2 = -1.3;
    int x1 = N / 4, y1 = N / 4, z1 = N / 4;
    int x2 = 3 * N / 4, y2 = N / 2, z2 = 3 * N / 4;
    
    source_field.get_real_local(x1, y1, z1) = mag1;
    source_field.get_real_local(x2, y2, z2) = mag2;
    
    // Apply GPU convolution
    conv_gpu->fft_backward_field_clean();
    int level_diff = 0;
    conv_gpu->apply_forward_add(*kernel_block, &lgf, level_diff, source_field);
    conv_gpu->apply_backward(*source_block, result_field, 1.0);
    
    // Verify results using linearity
    double l2_error = 0.0;
    int count = 0;
    
    for(int k = 0; k < N; ++k) {
        for(int j = 0; j < N; ++j) {
            for(int i = 0; i < N; ++i) {
                dims_t coord1, coord2;
                coord1[0] = i - x1; coord1[1] = j - y1; coord1[2] = k - z1;
                coord2[0] = i - x2; coord2[1] = j - y2; coord2[2] = k - z2;
                
                float_type numerical = result_field.get_real_local(i, j, k);
                float_type expected = mag1 * lgf.get(coord1) + mag2 * lgf.get(coord2);
                float_type error = std::abs(numerical - expected);
                
                l2_error += error * error;
                count++;
            }
        }
    }
    
    l2_error = std::sqrt(l2_error / count);
    
    EXPECT_LT(l2_error, 1e-4) << "GPU superposition error should be small";
}

TEST_F(LGFConvolutionGpu3DTest, UniformSourceGpu) {
    // Test: GPU convolution with uniform source field
    domain::DataField<float_type, Dim> source_field, result_field;
    source_field.initialize(*source_block);
    result_field.initialize(*source_block);
    
    // Initialize uniform source
    float_type source_value = 1.5;
    for(int k = 0; k < N; ++k) {
        for(int j = 0; j < N; ++j) {
            for(int i = 0; i < N; ++i) {
                source_field.get_real_local(i, j, k) = source_value;
                result_field.get_real_local(i, j, k) = 0.0;
            }
        }
    }
    
    // Apply GPU convolution
    conv_gpu->fft_backward_field_clean();
    int level_diff = 0;
    
    conv_gpu->apply_forward_add(*kernel_block, &lgf, level_diff, source_field);
    conv_gpu->apply_backward(*source_block, result_field, 1.0);
    
    // Verify result is non-trivial and symmetric
    double sum = 0.0;
    for(int k = 0; k < N; ++k) {
        for(int j = 0; j < N; ++j) {
            for(int i = 0; i < N; ++i) {
                sum += result_field.get_real_local(i, j, k);
            }
        }
    }
    
    EXPECT_GT(std::abs(sum), 0.0) << "GPU uniform source should produce non-zero result";
}

TEST_F(LGFConvolutionGpu3DTest, ZeroSourceGpu) {
    // Test: GPU convolution with zero source field
    domain::DataField<float_type, Dim> source_field, result_field;
    source_field.initialize(*source_block);
    result_field.initialize(*source_block);
    
    // Initialize to zero
    for(int k = 0; k < N; ++k) {
        for(int j = 0; j < N; ++j) {
            for(int i = 0; i < N; ++i) {
                source_field.get_real_local(i, j, k) = 0.0;
                result_field.get_real_local(i, j, k) = 0.0;
            }
        }
    }
    
    // Apply GPU convolution
    conv_gpu->fft_backward_field_clean();
    int level_diff = 0;
    
    conv_gpu->apply_forward_add(*kernel_block, &lgf, level_diff, source_field);
    conv_gpu->apply_backward(*source_block, result_field, 1.0);
    
    // Result should remain zero (or very small due to numerical errors)
    double max_val = 0.0;
    for(int k = 0; k < N; ++k) {
        for(int j = 0; j < N; ++j) {
            for(int i = 0; i < N; ++i) {
                max_val = std::max(max_val, std::abs(result_field.get_real_local(i, j, k)));
            }
        }
    }
    
    EXPECT_LT(max_val, 1e-10) << "GPU zero source should produce near-zero result";
}

TEST_F(LGFConvolutionGpu3DTest, ScalingPropertyGpu) {
    // Test: GPU convolution with scaled source field
    domain::DataField<float_type, Dim> source_field1, source_field2, result_field1, result_field2;
    source_field1.initialize(*source_block);
    source_field2.initialize(*source_block);
    result_field1.initialize(*source_block);
    result_field2.initialize(*source_block);
    
    // Create source field
    for(int k = 0; k < N; ++k) {
        for(int j = 0; j < N; ++j) {
            for(int i = 0; i < N; ++i) {
                float_type val = std::sin(2.0 * M_PI * i / N);
                source_field1.get_real_local(i, j, k) = val;
                source_field2.get_real_local(i, j, k) = 2.5 * val;
                result_field1.get_real_local(i, j, k) = 0.0;
                result_field2.get_real_local(i, j, k) = 0.0;
            }
        }
    }
    
    // Apply GPU convolution to first field
    conv_gpu->fft_backward_field_clean();
    int level_diff = 0;
    conv_gpu->apply_forward_add(*kernel_block, &lgf, level_diff, source_field1);
    conv_gpu->apply_backward(*source_block, result_field1, 1.0);
    
    // Apply GPU convolution to scaled field
    conv_gpu->fft_backward_field_clean();
    conv_gpu->apply_forward_add(*kernel_block, &lgf, level_diff, source_field2);
    conv_gpu->apply_backward(*source_block, result_field2, 1.0);
    
    // Verify scaling property: result2 ≈ 2.5 * result1
    double max_error = 0.0;
    for(int k = 0; k < N; ++k) {
        for(int j = 0; j < N; ++j) {
            for(int i = 0; i < N; ++i) {
                float_type val1 = result_field1.get_real_local(i, j, k);
                float_type val2 = result_field2.get_real_local(i, j, k);
                float_type error = std::abs(val2 - 2.5 * val1);
                max_error = std::max(max_error, error);
            }
        }
    }
    
    EXPECT_LT(max_error, 1e-8) << "GPU scaling property should hold";
}

// ============================================================================
// GPU Batch Processing Tests
// ============================================================================

TEST_F(LGFConvolutionGpu3DTest, BatchProcessingGpu) {
    // Test: Multiple applications before backward transform (batch accumulation)
    domain::DataField<float_type, Dim> source_field, result_field;
    source_field.initialize(*source_block);
    result_field.initialize(*source_block);
    
    // Initialize
    for(int k = 0; k < N; ++k) {
        for(int j = 0; j < N; ++j) {
            for(int i = 0; i < N; ++i) {
                source_field.get_real_local(i, j, k) = 0.0;
                result_field.get_real_local(i, j, k) = 0.0;
            }
        }
    }
    
    // Create multiple point sources
    source_field.get_real_local(N/4, N/4, N/4) = 1.0;
    source_field.get_real_local(3*N/4, N/2, N/2) = 1.0;
    source_field.get_real_local(N/2, 3*N/4, 3*N/4) = 1.0;
    
    conv_gpu->fft_backward_field_clean();
    int level_diff = 0;
    
    // Apply multiple forward transforms (should batch on GPU)
    for(int iter = 0; iter < 3; ++iter) {
        conv_gpu->apply_forward_add(*kernel_block, &lgf, level_diff, source_field);
    }
    
    // Single backward transform
    conv_gpu->apply_backward(*source_block, result_field, 1.0);
    
    // Verify batch was processed
    EXPECT_EQ(conv_gpu->number_fwrd_executed, 3) << "GPU should track batch size";
}

// ============================================================================
// GPU Performance and Resource Tests
// ============================================================================

TEST_F(LGFConvolutionGpu3DTest, CleanFieldResetGpu) {
    // Test: GPU field cleaning resets counters correctly
    domain::DataField<float_type, Dim> source_field;
    source_field.initialize(*source_block);
    
    for(int k = 0; k < N; ++k) {
        for(int j = 0; j < N; ++j) {
            for(int i = 0; i < N; ++i) {
                source_field.get_real_local(i, j, k) = 1.0;
            }
        }
    }
    
    // Apply some operations
    conv_gpu->fft_backward_field_clean();
    int level_diff = 0;
    conv_gpu->apply_forward_add(*kernel_block, &lgf, level_diff, source_field);
    
    EXPECT_GT(conv_gpu->number_fwrd_executed, 0);
    
    // Clean should reset
    conv_gpu->fft_backward_field_clean();
    EXPECT_EQ(conv_gpu->number_fwrd_executed, 0) << "GPU clean should reset counter";
    EXPECT_EQ(conv_gpu->current_batch_size_, 0) << "GPU clean should reset batch size";
}

TEST_F(LGFConvolutionGpu3DTest, LargerProblemSizeGpu) {
    // Test: GPU handling of larger problem size
    int N_large = 32;
    dims_t dims_large;
    dims_large[0] = N_large; dims_large[1] = N_large; dims_large[2] = N_large;
    
    coordinate_t lgf_base_large(-N_large + 1);
    coordinate_t base_large(0);
    coordinate_t extent_large(2*N_large - 1);
    
    blk_d_t kernel_block_large(lgf_base_large, extent_large);
    blk_d_t source_block_large(base_large, N_large);
    
    Convolution_GPU<Dim> conv_large(dims_large, dims_large);
    
    domain::DataField<float_type, Dim> source_field;
    source_field.initialize(source_block_large);
    
    // Delta at origin
    for(int k = 0; k < N_large; ++k) {
        for(int j = 0; j < N_large; ++j) {
            for(int i = 0; i < N_large; ++i) {
                source_field.get_real_local(i, j, k) = 0.0;
            }
        }
    }
    source_field.get_real_local(0, 0, 0) = 1.0;
    
    conv_large.fft_backward_field_clean();
    int level_diff = 0;
    conv_large.apply_forward_add(kernel_block_large, &lgf, level_diff, source_field);
}
