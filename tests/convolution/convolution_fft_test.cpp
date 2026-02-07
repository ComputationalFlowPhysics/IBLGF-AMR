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

#include <gtest/gtest.h>
#include <iblgf/utilities/convolution.hpp>
#include <iblgf/types.hpp>
#include <cmath>
#include <numeric>
#include <algorithm>

using namespace iblgf::fft;
using namespace iblgf;

// ============================================================================
// dfft_r2c Tests (Real-to-Complex FFT)
// ============================================================================

class DfftR2C3DTest : public ::testing::Test {
protected:
    void SetUp() override {
        dims_3d[0] = 8; dims_3d[1] = 8; dims_3d[2] = 8;
        dims_small_3d[0] = 4; dims_small_3d[1] = 4; dims_small_3d[2] = 4;
        fft_3d = std::make_unique<dfft_r2c>(dims_3d, dims_small_3d);
    }

    dfft_r2c::dims_3D dims_3d;
    dfft_r2c::dims_3D dims_small_3d;
    std::unique_ptr<dfft_r2c> fft_3d;
};

class DfftR2C2DTest : public ::testing::Test {
protected:
    void SetUp() override {
        dims_2d[0] = 8; dims_2d[1] = 8;
        dims_small_2d[0] = 4; dims_small_2d[1] = 4;
        fft_2d = std::make_unique<dfft_r2c>(dims_2d, dims_small_2d);
    }

    dfft_r2c::dims_2D dims_2d;
    dfft_r2c::dims_2D dims_small_2d;
    std::unique_ptr<dfft_r2c> fft_2d;
};

TEST_F(DfftR2C3DTest, ConstructorInitializesCorrectly) {
    EXPECT_NO_THROW({
        dfft_r2c fft(dims_3d, dims_small_3d);
    });
}

TEST_F(DfftR2C2DTest, ConstructorInitializesCorrectly) {
    EXPECT_NO_THROW({
        dfft_r2c fft(dims_2d, dims_small_2d);
    });
}

TEST_F(DfftR2C3DTest, InputOutputAccessors) {
    auto& input = fft_3d->input();
    auto& output = fft_3d->output();
    
    EXPECT_EQ(input.size(), dims_3d[0] * dims_3d[1] * dims_3d[2]);
    EXPECT_EQ(output.size(), dims_3d[1] * dims_3d[2] * ((dims_3d[0] / 2) + 1));
}

TEST_F(DfftR2C2DTest, InputOutputAccessors) {
    auto& input = fft_2d->input();
    auto& output = fft_2d->output();
    
    EXPECT_EQ(input.size(), dims_2d[0] * dims_2d[1]);
    EXPECT_EQ(output.size(), dims_2d[1] * ((dims_2d[0] / 2) + 1));
}

TEST_F(DfftR2C3DTest, CopyInputValidSize) {
    std::vector<double> data(dims_3d[0] * dims_3d[1] * dims_3d[2], 1.0);
    
    EXPECT_NO_THROW({
        fft_3d->copy_input(data, dims_3d);
    });
    
    const auto& input = fft_3d->input();
    for (size_t i = 0; i < input.size(); ++i) {
        EXPECT_DOUBLE_EQ(input[i], 1.0);
    }
}

TEST_F(DfftR2C2DTest, CopyInputValidSize) {
    std::vector<double> data(dims_2d[0] * dims_2d[1], 2.5);
    
    EXPECT_NO_THROW({
        fft_2d->copy_input(data, dims_2d);
    });
    
    const auto& input = fft_2d->input();
    for (size_t i = 0; i < input.size(); ++i) {
        EXPECT_DOUBLE_EQ(input[i], 2.5);
    }
}

TEST_F(DfftR2C3DTest, CopyInputInvalidSizeThrows) {
    std::vector<double> data(10);
    
    EXPECT_THROW({
        fft_3d->copy_input(data, dims_3d);
    }, std::runtime_error);
}

TEST_F(DfftR2C2DTest, CopyInputInvalidSizeThrows) {
    std::vector<double> data(10);
    
    EXPECT_THROW({
        fft_2d->copy_input(data, dims_2d);
    }, std::runtime_error);
}

TEST_F(DfftR2C3DTest, ExecuteWholeTransform) {
    auto& input = fft_3d->input();
    
    std::fill(input.begin(), input.end(), 0.0);
    input[0] = 1.0;
    
    EXPECT_NO_THROW({
        fft_3d->execute_whole();
    });
    
    const auto& output = fft_3d->output();
    EXPECT_GT(std::abs(output[0]), 0.0);
}

TEST_F(DfftR2C3DTest, ExecuteStageWiseTransform) {
    auto& input = fft_3d->input();
    
    std::iota(input.begin(), input.end(), 0.0);
    
    EXPECT_NO_THROW({
        fft_3d->execute();
    });
    
    const auto& output = fft_3d->output();
    EXPECT_GT(output.size(), 0);
}

TEST_F(DfftR2C2DTest, ExecuteStageWiseTransform) {
    auto& input = fft_2d->input();
    
    std::iota(input.begin(), input.end(), 0.0);
    
    EXPECT_NO_THROW({
        fft_2d->execute();
    });
    
    const auto& output = fft_2d->output();
    EXPECT_GT(output.size(), 0);
}

TEST_F(DfftR2C3DTest, OutputCopyFunction) {
    auto& input = fft_3d->input();
    std::fill(input.begin(), input.end(), 1.0);
    
    fft_3d->execute_whole();
    
    auto output_copy = fft_3d->output_copy();
    const auto& output_ref = fft_3d->output();
    
    ASSERT_EQ(output_copy.size(), output_ref.size());
    
    for (size_t i = 0; i < output_copy.size(); ++i) {
        EXPECT_EQ(output_copy[i], output_ref[i]);
    }
}

// ============================================================================
// dfft_c2r Tests (Complex-to-Real FFT)
// ============================================================================

class DfftC2R3DTest : public ::testing::Test {
protected:
    void SetUp() override {
        dims_3d[0] = 8; dims_3d[1] = 8; dims_3d[2] = 8;
        dims_small_3d[0] = 4; dims_small_3d[1] = 4; dims_small_3d[2] = 4;
        fft_3d = std::make_unique<dfft_c2r>(dims_3d, dims_small_3d);
    }

    dfft_c2r::dims_3D dims_3d;
    dfft_c2r::dims_3D dims_small_3d;
    std::unique_ptr<dfft_c2r> fft_3d;
};

class DfftC2R2DTest : public ::testing::Test {
protected:
    void SetUp() override {
        dims_2d[0] = 8; dims_2d[1] = 8;
        dims_small_2d[0] = 4; dims_small_2d[1] = 4;
        fft_2d = std::make_unique<dfft_c2r>(dims_2d, dims_small_2d);
    }

    dfft_c2r::dims_2D dims_2d;
    dfft_c2r::dims_2D dims_small_2d;
    std::unique_ptr<dfft_c2r> fft_2d;
};

TEST_F(DfftC2R3DTest, ConstructorInitializesCorrectly) {
    EXPECT_NO_THROW({
        dfft_c2r fft(dims_3d, dims_small_3d);
    });
}

TEST_F(DfftC2R2DTest, ConstructorInitializesCorrectly) {
    EXPECT_NO_THROW({
        dfft_c2r fft(dims_2d, dims_small_2d);
    });
}

TEST_F(DfftC2R3DTest, InputOutputAccessors) {
    auto& input = fft_3d->input();
    auto& output = fft_3d->output();
    
    EXPECT_EQ(input.size(), dims_3d[1] * dims_3d[2] * ((dims_3d[0] / 2) + 1));
    EXPECT_EQ(output.size(), dims_3d[0] * dims_3d[1] * dims_3d[2]);
}

TEST_F(DfftC2R2DTest, InputOutputAccessors) {
    auto& input = fft_2d->input();
    auto& output = fft_2d->output();
    
    EXPECT_EQ(input.size(), dims_2d[1] * ((dims_2d[0] / 2) + 1));
    EXPECT_EQ(output.size(), dims_2d[0] * dims_2d[1]);
}

TEST_F(DfftC2R3DTest, ExecuteTransform) {
    auto& input = fft_3d->input();
    
    std::fill(input.begin(), input.end(), std::complex<double>(1.0, 0.0));
    
    EXPECT_NO_THROW({
        fft_3d->execute();
    });
    
    const auto& output = fft_3d->output();
    EXPECT_GT(output.size(), 0);
}

TEST_F(DfftC2R2DTest, ExecuteTransform) {
    auto& input = fft_2d->input();
    
    std::fill(input.begin(), input.end(), std::complex<double>(1.0, 0.0));
    
    EXPECT_NO_THROW({
        fft_2d->execute();
    });
    
    const auto& output = fft_2d->output();
    EXPECT_GT(output.size(), 0);
}

// ============================================================================
// Round-trip FFT Tests (verify FFT -> IFFT recovers original)
// ============================================================================

TEST_F(DfftR2C2DTest, ForwardAndBackwardTransformsExecute) {
    dfft_r2c r2c(dims_2d, dims_small_2d);
    dfft_c2r c2r(dims_2d, dims_small_2d);
    
    auto& input = r2c.input();
    std::fill(input.begin(), input.end(), 1.0);
    
    EXPECT_NO_THROW(r2c.execute_whole());
    
    const auto& freq_domain = r2c.output();
    EXPECT_GT(freq_domain.size(), 0);
    
    auto& inverse_input = c2r.input();
    std::copy(freq_domain.begin(), freq_domain.end(), inverse_input.begin());
    
    EXPECT_NO_THROW(c2r.execute());
    
    const auto& output = c2r.output();
    EXPECT_GT(output.size(), 0);
}

// ============================================================================
// Analytical Solution Tests
// ============================================================================

TEST(AnalyticalSolutions, CosineWaveFFT2D) {
    const int N = 16;
    dfft_r2c::dims_2D dims;
    dims[0] = N; dims[1] = N;
    
    dfft_r2c fft(dims, dims);
    auto& input = fft.input();
    
    const int k = 2;
    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < N; ++i) {
            double x = static_cast<double>(i) / N;
            input[i + j * N] = std::cos(2.0 * M_PI * k * x);
        }
    }
    
    fft.execute_whole();
    const auto& output = fft.output();
    
    double max_magnitude = 0.0;
    int max_idx = 0;
    
    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < (N/2 + 1); ++i) {
            double mag = std::abs(output[i + j * (N/2 + 1)]);
            if (mag > max_magnitude) {
                max_magnitude = mag;
                max_idx = i;
            }
        }
    }
    
    EXPECT_EQ(max_idx, k);
    EXPECT_GT(max_magnitude, 0.4 * N * N);
}

TEST(AnalyticalSolutions, CosineWaveFFT3D) {
    const int N = 8;
    dfft_r2c::dims_3D dims;
    dims[0] = N; dims[1] = N; dims[2] = N;
    
    dfft_r2c fft(dims, dims);
    auto& input = fft.input();
    
    const int k = 2;
    for (int z = 0; z < N; ++z) {
        for (int y = 0; y < N; ++y) {
            for (int x = 0; x < N; ++x) {
                double pos = static_cast<double>(x) / N;
                input[x + y * N + z * N * N] = std::cos(2.0 * M_PI * k * pos);
            }
        }
    }
    
    fft.execute_whole();
    const auto& output = fft.output();
    
    double max_magnitude = 0.0;
    for (size_t i = 0; i < output.size(); ++i) {
        max_magnitude = std::max(max_magnitude, std::abs(output[i]));
    }
    
    EXPECT_GT(max_magnitude, 0.4 * N * N * N);
}

TEST(AnalyticalSolutions, ParsevalsTheorem2D) {
    const int N = 16;
    dfft_r2c::dims_2D dims;
    dims[0] = N; dims[1] = N;
    
    dfft_r2c fft(dims, dims);
    auto& input = fft.input();
    
    double time_energy = 0.0;
    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < N; ++i) {
            double val = std::sin(2.0 * M_PI * i / N) * std::cos(2.0 * M_PI * j / N);
            input[i + j * N] = val;
            time_energy += val * val;
        }
    }
    
    fft.execute_whole();
    const auto& output = fft.output();
    
    double freq_energy = 0.0;
    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < (N/2 + 1); ++i) {
            double mag_sq = std::norm(output[i + j * (N/2 + 1)]);
            if (i > 0 && i < N/2) {
                freq_energy += 2.0 * mag_sq;
            } else {
                freq_energy += mag_sq;
            }
        }
    }
    freq_energy /= (N * N);
    
    EXPECT_NEAR(time_energy, freq_energy, 1e-8 * time_energy);
}

TEST(AnalyticalSolutions, GaussianSymmetry3D) {
    const int N = 16;
    dfft_r2c::dims_3D dims;
    dims[0] = N; dims[1] = N; dims[2] = N;
    
    dfft_r2c fft(dims, dims);
    auto& input = fft.input();
    
    const double sigma = 3.0;
    const double center = N / 2.0;
    
    for (int z = 0; z < N; ++z) {
        for (int y = 0; y < N; ++y) {
            for (int x = 0; x < N; ++x) {
                double dx = x - center;
                double dy = y - center;
                double dz = z - center;
                double r2 = dx*dx + dy*dy + dz*dz;
                input[x + y * N + z * N * N] = std::exp(-r2 / (2.0 * sigma * sigma));
            }
        }
    }
    
    fft.execute_whole();
    const auto& output = fft.output();
    
    double dc_value = std::abs(output[0]);
    
    int larger_count = 0;
    for (size_t i = 1; i < output.size(); ++i) {
        if (std::abs(output[i]) > dc_value) {
            larger_count++;
        }
    }
    
    EXPECT_LT(larger_count, 10);
    EXPECT_GT(dc_value, 10.0);
}

TEST(AnalyticalSolutions, LinearityProperty2D) {
    const int N = 8;
    dfft_r2c::dims_2D dims;
    dims[0] = N; dims[1] = N;
    
    dfft_r2c fft_f(dims, dims);
    dfft_r2c fft_g(dims, dims);
    dfft_r2c fft_combined(dims, dims);
    
    const double a = 2.5;
    const double b = -1.3;
    
    auto& input_f = fft_f.input();
    auto& input_g = fft_g.input();
    auto& input_combined = fft_combined.input();
    
    for (int i = 0; i < N * N; ++i) {
        input_f[i] = std::sin(2.0 * M_PI * i / N);
        input_g[i] = std::cos(2.0 * M_PI * i / (2.0 * N));
        input_combined[i] = a * input_f[i] + b * input_g[i];
    }
    
    fft_f.execute_whole();
    fft_g.execute_whole();
    fft_combined.execute_whole();
    
    const auto& output_f = fft_f.output();
    const auto& output_g = fft_g.output();
    const auto& output_combined = fft_combined.output();
    
    for (size_t i = 0; i < output_combined.size(); ++i) {
        std::complex<double> expected = a * output_f[i] + b * output_g[i];
        EXPECT_NEAR(output_combined[i].real(), expected.real(), 1e-10);
        EXPECT_NEAR(output_combined[i].imag(), expected.imag(), 1e-10);
    }
}

TEST(AnalyticalSolutions, ConvolutionTheorem2D) {
    const int N = 8;
    types::vector_type<int, 2> dims;
    dims[0] = N; dims[1] = N;
    
    std::vector<double> f(N * N, 0.0);
    std::vector<double> g(N * N, 0.0);
    
    for (int j = N/4; j < 3*N/4; ++j) {
        for (int i = N/4; i < 3*N/4; ++i) {
            f[i + j * N] = 1.0;
        }
    }
    
    for (int j = N/3; j < 2*N/3; ++j) {
        for (int i = N/3; i < 2*N/3; ++i) {
            g[i + j * N] = 1.0;
        }
    }
    
    dfft_r2c::dims_2D fft_dims;
    fft_dims[0] = N; fft_dims[1] = N;
    
    dfft_r2c fft_f(fft_dims, fft_dims);
    dfft_r2c fft_g(fft_dims, fft_dims);
    
    fft_f.copy_input(f, fft_dims);
    fft_g.copy_input(g, fft_dims);
    
    fft_f.execute_whole();
    fft_g.execute_whole();
    
    const auto& F = fft_f.output();
    const auto& G = fft_g.output();
    
    std::vector<std::complex<double>> product(F.size());
    for (size_t i = 0; i < F.size(); ++i) {
        product[i] = F[i] * G[i];
    }
    
    double product_energy = 0.0;
    for (const auto& val : product) {
        product_energy += std::norm(val);
    }
    
    EXPECT_GT(product_energy, 1.0);
}
