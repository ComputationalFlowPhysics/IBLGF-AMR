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
// Convolution Tests
// ============================================================================

template <int Dim>
class ConvolutionTest : public ::testing::Test {
protected:
    using dims_t = types::vector_type<int, Dim>;
};

using ConvolutionTestTypes = ::testing::Types<
    std::integral_constant<int, 2>,
    std::integral_constant<int, 3>
>;

// Since we can't easily instantiate typed tests with dimension template parameter,
// we'll test 2D and 3D separately

class Convolution2DTest : public ::testing::Test {
protected:
    void SetUp() override {
        dims0[0] = 8; dims0[1] = 8;
        dims1[0] = 4; dims1[1] = 4;
        conv = std::make_unique<Convolution<2>>(dims0, dims1);
    }

    types::vector_type<int, 2> dims0;
    types::vector_type<int, 2> dims1;
    std::unique_ptr<Convolution<2>> conv;
};

class Convolution3DTest : public ::testing::Test {
protected:
    void SetUp() override {
        dims0[0] = 8; dims0[1] = 8; dims0[2] = 8;
        dims1[0] = 4; dims1[1] = 4; dims1[2] = 4;
        conv = std::make_unique<Convolution<3>>(dims0, dims1);
    }

    types::vector_type<int, 3> dims0;
    types::vector_type<int, 3> dims1;
    std::unique_ptr<Convolution<3>> conv;
};

TEST_F(Convolution2DTest, ConstructorInitializesCorrectly) {
    EXPECT_NO_THROW({
        Convolution<2> c(dims0, dims1);
    });
}

TEST_F(Convolution3DTest, ConstructorInitializesCorrectly) {
    EXPECT_NO_THROW({
        Convolution<3> c(dims0, dims1);
    });
}

TEST_F(Convolution2DTest, HelperNextPow2) {
    types::vector_type<int, 2> test_dims;
    test_dims[0] = 7; test_dims[1] = 9;
    auto result = conv->helper_next_pow_2(test_dims);
    
    // The function currently just copies values
    // (real next_pow_2 function would be in math utilities)
    EXPECT_EQ(result[0], 7);
    EXPECT_EQ(result[1], 9);
}

TEST_F(Convolution3DTest, HelperNextPow2) {
    types::vector_type<int, 3> test_dims;
    test_dims[0] = 7; test_dims[1] = 9; test_dims[2] = 5;
    auto result = conv->helper_next_pow_2(test_dims);
    
    EXPECT_EQ(result[0], 7);
    EXPECT_EQ(result[1], 9);
    EXPECT_EQ(result[2], 5);
}
TEST_F(Convolution2DTest, HelperAllProd) {
    types::vector_type<int, 2> test_dims;
    test_dims[0] = 4; test_dims[1] = 5;
    auto result = conv->helper_all_prod(test_dims);
    
    EXPECT_EQ(result, 20);
}

TEST_F(Convolution3DTest, HelperAllProd) {
    types::vector_type<int, 3> test_dims;
    test_dims[0] = 2; test_dims[1] = 3; test_dims[2] = 4;
    auto result = conv->helper_all_prod(test_dims);
    
    EXPECT_EQ(result, 24);
}

TEST_F(Convolution2DTest, FftBackwardFieldClean) {
    conv->number_fwrd_executed = 5;
    
    EXPECT_NO_THROW({
        conv->fft_backward_field_clean();
    });
    
    EXPECT_EQ(conv->number_fwrd_executed, 0);
}

TEST_F(Convolution3DTest, FftBackwardFieldClean) {
    conv->number_fwrd_executed = 3;
    
    EXPECT_NO_THROW({
        conv->fft_backward_field_clean();
    });
    
    EXPECT_EQ(conv->number_fwrd_executed, 0);
}

TEST_F(Convolution2DTest, SimdProdComplexAdd) {
    size_t size = 16;
    Convolution<2>::complex_vector_t a(size, std::complex<double>(2.0, 1.0));
    Convolution<2>::complex_vector_t b(size, std::complex<double>(3.0, 2.0));
    Convolution<2>::complex_vector_t res(size, std::complex<double>(0.0, 0.0));
    
    conv->simd_prod_complex_add(a, b, res);
    
    // Expected: (2+i) * (3+2i) = 6 + 4i + 3i + 2i^2 = 6 + 7i - 2 = 4 + 7i
    std::complex<double> expected(4.0, 7.0);
    
    for (size_t i = 0; i < size; ++i) {
        EXPECT_NEAR(res[i].real(), expected.real(), 1e-10);
        EXPECT_NEAR(res[i].imag(), expected.imag(), 1e-10);
    }
}

TEST_F(Convolution3DTest, SimdProdComplexAdd) {
    size_t size = 32;
    Convolution<3>::complex_vector_t a(size, std::complex<double>(1.0, 0.5));
    Convolution<3>::complex_vector_t b(size, std::complex<double>(2.0, 1.0));
    Convolution<3>::complex_vector_t res(size, std::complex<double>(1.0, 1.0));
    
    conv->simd_prod_complex_add(a, b, res);
    
    // Expected: res = (1+i) + (1+0.5i) * (2+i)
    // (1+0.5i) * (2+i) = 2 + i + i + 0.5i^2 = 2 + 2i - 0.5 = 1.5 + 2i
    // res = (1+i) + (1.5+2i) = 2.5 + 3i
    std::complex<double> expected(2.5, 3.0);
    
    for (size_t i = 0; i < size; ++i) {
        EXPECT_NEAR(res[i].real(), expected.real(), 1e-10);
        EXPECT_NEAR(res[i].imag(), expected.imag(), 1e-10);
    }
}

TEST_F(Convolution2DTest, OutputAccessor) {
    auto& output = conv->output();
    EXPECT_GT(output.size(), 0);
}

TEST_F(Convolution3DTest, OutputAccessor) {
    auto& output = conv->output();
    EXPECT_GT(output.size(), 0);
}

// ============================================================================
// Edge Cases and Error Handling
// ============================================================================

TEST(ConvolutionEdgeCases, SmallDimensions) {
    types::vector_type<int, 2> small_dims;
    small_dims[0] = 2; small_dims[1] = 2;
    
    EXPECT_NO_THROW({
        Convolution<2> conv(small_dims, small_dims);
    });
}

TEST(ConvolutionEdgeCases, ZeroInitialization) {
    types::vector_type<int, 2> dims;
    dims[0] = 8; dims[1] = 8;
    Convolution<2> conv(dims, dims);
    
    // After clean, everything should be zeroed
    conv.fft_backward_field_clean();
    EXPECT_EQ(conv.number_fwrd_executed, 0);
}

