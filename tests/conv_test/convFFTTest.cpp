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

#include <mpi.h>
#include <gtest/gtest.h>

#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <memory>

#include <vector>
#include <array>
#include <map>
#include <memory>
#include <cassert>
#include <cstddef>
#include <cmath>
#include <iostream>
#include <iblgf/types.hpp>
#include <iblgf/utilities/convolution.hpp>
#include <iblgf/domain/dataFields/blockDescriptor.hpp>
#include <iblgf/lgf/lgf_gl.hpp>
#include <chrono>

using namespace iblgf;

// class for test kernel (not LGF kernel)
template<std::size_t Dim>
class TestKernel3X3
{
public:
    using float_type = iblgf::types::float_type;
    using dims_t = iblgf::types::vector_type<int, Dim>;
    using block_descriptor_t = iblgf::domain::BlockDescriptor<int, Dim>;

    // Match Convolution's complex vector type
    using conv_t = iblgf::fft::Convolution<Dim>;
    using complex_vector_t = typename conv_t::complex_vector_t;

    // 3x3x3 kernel values in dx,dy,dz in {-1,0,1} flattened to length 27
    explicit TestKernel3X3(std::vector<float_type> kernel_values)
        : K_(std::move(kernel_values))
    {
        assert(K_.size() == 27);
    }

    // Mimic dft function in LGF_BASE to be able to used with Convolution's execute_fwrd_field 
    template<class Convolutor, int Dim1 = (int)Dim>
    auto& dft(const block_descriptor_t& lgf_block,
              dims_t extended_dims,
              Convolutor* conv,
              typename std::enable_if<Dim1 == 3, int>::type level_diff)
    {
        (void)level_diff;

        auto base = lgf_block.base();
        key_t key = std::make_tuple(base[0], base[1], base[2],
                                    extended_dims[0], extended_dims[1], extended_dims[2]);

        auto it = cache_.find(key);
        if (it != cache_.end())
            return *(it->second);

        const int Nx = extended_dims[0];
        const int Ny = extended_dims[1];
        const int Nz = extended_dims[2];

        const std::size_t total =
            (std::size_t)Nx * (std::size_t)Ny * (std::size_t)Nz;

        std::vector<float_type> padded(total, float_type(0));

        // Kernel index flatten: (dx+1)+3(dy+1)+9(dz+1)
        auto kidx = [](int dx, int dy, int dz) {
            return (dx + 1) + 3*(dy + 1) + 9*(dz + 1);
        };

        auto wrap = [](int d, int n) {
            return (d >= 0) ? d : (n + d);
        };

        for (int dz = -1; dz <= 1; ++dz)
        for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx)
        {
            auto base = lgf_block.base(); // should be (-N+1, -N+1, -N+1)
            int ix = dx - base[0];
            int iy = dy - base[1];
            int iz = dz - base[2];

            std::size_t lin =
                (std::size_t)ix
                + (std::size_t)Nx * (std::size_t)iy
                + (std::size_t)Nx * (std::size_t)Ny * (std::size_t)iz;

            padded[lin] = K_[kidx(dx,dy,dz)];
        }

        // Use Convolution's FFT helper 
        auto& spec_ref = conv->dft_r2c(padded);

        auto ins = cache_.emplace(key, std::make_unique<complex_vector_t>(spec_ref));
        return *(ins.first->second);
    }

private:
    std::vector<float_type> K_;

    using key_t = std::tuple<int,int,int,int,int,int>;
    std::map<key_t, std::unique_ptr<complex_vector_t>> cache_;
};

TEST(convolution_fft_test, FFTMatchesNaive)
{
    static constexpr std::size_t Dim = 3;
    using float_type = types::float_type;
    using dims_t = types::vector_type<int, Dim>;
    using blk_d_t = domain::BlockDescriptor<int, Dim>;
    using coordinate_t = blk_d_t::coordinate_type;
    using k_type = std::vector<float_type>; // using custom kernel type

    const int N = 8; // domain size
    dims_t dims(N); // cube domain

    const int R = 1; // kernel radius
    const int K_SIZE = 2*R + 1; //{-1, 0, 1}

    auto start = std::chrono::high_resolution_clock::now(); // start timer

    fft::Convolution<Dim> conv(dims, dims);
    conv.fft_backward_field_clean();

    coordinate_t kernel_base(-N+1);
    coordinate_t kernel_extent(2*N-1);
    blk_d_t kernel_block(kernel_base, kernel_extent);
    blk_d_t source_block(kernel_base, N);

    // make source field
    domain::DataField<float_type, Dim> source_field;
    source_field.initialize(source_block);

    // fill source field with formula with known FFT
    // sin(i) + 0.1 * j - 0.2 * k
    for(int k=0; k<N; ++k)
    for(int j=0; j<N; ++j)
    for(int i=0; i<N; ++i)
        source_field.get_real_local(i,j,k) = std::sin(i) + 0.1 * j - 0.2 * k;

    // make kernel 
    k_type kernel(K_SIZE*K_SIZE*K_SIZE);

    // flatten indices to access kernel elements
    auto k_idx = [K_SIZE, R](int dx, int dy, int dz)
    {
        return (dx + R) + K_SIZE * ((dy + R) + K_SIZE * (dz + R));
    };

    // fill kernel with simple function
    // K(dx, dy, dz) = 0.3 + 0.1*dx + 0.05dy - 0.02*dz
    for (int dz = -1; dz <= 1; ++dz)
    for (int dy = -1; dy <= 1; ++dy)
    for (int dx = -1; dx <= 1; ++dx) {
        kernel[k_idx(dx,dy,dz)] = 0.3 + 0.1*dx + 0.05*dy - 0.02*dz;
    }

    // make center distinct for debugging
    kernel[k_idx(0,0,0)] = 1.2345;

    // convert into format Convolution class can use
    TestKernel3X3<Dim> test_kernel(kernel);

    // apply convolution
    domain::DataField<float_type, Dim> result_field;
    result_field.initialize(source_block);
    result_field = 0.0;
    conv.apply_forward_add(kernel_block, &test_kernel, 0, source_field);
    conv.apply_backward(source_block, result_field, 1.0);

    //end timer and print elapsed time
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    
    // compute naive convolution
    float_type l2_error = 0.0;
    float_type linf_error = 0.0;
    for(int k=0; k<N; ++k)
    for(int j=0; j<N; ++j)
    for(int i=0; i<N; ++i)
    {
        float_type sum = 0.0;

        auto wrapN = [N](int a) {
            a %= N;
            if (a < 0) a += N;
            return a;
        };

        for (int dz=-R; dz<=R; ++dz)
        for (int dy=-R; dy<=R; ++dy)
        for (int dx=-R; dx<=R; ++dx)
        {
            int ii = i - dx;
            int jj = j - dy;
            int kk = k - dz;

            if (ii >= 0 && ii < N && jj >= 0 && jj < N && kk >= 0 && kk < N) 
                sum += kernel[k_idx(dx, dy, dz)] * source_field.get_real_local(ii, jj, kk);
        }
        
        float_type error = std::abs(result_field.get_real_local(i,j,k) - sum);
        l2_error += error * error;
        linf_error = std::max(linf_error, error);
    }

    EXPECT_LT(linf_error, 1e-8);
} 

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);

    int rc = RUN_ALL_TESTS();

    MPI_Finalize();
    return rc;
}
