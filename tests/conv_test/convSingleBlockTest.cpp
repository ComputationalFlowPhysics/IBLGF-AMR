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
#include <cmath>
#include <iostream>
#include <algorithm>
#include <chrono>

#include <iblgf/types.hpp>
#include <iblgf/utilities/convolution.hpp>
#include <iblgf/domain/dataFields/blockDescriptor.hpp>
#include <iblgf/domain/dataFields/dataField.hpp>
#include <iblgf/lgf/lgf_gl.hpp>

using namespace iblgf;

TEST(convolution_single_block_test, DeltaConvolutionMatchesLGF)
{
    static constexpr std::size_t Dim = 3;
    using float_type = types::float_type;
    using dims_t = types::vector_type<int, Dim>;
    using blk_d_t = domain::BlockDescriptor<int, Dim>;
    using coordinate_t = blk_d_t::coordinate_type;
    using k_type = lgf::LGF_GL<Dim>;

    const int N = 16;
    dims_t dims(N);

    auto start = std::chrono::high_resolution_clock::now();

    fft::Convolution<Dim> conv(dims, dims);
    conv.fft_backward_field_clean();

    coordinate_t base(0);
    coordinate_t lgf_base(-N + 1);
    coordinate_t kernel_extent(2 * N - 1);

    blk_d_t source_block(base, N);
    blk_d_t kernel_block(lgf_base, kernel_extent);

    domain::DataField<float_type, Dim> source_field;
    source_field.initialize(source_block);

    for (int k = 0; k < N; ++k)
    for (int j = 0; j < N; ++j)
    for (int i = 0; i < N; ++i)
        source_field.get_real_local(i, j, k) = 0.0;

    const int cx = N / 3;
    const int cy = N / 4;
    const int cz = N / 2;
    source_field.get_real_local(cx, cy, cz) = 1.0;

    k_type lgf;
    const int level_diff = 0;

    domain::DataField<float_type, Dim> result_field;
    result_field.initialize(source_block);

    conv.apply_forward_add(kernel_block, &lgf, level_diff, source_field);
    conv.apply_backward(source_block, result_field, 1.0);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Elapsed time: " << elapsed.count() << " seconds\n";

    float_type l2_error = 0.0;
    float_type linf_error = 0.0;

    for (int k = 0; k < N; ++k)
    for (int j = 0; j < N; ++j)
    for (int i = 0; i < N; ++i)
    {
        coordinate_t coord(0);
        coord[0] = i - cx;
        coord[1] = j - cy;
        coord[2] = k - cz;

        float_type numerical = result_field.get_real_local(i, j, k);
        float_type exact = lgf.get(coord);
        float_type error = std::abs(numerical - exact);

        l2_error += error * error;
        linf_error = std::max(linf_error, error);
    }

    l2_error = std::sqrt(l2_error / static_cast<float_type>(N * N * N));

    EXPECT_LT(l2_error, 1e-8);
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
