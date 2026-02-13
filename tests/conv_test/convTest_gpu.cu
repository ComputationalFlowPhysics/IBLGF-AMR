#ifndef IBLGF_COMPILE_CUDA
#define IBLGF_COMPILE_CUDA
#endif

#include <boost/mpi.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <iostream>
#include <vector>
#include <complex>
#include <iblgf/utilities/convolution_GPU.hpp> // includes Convolution_GPU and dfft_r2c_gpu
#include <iblgf/domain/dataFields/blockDescriptor.hpp>
#include <iblgf/lgf/lgf_gl.hpp>
#include <iblgf/simulation.hpp>
#include <iblgf/domain/domain.hpp>
#include <iblgf/types.hpp>
#include <chrono>
using namespace iblgf;

int main(int argc, char *argv[]) {
    // static constexpr std::size_t Dim = 3;
    // using float_type = types::float_type;
    // using dims_t = types::vector_type<int, Dim>;
    // // using blk_d_t = domain::BlockDescriptor<int, Dim>;
    // // using coordinate_t = blk_d_t::coordinate_type;
    // // using k_type = lgf::LGF_GL<Dim>;
    // cudaSetDevice(0); // Set GPU device
    // const int N = 10;             // domain size
    // dims_t dims(N);                // cube domain
    // dims_t dims0(N);
    // // Define dimensions for two small fields

    // // Initialize a simple 3D input field
    // const int N_field=N*2-1;
    // std::vector<float_type> field0(N_field  * N_field * N_field, 0.0);
    // // for (int z = 0; z < N_field; ++z) {
    // //     for (int y = 0; y < N_field; ++y) {
    // //         for (int x = 0; x < N_field; ++x) {
    // //             field0[x + y * N_field + z * N_field * N_field] = static_cast<float_type>(x + y + z);
    // //         }
    // //     }
    // // }  
    // field0[0] = 1.0; 
    // // Construct the Convolution_GPU object
    // fft::Convolution_GPU<Dim> conv(dims, dims);

    // // Perform forward FFT of field0 using Convolution_GPU
    // auto fft_output = conv.dft_r2c(field0);

    // // Print output (complex numbers)
    // std::cout << "FFT output of field0:\n";
    // for (size_t i = 0; i < fft_output.size(); ++i) {
    //     std::cout << fft_output[i] << " ";
    //     // Add a newline every "slice" for readability
    //     if ((i+1) % ((2*N - 1)) == 0) std::cout << "\n";
    // }

    boost::mpi::environment env(argc, argv);
	boost::mpi::communicator world;
    static constexpr std::size_t Dim = 3;
    using float_type = types::float_type;
    using dims_t = types::vector_type<int, Dim>;
    using blk_d_t = domain::BlockDescriptor<int, Dim>;
    using coordinate_t = blk_d_t::coordinate_type;
    using k_type = lgf::LGF_GL<Dim>;
    cudaSetDevice(0); // Set GPU device
    const int N = 16;
    dims_t dims(N);                // cube domain
    auto start = std::chrono::high_resolution_clock::now();
    fft::Convolution_GPU<Dim> conv(dims, dims);
    conv.fft_backward_field_clean();

    coordinate_t lgf_base(-N+1);
    coordinate_t base(0);
    coordinate_t extent(2*N-1);
    blk_d_t kernel_block(lgf_base, extent);
    blk_d_t source_block(base, N);

    domain::DataField<float_type, Dim> source_field;
    source_field.initialize(source_block);

    // --- Single point source at (0,0,0) ---
    int cx = N / 3;
    int cy = N / 4;
    int cz = N / 2;

    for(int k=0; k<N; ++k)
    for(int j=0; j<N; ++j)
    for(int i=0; i<N; ++i)
        source_field.get_real_local(i,j,k) = 0.0;

    source_field.get_real_local(cx, cy, cz) = 1.0;

    

    k_type lgf;
    int level_diff = 0;
    

    domain::DataField<float_type, Dim> result_field;
    result_field.initialize(source_block);
    conv.apply_forward_add(kernel_block, &lgf, level_diff, source_field);
    conv.apply_backward(source_block, result_field, 1.0);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Convolution completed in " << elapsed.count() << " seconds." << std::endl;
    //compute l2 and linf error norms
    float_type l2_error = 0.0;
    float_type linf_error = 0.0;
    for(int k=0; k<N; ++k)
    for(int j=0; j<N; ++j)
    for(int i=0; i<N; ++i)
    {
        dims_t coord(0);
        coord[0] = i - cx;
        coord[1] = j - cy;
        coord[2] = k - cz;
        float_type r = std::sqrt(
            static_cast<float_type>(coord[0] * coord[0] +
                                    coord[1] * coord[1] +
                                    coord[2] * coord[2]));
        float_type analytic = (r == 0.0) ? 0.0 : (-1.0 / (4.0 * M_PI * r));
        float_type numerical = result_field.get_real_local(i, j, k);
        float_type kval=lgf.get(coord);
        float_type error = std::abs(numerical - kval);
        l2_error += error * error;
        linf_error = std::max(linf_error, error);
    }
    l2_error = std::sqrt(l2_error / (N * N * N));
    std::cout << "L2 error norm: " << l2_error << std::endl;
    std::cout << "Linf error norm: " << linf_error << std::endl;

    // // --- Compare numerical result with analytic 1/(-4*pi*r) ---
    // for(int k=0; k<N; ++k)
    // for(int j=0; j<N; ++j)
    // for(int i=0; i<N; ++i)
    // {
    //     dims_t coord(0);
    //     coord[0] = i - cx;
    //     coord[1] = j - cy;
    //     coord[2] = k - cz;
    //     float_type r = std::sqrt(
    //         static_cast<float_type>(coord[0] * coord[0] +
    //                                 coord[1] * coord[1] +
    //                                 coord[2] * coord[2]));
    //     float_type analytic = (r == 0.0) ? 0.0 : (1.0 / (4.0 * M_PI * r));
    //     float_type numerical = result_field.get_real_local(i, j, k);
    //     float_type kval=lgf.get(coord);
    //     std::cout << "Point (" << i << "," << j << "," << k << "): "
    //                 << "Numerical = " << numerical<<", LGF = " << kval
    //                 << ", Analytic = " << analytic << std::endl;
    // }

    return 0;
}
