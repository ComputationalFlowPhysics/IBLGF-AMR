// #include <vector>
// #include <cmath>
// #include <iostream>
// #include <iblgf/types.hpp>
// #include <iblgf/lgf/lgf_gl.hpp>

// #include <iblgf/utilities/convolution.hpp>
// #include <iblgf/domain/dataFields/blockDescriptor.hpp>

// using namespace iblgf;

// int main()
// {
//     static constexpr std::size_t Dim = 3;
//     using float_type = types::float_type;
//     using dims_t = types::vector_type<int, Dim>;
//     using blk_d_t= domain::BlockDescriptor<int, Dim>;
//     using coordinate_type=blk_d_t::coordinate_type;
//     using k_type=typename lgf::LGF_GL<Dim>;
//     const int N = 8;
//     const float_type L=2.0*M_PI;
//     dims_t dims(N);
//     std::cout<<dims[0]<<" "<<dims[1]<<" "<<dims[2]<<std::endl;
    
//     fft::Convolution<Dim> conv(dims, dims);
    
//     std::cout << "Convolution object created\n"<<std::endl;
//     coordinate_type base(0);
//     coordinate_type extent(2*N-1);
//     blk_d_t kernel_block(base, extent);
//     blk_d_t source_block(base, N);
//     domain::DataField<float_type, Dim> source_field;
//     source_field.initialize(source_block);
//     std::cout << "DataField object created\n"<<std::endl;
//     k_type lgf;
//     int level_diff=0;
//     std::cout<<"LGF object created\n"<<std::endl;
//     std::cout<<lgf.neighbor_only()<<std::endl;
//     std::cout << "BlockDescriptor object created\n"<<std::endl;
//     // conv.fft_backward_field_clean();
//     for(int k=0;k<N;k++)
//     for(int j=0;j<N;j++)
//         for(int i=0;i<N;i++)
//         {
//             source_field.get_real_local(i,j,k) = std::sin(i) + std::cos(j) + std::sin(k);
//         }
//     conv.fft_backward_field_clean();
//     conv.apply_forward_add(kernel_block, &lgf, level_diff, source_field);
//     domain::DataField<float_type, Dim> result_field;
//     result_field.initialize(source_block);
//     conv.apply_backward(source_block,result_field,1.0);
//     return 0;
    
// }
#include <vector>
#include <cmath>
#include <iostream>
#include <iblgf/types.hpp>
#include <iblgf/utilities/convolution.hpp>
#include <iblgf/domain/dataFields/blockDescriptor.hpp>
#include <iblgf/lgf/lgf_gl.hpp>
#include <chrono>

using namespace iblgf;

int main()
{
    static constexpr std::size_t Dim = 3;
    using float_type = types::float_type;
    using dims_t = types::vector_type<int, Dim>;
    using blk_d_t = domain::BlockDescriptor<int, Dim>;
    using coordinate_t = blk_d_t::coordinate_type;
    using k_type = lgf::LGF_GL<Dim>;

    const int N = 16;             // domain size
    dims_t dims(N);                // cube domain
    auto start = std::chrono::high_resolution_clock::now();

    fft::Convolution<Dim> conv(dims, dims);
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
    std::cout << "Elapsed time: " << elapsed.count() << " seconds\n";
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
    // --- Compare numerical result with analytic 1/(4*pi*r) ---
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
