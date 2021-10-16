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

#ifndef IBLGF_INCLUDED_VORTEXRINGS_HPP
#define IBLGF_INCLUDED_VORTEXRINGS_HPP

//#ifdef IBLGF_VORTEX_RUN_ALL

#define POISSON_TIMINGS

#include <iostream>
#include <chrono>
#include <vector>
#include <complex>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <math.h>

// IBLGF-specific
#include "../../setups/setup_base.hpp"
#include <iblgf/solver/time_integration/HelmholtzFFT.hpp>

namespace iblgf
{
    class Helmholtz_FFT {
    public:
        Helmholtz_FFT(int _padded_dim, int _dim_nonzero, int _dim_0, int _dim_1)
            : dim_0(_dim_0)
            , dim_1(_dim_1)
            , r2cFunc(_padded_dim, _dim_nonzero, _dim_0, _dim_1, 2)
            , c2rFunc(_padded_dim, _dim_nonzero, _dim_0, _dim_1, 2)
            , padded_dim(_padded_dim)
            , dim_nonzero(_dim_nonzero) {
            std::cout << "Constructor done" << std::endl;
        }

        void testingTransform() {
            int num_transform = 2 * dim_0 * dim_1;
            int numCells = dim_0 * dim_1;
            std::vector<double> v(padded_dim * num_transform, 0.0);
            std::cout << "start to transform" << std::endl;
            /*for (int k = 0; k < 2; k++)
            {*/
                for (int i = 0; i < padded_dim; i++)
                {
                    for (int j = 0; j < num_transform; j++)
                    {
                        v[i + j * padded_dim] =
                            std::sin(static_cast<double>(i) /
                                static_cast<double>(padded_dim) * 2.0 *
                                M_PI);
                    }
                    //v[i] = std::cos(static_cast<double>(i)/static_cast<double>(padded_dim)*2.0*M_PI);
                }
            //}
            std::cout << "end allocation" << std::endl;
            r2cFunc.copy_field(v);
            std::cout << "end copying" << std::endl;
            r2cFunc.execute();
            std::cout << "end r2c" << std::endl;
            std::vector<std::complex<double>> res;
            r2cFunc.output_field(res);
            std::cout << "real" << std::endl;
            for (int i = 0; i < res.size(); i++) {
                std::cout << res[i].real() << std::endl;
            }
            std::cout << "imag" << std::endl;
            for (int i = 0; i < res.size(); i++) {
                std::cout << res[i].imag() << std::endl;
            }

            c2rFunc.copy_field(res);
            c2rFunc.execute();
            std::vector<double> realRes;
            c2rFunc.output_field_padded(realRes);
            std::cout << "inverse transform" << std::endl;
            std::vector<double> v_fine(padded_dim * num_transform, 0.0);
            for (int i = 0; i < padded_dim; i++)
            {
                for (int j = 0; j < num_transform; j++)
                {
                    v_fine[i + j * padded_dim] =
                        std::sin(static_cast<double>(i) /
                            static_cast<double>(padded_dim) * 2.0 *
                            M_PI);
                }
                //v[i] = std::cos(static_cast<double>(i)/static_cast<double>(padded_dim)*2.0*M_PI);
            }
            for (int i = 0; i < realRes.size(); i++) {
                std::cout << (realRes[i] / static_cast<double>(padded_dim) - v_fine[i]) << std::endl;
            }
        }


    private:
        fft::helm_dfft_r2c r2cFunc;
        fft::helm_dfft_c2r c2rFunc;
        int padded_dim;
        int dim_nonzero;
        int dim_0;
        int dim_1;

    };

} // namespace iblgf
#endif // IBLGF_INCLUDED_POISSON_HPP
