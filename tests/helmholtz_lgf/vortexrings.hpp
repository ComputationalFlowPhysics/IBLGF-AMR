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
#include <iostream>
#include <fstream>
#include <iomanip>

// IBLGF-specific
#include "../../setups/setup_base.hpp"
#include <iblgf/lgf/lgf.hpp>
#include <iblgf/lgf/helmholtz.hpp>

namespace iblgf
{
    class Helmholtz_LGF_generate {
        public:
        Helmholtz_LGF_generate(float_type C) : c(C), H_lgf(C) {}

        void writing_LGF(int N, int M) {
            float_type origin_val = H_lgf.origin;
            //std::cout << "origin is " << origin_val << std::endl;
            std::ofstream file1;
            file1.open("Helmholtz.txt");
            auto start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < N; i++) {
                std::cout << i << std::endl;
                for (int j = 0 ; j < M; j++) {
                    
                    float_type LGF_val = H_lgf.get(i,j);
                    lgf_vec.emplace_back(LGF_val);
                    file1 << std::setprecision(20) << LGF_val << std::endl;
                    //if (j == 0) std::cout << LGF_val << std::endl;
                }
            }
            auto stop = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<milliseconds>(stop - start);
            std::cout << "time taken is [ms]: " << duration.count() << std::endl;

            file1.close();
        }
        private:
        float_type c;
        lgf::Helmholtz<2> H_lgf;
        std::vector<float_type> lgf_vec;
    };

} // namespace iblgf
#endif // IBLGF_INCLUDED_POISSON_HPP
