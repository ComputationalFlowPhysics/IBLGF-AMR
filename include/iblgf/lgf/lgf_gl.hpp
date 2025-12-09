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

#ifndef INCLUDED_LGFS_GL_HPP
#define INCLUDED_LGFS_GL_HPP

#include <stdio.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <complex>
#include <boost/math/special_functions/bessel.hpp>
#include <boost/align/aligned_allocator_adaptor.hpp>

#include <iblgf/global.hpp>
#include <iblgf/domain/dataFields/dataBlock.hpp>
#include <iblgf/domain/dataFields/datafield.hpp>
#include <iblgf/lgf/lgf_gl_lookup.hpp>
#include <iblgf/lgf/lgf.hpp>

namespace iblgf
{
namespace lgf
{
using namespace domain;

template<std::size_t Dim>
class LGF_GL : public LGF_Base<Dim, LGF_GL<Dim>>
{
  public: //Ctor:
    using super_type = LGF_Base<Dim, LGF_GL<Dim>>;
    using block_descriptor_t = BlockDescriptor<int, Dim>;
    using coordinate_t = typename block_descriptor_t::coordinate_type;
    // using complex_vector_t = typename super_type::complex_vector_gpu_t;// set this back to get cpu code to work
    // #ifdef __CUDACC__   // NVCC device code
    #ifdef IBLGF_COMPILE_CUDA // Compiling for CUDA
    #pragma message("Compiling for CUDA: using GPU complex vector")
    using complex_vector_t = typename super_type::complex_vector_gpu_t;
    #else               // CPU code
    #pragma message("Compiling for CPU: using CPU complex vector")

    using complex_vector_t = typename super_type::complex_vector_t;
    #endif
    // using complex_vector_t = typename super_type::complex_vector_t;

    using key_3D = std::tuple<int, int, int>;
    using key_2D = std::tuple<float, int, int>;
    //using key_2D = std::tuple<int, int>;
    using level_map_3D_t = std::map<key_3D, std::unique_ptr<complex_vector_t>>;
    using level_map_2D_t = std::map<key_2D, std::unique_ptr<complex_vector_t>>;

  public: //Ctor:
    LGF_GL()
    : dft_level_maps_3D(super_type::max_lgf_map_level)
    , dft_level_maps_2D(super_type::max_lgf_map_level)
    {
        this->LaplaceLGF_ = true;
    }
    template<int Dim1 = Dim>
    auto get_key(const block_descriptor_t&            _b,
        typename std::enable_if<Dim1 == 3, int>::type _level_diff)
        const noexcept
    {
        const auto base = _b.base();
        return key_3D(base[0], base[1], base[2]);
    }

    template<int Dim1 = Dim>
    auto get_key(const block_descriptor_t&            _b,
        typename std::enable_if<Dim1 == 2, int>::type _level_diff)
        const noexcept
    {
        const auto base = _b.base();
        return key_2D(c_diff, base[0], base[1]);
    }
    void build_lt() {}
    float_type return_c_impl() {return 0.0;}
    void change_level_impl(int _level_diff)
    {
        if (Dim == 2)
            c_diff = -static_cast<float_type>(_level_diff) / 2.0 / M_PI *
                     std::log(2.0);
    }

    auto get(const coordinate_t& _c) const noexcept
    {
        if (Dim == 2) return LGF_GL_Lookup::get<coordinate_t>(_c) + c_diff;
        else
        {
            return LGF_GL_Lookup::get<coordinate_t>(_c);
        }
    }

  public:
    std::vector<level_map_3D_t>
        dft_level_maps_3D; ///<lgf map for octants per level
    std::vector<level_map_2D_t>
        dft_level_maps_2D; ///<lgf map for octants per level
  private:
    float_type c_diff = 0.0;
};

} // namespace lgf
} // namespace iblgf

#endif
