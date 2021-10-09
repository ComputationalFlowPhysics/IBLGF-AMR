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

#ifndef INCLUDED_HELMHOLTZ_HPP
#define INCLUDED_HELMHOLTZ_HPP

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
#include <iblgf/lgf/helmholtz_lookup.hpp>
#include <iblgf/lgf/lgf.hpp>

namespace iblgf
{
namespace lgf
{
using namespace domain;

template<std::size_t Dim>
class Helmholtz : public LGF_Base<Dim, Helmholtz<Dim>>
{
  public: //Ctor:
    using super_type = LGF_Base<Dim, Helmholtz<Dim>>;
    using block_descriptor_t = BlockDescriptor<int, Dim>;
    using coordinate_t = typename block_descriptor_t::coordinate_type;
    using complex_vector_t = typename super_type::complex_vector_t;

    using key_3D = std::tuple<float_type, int, int, int>;
    using key_2D = std::tuple<float_type, int, int>;
    using level_map_3D_t = std::map<key_3D, std::unique_ptr<complex_vector_t>>;
    using level_map_2D_t = std::map<key_2D, std::unique_ptr<complex_vector_t>>;

  public: //Ctor:
    Helmholtz(float_type C)
    : dft_level_maps_3D(super_type::max_lgf_map_level)
    , dft_level_maps_2D(super_type::max_lgf_map_level)
    , c(C)
    , c_level(C)
    {
        origin = Helmholtz_Lookup::origin_val(c);
    }

    Helmholtz()
    : dft_level_maps_3D(super_type::max_lgf_map_level)
    , dft_level_maps_2D(super_type::max_lgf_map_level)
    {
        c = 0.1;
        c_level = c;
        origin = 0.0;
    }

    void change_c(float_type C)
    {
        c = C;
        c_level = c;
        origin = Helmholtz_Lookup::origin_val(c_level);
    }

    template<int Dim1 = Dim>
    auto get_key(const block_descriptor_t&            _b,
        typename std::enable_if<Dim1 == 3, int>::type _level_diff)
        const noexcept
    {
        const auto base = _b.base();
        return key_3D(c_level, base[0], base[1], base[2]);
    }

    template<int Dim1 = Dim>
    auto get_key(const block_descriptor_t&            _b,
        typename std::enable_if<Dim1 == 2, int>::type _level_diff)
        const noexcept
    {
        const auto base = _b.base();
        return key_2D(c_level, base[0], base[1]);
    }
    void build_lt() {}
    float_type return_c_impl() {return c_level;}
    void change_level_impl(int _level_diff)
    {
        c_level = c*std::pow(0.5, _level_diff);
        origin = Helmholtz_Lookup::origin_val(c_level);
    }

    auto get(const coordinate_t& _c) const noexcept
    {
        if (Dim == 2)
            return Helmholtz_Lookup::get<coordinate_t>(_c, c_level) - origin;
        else
        {
            return Helmholtz_Lookup::get<coordinate_t>(_c, c_level) - origin;
        }
    }

    auto get(int N, int M) const noexcept
    {
        return Helmholtz_Lookup::get<coordinate_t>(coordinate_t({N, M}), c_level) -
               origin;
        //else {return Helmholtz_Lookup::get<coordinate_t>(_c) - origin;}
    }

  public:
    std::vector<level_map_3D_t>
        dft_level_maps_3D; ///<lgf map for octants per level
    std::vector<level_map_2D_t>
               dft_level_maps_2D; ///<lgf map for octants per level
    float_type origin = 0.0;

  private:
    float_type c_diff = 0.0;

    float_type c = 0.0;
    float_type c_level = 0.0; //for helmholtz equation solver

};

} // namespace lgf
} // namespace iblgf

#endif
