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

    using key_3D = std::tuple<bool, float_type, int, int, int>; //the boolean is needed to pass the information if the LGF need to be evaluated or not
    using key_2D = std::tuple<bool, float_type, int, int>;
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
        this->HelmholtzLGF_ = true;
        this->emptyVec.resize(0);
        
    }

    Helmholtz()
    : dft_level_maps_3D(super_type::max_lgf_map_level)
    , dft_level_maps_2D(super_type::max_lgf_map_level)
    {
        c = 0.1;
        c_level = c;
        origin = 0.0;

        this->HelmholtzLGF_ = true;
        this->emptyVec.resize(0);

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
        return key_3D(true, c_level, base[0], base[1], base[2]);
    }

    template<int Dim1 = Dim>
    auto get_key(const block_descriptor_t&            _b,
        typename std::enable_if<Dim1 == 2, int>::type _level_diff)
        const noexcept
    {
        const auto base = _b.base();
        const auto max =  _b.max();

        int k_max = 0;

        if (int i = 0; i < 2) {
            if (std::abs(base[i]) > std::abs(max[i])) {
                k_max += std::abs(max[i]);
            }
            else {
                k_max += std::abs(base[i]);
            }
        }
        float_type factor = 4.0/(4.0 + c_level * c_level);
        float_type eps = std::pow(factor, (k_max - 1));
        //float_type a = 2 + c_level*c_level/2.0;
        //float_type factor = (a / 2.0 + std::sqrt(a*a/4.0 - 1));
        //float_type eps = std::pow(factor, -k_max) * std::pow((a*a - 4), 0.25) / std::sqrt(k_max);
        bool tmp = true;
        if (eps < 1e-13) {
            tmp = false;
        }

        return key_2D(true, c_level, base[0], base[1]);
    }
    void build_lt() {}
    float_type return_c_impl() {return c_level;}
    void change_level_impl(int _level_diff)
    {
        c_level = c*std::pow(0.5, _level_diff);
        if ((ori_value.size()) < (_level_diff + 1)) {
            auto tmp = ori_value;
            auto tmp_bool = ori_bool;
            ori_value.resize((_level_diff + 1));
            ori_bool.resize((_level_diff + 1));
            for (int i = 0; i < tmp.size(); i++) {
                ori_value[i] = tmp[i];
                ori_bool[i] = tmp_bool[i];
            }
            for (int i = tmp.size(); i < ori_value.size(); i++) {
                ori_bool[i] = false;
            }
            origin = Helmholtz_Lookup::origin_val(c_level);
            ori_value[_level_diff] = origin;
            ori_bool[_level_diff] = true;
        }
        else if (!(ori_bool[_level_diff])){
            origin = Helmholtz_Lookup::origin_val(c_level);
            ori_value[_level_diff] = origin;
            ori_bool[_level_diff] = true;
        }
        else {
            origin = ori_value[_level_diff];
        }
        if (_level_diff > (ori_value.size())) {
            throw std::runtime_error("Level diff is too large in helmholtz");
        }
        //origin = Helmholtz_Lookup::origin_val(c_level);
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

    float_type c = 0.0;       //c is the wave number at the coarse level
    float_type c_level = 0.0; //for helmholtz equation solver
    std::vector<float_type> ori_value;
    std::vector<bool> ori_bool;


};

} // namespace lgf
} // namespace iblgf

#endif
