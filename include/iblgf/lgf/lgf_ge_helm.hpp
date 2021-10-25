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

#ifndef INCLUDED_LGF_GE_HELM_HPP
#define INCLUDED_LGF_GE_HELM_HPP

#include <stdio.h>
#include <iostream>
#include <vector>
#include <boost/math/special_functions/bessel.hpp>
#include <cmath>

#include <iblgf/global.hpp>
#include <iblgf/lgf/lgf.hpp>
#include <iblgf/domain/dataFields/blockDescriptor.hpp>

namespace iblgf
{
namespace lgf
{
using namespace domain;

template<std::size_t Dim = 3>
class LGF_GE_HELM : public LGF_Base<Dim, LGF_GE<Dim>>
{
  public: //Ctor:
    using super_type = LGF_Base<Dim, LGF_GE<Dim>>;
    using block_descriptor_t = BlockDescriptor<int, Dim>;
    using coordinate_t = typename block_descriptor_t::coordinate_type;

    using complex_vector_t = typename super_type::complex_vector_t;

    using key_3D = std::tuple<float_type, int, int, int>;
    using level_map_3D_t = std::map<key_3D, std::unique_ptr<complex_vector_t>>;

    using key_2D = std::tuple<float_type, int, int>;
    using level_map_2D_t = std::map<key_2D, std::unique_ptr<complex_vector_t>>;

  public: //Ctor:
    LGF_GE_HELM()
    : dft_level_maps_3D(20)
    , dft_level_maps_2D(20)
    {
        this->neighbor_only_ = true;
    }

    LGF_GE_HELM(float_type _omega)
    : dft_level_maps_3D(20)
    , dft_level_maps_2D(20)
    , omega(_omega)
    {
        this->neighbor_only_ = true;
    }


  public:
    void build_lt() { build_lt(alpha_); }
    float_type return_c_impl() {return 0.0;}
    template<int Dim1 = Dim>
    auto get_key(const block_descriptor_t& _b, typename std::enable_if<Dim1 == 3, int>::type _level_diff) const noexcept
    {
        const auto base = _b.base();
        return key_3D(alpha_, base[0], base[1], base[2]);
    }


    template<int Dim1 = Dim>
    auto get_key(const block_descriptor_t& _b, typename std::enable_if<Dim1 == 2, int>::type _level_diff) const noexcept
    {
        const auto base = _b.base();
        return key_2D(alpha_, base[0], base[1]);
    }

    void flip_alpha() noexcept { alpha_ = -alpha_; }

    void change_level_impl(int _level_diff) noexcept
    {
        alpha_ = alpha_base_level_ * std::pow(4, _level_diff);
    }

    template<class Coordinate>
    typename std::enable_if<Coordinate::size() == 3, float_type>::type get(const Coordinate& _c) const noexcept
    {
        int x = std::abs(static_cast<int>(_c.x()));
        int y = std::abs(static_cast<int>(_c.y()));
        int z = std::abs(static_cast<int>(_c.z()));

        if (x <= N_max && y <= N_max && z <= N_max) // using table
        {
            if (x < y) std::swap(x, y);
            if (x < z) std::swap(x, z);
            if (y < z) std::swap(y, z);

            return table_3D[(x * (2 + 3 * x + x * x)) / 6 + y * (y + 1) / 2 +
                          z]; // indexing
        }
        else
        {
            return compute_lgf(x, y, z, alpha_);
        }
    }


    template<class Coordinate>
    typename std::enable_if<Coordinate::size() == 2, float_type>::type get(const Coordinate& _c) const noexcept
    {
        int x = std::abs(static_cast<int>(_c.x()));
        int y = std::abs(static_cast<int>(_c.y()));

        if (x <= N_max && y <= N_max) // using table
        {
            if (x < y) std::swap(x, y);

            return table_2D[x*(N_max+1) + y]; // indexing
        }
        else
        {
            return compute_lgf(x, y, alpha_);
        }
    }

    auto&       alpha_base_level() noexcept { return alpha_base_level_; }
    const auto& alpha_base_level() const noexcept { return alpha_base_level_; }

  private:
    template<int Dim1 = Dim>
    typename std::enable_if<Dim1 == 3, void>::type build_lt(float_type _alpha) noexcept
    {
        table_3D.clear();
        for (int i = 0; i < N_max + 1; ++i)
        {
            for (int j = 0; j < i + 1; ++j)
            {
                for (int k = 0; k < j + 1; ++k)
                { table_3D.push_back(compute_lgf(i, j, k, _alpha)); }
            }
        }
    }

    template<int Dim1 = Dim>
    typename std::enable_if<Dim1 == 2, void>::type build_lt(float_type _alpha) noexcept
    {
        table_2D.clear();
        for (int i = 0; i < N_max + 1; ++i)
        {
            for (int j = 0; j < N_max + 1; ++j)
            {
                table_2D.push_back(compute_lgf(i, j, _alpha));
            }
        }
    }

    static float_type compute_lgf(
        int _n1, int _n2, int _n3, float_type _alpha) noexcept
    {
        return std::exp(-6 * _alpha) *
               boost::math::cyl_bessel_i(
                   static_cast<float_type>(_n1), 2 * _alpha) *
               boost::math::cyl_bessel_i(
                   static_cast<float_type>(_n2), 2 * _alpha) *
               boost::math::cyl_bessel_i(
                   static_cast<float_type>(_n3), 2 * _alpha);
    }

    static float_type compute_lgf(
        int _n1, int _n2, float_type _alpha) noexcept
    {
        return std::exp(-4 * _alpha) *
               boost::math::cyl_bessel_i(
                   static_cast<float_type>(_n1), 2 * _alpha) *
               boost::math::cyl_bessel_i(
                   static_cast<float_type>(_n2), 2 * _alpha);
    }

  public:
    std::vector<level_map_3D_t> dft_level_maps_3D; ///<lgf map for octants per level
    std::vector<level_map_2D_t> dft_level_maps_2D; ///<lgf map for octants per level
    static constexpr int     N_max = 25;
    std::vector<float_type>  table_3D;
    std::vector<float_type>  table_2D;
    float_type               alpha_ = 0;
    float_type               alpha_base_level_ = 0;
    float_type               omega;
};

} // namespace lgf
} // namespace iblgf
#endif
