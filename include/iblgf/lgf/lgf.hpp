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

#ifndef INCLUDED_LGFS_HPP
#define INCLUDED_LGFS_HPP

#include <iostream>
#include <vector>
#include <cmath>
#include <complex>
#include <boost/math/special_functions/bessel.hpp>

#include <iblgf/global.hpp>
#include <iblgf/utilities/crtp.hpp>
#include <iblgf/lgf/lgf_gl_lookup.hpp>

namespace iblgf
{
namespace lgf
{
using namespace domain;

template<std::size_t Dim, class Derived>
class LGF_Base : public crtp::Crtps<Derived, LGF_Base<Dim, Derived>>
{
  public:
    using block_descriptor_t = BlockDescriptor<int, Dim>;
    using coordinate_t = typename block_descriptor_t::coordinate_type;
    using dims_t = types::vector_type<int, Dim>;

    using complex_vector_t = std::vector<std::complex<float_type>,
        xsimd::aligned_allocator<std::complex<float_type>, 32>>;

    template<class Convolutor, int Dim1 = Dim>
    auto& dft(const block_descriptor_t& _lgf_block, dims_t _extended_dims,
        Convolutor* _conv, typename std::enable_if<Dim1 == 3, int>::type level_diff)
    {
        auto k_ = this->derived().get_key(_lgf_block, level_diff);
        auto it = this->derived().dft_level_maps_3D[level_diff].find(k_);

        //Check if lgf is already stored
        if (it == this->derived().dft_level_maps_3D[level_diff].end())
        {
            this->get_subblock(
                _lgf_block, _extended_dims, lgf_buffer_, level_diff);
            auto& dft = _conv->dft_r2c(lgf_buffer_);
            this->derived().dft_level_maps_3D[level_diff].emplace(
                k_, std::make_unique<complex_vector_t>(dft));
            return dft;
        }
        else
        {
            return *(it->second).get();
        }
    }


    template<class Convolutor, int Dim1 = Dim>
    auto& dft(const block_descriptor_t& _lgf_block, dims_t _extended_dims,
        Convolutor* _conv, typename std::enable_if<Dim1 == 2, int>::type level_diff)
    {
        auto k_ = this->derived().get_key(_lgf_block, level_diff);
        auto it = this->derived().dft_level_maps_2D[level_diff].find(k_);

        //Check if lgf is already stored
        if (it == this->derived().dft_level_maps_2D[level_diff].end()/* || this->LaplaceLGF()  !this->neighbor_only()*/)
        {
            this->get_subblock(
                _lgf_block, _extended_dims, lgf_buffer_, level_diff);
            auto& dft = _conv->dft_r2c(lgf_buffer_);
            this->derived().dft_level_maps_2D[level_diff].emplace(
                k_, std::make_unique<complex_vector_t>(dft));
            return dft;
        }
        else
        {
            return *(it->second).get();
        }
    }

    void change_level(int _level_diff) noexcept
    {
        this->derived().change_level_impl(_level_diff);
    }

    bool neighbor_only() { return neighbor_only_; }

    bool LaplaceLGF() {return LaplaceLGF_;}

    float_type return_c_level() {
        return this->derived().return_c_impl();
    }


  protected:
    template<int Dim1 = Dim>
    typename std::enable_if<Dim1 == 3, void>::type get_subblock(const block_descriptor_t& _b, dims_t _extended_dims,
        std::vector<float_type>& _lgf, int level_diff = 0) noexcept
    {
        this->derived().build_lt();

        const auto base = _b.base();
        const auto max = _b.max();
        int        step = pow(2, level_diff);

        std::vector<float_type> _lgf_small;
        _lgf_small.resize(_b.size());

        for (auto k = base[2]; k <= max[2]; ++k)
        {
            for (auto j = base[1]; j <= max[1]; ++j)
            {
                for (auto i = base[0]; i <= max[0]; ++i)
                {
                    //get view
                    _lgf_small[_b.index(i, j, k)] = this->derived().get(
                        coordinate_t({i * step, j * step, k * step}));
                }
            }
        }

        block_descriptor_t _b_pad(base, _extended_dims);

        _lgf.resize(_b_pad.size());
        std::fill(_lgf.begin(), _lgf.end(), 0.0);

        for (auto k = base[2]; k <= max[2]; ++k)
        {
            for (auto j = base[1]; j <= max[1]; ++j)
            {
                for (auto i = base[0]; i <= max[0]; ++i)
                {
                    _lgf[_b_pad.index(i, j, k)] = _lgf_small[_b.index(i, j, k)];
                }
            }
        }
    }

    template<int Dim1 = Dim>
    typename std::enable_if<Dim1 == 2, void>::type get_subblock(const block_descriptor_t& _b, dims_t _extended_dims,
        std::vector<float_type>& _lgf, int level_diff = 0) noexcept
    {
        this->derived().build_lt();

        const auto base = _b.base();
        const auto max = _b.max();
        int        step = pow(2, level_diff);

        std::vector<float_type> _lgf_small;
        _lgf_small.resize(_b.size());

            for (auto j = base[1]; j <= max[1]; ++j)
            {
                for (auto i = base[0]; i <= max[0]; ++i)
                {
                    //get view
                    _lgf_small[_b.index(i, j)] = this->derived().get(
                        coordinate_t({i * step, j * step}));
                }
            }

        block_descriptor_t _b_pad(base, _extended_dims);

        _lgf.resize(_b_pad.size());
        std::fill(_lgf.begin(), _lgf.end(), 0.0);

            for (auto j = base[1]; j <= max[1]; ++j)
            {
                for (auto i = base[0]; i <= max[0]; ++i)
                {
                    _lgf[_b_pad.index(i, j)] = _lgf_small[_b.index(i, j)];
                }
            }
        
    }

    std::vector<float_type> lgf_buffer_; ///<lgf buffer
    const int               max_lgf_map_level = 20;
    bool                    neighbor_only_ = false;
    bool                    LaplaceLGF_ = false;
    
};

} // namespace lgf
} // namespace iblgf
#endif
