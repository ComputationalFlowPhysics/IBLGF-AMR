#ifndef INCLUDED_LGFS_HPP
#define INCLUDED_LGFS_HPP

#include <stdio.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <boost/math/special_functions/bessel.hpp>
#include <boost/align/aligned_allocator_adaptor.hpp>


#include <complex>
#include <domain/dataFields/dataBlock.hpp>
#include <domain/dataFields/datafield.hpp>
#include <global.hpp>

namespace lgf
{

using namespace domain;

template<class Policy, std::size_t Dim=3>
class LGF : public Policy
{

public: //Ctor:

    using block_descriptor_t = BlockDescriptor<int,Dim>;
    using coordinate_t = typename block_descriptor_t::base_t;

    using complex_vector_t = std::vector<std::complex<float_type>,
          boost::alignment::aligned_allocator_adaptor<
              std::allocator<std::complex<float_type>>,32>> ;

    using real_vector_t = std::vector<float_type,
          boost::alignment::aligned_allocator_adaptor<
              std::allocator<float_type>,32>>;

    using key_t = std::tuple<int, int, int>;
    using level_map_t = std::map<key_t,std::unique_ptr<complex_vector_t>>;

public: //Ctor:

    LGF()
    {
        const int max_lgf_map_level = 20;
        dft_level_maps_.clear();
        dft_level_maps_.resize(max_lgf_map_level);
    }
    static_assert(Dim==3, "LGF only implemented for D=3");


    template<class Convolutor>
    auto& dft(const block_descriptor_t& _lgf_block, 
                  Convolutor* _conv,  int level_diff)
    {
        const auto base = _lgf_block.base();
        key_t k_(base[0],base[1],base[2]);
        auto it = dft_level_maps_[level_diff].find( k_ );

        //Check if lgf is already stored
        if ( it == dft_level_maps_[level_diff].end() )
        {
            this->get_subblock( _lgf_block, lgf_buffer_, level_diff);
            auto& dft=_conv->dft_r2c(lgf_buffer_);
            dft_level_maps_[level_diff].emplace(k_,
                    std::make_unique<complex_vector_t>( dft ));
            return dft;
        } 
        else
        {
            return *(it->second).get();
        }
    }


private:

    template<class Coordinate>
    static auto get(const Coordinate& _coord) noexcept
    {
        return Policy::get(_coord);
    }

    void get_subblock(const block_descriptor_t& _b,
                      std::vector<float_type>&  _lgf, 
                      int level_diff = 0) noexcept
    {
        const auto base = _b.base();
        const auto max  = _b.max();
        int step = pow(2, level_diff);

        _lgf.resize(_b.size());
        for (auto k = base[2]; k <= max[2] ; ++k)
        {
            for (auto j = base[1]; j <= max[1]; ++j)
            {
                for (auto i = base[0]; i <= max[0]; ++i)
                {
                    //get view
                    _lgf[_b.index(i,j,k)] =
                        Policy::get(coordinate_t({i*step, j*step, k*step}));
                }
            }
        }
    }

private:
    std::vector<level_map_t> dft_level_maps_;   ///<lgf map for octants per level
    std::vector<float_type> lgf_buffer_;   ///<lgf map for octants per level
};

}

#endif
