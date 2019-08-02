#ifndef INCLUDED_LGF_GE_HPP
#define INCLUDED_LGF_GE_HPP

#include <stdio.h>
#include <iostream>
#include <vector>
#include <cmath>

#include <global.hpp>
#include <lgf/lgf.hpp>
#include <domain/dataFields/blockDescriptor.hpp>
#include <boost/math/special_functions/bessel.hpp>

namespace lgf
{

using namespace domain;

template<std::size_t Dim=3>
class LGF_GE : public LGF_Base<Dim,LGF_GE<Dim>>
{

public: //Ctor:

    using super_type = LGF_Base<Dim,LGF_GE<Dim>>;
    using block_descriptor_t = BlockDescriptor<int,Dim>;
    using coordinate_t = typename block_descriptor_t::base_t;

    using complex_vector_t = typename super_type::complex_vector_t;

    using key_t = std::tuple<float_type, int, int, int>;
    using level_map_t = std::map<key_t,std::unique_ptr<complex_vector_t>>;

public: //Ctor:


    LGF_GE()
    :dft_level_maps_(20)
    {
        this->neighbor_only_=true;
    }

    static_assert(Dim==3, "LGFE only implemented for D=3");

public:

    void  build_lt(  )
    {
        build_lt(alpha_);
    }
    auto get_key( const block_descriptor_t& _b, int _level_diff )const noexcept
    {
        const auto base = _b.base();
        return key_t(alpha_,base[0],base[1],base[2]);
    }

    void flip_alpha() noexcept
    {
        alpha_ = -alpha_;
    }

    void change_level_impl( int _level_diff) noexcept
    {
        alpha_ = alpha_base_level_ * std::pow(4, _level_diff);
    }


    template<class Coordinate>
    auto get(const Coordinate& _c) const noexcept
    {
        int x = std::abs(static_cast<int>(_c.x()));
        int y = std::abs(static_cast<int>(_c.y()));
        int z = std::abs(static_cast<int>(_c.z()));

        if (x<= N_max && y<=N_max && z<=N_max) // using table
        {
            if (x < y)
                std::swap(x, y);
            if (x < z)
                std::swap(x, z);
            if (y < z)
                std::swap(y, z);

            return table_[(x * (2 + 3*x + x * x))/6 + y*(y+1)/2 + z]; // indexing
        }
        else
        {
            return compute_lgf(x,y,z,alpha_);
        }
   }

    auto& alpha_base_level() noexcept {return alpha_base_level_;}
    const auto& alpha_base_level() const noexcept {return alpha_base_level_;}

private:

    void build_lt( float_type _alpha  ) noexcept
    {
        table_.clear();
        for(int i=0;i<N_max+1;++i)
        {
            for(int j=0;j<i+1;++j)
            {
                for(int k=0;k<j+1;++k)
                {
                    table_.push_back(compute_lgf(i,j,k,_alpha));
                }
            }
        }
    }

    static float_type compute_lgf(int _n1, int _n2, int _n3,
                                    float_type _alpha) noexcept
    {
        return std::exp(-6*_alpha)*
                boost::math::cyl_bessel_i(static_cast<float_type>(_n1), 2*_alpha)*
                boost::math::cyl_bessel_i(static_cast<float_type>(_n2), 2*_alpha)*
                boost::math::cyl_bessel_i(static_cast<float_type>(_n3), 2*_alpha);
    }

public:
    std::vector<level_map_t> dft_level_maps_;   ///<lgf map for octants per level
    static constexpr int N_max=25;
    std::vector<float_type> table_;
    float_type alpha_=0;
    float_type alpha_base_level_=0;
};

}

#endif
