#ifndef INCLUDED_LGFS_GL_HPP
#define INCLUDED_LGFS_GL_HPP

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
#include <lgf/lgf_gl_lookup.hpp>
#include <lgf/lgf.hpp>

namespace lgf
{

using namespace domain;


template<std::size_t Dim>
class LGF_GL : public LGF_Base<Dim,LGF_GL<Dim>>
{

public: //Ctor:
    using super_type = LGF_Base<Dim,LGF_GL<Dim>>;
    using block_descriptor_t = BlockDescriptor<int,Dim>;
    using coordinate_t = typename block_descriptor_t::base_t;
    using complex_vector_t = typename super_type::complex_vector_t;

    using key_t = std::tuple<int, int, int>;
    using level_map_t = std::map<key_t,std::unique_ptr<complex_vector_t>>;

public: //Ctor:

    LGF_GL()
    :dft_level_maps_(super_type::max_lgf_map_level)
    {
    }
    static_assert(Dim==3, "LGF_GL only implemented for D=3");


    auto get_key( const block_descriptor_t& _b, int _level_diff )const noexcept
    {
        const auto base = _b.base();
        return key_t(base[0],base[1],base[2]);
    }
    void build_lt(){}
    void change_level_impl( int _level_diff){}

    auto get(const coordinate_t& _c) const noexcept
    {
        return LGF_GL_Lookup::get(_c);
    }

public:
    std::vector<level_map_t> dft_level_maps_;   ///<lgf map for octants per level
};

}

#endif
