#ifndef INCLUDED_LGFS_HPP
#define INCLUDED_LGFS_HPP

#include <stdio.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <boost/math/quadrature/gauss_kronrod.hpp>
#include <boost/math/special_functions/bessel.hpp>

// IBLGF-specific
#include <lgf/lgf_lookup.hpp>
#include <lgf/lgf_integrator.hpp>

namespace lgf
{

template<class Policy>
class LGF : public Policy
{
public:
    
    make_field_type(lgf, float_type)
    using datablock_t = DataBlock<Dim, node, lgf>;
    
    template<class Coordinate>
    static auto get(const Coordinate& _coord) noexcept
    {
        return Policy::get(_coord);
    }
    
    datablock_t lgf_container;
};

}

#endif
