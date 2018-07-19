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
    template<class Coordinate>
    static auto get(const Coordinate& _coord) noexcept
    {
        return Policy::get(_coord);
    }
};

}

#endif
