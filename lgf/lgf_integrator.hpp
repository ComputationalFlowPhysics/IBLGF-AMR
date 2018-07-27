#ifndef INCLUDED_LGF_INTEGRATOR_HPP
#define INCLUDED_LGF_INTEGRATOR_HPP

#include <stdio.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <boost/math/quadrature/gauss_kronrod.hpp>
#include <boost/math/special_functions/bessel.hpp>

// IBLGF-specific
#include <global.hpp>

namespace lgf
{

class Integrator
{
    
public:
    template<class Coordinate>
    static auto get(const Coordinate& _coord) noexcept
    {
        float_type error;
        auto BesselIntegrand = [&_coord](float_type t) 
        { 
            return getBesselIntegrand(_coord, t); 
        };

        return 
            - boost::math::quadrature::
                gauss_kronrod<float_type, 15>::integrate(
                BesselIntegrand, 
                0, std::numeric_limits<float_type>::infinity(),
                0, 0, &error);
    }
private:
    template<class Coordinate>
    static auto getBesselIntegrand(const Coordinate& _coord, float_type t)
    {
        return exp(-6 * t) * boost::math::cyl_bessel_i(_coord.x(), 2*t) *
                             boost::math::cyl_bessel_i(_coord.y(), 2*t) *
                             boost::math::cyl_bessel_i(_coord.z(), 2*t);
    }
    
    
};



} //namepsace
#endif
