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
        //long double error;
        //auto BesselIntegrand = [&_coord](float_type t) 
        //{ 
        //    return getBesselIntegrand(_coord, t); 
        //};

        //std::cout << _coord.x() << ", "<< _coord.y()<< ", " << _coord.z() << std::endl;

        //long double tmp = 
        //     boost::math::quadrature::
        //        gauss_kronrod<long double, 15>::integrate(
        //        BesselIntegrand, 
        //        0, std::numeric_limits<float>::infinity(),
        //        0, 1.0e-14, &error);

        //std::cout << "value = " << tmp << " error = " << error << std::endl;


        return 0;

    }
private:
    template<class Coordinate>
    static auto getBesselIntegrand(const Coordinate& _coord, long double t)
    {
        long double x = _coord.x();
        long double y = _coord.y();
        long double z = _coord.z();

        return - exp( -6 * t) * 
                            boost::math::cyl_bessel_i(x , 2*t) *
                            boost::math::cyl_bessel_i(y , 2*t) *
                            boost::math::cyl_bessel_i(z , 2*t);
    }
    
    
};



} //namepsace
#endif
