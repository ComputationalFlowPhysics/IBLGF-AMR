#ifndef INCLUDED_LGFS_HPP
#define INCLUDED_LGFS_HPP

#include <stdio.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <boost/math/quadrature/gauss_kronrod.hpp>
#include <boost/math/special_functions/bessel.hpp>

#include "global.hpp"

template <typename T>
class LGF
{
public:
    auto retrieve(index_type i, index_type j, index_type k)
    {
        T* pT = static_cast<T*>(this);
        return pT->lgfRetrival(i, j, k);
    }
    
};

class Bessel : public LGF<Bessel>
{
private:
    static auto getBesselIntegrand(index_type i, index_type j, index_type k,
        float_type t)
    {
        return exp(-6 * t) * boost::math::cyl_bessel_j(i, 2*t) *
                             boost::math::cyl_bessel_j(j, 2*t) *
                             boost::math::cyl_bessel_j(k, 2*t);
    }
    
    
public:
    auto lgfRetrival(index_type i, index_type j, index_type k)
    {
        float_type error;
        auto BesselIntegrand = [&i,&j,&k](double t) { 
            return getBesselIntegrand(i,j,k,t); };

        float_type Q =
            boost::math::quadrature::gauss_kronrod<float_type, 15>::integrate(
                BesselIntegrand, 0, std::numeric_limits<float_type>::infinity(),
                0, 0, &error);

            return Q;
        
    }
};

class Lookup : public LGF<Lookup>
{
public:
};


#endif
