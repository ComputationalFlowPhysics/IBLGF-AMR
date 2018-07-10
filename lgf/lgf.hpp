#ifndef INCLUDED_LGFS_HPP
#define INCLUDED_LGFS_HPP

#include <stdio.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <boost/math/quadrature/gauss_kronrod.hpp>
#include <boost/math/special_functions/bessel.hpp>


template <typename T>
class LGF
{
public:
    void retrieve(index_type i, index_type j, index_type k)
    {
        T* pT = static_cast<T*>(this);
        pT->lgfRetrival(index_type i, index_type j, index_type k);
    }
    
    void lgfRetrival() { std::cout << "LGF::lgfRetrival" << std::endl; }
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
    void lgfRetrival(index_type i, index_type j, index_type k)
    {
        std::cout << "Calculating LGF via Bessel integral\n";
        float_type error;
        index_type i = 1;
        index_type j = 1;
        index_type k = 1;
        float_type t = 1.;
        
        
        auto BesselIntegrand = [&i,&j,&k](double t) { return getBesselIntegrand(i,j,k,t); };

        float_type Q =
            boost::math::quadrature::gauss_kronrod<float_type, 15>::integrate(
                BesselIntegrand, 0, numeric_limits<float_type>::infinity(),
                0, 0, &error);
        
        // Regular modified cylindrical Bessel function
        std::cout << "I_0(" << v << "," << x1 << ") = "
             << boost::math::cyl_bessel_i(v, x1)
             << std::ndl;
        
        // Irregular modified cylindrical Bessel function also
        // known as modified Bessel fucntion of the second kind
        std::cout << "K_0(" << v << "," << x1 << ") = "
             << boost::math::cyl_bessel_k(v, x1)
             << std::endl;
    }
};

class Lookup : public LGF<Lookup>
{
public:
    void lgfRetrival()
    {
        std::cout << "Lookup::lgfRetrival" << std::endl;
    }
};


//int main()
//{
//    Bessel b;
//    b.retrieve();
//    
//    Lookup l;
//    l.retrieve();
//    
//    return 0;
//}
#endif
