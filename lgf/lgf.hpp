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
    void retrieve()
    {
        T* pT = static_cast<T*>(this);
        pT->lgfRetrival();
    }
    
    void lgfRetrival() { std::cout << "LGF::lgfRetrival" << std::endl; }
};

class Bessel : public LGF<Bessel>
{
private:
    static auto getBesselIntegrand(float_type v, float_type x1, float_type x2, float_type x3)
    {
        return exp(-6 * x1) * boost::math::cyl_bessel_j(v, x1) *
                              boost::math::cyl_bessel_j(v, x2) *
                              boost::math::cyl_bessel_j(v, x3);
    }
    
public:
    void lgfRetrival()
    {
        std::cout << "Calculating LGF via Bessel integral\n";
        float_type error;
        float_type v = 1.;
        float_type x1 = 1.;
        float_type x2 = 2.;
        float_type x3 = 3.;
        
        // Cylindrical Bessel function of the first kind
        std::cout << "J_0(" << v << "," << x1 << ") = " 
                  << getBesselIntegrand(v,x1,x2,x3) << std::endl;
        
        float_type Q = boost::math::quadrature::gauss_kronrod<float_type, 15>::integrate(
            getBesselIntegrand(v,x1,x2,x3), 0,
            numeric_limits<float_type>::infinity(),
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
