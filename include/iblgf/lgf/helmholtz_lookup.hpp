//      ▄▄▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄▄▄   ▄            ▄▄▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄▄▄▄
//     ▐░░░░░░░░░░░▌▐░░░░░░░░░░▌ ▐░▌          ▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌
//      ▀▀▀▀█░█▀▀▀▀ ▐░█▀▀▀▀▀▀▀█░▌▐░▌          ▐░█▀▀▀▀▀▀▀▀▀ ▐░█▀▀▀▀▀▀▀▀▀
//          ▐░▌     ▐░▌       ▐░▌▐░▌          ▐░▌          ▐░▌
//          ▐░▌     ▐░█▄▄▄▄▄▄▄█░▌▐░▌          ▐░▌ ▄▄▄▄▄▄▄▄ ▐░█▄▄▄▄▄▄▄▄▄
//          ▐░▌     ▐░░░░░░░░░░▌ ▐░▌          ▐░▌▐░░░░░░░░▌▐░░░░░░░░░░░▌
//          ▐░▌     ▐░█▀▀▀▀▀▀▀█░▌▐░▌          ▐░▌ ▀▀▀▀▀▀█░▌▐░█▀▀▀▀▀▀▀▀▀
//          ▐░▌     ▐░▌       ▐░▌▐░▌          ▐░▌       ▐░▌▐░▌
//      ▄▄▄▄█░█▄▄▄▄ ▐░█▄▄▄▄▄▄▄█░▌▐░█▄▄▄▄▄▄▄▄▄ ▐░█▄▄▄▄▄▄▄█░▌▐░▌
//     ▐░░░░░░░░░░░▌▐░░░░░░░░░░▌ ▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░▌
//      ▀▀▀▀▀▀▀▀▀▀▀  ▀▀▀▀▀▀▀▀▀▀   ▀▀▀▀▀▀▀▀▀▀▀  ▀▀▀▀▀▀▀▀▀▀▀  ▀

#ifndef INCLUDED_HELMHOLTZ_LOOKUP_HPP
#define INCLUDED_HELMHOLTZ_LOOKUP_HPP

#include <algorithm>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <array>
#include <cmath>
#include <boost/math/quadrature/trapezoidal.hpp>
//#include <boost/math/special_functions/hypergeometric_pFq.hpp>

#include <iblgf/global.hpp>

namespace iblgf
{
namespace lgf
{
class Helmholtz_Lookup
{
  public:
    template<class Coordinate>
    static typename std::enable_if<Coordinate::size() == 3, float_type>::type get(const Coordinate& _c) noexcept
    {
        int x = std::abs(static_cast<int>(_c.x()));
        int y = std::abs(static_cast<int>(_c.y()));
        int z = std::abs(static_cast<int>(_c.z()));

        if (x <= N_max_3D && y <= N_max_3D && z <= N_max_3D) // using table
        {
            if (x < y) std::swap(x, y);
            if (x < z) std::swap(x, z);
            if (y < z) std::swap(y, z);

            return -table_3D[(x * (2 + 3 * x + x * x)) / 6 + y * (y + 1) / 2 +
                           z]; // indexing
        }
        else
        {
            return asym(x, y, z);
        }
    }

    template<class Coordinate>
    static typename std::enable_if<Coordinate::size() == 2, float_type>::type
    get(const Coordinate& _c, float_type c) noexcept
    {
        int x = std::abs(static_cast<int>(_c.x()));
        int y = std::abs(static_cast<int>(_c.y()));

        if (x < y) std::swap(x, y);
        
        float_type res = integralHelmholtzTrap(x, y, c);
        //std::cout << "getting values from correct function " << x << " " << y << " res is " << res << std::endl;
        return res;
    }

    class integrand {
        public:
        integrand(const float_type C, int N, int M) : c(C) {
            n = std::abs(N);
            m = std::abs(m);
        }
        float_type operator()(const float_type& x) const {
            float_type a = 4.0 - 2.0*std::cos(x)+c*c;
            float_type K = (a + std::sqrt(a*a - 4.0))/2.0;
            float_type int_val = (1.0 - std::cos(x*m))/(std::pow(K,n))*(1/(K - 1.0/K));
            return int_val/2.0/M_PI;
        }

        private:
        int n;
        int m;
        float_type c;
    };

    static float_type asym(int n1, int n2) {
	    float_type Cfund = -0.18124942796; // constant for asymptotic expansion (C_fund)
	    float_type Integral_val = -0.076093998454228; // constant for asymptotical expansion (integral)	
	    float_type r = std::sqrt(n1*n1 + n2*n2);

	    float_type theta = atan2(n2,n1);
	    float_type second_term = 1.0/24.0/M_PI*cos(4.0*theta)/r/r;
	    float_type res = -0.5/M_PI*log(r) + Cfund + Integral_val + second_term;
	    return res;
    }

    static float_type origin_val(float_type c) {
        float_type z = 2.0/(2.0+c*c/2.0) * 2.0/(2.0+c*c/2.0);
        return 0.5/(2.0+c*c/2.0)*hypergeometric(0.5, 0.5,1, z, 1.0e-20);
    }

    static float_type hypergeometric(float_type a, float_type b, float_type c, float_type x, float_type tol)
    {
        const float_type TOLERANCE = tol;
        float_type       term = a * b * x / c;
        float_type       value = 1.0 + term;
        int              n = 1;

        while (abs(term) > TOLERANCE)
        {
            a++, b++, c++, n++;
            term *= a * b * x / c / n;
            value += term;
        }

        return value;
    }

    template<typename function_type>
    static float_type integral(const float_type a, const float_type b,
        const float_type tol, function_type func)
    {
        unsigned n = 1U;

        float_type h = (b - a);
        float_type I = (func(a) + func(b)) * (h / 2);

        for (unsigned k = 0U; k < 10000U; k++)
        {
            h /= 2;

            float_type sum(0);
            for (unsigned j = 1U; j <= n; j++)
            {
                sum += func(a + (float_type((j * 2) - 1) * h));
            }

            const float_type I0 = I;
            I = (I / 2) + (h * sum);

            const float_type ratio = I0 / I;
            const float_type delta = ratio - 1;
            const float_type delta_abs = ((delta < 0) ? -delta : delta);

            if ((k > 1U) && (delta_abs < tol)) { break; }

            n *= 2U;
        }

        return I;
    }

    //TODO: Vectorize this function. Takes a lot of time!
    static float_type integralHelmholtz(int n1, int n2, float_type c) {
        return integral(-M_PI, M_PI, 1.0e-14, integrand(c, n1, n2));
    }

    static float_type integrandFcn(int n, int m, float_type c, float_type x) {
        float_type a = 4.0 - 2.0*std::cos(x)+c*c;
        float_type K = (a + std::sqrt(a*a - 4.0))/2.0;
        float_type int_val = (1.0 - std::cos(x*m)/std::pow(K,n))*(1/(K - 1.0/K));
        return int_val/2.0/M_PI;
    }

    static float_type integralHelmholtzTrap(int n1, int n2, float_type c) {
        
        auto f = [&](float_type x) {
            
            return integrandFcn(n1, n2, c, x);
        };
        return boost::math::quadrature::trapezoidal(f, -M_PI, M_PI, 1.0e-20);
    }

  private:
    static const int                     N_max_3D;
    static const std::vector<float_type> table_3D;

    
    static const int                     N_max_2D;
    static const std::vector<float_type> table_2D;

};

//decltype(Lookup::N_max) Lookup::N_max(100);

} // namespace lgf
} // namespace iblgf
#endif
