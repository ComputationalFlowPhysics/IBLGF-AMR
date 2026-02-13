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

#ifndef INCLUDED_LGF_GL_LOOKUP_HPP
#define INCLUDED_LGF_GL_LOOKUP_HPP

#include <algorithm>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <array>
#include <cmath>

#include <iblgf/global.hpp>

namespace iblgf
{
namespace lgf
{
class LGF_GL_Lookup
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
    static typename std::enable_if<Coordinate::size() == 2, float_type>::type get(const Coordinate& _c) noexcept
    {
	int x = std::abs(static_cast<int>(_c.x()));
	int y = std::abs(static_cast<int>(_c.y()));

	if (x <= N_max_2D && y <= N_max_2D) // using table
	{
		if (x < y) std::swap(x, y);
		return -table_2D[x*(N_max_2D+2)+y]; 
	}
	else
	{
		return -asym(x, y);
	}
    }

    //TODO: Vectorize this function. Takes a lot of time!
    static float_type asym(int n1, int n2, int n3)
    {
        const float_type n1_2 = n1 * n1, n2_2 = n2 * n2, n3_2 = n3 * n3;
        const float_type n_abs = sqrt(n1_2 + n2_2 + n3_2);

        const float_type n1_6 = n1_2 * n1_2 * n1_2;
        const float_type n2_6 = n2_2 * n2_2 * n2_2;
        const float_type n3_6 = n3_2 * n3_2 * n3_2;

        const float_type n1_8 = n1_6 * n1_2;
        const float_type n2_8 = n2_6 * n2_2;
        const float_type n3_8 = n3_6 * n3_2;

        const float_type n_abs_6 =
            n_abs * n_abs * n_abs * n_abs * n_abs * n_abs;
        const float_type n_abs_7 = n_abs_6 * n_abs;
        const float_type n_abs_13 = n_abs_6 * n_abs_6 * n_abs;

        float_type tmp = -1.0 / 4.0 / M_PI / n_abs; // the first asymp term
        tmp =
            tmp - (n1_2 * n1_2 + n2_2 * n2_2 + n3_2 * n3_2 - 3.0 * n1_2 * n2_2 -
                      3.0 * n2_2 * n3_2 - 3.0 * n3_2 * n1_2) /
                      16.0 / M_PI / n_abs_7; // the second asymp term

        if (n1 < 600 || n2 < 600 || n3 < 600) // add the third term
        {
            float_type       tmp2;
            const float_type coef = 8.0 / (768.0 * M_PI);

            tmp2 =
                -3.0 * (23 * (n1_8 + n2_8 + n3_8) -
                           244 * (n2_6 * n3_2 + n3_6 * n2_2 + n1_6 * n2_2 +
                                     n2_6 * n1_2 + n1_6 * n3_2 + n3_6 * n1_2) +
                           621 * ((n1_2 * n1_2) * (n2_2 * n2_2) +
                                     (n2_2 * n2_2) * (n3_2 * n3_2) +
                                     (n3_2 * n3_2) * (n1_2 * n1_2)) -
                           228 * ((n1_2 * n1_2) * n2_2 * n3_2 +
                                     n1_2 * (n2_2 * n2_2) * n3_2 +
                                     n1_2 * n2_2 * (n3_2 * n3_2)));

            tmp2 = tmp2 / n_abs_13 / 4.0 * coef;
            tmp += tmp2;
        }

        return tmp;
    }

    static float_type asym(int n1, int n2) {
	    float_type Cfund = -0.18124942796; // constant for asymptotic expansion (C_fund)
	    float_type Integral_val = -0.076093998454228; // constant for asymptotical expansion (integral)	
	    float_type r = std::sqrt(n1*n1 + n2*n2);

	    float_type theta = atan2(n2,n1);
	    float_type second_term = 1.0/24.0/M_PI*cos(4.0*theta)/r/r;
	    float_type res = -0.5/M_PI*log(r) + Cfund + Integral_val + second_term;
	    return res;
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
