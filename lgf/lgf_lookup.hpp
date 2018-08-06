#ifndef INCLUDED_LGF_LOOKUP_HPP
#define INCLUDED_LGF_LOOKUP_HPP

#include <algorithm>    // std::swap
#include <stdio.h>
#include <iostream>
#include <vector>
#include <array>
#include <cmath>


namespace lgf
{

class Lookup
{
public:

    template<class Coordinate>
    static auto get(const Coordinate& _c) noexcept
    {
        int x = abs(_c.x());
        int y = abs(_c.y());
        int z = abs(_c.z());

        if (x<= N_max && y<=N_max && z<=N_max) 
        {
            if (x < y)
                std::swap(x, y);
            if (x < z)
                std::swap(x, z);
            if (y < z)
                std::swap(y, z);

            return -Q[(x * (2 + 3*x + x * x))/6 + y*(y+1)/2 + z];

        } else
        {
            
            return asym(x,y,z);
        }

   }

    static auto asym(int n1, int n2, int n3)
    {
        double n1_2 = n1 * n1, n2_2 = n2 * n2, n3_2 = n3 * n3;
        double n_abs = sqrt(n1_2 + n2_2 + n3_2);
        double tmp;
        
        tmp = -1.0/4.0/M_PI/n_abs; // first term
        tmp = tmp - (n1_2 * n1_2 + n2_2 * n2_2 + n3_2 * n3_2 
                    - 3.0 * n1_2 * n2_2 
                    - 3.0 * n2_2 * n3_2 
                    - 3.0 * n3_2 * n1_2)/16.0/M_PI/pow(n_abs,7.0);
                                    // second term

        // n3 is the smallest by definition
        if (n1<600 || n2<600 || n3<600) // add third term
        {
            double tmp2;
            const double coef = 8.0/(768.0 * M_PI);

            tmp2 = -3.0 * (   23 * (pow(n1_2,4.0) + pow(n2_2,4.0) + pow(n3_2,4.0)) 
                    - 244 * (pow(n2_2,3.0) * n3_2 + pow(n3_2,3.0) * n2_2 + 
                             pow(n1_2,3.0) * n2_2 + pow(n2_2,3.0) * n1_2 + 
                             pow(n1_2,3.0) * n3_2 + pow(n3_2,3.0) * n1_2  
                             )
                    + 621 * ((n1_2 * n1_2) * (n2_2 * n2_2) 
                        + (n2_2 * n2_2) * (n3_2 * n3_2) 
                        + (n3_2 * n3_2) * (n1_2 * n1_2) )
                    - 228 * ( (n1_2 * n1_2) * n2_2 * n3_2 
                             + n1_2 * (n2_2 * n2_2) * n3_2
                             + n1_2 * n2_2 * (n3_2 * n3_2)));

            tmp2 = tmp2 / pow(n_abs, 13.0) /4.0 * coef;
            tmp += tmp2;
        }

        return tmp;
    }

private:

    static constexpr std::array<double, 23426> Q
    {{
    #include "lgf/lgf_table_50.hpp"
    }};

    static constexpr int N_max=50;

};

constexpr decltype(Lookup::Q) Lookup::Q;

} //namepsace
#endif
