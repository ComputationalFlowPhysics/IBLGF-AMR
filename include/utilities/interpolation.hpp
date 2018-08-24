#ifndef INCLUDED_INTERPOLATION_HPP
#define INCLUDED_INTERPOLATION_HPP


#include <cassert>
#include <cstring>
#include <vector>
#include <complex.h>
#include <global.hpp>

namespace interpolation
{


template<long unsigned int N>
float_type lagrange_interpolate_1D(float_type x, const std::array<float_type, N>& y)
{
    float_type res = 0;
    for (int j=0; j<(int)N; ++j)
    {
        float_type l_j = 1.0;
        for (int m=0; m<(int)N; ++m)
            if (m!=j)
                l_j *= (x-m)/(j-m);
        res += y[j]*l_j;
    }
    return res;
}

template<class Field>
inline float_type interpolate(int min_x, int min_y, int min_z, 
                              float_type x, float_type  y, float_type z,const Field& _field )
{
   
    //constexpr long unsigned int N = 4;
    //std::array<int, N> x_pos{{(min_x-1), (min_x), (min_x+1), (min_x+2)}};
    //std::array<int, N> y_pos{{(min_y-1), (min_y), (min_y+1), (min_y+2)}};
    //std::array<int, N> z_pos{{(min_z-1), (min_z), (min_z+1), (min_z+2)}};
    
    constexpr long unsigned int N = 2;
    std::array<int, N> x_pos{{(min_x), (min_x+1)}};
    std::array<int, N> y_pos{{(min_y), (min_y+1)}};
    std::array<int, N> z_pos{{(min_z), (min_z+1)}};
    
    const float_type x_ref = x - x_pos[0];
    const float_type y_ref = y - y_pos[0];
    const float_type z_ref = z - z_pos[0];


    std::array<float_type,N> x_values;
    for (unsigned int i=0; i<N; ++i)
    {
        std::array<float_type,N> y_values;
        for (unsigned int j=0; j<N; ++j)
        {
            std::array<float_type,N> z_values;
            for (unsigned int k=0; k<N; ++k)
            {
                z_values[k] = _field.get( x_pos[i], y_pos[j], z_pos[k]);
            }
            y_values[j] = lagrange_interpolate_1D(z_ref, z_values);
        }
        x_values[i] = lagrange_interpolate_1D(y_ref, y_values);
    }
    return lagrange_interpolate_1D(x_ref, x_values);
}
} //namespace
#endif

