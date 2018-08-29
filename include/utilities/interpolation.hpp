#ifndef INCLUDED_INTERPOLATION_HPP
#define INCLUDED_INTERPOLATION_HPP


#include <cassert>
#include <cstring>
#include <vector>
#include <complex.h>
#include <global.hpp>

namespace interpolation
{

template<class Values, class Stencil>
float_type lagrange_interpolation_1D(const Values& _values, const Stencil& _stencil,  float_type _p )
{
    float_type res=0;
    for(std::size_t i=0; i<_stencil.size();++i)
    {
        float_type l = 1.0;
        auto xi =_stencil[i];
        for(std::size_t j=0; j<_stencil.size();++j)
        {
            if(i==j)continue;
            auto xj =_stencil[j];
            l*= (_p-xj)/static_cast<float_type>((xi-xj));
        }
        res+=l*_values[i];
    }
    return res;
}



template<class Field>
inline float_type interpolate(int min_x, int min_y, int min_z, 
                               float_type x, float_type  y, float_type z,
                               const Field& _field, int _stride=1)
{
    constexpr long unsigned int N = 2;
    std::array<int, N> x_pos{{(min_x), (min_x+_stride)}};
    std::array<int, N> y_pos{{(min_y), (min_y+_stride)}};
    std::array<int, N> z_pos{{(min_z), (min_z+_stride)}};
    
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
            y_values[j] = lagrange_interpolation_1D(z_values, z_pos, z);
        }
        x_values[i] = lagrange_interpolation_1D(y_values, y_pos, y);
    }
    return lagrange_interpolation_1D( x_values,x_pos,x);
}

} //namespace
#endif
