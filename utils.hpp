#ifndef INCLUDED_UTILS_HPP
#define INCLUDED_UTILS_HPP

#include <iostream>
#include <iomanip>

namespace octree{

template<typename T>
constexpr T pow(const T& base, const int exp)
{
    return  exp == 0 ? 1 : base*pow(base,exp-1);
}

template<int Dim, int D=Dim-1>
struct rcIterator
{
    template<class ArrayType, class Function>
    static void apply(const ArrayType& _base,
                      const ArrayType& _extent,
                      const Function& f)
    {
        auto p = _base;
        rcIterator<Dim, D>::apply_impl(p,f,_base, _extent);
    }
    template<class ArrayType, class Function>
    static void apply_impl(ArrayType& _p, 
                           const Function& f,
                           const ArrayType& _base, 
                           const ArrayType& _extent)
    {
        for(std::size_t k = 0; k < static_cast<std::size_t>(_extent[D]); ++k)
        {
            _p[D] = _base[D]+k;
            rcIterator<Dim, D-1>::apply_impl(_p,f, _base, _extent);
        }
    }
};

template<int Dim>
struct rcIterator<Dim,0>
{
    template<class ArrayType, class Function>
    static void apply_impl(ArrayType& _p, 
                           const Function& f,
                           const ArrayType& _base, 
                           const ArrayType& _extent)
    {
        for(std::size_t k = 0; k < static_cast<std::size_t>(_extent[0]); ++k)
        {
            _p[0] = _base[0]+k;
            f(_p);
        }
    }
};

} //namespace octree

#endif //INCLUDED_UTILS_HPP
