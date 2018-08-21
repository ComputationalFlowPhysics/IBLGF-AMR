#ifndef INCLUDED_TYPES_HPP
#define INCLUDED_TYPES_HPP

#include <cstddef>
#include <tensor/vector.hpp>


namespace types
{
    using float_type =double;
    using size_type = std::size_t;
    using uint_type = unsigned int;
    using index_type = int;

    template<typename U,std::size_t Dim>
    using vector_type = math::vector<U,Dim>;

    template<typename U,std::size_t Dim>
    using coordinate_type = vector_type<U,Dim>;

    template<class D=void>
    class void_mixin{};


}//namespace types



#endif
