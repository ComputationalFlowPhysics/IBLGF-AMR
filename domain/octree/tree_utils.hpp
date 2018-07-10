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
        auto p=_base;
        rcIterator<Dim, D>::apply_impl(p,f,_base, _extent );
    }
    template<class ArrayType, class Function>
    static void apply_impl(ArrayType& _p, 
                           const Function& f,
                           const ArrayType& _base, 
                           const ArrayType& _extent
                       )
    {
        for(std::size_t k=0; k<static_cast<std::size_t>(_extent[D]);++k)
        {
            _p[D]=_base[D]+k;
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
                           const ArrayType& _extent )
    {
        for(std::size_t k=0; k<static_cast<std::size_t>(_extent[0]);++k)
        {
            _p[0]=_base[0]+k;
            f(_p);
        }
    }
};

template<class MapType>
class MapKeyIterator : public MapType::iterator 
{

public:
    using typename MapType::mapped_type;
    using typename MapType::key_type;
    using iterator_t =typename MapType::iterator;

public:

    MapKeyIterator()
    :iterator_t(){};

    MapKeyIterator(iterator_t it_ ) 
    :iterator_t(it_){};

public:
    key_type* operator->() noexcept
    {
        return (key_type* const )&( iterator_t::operator -> ( )->first ); 
    }
    const key_type& operator*(){return iterator_t::operator*().first; }
};

template<class MapType>
class MapValueIterator : public MapType::iterator 
{
public:
    using mapped_type =typename MapType::mapped_type;
    using key_type =typename MapType::key_type;
    using iterator_t =typename MapType::iterator;

public:
    MapValueIterator()
    : iterator_t(){}

    MapValueIterator( const iterator_t& _it )
    : iterator_t(_it){}

    mapped_type* operator->() noexcept
    { 
        return (mapped_type* const )&( iterator_t::operator->()->second ); 
    }
    const mapped_type& operator*() { return iterator_t::operator*().second; }
};

namespace tuple_utils
{

    template <typename Tuple, typename F, std::size_t ...Indices>
    void for_each_impl(Tuple&& tuple, F&& f, std::index_sequence<Indices...>) 
    {
        using swallow = int[];
        (void)swallow{1, (f(std::get<Indices>(std::forward<Tuple>(tuple))), void(), int{})...  };
    }

    template <typename Tuple, typename F>
    void for_each(Tuple&& tuple, F&& f) 
    {
        constexpr std::size_t N = std::tuple_size<std::remove_reference_t<Tuple>>::value;
        for_each_impl(std::forward<Tuple>(tuple), std::forward<F>(f),
                std::make_index_sequence<N>{});
    }
}


} //namespace octree

#endif //INCLUDED_UTILS_HPP
