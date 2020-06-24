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

#ifndef INCLUDED_LGF_DOMAIN_DATAFIELD_UITILS_HPP
#define INCLUDED_LGF_DOMAIN_DATAFIELD_UITILS_HPP

#include <cstddef>
#include <tuple>
#include <utility>

namespace iblgf
{

namespace domain
{
namespace tuple_utils
{
template<class... T>
struct typelist
{
};

/***************************************************************************/
//Iterators:

/**
 * \brief Implemnetation to iterate over elements of a tuple 
 */
template<typename Tuple, typename F, std::size_t... Indices>
void
for_each_impl(Tuple&& tuple, F&& f, std::index_sequence<Indices...>)
{
    using swallow = int[];
    (void)swallow{1,
        (f(std::get<Indices>(std::forward<Tuple>(tuple))), void(), int{})...};
}

/**
 * \brief Iterate over all elements in a tuple and apply the functor f
 */
template<typename Tuple, typename F>
void
for_each(Tuple&& tuple, F&& f)
{
    constexpr std::size_t N =
        std::tuple_size<std::remove_reference_t<Tuple>>::value;
    for_each_impl(std::forward<Tuple>(tuple), std::forward<F>(f),
        std::make_index_sequence<N>{});
}

/**
 * \brief Iterate over all types in a tuple and apply the functor f
 */
template<typename... Ts>
struct TypeIterator
{
    template<typename F>
    static constexpr void for_types(F&& f)
    {
        (f.template operator()<Ts>(), ...);
    }
};

/***************************************************************************/
/**
 * \brief Concatenate tuples
 */
template<class... Tuples>
struct concat
{
    using type = decltype(std::tuple_cat(std::declval<Tuples>()...));
};

/**
 * \brief Append types to tuple
 */
template<class Tuple, class... Types>
struct append
{
    using type = decltype(std::tuple_cat(
        std::declval<Tuple>(), std::declval<std::tuple<Types...>>()));
};

/**
 * \brief Concatenate typelist of tuples and Types to one tuple
 */
template<class Tuples, class Types>
struct concat_all
{
};

template<class... Tuples, class... Types>
struct concat_all<typelist<Tuples...>, typelist<Types...>>
{
    using concat_tuple_t = typename concat<Tuples...>::type;
    using type = typename append<concat_tuple_t, Types...>::type;
};

/***************************************************************************/
/**
 * \brief Implements selet_tuple_element
 */
template<class U, class V>
struct select_tuple_element_impl;

template<class Tuple, size_t... Is>
struct select_tuple_element_impl<Tuple, std::index_sequence<Is...>>
{
    using type = std::tuple<typename std::tuple_element<Is, Tuple>::type...>;
};

/**
 * \brief Select the first N tuple elements and store it again
 */
template<class Tuple, std::size_t N = std::tuple_size<Tuple>::value>
struct select_tuple_elements
{
    using type = typename select_tuple_element_impl<Tuple,
        std::make_index_sequence<N>>::type;
};

/***************************************************************************/
/**
 * \brief Implementation of make_from_tuple
 */
template<template<typename...> class T, class Tuple, class V>
struct make_from_tuple_impl;

template<template<typename...> class T, class Tuple, size_t... Is>
struct make_from_tuple_impl<T, Tuple, std::index_sequence<Is...>>
{
    using type = T<typename std::tuple_element<Is, Tuple>::type...>;
};

/**
 * \brief Get type, which takes a parameter pack, from tuple
 */
template<template<typename...> class T, class Tuple>
struct make_from_tuple
{
    static constexpr std::size_t N =
        std::tuple_size<std::decay_t<Tuple>>::value;
    using type = typename make_from_tuple_impl<T, Tuple,
        std::make_index_sequence<N>>::type;
};

} // namespace tuple_utils

/**
 * \brief Assign fields 
 */
template<class Field0, class Field1, class BlockDescriptor, class Stride>
void
assign(const Field0& src, const BlockDescriptor& view_src, const Stride& strd_s,
    Field1& target, const BlockDescriptor& view_target, const Stride& strd_t)
{
    auto idxs = view_src.base();

    for (auto k = view_target.base()[2]; k <= view_target.max()[2];
         k += strd_t[2])
    {
        idxs[1] = view_src.base()[1];
        for (auto j = view_target.base()[1]; j <= view_target.max()[1];
             j += strd_t[1])
        {
            idxs[0] = view_src.base()[0];
            for (auto i = view_target.base()[0]; i <= view_target.max()[0];
                 i += strd_t[0])
            {
                target.get(i, j, k) = src.get(idxs[0], idxs[1], idxs[2]);
                idxs[0] += strd_s[0];
            }
            idxs[1] += strd_s[1];
        }
        idxs[2] += strd_s[2];
    }
}

/***************************************************************************/
/**
 * @brief Static id the generate/idnetify fields in tuple
 */
struct tuple_tag_h
{
    template<std::size_t N>
    constexpr tuple_tag_h(const char(&a)[N]) :id_(a), size_(N-1){}

    constexpr std::size_t size() const { return size_; }
    constexpr auto id() const { return id_; }
private:
    const char* const id_;
    const std::size_t size_;
};

template <char... id>
struct tuple_tag{
    static char const * c_str() {
        static constexpr char str[]={id...,'\0'};
        return str;
    }
    using tag_type=tuple_tag;
};

template<tuple_tag_h const& str,std::size_t... I>
auto constexpr expand(std::index_sequence<I...>){
    return tuple_tag<str.id()[I]...>{};
}
template <tuple_tag_h const& str>
using tag_type =
    decltype(expand<str>(std::make_index_sequence<str.size()>{}));

template< size_t I, typename T, typename Tuple_t>
constexpr size_t tagged_tuple_index_impl()
{
    typedef typename std::tuple_element<I,Tuple_t>::type el;
    if constexpr(std::is_same<T,typename el::tag_type>::value ){
        return I;
    }else
    {
        return tagged_tuple_index_impl<I+1,T,Tuple_t>();
    }
}

template<typename T, typename Tuple_t>
struct tagged_tuple_index
{
    static constexpr size_t value = tagged_tuple_index_impl<0,T,Tuple_t>();
};


} // namespace domain
} // namespace iblgf

#endif
