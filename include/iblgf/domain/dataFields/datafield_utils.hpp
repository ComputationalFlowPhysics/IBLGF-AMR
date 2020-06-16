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


} // namespace domain
} // namespace iblgf

#endif
