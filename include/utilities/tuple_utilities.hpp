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

#ifndef INCLUDED_LGF_TUPLE_UTILITIES_HPP
#define INCLUDED_LGF_TUPLE_UTILITIES_HPP

namespace tuple_utils
{
template<typename Tuple, typename F, std::size_t... Indices>
void
for_each_impl(Tuple&& tuple, F&& f, std::index_sequence<Indices...>)
{
    using swallow = int[];
    (void)swallow{1,
        (f(std::get<Indices>(std::forward<Tuple>(tuple))), void(), int{})...};
}

template<typename Tuple, typename F>
void
for_each(Tuple&& tuple, F&& f)
{
    constexpr std::size_t N =
        std::tuple_size<std::remove_reference_t<Tuple>>::value;
    for_each_impl(std::forward<Tuple>(tuple), std::forward<F>(f),
        std::make_index_sequence<N>{});
}

template<typename... Ts>
struct TypeIterator
{
    template<typename F>
    static constexpr void for_types(F&& f)
    {
        (f.template operator()<Ts>(), ...);
    }
};

template<template<typename...> class T, class Tuple, class V>
struct make_from_tuple_impl;

template<template<typename...> class T, class Tuple, size_t... Is>
struct make_from_tuple_impl<T, Tuple, std::index_sequence<Is...>>
{
    using type = T<typename std::tuple_element<Is, Tuple>::type...>;
};
template<template<typename...> class T, class Tuple>
struct make_from_tuple
{
    static constexpr std::size_t N =
        std::tuple_size<std::decay_t<Tuple>>::value;
    using type = typename make_from_tuple_impl<T, Tuple,
        std::make_index_sequence<N>>::type;
};

} // namespace tuple_utils

#endif
