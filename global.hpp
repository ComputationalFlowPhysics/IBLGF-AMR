#ifndef OCTREE_INCLUDED_GLOBAL_HPP
#define OCTREE_INCLUDED_GLOBAL_HPP

#include "types.hpp"

using namespace types;
   template <typename Tuple, typename F, std::size_t ...Indices>
    void for_each_impl2(Tuple&& tuple, F&& f, std::index_sequence<Indices...>) 
    {
        using swallow = int[];
        (void)swallow{1, (f(std::get<Indices>(std::forward<Tuple>(tuple))), void(), int{})...  };
    }

    template <typename Tuple, typename F>
    void for_each2(Tuple&& tuple, F&& f) 
    {
        constexpr std::size_t N = std::tuple_size<std::remove_reference_t<Tuple>>::value;
        for_each_impl2(std::forward<Tuple>(tuple), std::forward<F>(f),
                std::make_index_sequence<N>{});
    }



#endif // OCTREE_INCLUDED_GLOBAL_HPP
