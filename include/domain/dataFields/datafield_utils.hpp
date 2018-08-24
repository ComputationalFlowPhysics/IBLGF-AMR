#ifndef INCLUDED_LGF_DOMAIN_DATAFIELD_UITILS_HPP
#define INCLUDED_LGF_DOMAIN_DATAFIELD_UITILS_HPP

#include <cstddef>
#include <tuple>
#include <utility>

namespace domain
{
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


template <typename... Ts>
struct TypeIterator
{
    template<typename F>
    static constexpr void for_types(F&& f)
    {
        (f.template operator()<Ts>(), ...);
    }
};

}

template<class Field0, class Field1, class BlockDescriptor, class Stride>
void assign(const Field0& src, const BlockDescriptor& view_src, 
            const Stride& strd_s,
            Field1& target, const BlockDescriptor& view_target, 
            const Stride& strd_t)
{
    auto idxs=view_src.base();

    for(auto k = view_target.base()[2]; 
             k <= view_target.max()[2]; k+=strd_t[2])
    {
        idxs[1]=view_src.base()[1];
        for(auto j = view_target.base()[1]; 
                 j <= view_target.max()[1]; j+=strd_t[1])
        { 
            idxs[0]=view_src.base()[0];
            for(auto i = view_target.base()[0]; 
                     i <= view_target.max()[0]; i+=strd_t[0])
            {
                target.get(i,j,k)=src.get(idxs[0], idxs[1], idxs[2]);
                idxs[0]+=strd_s[0];
            }
            idxs[1]+=strd_s[1];
        }
        idxs[2]+=strd_s[2];
    }
}


#define crtp_helper(DerivedType,func_name) \
DerivedType* func_name(){return static_cast<DerivedType*>(this);} \
const DerivedType* func_name()const {return static_cast<DerivedType*>(this);} 

}

#endif 
