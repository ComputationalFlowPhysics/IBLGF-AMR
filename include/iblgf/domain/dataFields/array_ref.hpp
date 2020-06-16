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


#ifndef INCLUDED_LGF_DOMAIN_ARRAYREF_HPP
#define INCLUDED_LGF_DOMAIN_ARRAYREF_HPP

#include <vector>
#include <iostream>
#include <boost/multi_array.hpp>

// IBLGF-specific
#include <iblgf/types.hpp>
#include <iblgf/domain/dataFields/datafield_utils.hpp>
#include <iblgf/domain/dataFields/blockDescriptor.hpp>

namespace iblgf
{
namespace domain
{
//Exent generators
namespace detail
{
template<std::size_t Dimension>
using extent_type = boost::detail::multi_array::extent_gen<Dimension>;

template<std::size_t NR, std::size_t ND>
using index_gen_type = boost::detail::multi_array::index_gen<NR, ND>;

using extent_range = boost::multi_array_types::extent_range;
using index_range = boost::multi_array_types::index_range;

template<std::size_t ND, std::size_t NumDims>
struct extent_generator
{
    template<typename Extent>
    static extent_type<ND> apply(
        const Extent& index_base, const Extent& extents)
    {
        return extent_generator<ND - 1, NumDims>::apply(
            index_base, extents)[extent_range(
            index_base[ND - 1], index_base[ND - 1] + extents[ND - 1])];
    }
};

template<std::size_t NumDims>
struct extent_generator<1, NumDims>
{
    template<typename Extent>
    static extent_type<1> apply(const Extent& index_base, const Extent& extents)
    {
        return boost::extents[extent_range(
            index_base[0], index_base[0] + extents[0])];
    }
};

template<std::size_t ND, std::size_t NumDims>
struct index_generator
{
    template<typename Extent>
    static index_gen_type<ND, ND> apply(
        const Extent& index_base, const Extent& extents, const Extent& strides)
    {
        return index_generator<ND - 1, NumDims>::apply(
            index_base, extents, strides)[index_range(index_base[ND - 1],
            index_base[ND - 1] + extents[ND - 1], strides[ND - 1])];
    }

    template<typename Extent>
    static index_gen_type<ND, ND> apply(
        const Extent& index_base, const Extent& extents)
    {
        return index_generator<ND - 1, NumDims>::apply(
            index_base, extents)[index_range(
            index_base[ND - 1], index_base[ND - 1] + extents[ND - 1])];
    }
};

template<std::size_t NumDims>
struct index_generator<1, NumDims>
{
    template<typename Extent>
    static index_gen_type<1, 1> apply(
        const Extent& index_base, const Extent& extents, const Extent& strides)
    {
        return boost::indices[index_range(
            index_base[0], index_base[0] + extents[0], strides[0])];
    }

    template<typename Extent>
    static index_gen_type<1, 1> apply(
        const Extent& index_base, const Extent& extents)
    {
        return boost::indices[index_range(
            index_base[0], index_base[0] + extents[0])];
    }
};

} // namespace detail

template<class DataType, int Dim>
class Multiarray_ref
{
  public: //member types
    using data_type = DataType;
    using boost_multi_array_ref = boost::multi_array_ref<DataType, Dim>;
    using view_type =
        typename boost_multi_array_ref::template array_view<Dim>::type;
    using block_type = BlockDescriptor<int, Dim>;
    using extent_t = typename block_type::extent_t;

  public: //Ctors:
    Multiarray_ref() = default;
    ~Multiarray_ref() = default;
    Multiarray_ref(const Multiarray_ref& rhs) = default;
    Multiarray_ref& operator=(const Multiarray_ref&) & = default;

    Multiarray_ref(Multiarray_ref&& rhs) = default;
    Multiarray_ref& operator=(Multiarray_ref&&) & = default;

    Multiarray_ref(DataType* _d, block_type _domain)
    : multi_array_ref_(_d,
          detail::extent_generator<Dim, Dim>::apply(
              _domain.base(), _domain.extent()),
          boost::fortran_storage_order())
    {
    }

    //* @brief get view of the domain */
    view_type get_view(const block_type& _b, extent_t _stride) noexcept
    {
        return multi_array_ref_[detail::index_generator<Dim, Dim>::apply(
            _b.base(), _b.extent(), _stride)];
    }

    view_type get_view(const block_type& _b) noexcept
    {
        return multi_array_ref_[detail::index_generator<Dim, Dim>::apply(
            _b.base(), _b.extent())];
    }

  public:  //member functions
  private: //protected memeber:
    boost_multi_array_ref multi_array_ref_;
};

} //namespace domain
} // namespace iblgf

#endif
