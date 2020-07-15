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

#ifndef INCLUDED_IBLGF_BLOCK_ITERATOR_HPP
#define INCLUDED_IBLGF_BLOCK_ITERATOR_HPP

namespace iblgf
{
/**
 *  @brief Block iterator for ijk-iteration from base to extent
 *
 *  @tparam Dim Spatial dimesnion
 *  @tparam D   Dim-1 as defaults
 */
template<int Dim, int D = Dim - 1>
struct BlockIterator
{
    /** @{
     *  @brief Iterate from base to extent (with stride) and apply a functor
     *  to the coordinate
     *
     *  @tparam ArrayType Any array, vector or container supporting []-operator.
     *  @tparam Funtion Any Functor with signature void(ArrayType _p)
     */
    template<class ArrayType, class Function>
    static void iterate(
        const ArrayType& _base, const ArrayType& _extent, const Function& f)
    {
        auto p = _base;
        BlockIterator<Dim, D>::iterate_impl(p, f, _base, _extent, ArrayType(1));
    }
    template<class ArrayType, class Function>
    static void iterate(const ArrayType& _base, const ArrayType& _extent,
        const ArrayType& _stride, const Function& f)
    {
        auto p = _base;
        BlockIterator<Dim, D>::iterate_impl(p, f, _base, _extent, _stride);
    }
    template<class ArrayType, class Function>
    static void iterate(const ArrayType& _extent, const Function& f)
    {
        const ArrayType base(0);
        auto            p = base;
        BlockIterator<Dim, D>::iterate_impl(p, f, base, _extent, ArrayType(1));
    }
    /** @} */

    template<class ArrayType, class Function>
    static void iterate_impl(ArrayType& _p, const Function& f,
        const ArrayType& _base, const ArrayType& _extent,
        const ArrayType& _stride)
    {
        for (std::size_t k = 0; k < static_cast<std::size_t>(_extent[D]);
             k += _stride[D])
        {
            _p[D] = _base[D] + k;
            BlockIterator<Dim, D - 1>::iterate_impl(
                _p, f, _base, _extent, _stride);
        }
    }
};

template<int Dim>
struct BlockIterator<Dim, 0>
{
    template<class ArrayType, class Function>
    static void iterate_impl(ArrayType& _p, const Function& f,
        const ArrayType& _base, const ArrayType& _extent,
        const ArrayType& _stride)
    {
        for (std::size_t k = 0; k < static_cast<std::size_t>(_extent[0]);
             k += _stride[0])
        {
            _p[0] = _base[0] + k;
            f(_p);
        }
    }
};
} // namespace iblgf

#endif
