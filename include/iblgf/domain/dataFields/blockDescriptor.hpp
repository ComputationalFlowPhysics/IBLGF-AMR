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

#ifndef INCLUDED_LGF_DOMAIN_BLOCKDESCRIPTOR_HPP
#define INCLUDED_LGF_DOMAIN_BLOCKDESCRIPTOR_HPP

#include <algorithm>
#include <ostream>
#include <iostream>
#include <cmath>
#include <array>
#include <vector>
#include <fstream>
#include <iomanip>

#include <iblgf/types.hpp>
#include <iblgf/utilities/block_iterator.hpp>

namespace iblgf
{
namespace domain
{
namespace detail
{
template<std::size_t ND, std::size_t Dim, class BlockDescriptor>
struct subdivision_generator
{
    template<class CutIdx>
    static void apply(const std::array<std::vector<CutIdx>, Dim>& cuts,
        std::vector<BlockDescriptor>& blocks, int level = 0)
    {
        std::array<std::size_t, Dim> idx;
        apply_impl(cuts, idx, blocks, level);
    }

    template<class CutIdx>
    static void apply_impl(const std::array<std::vector<CutIdx>, Dim>& cuts,
        std::array<std::size_t, Dim>& idx, std::vector<BlockDescriptor>& blocks,
        int level)
    {
        idx[ND - 1] = 0;
        for (std::size_t i = 0; i < cuts[ND - 1].size() - 1; ++i)
        {
            subdivision_generator<ND - 1, Dim, BlockDescriptor>::apply_impl(
                cuts, idx, blocks, level);
            ++idx[ND - 1];
            idx[ND - 2] = 0;
        }
    }
};

template<std::size_t Dim, class BlockDescriptor>
struct subdivision_generator<1, Dim, BlockDescriptor>
{
    template<class CutIdx>
    static void apply_impl(const std::array<std::vector<CutIdx>, Dim>& cuts,
        std::array<std::size_t, Dim>& idx, std::vector<BlockDescriptor>& blocks,
        int level)
    {
        for (std::size_t i = 0; i < cuts[0].size() - 1; ++i)
        {
            BlockDescriptor b;
            auto            extent = b.extent();
            auto            base = b.extent();
            extent[0] = cuts[0][i + 1] - cuts[0][i];
            base[0] = cuts[0][i];
            for (std::size_t j = 1; j < Dim; ++j)
            {
                extent[j] = cuts[j][idx[j] + 1] - cuts[j][idx[j]];
                base[j] = cuts[j][idx[j]];
            }
            blocks.emplace_back(base, extent, level);
        }
    }
};

template<int Dim, int D = 0, class Enable = void>
struct get_corners_helper
{
    template<class ArrayType>
    static void apply(ArrayType& _p, const ArrayType& _base,
        const ArrayType& _extent, std::vector<ArrayType>& _points)
    {
        for (std::size_t k = 0; k < 2; ++k)
        {
            _p[D] = _base[D] + k * _extent[D];
            get_corners_helper<Dim, D + 1>::apply(_p, _base, _extent, _points);
        }
    }
};

template<int Dim, int D>
struct get_corners_helper<Dim, D, typename std::enable_if<D == Dim - 1>::type>
{
    template<class ArrayType>
    static void apply(ArrayType& _p, const ArrayType& _base,
        const ArrayType& _extent, std::vector<ArrayType>& _points)
    {
        for (std::size_t k = 0; k < 2; ++k)
        {
            _p[D] = _base[D] + k * _extent[D];
            _points.push_back(_p);
        }
    }
};

} //namespace detail

template<typename T, std::size_t Dim>
class BlockDescriptor
{
  public: //membery types:
    using data_type = T;
    using coordinate_type = typename types::vector_type<T, Dim>;
    using size_type = types::size_type;

  private: //Static members
    static constexpr auto dimension() { return Dim; }

  public:
    static std::vector<BlockDescriptor> generate_blocks_from_cuts(
        std::array<std::vector<T>, dimension()>& cut_locations, int _level = 0)
    {
        std::vector<BlockDescriptor> res;
        detail::subdivision_generator<dimension(), dimension(),
            BlockDescriptor>::apply(cut_locations, res, _level);
        return res;
    }

  public: //Ctor:
    BlockDescriptor() = default;
    ~BlockDescriptor() = default;

    BlockDescriptor(const BlockDescriptor& rhs) = default;
    BlockDescriptor& operator=(const BlockDescriptor& rhs) = default;

    BlockDescriptor(BlockDescriptor&& rhs) = default;
    BlockDescriptor& operator=(BlockDescriptor&& rhs) & = default;

    BlockDescriptor(
        coordinate_type _base, coordinate_type _extent, int _level = 0)
    : base_(_base)
    , extent_(_extent)
    , level_(_level)
    {
    }

  public: //Access
    coordinate_type&       base() noexcept { return base_; }
    const coordinate_type& base() const noexcept { return base_; }
    void base(const coordinate_type& _base) noexcept { base_ = _base; }

    coordinate_type&       extent() noexcept { return extent_; }
    const coordinate_type& extent() const noexcept { return extent_; }
    void extent(const coordinate_type& _extent) noexcept { extent_ = _extent; }

    coordinate_type&       min() noexcept { return base_; }
    const coordinate_type& min() const noexcept { return base_; }
    void min(const coordinate_type& _base) noexcept { base_ = _base; }

    coordinate_type max() const noexcept { return base_ + extent_ - 1; }
    void            max(const coordinate_type& _max) noexcept
    {
        extent_ = _max - base_ + 1;
    }

    const int& level() const noexcept { return level_; }
    int&       level() noexcept { return level_; }
    void       level(int _level) noexcept { level_ = _level; }

    /** @brief Get number of integer points within block */
    std::size_t size() const noexcept
    {
        std::size_t ext = 1;
        for (auto e : extent_) ext *= e;
        return ext;
    }

  public: //members
    /** @brief Check block is not empty
     *  @return True if block is empty
     */
    bool is_empty() const noexcept { return extent_ <= coordinate_type(0); }

    /** @brief Grow left and right corners of block
     *  @param _left_buffer Amount to grow the left corner
     *  @param _right_buffer Amount to grow the right corner
     */
    template<typename Btype>
    void grow(Btype _lBuffer, Btype _rBuffer) noexcept
    {
        base_ -= _lBuffer;
        extent_ += _lBuffer + _rBuffer;
    }

    /** @brief Scale block_descriptor to certain level.
     *
     *  Note that if exact scaling down is not possible, the block is shrunk and yields
     *  base+=1 and max-=1. Then, scaling down and up will not yield the original block.
     *
     *   @param[in] _level:  Level to wich block is scaled
     */
    void level_scale(int _level) noexcept
    {
        const auto levelDifference = std::abs(level_ - _level);
        const auto factor = static_cast<int>(std::pow(2, levelDifference));
        if (_level > level_)
        {
            base_ *= factor;
            extent_ *= factor;
        }
        else
        {
            //If no exact conversion possible:
            // => shrink it, e.g. base+=1, max-=1;
            auto max = this->max();

            for (std::size_t d = 0; d < extent_.size(); ++d)
            {
                if (base_[d] < 0) base_[d] = (base_[d] + 1) / factor - 1;
                else
                    base_[d] = base_[d] / factor;

                if (max[d] < 0) max[d] = (max[d] + 1) / factor - 1;
                else
                    max[d] = max[d] / factor;
            }
            this->max(max);
        }
        level_ = _level;
    }

    /** @brief Shift base of block_descriptor
     *  @param _shift Amount the base is shifted.
     *  @tparam Shift Type of shift. Any coordinate_type for which +=-operator with coordinate_type
     *   is overloaded.
     * */
    template<class Shift>
    void shift(Shift _shift) noexcept
    {
        base_ += _shift;
    }

    /** @brief Get a plane of the block.
     *  @param idx index of normal, i.e. 0=x-normal, 1=y-normal, 2 is z-normal
     *  @param dir=1; plane with positive normal, else negative normal
     */
    BlockDescriptor bcPlane(int idx, int dir) noexcept
    {
        auto p = *this;
        for (std::size_t d = 0; d < Dim; ++d)
        {
            if (dir == 1) p.base()[idx] = max()[idx];
            p.extent()[idx] = 1;
        }
        return p;
    }

    /** @brief Enlarge block such that _block is inside.
     *  @param _block Block to be included. */
    void enlarge_to_fit(const BlockDescriptor& _b) noexcept
    {
        auto max = this->max();
        for (std::size_t d = 0; d < Dim; ++d)
        {
            base_[d] = std::min(base_[d], _b.base()[d]);
            max[d] = std::max(max[d], _b.max()[d]);
        }
        this->max(max);
    }

    /** @brief Check if Point p is inside the block.
     *  @param Query point
     *  @return true if point is on block boundary
     */
    template<class PointType>
    bool is_inside(const PointType& p) const noexcept
    {
        for (std::size_t d = 0; d < p.size(); ++d)
        {
            if (p[d] < base_[d] || p[d] > max()[d]) return false;
        }
        return true;
    }

    /** @brief Check if Point p is on the block boundary
     *  @param Query point
     *  @return true if point is on block boundary
     */
    template<class PointType>
    bool on_boundary(const PointType& p) const noexcept
    {
        return (on_max_boundary(p) || on_min_boundary());
    }

    /** @brief Check if Point p is on the min block boundary
     *  @param Query point
     *  @return true if point is on min block boundary
     */
    template<class PointType>
    bool on_min_boundary(const PointType& p) const noexcept
    {
        for (std::size_t d = 0; d < p.size(); ++d)
        {
            if (p[d] == base()[d]) return true;
        }
        return false;
    }

    /** @brief Check if Point p is on the max block boundary
     *  @param Query point
     *  @return true if point is on max block boundary
     */
    template<class PointType>
    bool on_max_boundary(const PointType& p) const noexcept
    {
        for (std::size_t d = 0; d < p.size(); ++d)
        {
            if (p[d] == max()[d]) return true;
        }
        return false;
    }

  public: // indices & iterate
    /** @{
     *  @brief Compute flat index of point in 3D or 2D-block
     *
     *  Flat index takes base into account. That mean base() has index=0, which
     *  does not necessarily correspond the coordinate_type(0)
     *
     *  @param Point, either as point-type or ijk 
     *  @return Flat index
     */
    template<class PointType, int D = Dim,
        typename std::enable_if<D == 3, void>::type* = nullptr>
    inline size_type index(const PointType& p) const noexcept
    {
        return p[0] - base_[0] + extent_[0] * (p[1] - base_[1]) +
               extent_[0] * extent_[1] * (p[2] - base_[2]);
    }
    inline size_type index(int i, int j, int k) const noexcept
    {
        return i - base_[0] + extent_[0] * (j - base_[1]) +
               extent_[0] * extent_[1] * (k - base_[2]);
    }
    template<class PointType, int D = Dim,
        typename std::enable_if<D == 2, void>::type* = nullptr>
    inline size_type index(const PointType& p) const noexcept
    {
        return p[0] - base_[0] + extent_[0] * (p[1] - base_[1]);
    }
    inline size_type index(int i, int j) const noexcept
    {
        return i - base_[0] + extent_[0] * (j - base_[1]);
    }
    /** @} */

    /** @{
     *  @brief Compute flat index of point in 3D or 2D-block without considering base of the block
     *
     *  Flat index does not take into account base. That mean coordinate_type(0) does always
     *  have the index 0
     *
     *  @param Point Either as point-type (type supporting []-operator) or ijk 
     *  @return Flat index
     */
    template<class PointType, int D = Dim,
        typename std::enable_if<D == 3, void>::type* = nullptr>
    inline size_type index_zeroBase(const PointType& p) const noexcept
    {
        return p[0] + extent_[0] * p[1] + extent_[0] * extent_[1] * p[2];
    }
    inline size_type index_zeroBase(int i, int j, int k) const noexcept
    {
        return i + extent_[0] * (j) + extent_[0] * extent_[1] * (k);
    }
    template<class PointType, int D = Dim,
        typename std::enable_if<D == 2, void>::type* = nullptr>
    inline size_type index_zeroBase(const PointType& p) const noexcept
    {
        return p[0] + extent_[0] * p[1];
    }
    inline size_type index_zeroBase(int i, int j) const noexcept
    {
        return i + extent_[0] * j;
    }
    /** @} */

    /** @brief Get the overlap block with input block
     *  @param other Block with whom the overlap is computed
     *  @param overlap Overlap block between this and other.
     */
    template<class BlockType, class OverlapType>
    bool overlap(BlockType other, OverlapType& overlap) const noexcept
    {
        if (other.level() != level_) other.level_scale(level_);

        overlap = other;

        for (std::size_t d = 0; d < overlap.extent().size(); ++d)
        {
            overlap.base()[d] = std::max(base_[d], other.base()[d]);
            overlap.extent()[d] = std::min(base_[d] + extent_[d],
                                      other.base()[d] + other.extent()[d]) -
                                  overlap.base()[d];

            if (overlap.extent()[d] < 1) return false;
        }
        return true;
    }

    /** @brief Get the overlap block with input block
     *  @param other Block with whom the overlap is computed
     *  @param overlap Overlap block between this and other.
     *  @param _level Level at which overlap is computed
     */
    template<class BlockType, class OverlapType>
    bool overlap(
        const BlockType& other, OverlapType& overlap, int _level) const noexcept
    {
        auto this_scaled = *this;
        auto other_scaled = other;
        this_scaled.level_scale(_level);
        other_scaled.level_scale(_level);
        if (this_scaled.overlap(other_scaled, overlap)) return true;
        return false;
    }

    /** @brief Get corner points/coordinates of this block */
    auto get_corners() const noexcept
    {
        std::vector<coordinate_type> points;
        coordinate_type              p = base_;
        detail::get_corners_helper<Dim>::apply(p, base_, extent_, points);
        return points;
    }
    /** @brief Divide this block into block with extent e */
    auto divide_into(const coordinate_type& _e) const
    {
        std::array<std::vector<T>, dimension()> cuts;
        for (std::size_t d = 0; d < _e.size(); ++d)
        {
            if (extent_[d] < _e[d])
            {
                throw std::runtime_error(
                    "BlockDrscriptor: Cannot divide box, too small");
            }
            int nBoxes = (extent_[d] + _e[d] - 1) / _e[d];
            for (int i = 0; i < nBoxes; ++i)
            { cuts[d].push_back(i * _e[d] + base_[d]); }
            cuts[d].push_back(base_[d] + extent_[d]);
        }

        return generate_blocks_from_cuts(cuts, this->level());
    }

    /** @brief Iterate over points in block in ijk fashion
     *
     *  @tparam Function - function type, i.e. lambda
     *  @param _f Functor which is called as  _f(p)
     */
    template<class Function>
    void ijk_iterate(const Function& _f) const noexcept
    {
        BlockIterator<Dim>::iterate(base_, extent_, _f);
    }

  public:
    /** @brief Output operator */
    friend std::ostream& operator<<(std::ostream& os, const BlockDescriptor& b)
    {
        os << "Base: " << b.base_ << " extent: " << b.extent_
           << " max: " << b.max() << " level: " << b.level();
        return os;
    }

  protected:
    coordinate_type base_;      ///< lower-left corner of block
    coordinate_type extent_;    ///< extent of block
    int             level_ = 0; ///< refinement level of block
};

} // namespace domain
} // namespace iblgf

#endif
