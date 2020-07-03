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
    template<typename U>
    using vector_t = typename types::vector_type<U, Dim>;

    using coordinate_type = typename types::vector_type<T, Dim>;

    //using base_t = vector_t<T>;
    //using extent_t = coordinate_type;

    using min_t = coordinate_type;
    using max_t = coordinate_type;
    using data_type = T;

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

    min_t&       min() noexcept { return base_; }
    const min_t& min() const noexcept { return base_; }
    void         min(const min_t& _base) noexcept { base_ = _base; }

    max_t max() const noexcept { return base_ + extent_ - 1; }
    void  max(const max_t& _max) noexcept { extent_ = _max - base_ + 1; }

    const int& level() const noexcept { return level_; }
    int&       level() noexcept { return level_; }
    void       level(int _level) noexcept { level_ = _level; }

    auto size() const noexcept
    {
        size_type size = 1;
        for (std::size_t d = 0; d < extent().size(); ++d) size *= extent()[d];
        return size;
    }

  public: //members
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

    template<class Shift>
    void shift(Shift _shift) noexcept
    {
        base_ += _shift;
    }

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

    template<class PointType>
    bool is_inside(const PointType& p) const noexcept
    {
        for (std::size_t d = 0; d < p.size(); ++d)
        {
            if (p[d] < base_[d] || p[d] > max()[d]) return false;
        }
        return true;
    }
    template<class PointType>
    bool on_boundary(const PointType& p) const noexcept
    {
        return (on_max_boundary(p) || on_min_boundary());
    }

    template<class PointType>
    bool on_min_boundary(const PointType& p) const noexcept
    {
        for (std::size_t d = 0; d < p.size(); ++d)
        {
            if (p[d] == base()[d]) return true;
        }
        return false;
    }

    template<class PointType>
    bool on_max_boundary(const PointType& p) const noexcept
    {
        for (std::size_t d = 0; d < p.size(); ++d)
        {
            if (p[d] == max()[d]) return true;
        }
        return false;
    }

    //Flat indices
    template<class PointType, int D = Dim,
        typename std::enable_if<D == 3, void>::type* = nullptr>
    inline size_type index(const PointType& p) const noexcept
    {
        return p[0] - base_[0] + extent_[0] * (p[1] - base_[1]) +
               extent_[0] * extent_[1] * (p[2] - base_[2]);
    }
    template<class PointType, int D = Dim,
        typename std::enable_if<D == 3, void>::type* = nullptr>
    inline size_type index_zeroBase(const PointType& p) const noexcept
    {
        return p[0] + extent_[0] * p[1] + extent_[0] * extent_[1] * p[2];
    }
    template<class PointType, int D = Dim,
        typename std::enable_if<D == 2, void>::type* = nullptr>
    inline size_type index(const PointType& p) const noexcept
    {
        return p[0] - base_[0] + extent_[0] * (p[1] - base_[1]);
    }
    template<class PointType, int D = Dim,
        typename std::enable_if<D == 2, void>::type* = nullptr>
    inline size_type index_zeroBase(const PointType& p) const noexcept
    {
        return p[0] + extent_[0] * p[1];
    }
    inline size_type index(int i, int j, int k) const noexcept
    {
        return i - base_[0] + extent_[0] * (j - base_[1]) +
               extent_[0] * extent_[1] * (k - base_[2]);
    }
    inline size_type index_zeroBase(int i, int j, int k) const noexcept
    {
        return i + extent_[0] * (j) + extent_[0] * extent_[1] * (k);
    }
    inline size_type index(int i, int j) const noexcept
    {
        return i - base_[0] + extent_[0] * (j - base_[1]);
    }
    inline size_type index_zeroBase(int i, int j) const noexcept
    {
        return i + extent_[0] * j;
    }

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

    template<class BlockType, class OverlapType>
    bool overlap(const BlockType& other, OverlapType& overlap, int _level) const
        noexcept
    {
        auto this_scaled = *this;
        auto other_scaled = other;
        this_scaled.level_scale(_level);
        other_scaled.level_scale(_level);
        if (this_scaled.overlap(other_scaled, overlap)) return true;
        return false;
    }

    bool is_empty() const noexcept { return extent_ <= coordinate_type(0); }

    template<typename Btype>
    void grow(Btype _lBuffer, Btype _rBuffer) noexcept
    {
        base_ -= _lBuffer;
        extent_ += _lBuffer + _rBuffer;
    }

    std::size_t nPoints() const noexcept
    {
        std::size_t ext = 1;
        for (auto e : extent_) ext *= e;
        return ext;
    }

    friend std::ostream& operator<<(std::ostream& os, const BlockDescriptor& b)
    {
        os << "Base: " << b.base_ << " extent: " << b.extent_
           << " max: " << b.max() << " level: " << b.level();
        return os;
    }

    auto get_corners() const noexcept
    {
        std::vector<coordinate_type> points;
        coordinate_type              p = base_;
        detail::get_corners_helper<Dim>::apply(p, base_, extent_, points);
        return points;
    }
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

  protected:
    coordinate_type base_;
    coordinate_type extent_;
    int             level_ = 0;
};

} // namespace domain
} // namespace iblgf

#endif
