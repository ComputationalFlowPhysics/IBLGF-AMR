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

#ifndef INCLUDED_KEY_HPP
#define INCLUDED_KEY_HPP

#include <array>
#include <iomanip>
#include <cmath>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/access.hpp>

// IBLGF-specific
#include <iblgf/global.hpp>
#include <iblgf/tensor/vector.hpp>
#include <iblgf/domain/octree/bitmasks.hpp>
#include <iblgf/utilities/block_iterator.hpp>

namespace iblgf
{
namespace octree
{

/**
 *  @brief Morton key to be used in a binary, quad or octree.
 *
 *  Constructed based on an integer coordinate and the tree level.
 *
 *  @tparam Dim Spatial dimension
 */
template<int Dim>
struct Key
{
  public: // member types
    using bitmask_t = Bitmasks<Dim>;
    using value_type = typename bitmask_t::index_type;
    using level_type = typename bitmask_t::level_type;
    using scalar_coordinate_type = typename bitmask_t::scalar_coordinate_type;
    using coordinate_type = types::vector_type<scalar_coordinate_type, Dim>;
    using real_coordinate_type = types::vector_type<types::float_type, Dim>;

    static constexpr std::size_t nChildren = pow(2, Dim);

  public: //static members
    /** @brief Compute the morton index on the minimum required level and based on
     *         coordinate only.
     */
    static constexpr value_type compute_index(
        const coordinate_type& _c) noexcept
    {
        return compute_index(_c, minimum_level(_c));
    }

    /** @brief Compute the morton index based on coordinate as well as the level.  */
    static constexpr value_type compute_index(
        const coordinate_type& _c, int _level) noexcept
    {
        value_type idx = split_bits(static_cast<value_type>(_c.x()));
        for (int d = 1; d < Dim; ++d)
        { idx |= (split_bits(static_cast<value_type>(_c[d])) << d); }
        idx <<= (bitmask_t::nLevelBits + bitmask_t::nFlagBits);
        idx <<= (bitmask_t::max_level - _level) * Dim;
        idx |= (static_cast<value_type>(_level) << bitmask_t::nFlagBits);
        idx |= 1;
        return idx;
    }
    /** @brief Get the maximum element of coordinate. */
    static constexpr scalar_coordinate_type max_element(
        const coordinate_type& x) noexcept
    {
        scalar_coordinate_type res =
            std::numeric_limits<scalar_coordinate_type>::lowest();
        for (auto& e : x)
            if (e > res) res = e;
        return res;
    }
    /** @brief Get the minimum element of coordinate. */
    static constexpr scalar_coordinate_type min_element(
        const coordinate_type& x) noexcept
    {
        scalar_coordinate_type res =
            std::numeric_limits<scalar_coordinate_type>::max();
        for (auto& e : x)
            if (e < res) res = e;
        return res;
    }
    /** @brief Check if a coordinate _x, can be represented on level _level.
     *  @return True of coordinate _x can be respresented on level _level.
     * */
    static constexpr bool representable(
        const coordinate_type& _x, level_type _level)
    {
        for (auto& e : _x)
            if (e < 0) return false;
        scalar_coordinate_type m = max_element(_x);
        scalar_coordinate_type mini = min_element(_x);
        if (m <= bitmask_t::max_coord_array[_level] - 1 && mini >= 0)
            return true;
        return false;
    }
    /** @brief Get the minimum level required to represent coordinate x. */
    static constexpr level_type minimum_level(const coordinate_type& x) noexcept
    {
        scalar_coordinate_type m = max_element(x);
        level_type             _level = 0;
        while (m >= bitmask_t::max_coord_array[_level]) { ++_level; }
        return _level;
    }
    /** @brief Get the level of the morton key _mkey */
    static constexpr level_type level(const value_type& _mkey) noexcept
    {
        return ((bitmask_t::level_mask & _mkey) >> bitmask_t::nFlagBits);
    }
    /** @brief Get the coordinate of the morton key _mkey */
    static constexpr auto coordinate(const value_type& _mkey) noexcept
    {
        coordinate_type c(0);
        for (int d = 0; d < Dim; ++d)
            c[d] = compress_bits(
                _mkey >> (d + bitmask_t::ShiftToCoord +
                             (bitmask_t::max_level - level(_mkey)) * Dim));
        return c;
    }
    /** @brief Get maximum level representable by the bitmask_t.*/
    static constexpr level_type max_level() noexcept
    {
        return bitmask_t::max_level;
    }
    /** @brief Get the first morton key of level _level.*/
    static constexpr Key begin(level_type _level) noexcept
    {
        return Key(bitmask_t::min_array[_level]);
    }
    /** @brief Get the morton key, which is one beyond the largest morton key of level _level.*/
    static constexpr Key end(level_type _level) noexcept
    {
        return Key((bitmask_t::min_array[_level] | bitmask_t::coord_mask) + 1u);
    }
    /** @brief Reverse begin. Get the last/largest morton key of level _level.*/
    static constexpr Key rbegin(level_type _level) noexcept
    {
        return Key(bitmask_t::max_arr[_level]);
    }
    /** @brief Reverse end. Get the morton key, which is one preceeding the begining. */
    static Key rend(level_type _level) noexcept
    {
        return Key(bitmask_t::min_array[_level] - 1u);
    }
    /** @brief Get the minimum morton key at level _level. */
    static constexpr Key min(level_type _level) noexcept
    {
        return Key(bitmask_t::min_array[_level]);
    }
    /** @brief Get the maximum morton key at level _level. */
    static constexpr Key max(level_type _level) noexcept
    {
        return Key(bitmask_t::max_array[_level]);
    }

  public: // Ctors
    /** @brief Default constructor. */
    Key() noexcept
    : index_(bitmask_t::min_array[0])
    {
    }

    /** @brief Constructor, based on Key. */
    Key(value_type idx) noexcept
    : index_(idx)
    {
    }

    /** @brief Constructor, based coordinate x using the minimum level required to represent x. */
    Key(coordinate_type x) noexcept
    : index_(compute_index(x))
    {
    }
    /** @brief Constructor, based coordinate x as well as the level. */
    Key(coordinate_type x, level_type _level) noexcept
    : index_(compute_index(x, _level))
    {
    }

    Key(const Key&) = default;
    Key(Key&&) = default;
    Key& operator=(const Key&) & = default;
    Key& operator=(Key&&) & = default;

  public: //Access

    auto        coordinate() const noexcept { return coordinate(index_); }

    const auto& id() const { return index_; }
    auto        id() { return index_; }

    /** @brief Get neighbor morton key, which has the offset _offset to the morton key. */
    Key neighbor(const coordinate_type& _offset) const noexcept
    {
        return Key(coordinate() + _offset, level());
    }

    /***********************************************************************/
    //FIXME: This is application code, please put these two functions somewhere else.
    /** @brief Get all neighbors of distance 1 */
    auto get_neighbor_keys(int distance = 1) const noexcept
    {
        std::vector<Key> res;
        coordinate_type  offset(distance);
        BlockIterator<Dim>::iterate(
            -1 * offset, 2 * offset + 1, [&](const coordinate_type& _p) {
                res.emplace_back(this->neighbor(_p));
            });
        return res;
    }

    /** @brief Get all influence keys */
    auto get_infl_keys() const noexcept
    {
        std::vector<Key> res;
        if (this->level() == 0) return res;
        auto p_k = this->parent();

        auto p_n_k = p_k.get_neighbor_keys();
        for (auto& k : p_n_k)
        {
            if (k.is_end()) continue;
            const auto coord = this->coordinate();

            for (std::size_t p_n_child_id = 0; p_n_child_id < nChildren;
                 ++p_n_child_id)
            {
                const auto child_key = k.child(p_n_child_id);
                const auto p_n_c_coord = child_key.coordinate();
                if ((std::abs(p_n_c_coord.x() - coord.x()) > 1) ||
                    (std::abs(p_n_c_coord.y() - coord.y()) > 1) ||
                    ((Dim == 3) && std::abs(p_n_c_coord.z() - coord.z()) > 1))
                { res.emplace_back(child_key); }
            }
        }
        return res;
    }
    /***********************************************************************/

    /** @brief Get parent of the morton key. */
    Key parent() const noexcept
    {
        const auto _level(level());
        if (_level == 0) return *this;
        return {
            ((bitmask_t::coord_mask_array[_level - 1] & index_) |
                (static_cast<value_type>(_level - 1) << bitmask_t::nFlagBits)) +
            1u};
    }
    /** @brief Get the i-th child of the morton key. */
    Key child(int i) const noexcept
    {
        const auto _level(level());
        if (_level == bitmask_t::max_level) return *this;
        return {
            ((bitmask_t::coord_mask_array[_level] & index_) |
                (static_cast<value_type>(i)
                    << ((bitmask_t::max_level - (_level + 1)) * Dim +
                           bitmask_t::ShiftToCoord)) |
                (static_cast<value_type>(_level + 1) << bitmask_t::nFlagBits)) +
            1u};
    }

    /** @brief Determine which child the morton key is from is parents. Return -1 if root. */
    int child_number() const noexcept
    {
        const auto level(this->level());
        if (level == 0) return -1;
        return child_number(level);
    }

    /** @brief Determine which child the morton key is from is parents at level _level.
     * Return -1 if root. */
    int child_number(int _level) const noexcept
    {
        if (_level == 0) return -1;
        return (
            (bitmask_t::level_coord_mask_array[_level] & index_) >>
            (bitmask_t::ShiftToCoord + Dim * (bitmask_t::max_level - _level)));
    }

    /** @brief Get level of current morton key. */
    level_type level() const noexcept
    {
        return ((bitmask_t::level_mask & index_) >> bitmask_t::nFlagBits);
    }

    /** @brief Get coordinate, which is scaled to the unit box on level of the current key. */
    real_coordinate_type unit_coordinate() const noexcept
    {
        return real_coordinate_type(coordinate()) /
               bitmask_t::max_coord_array[level()];
    }

    /** @brief Check if current key is end(). */
    bool is_end() const noexcept { return index_ >= end(level()).index_; }
    /** @brief Check if current key is the reverse end. */
    bool is_rend() const noexcept { return index_ <= rend(level()).index_; }

  public: //Operators
    friend constexpr bool operator==(const Key& _lhs, const Key& _rhs) noexcept
    {
        return _lhs.index_ == _rhs.index_;
    }

    friend constexpr bool operator!=(const Key& _lhs, const Key& _rhs) noexcept
    {
        return _lhs.index_ != _rhs.index_;
    }

    friend constexpr bool operator<(const Key& _lhs, const Key& _rhs) noexcept
    {
        return _lhs.index_ < _rhs.index_;
    }

    friend constexpr bool operator<=(const Key& _lhs, const Key& _rhs) noexcept
    {
        return _lhs.index_ <= _rhs.index_;
    }

    friend constexpr bool operator>(const Key& _lhs, const Key& _rhs) noexcept
    {
        return _lhs.index_ > _rhs.index_;
    }

    friend constexpr bool operator>=(const Key& _lhs, const Key& _rhs) noexcept
    {
        return _lhs.index_ >= _rhs.index_;
    }

    Key& operator++() noexcept
    {
        const level_type _level = level();
        if (index_ >= max(_level).index_)
        {
            index_ = end(_level).index_;
            return *this;
        };
        if (index_ <= rend(_level).index_)
        {
            index_ = min(_level).index_;
            return *this;
        };

        const auto shift =
            (bitmask_t::max_level - _level) * Dim + bitmask_t::ShiftToCoord;
        index_ = (((index_ >> shift) + 1u) << shift) |
                 (index_ & ~bitmask_t::coord_mask_array[_level]);
        return *this;
    }
    Key operator++(int) noexcept
    {
        Key   tmp(*this);
        this->operator++();
        return tmp;
    }

    Key& operator--() noexcept
    {
        const level_type _level = level();
        if (index_ >= end(_level).index_)
        {
            index_ = max(_level).index_;
            return *this;
        };
        if (index_ <= min(_level).index_)
        {
            index_ = rend(_level).index_;
            return *this;
        };
        const auto shift =
            (bitmask_t::max_level - _level) * Dim + bitmask_t::ShiftToCoord;
        index_ = (((index_ >> shift) - 1u) << shift) |
                 (index_ & ~bitmask_t::coord_mask_array[_level]);
        return *this;
    }

    Key operator--(int) noexcept
    {
        Key   tmp(*this);
        this->operator--();
        return tmp;
    }

    friend std::ostream& operator<<(std::ostream& os, const Key& t)
    {
        auto val = t.index_;
        auto bs = std::bitset<bitmask_t::nBits>(val);
        for (std::size_t i = 0; i < bitmask_t::nCoordinateBits; ++i)
        {
            os << bs[bitmask_t::nBits - 1 - i];
            if ((i + 1) % Dim == 0) os << " ";
        }
        os << " ";
        for (std::size_t i = bitmask_t::nCoordinateBits;
             i < bitmask_t::nCoordinateBits + bitmask_t::nLevelBits; ++i)
        { os << bs[bitmask_t::nBits - 1 - i]; }
        os << " ";
        for (std::size_t i = bitmask_t::nCoordinateBits + bitmask_t::nLevelBits;
             i < bitmask_t::nBits; ++i)
        { os << bs[bitmask_t::nBits - 1 - i]; }

        os << " = " << std::setw(20) << std::left << val << " " << std::setw(5)
           << std::left << "level " << std::setw(2) << t.level() << ", coord "
           << "="
           << " ( ";
        for (int d = 0; d < Dim - 1; ++d)
            os << std::setw(4) << t.coordinate()[d] << " ";
        os << std::left << std::setw(4) << t.coordinate()[Dim - 1] << ")"
           << std::right << std::setw(4) << " = " << std::fixed << " ("
           << std::setprecision(9) << t.unit_coordinate() << ")" << std::left
           << std::defaultfloat;
        return os;
    }

  private: //private static members
    /** @brief Split bits for interleaving the coordinate bits, basically distribute them
     * such that there is space for the other coordinates */
    template<int nDim = Dim>
    static value_type split_bits(value_type w)
    {
        using tag = std::integral_constant<int, nDim>;
        return split_bits_impl(w, tag());
    }
    /** @brief Revert the split bits function. */
    template<int nDim = Dim>
    static value_type compress_bits(value_type w)
    {
        using tag = std::integral_constant<int, nDim>;
        return compress_bits_impl(w, tag());
    }

    /** @brief Implementation of the bit splitting using the magic numbers in 3D. */
    static value_type split_bits_impl(
        value_type w, std::integral_constant<int, 3>) noexcept
    {
        w &= 0x00000000001fffff;
        w = (w | w << 32) & 0x001f00000000ffff;
        w = (w | w << 16) & 0x001f0000ff0000ff;
        w = (w | w << 8) & 0x010f00f00f00f00f;
        w = (w | w << 4) & 0x10c30c30c30c30c3;
        w = (w | w << 2) & 0x1249249249249249;
        return w;
    }
    /** @brief Implementation of bit compression using the magic numbers in 3D. */
    static scalar_coordinate_type compress_bits_impl(
        value_type w, std::integral_constant<int, 3>) noexcept
    {
        w &= 0x1249249249249249;
        w = (w ^ (w >> 2)) & 0x30c30c30c30c30c3;
        w = (w ^ (w >> 4)) & 0xf00f00f00f00f00f;
        w = (w ^ (w >> 8)) & 0x00ff0000ff0000ff;
        w = (w ^ (w >> 16)) & 0x00ff00000000ffff;
        w = (w ^ (w >> 32)) & 0x00000000001fffff;
        return static_cast<scalar_coordinate_type>(w);
    }
    /** @brief Implementation of the bit splitting using the magic numbers in 2D. */
    static value_type split_bits_impl(
        value_type x, std::integral_constant<int, 2>) noexcept
    {
        x = (x | (x << 16)) & 0x0000FFFF0000FFFF;
        x = (x | (x << 8)) & 0x00FF00FF00FF00FF;
        x = (x | (x << 4)) & 0x0F0F0F0F0F0F0F0F;
        x = (x | (x << 2)) & 0x3333333333333333;
        x = (x | (x << 1)) & 0x5555555555555555;
        return x;
    }
    /** @brief Implementation of bit compression using the magic numbers in 2D. */
    static scalar_coordinate_type compress_bits_impl(
        value_type w, std::integral_constant<int, 2>) noexcept
    {
        w &= 0x5555555555555555;
        w = (w ^ (w >> 1)) & 0x3333333333333333;
        w = (w ^ (w >> 2)) & 0x0f0f0f0f0f0f0f0f;
        w = (w ^ (w >> 4)) & 0x00ff00ff00ff00ff;
        w = (w ^ (w >> 8)) & 0x0000ffff0000ffff;
        w = (w ^ (w >> 16)) & 0x00000000ffffffff;
        return static_cast<scalar_coordinate_type>(w);
    }

  private:
    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive& ar, const unsigned int version)
    {
        ar& index_;
    }

  private:
    value_type index_;
};

} //namespace octree
} // namespace iblgf

namespace std
{
template<int Dim>
struct hash<iblgf::octree::Key<Dim>>
{
    auto operator()(const iblgf::octree::Key<Dim>& k) const { return k.id(); }
};

} // namespace std
#endif
