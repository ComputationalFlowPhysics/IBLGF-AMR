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

#ifndef OCTREE_INCLUDED_OCT_BIT_MASKS_HPP
#define OCTREE_INCLUDED_OCT_BIT_MASKS_HPP

#include <stdint.h>
#include <array>
#include <bitset>

namespace iblgf
{
namespace octree
{
/** @brief Static function returning the number of bit
 * to needed for n, i.e. similar to ceil of log2*/
constexpr std::size_t
clog2(std::size_t n)
{
    return ((n < 2) ? 1 : 1 + clog2(n >> 1));
}
/** @brief Static power function*/
template<typename T>
constexpr T
pow(const T& base, const int exp)
{
    return exp == 0 ? 1 : base * pow(base, exp - 1);
}

/** @brief  Helper class to generate some of the masks for the morton keys.
 *
 *  Most using static constexpr function to generate to bit masks efficiently.
 *  @tparam Dim Spatial dimension
 * */
template<int Dim>
class Bitmasks
{
  public: //Member types
    using index_type = unsigned long long int;
    using scalar_coordinate_type = int;
    using level_type = int;
    using difference_type = long long int;

  public: //Constexpr members
    static std::size_t constexpr nBits = 64;

    static constexpr std::size_t maxLevel()
    {
        int max_level = 0, nbits = 0, flagbits = 2;
        while (true)
        {
            nbits = max_level * Dim + clog2(max_level) + flagbits;
            if (nbits > static_cast<int>(nBits))
            {
                flagbits = nBits - clog2(max_level - 1) - (max_level - 1) * Dim;
                return max_level - 1;
            }
            ++max_level;
        }
        return max_level;
    }
    static constexpr std::size_t max_level = maxLevel();
    static constexpr std::size_t nLevelBits = clog2(max_level);
    static constexpr std::size_t nFlagBits =
        nBits - nLevelBits - (max_level)*Dim;
    static constexpr std::size_t nCoordinateBits =
        nBits - nLevelBits - nFlagBits;
    static constexpr std::size_t ShiftToCoord = nLevelBits + nFlagBits;

    static constexpr index_type reverseBits(index_type _number)
    {
        index_type reverse = 0;
        for (std::size_t i = 0; i < nBits; ++i)
        {
            const index_type tmp =
                (_number & (static_cast<index_type>(1) << i));
            if (tmp)
                reverse |= (static_cast<index_type>(1) << ((nBits - 1) - i));
        }
        return reverse;
    }

  private: //Static constexpr functions
    static constexpr index_type get_level_mask()
    {
        index_type idx = (1 << nLevelBits) - 1;
        idx <<= nFlagBits;
        return idx;
    }
    static constexpr index_type setBit(int idx, index_type _number = 0)
    {
        return _number |= static_cast<index_type>(1) << idx;
    }
    static constexpr index_type setBits(
        int _start, int _end, index_type _number = 0)
    {
        //Least significant bit is bit[0]
        if (_start > _end) return index_type(0);
        index_type idx =
            (static_cast<index_type>(1) << (_end - _start + 1)) - 1;
        idx <<= _start;
        return idx | _number;
    }

    static constexpr index_type get_coord_mask()
    {
        index_type idx(0);
        idx = setBits(nLevelBits + nFlagBits, nBits);
        return idx;
    }

    static constexpr index_type get_lo_mask() noexcept { return setBit(0); }
    static constexpr index_type get_hi_mask() noexcept
    {
        return setBit(nBits - 1);
    }

    static constexpr auto get_min_array() noexcept
    {
        std::array<index_type, max_level + 1> min_array{0};
        for (std::size_t i = 0; i <= max_level; ++i)
        {
            index_type tmp(i);
            tmp <<= 2;
            min_array[i] = setBit(0, tmp);
            ;
        }
        return min_array;
    }

    static constexpr auto get_coord_mask_arr() noexcept
    {
        std::array<index_type, max_level + 1> coord_mask_array{0};
        for (std::size_t i = 0; i <= max_level; ++i)
        { coord_mask_array[i] = setBits(nBits - i * Dim, nBits - 1); }
        return coord_mask_array;
    }
    static constexpr auto get_level_coord_mask_arr() noexcept
    {
        std::array<index_type, max_level + 1> coord_mask_array{0};
        for (std::size_t i = 1; i <= max_level; ++i)
        {
            coord_mask_array[i] =
                setBits(nBits - i * Dim, nBits - i * Dim + Dim - 1);
        }
        return coord_mask_array;
    }

    static constexpr auto get_max_array() noexcept
    {
        auto max_array = get_min_array();
        auto coord_mask_array = get_coord_mask_arr();
        for (std::size_t i = 0; i <= max_level; ++i)
        { max_array[i] |= coord_mask_array[i]; }
        return max_array;
    }

    static constexpr auto get_max_coord_arr() noexcept
    {
        std::array<scalar_coordinate_type, max_level + 1> max_coord_array{0};
        for (std::size_t i = 0; i < max_coord_array.size(); ++i)
        { max_coord_array[i] = (static_cast<index_type>(1) << i); }
        return max_coord_array;
    }

  public: //operators
    friend std::ostream& operator<<(std::ostream& os, Bitmasks b)
    {
        os << " nBits " << Bitmasks::nBits << std::endl;
        os << "level_mask " << std::endl;
        os << std::bitset<64>(Bitmasks::level_mask) << std::endl;
        os << "coord_mask " << std::endl;
        os << std::bitset<64>(Bitmasks::coord_mask) << std::endl;
        os << "lo_mask " << std::endl;
        os << std::bitset<64>(Bitmasks::lo_mask) << std::endl;
        os << "hi_mask " << std::endl;
        os << std::bitset<64>(Bitmasks::hi_mask) << std::endl;

        std::cout << "min_array " << std::endl;
        for (std::size_t i = 0; i <= Bitmasks::max_level; ++i)
            os << std::bitset<64>(Bitmasks::min_array[i]) << " " << i
               << std::endl;
        os << "max_array " << std::endl;
        for (std::size_t i = 0; i <= Bitmasks::max_level; ++i)
            os << std::bitset<64>(Bitmasks::max_array[i]) << " " << i
               << std::endl;
        os << "coord_mask_array " << std::endl;
        for (std::size_t i = 0; i <= Bitmasks::max_level; ++i)
            os << std::bitset<64>(Bitmasks::coord_mask_array[i]) << " " << i
               << std::endl;
        os << "level coord_mask_array " << std::endl;
        for (std::size_t i = 0; i <= Bitmasks::max_level; ++i)
            os << std::bitset<64>(Bitmasks::level_coord_mask_array[i]) << " "
               << i << std::endl;
        return os;
    }

  public: //static memebers:
    static constexpr index_type coord_mask = get_coord_mask();
    static constexpr index_type level_mask = get_level_mask();
    static constexpr index_type lo_mask = get_lo_mask();
    static constexpr index_type hi_mask = get_hi_mask();
    static constexpr std::array<index_type, max_level + 1> min_array =
        get_min_array();
    static constexpr std::array<index_type, max_level + 1> max_array =
        get_max_array();
    static constexpr std::array<index_type, max_level + 1> coord_mask_array =
        get_coord_mask_arr();
    static constexpr std::array<index_type, max_level + 1>
        level_coord_mask_array = get_level_coord_mask_arr();
    static constexpr std::array<scalar_coordinate_type, max_level + 1>
        max_coord_array = get_max_coord_arr();
};

} // namespace octree
} // namespace iblgf

#endif // LB_INCLUDED_OCT_BIT_MASKS_HPP
