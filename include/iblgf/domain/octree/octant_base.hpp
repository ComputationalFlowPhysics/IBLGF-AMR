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

#ifndef OCTREE_INCLUDED_NODE_HPP
#define OCTREE_INCLUDED_NODE_HPP

#include <vector>
#include <memory>
#include <cmath>
#include <set>
#include <string>
#include <map>

// IBLGF-specific
#include <iblgf/global.hpp>
#include <iblgf/domain/octree/key.hpp>

namespace octree
{
enum class node_flag : int
{
    octant,
    hanging,
    boundary_node
};

template<int Dim, class DataType>
class Tree;

template<int Dim, class DataType>
class Octant_base
{
  public:
    using key_type = Key<Dim>;
    using coordinate_type = typename key_type::coordinate_type;
    using real_coordinate_type = types::coordinate_type<float_type, Dim>;
    using key_index_type = typename key_type::value_type;

    using tree_type = Tree<Dim, DataType>;
    static constexpr int num_children() { return pow(2, Dim); };

  public:
    Octant_base() = delete;
    Octant_base(const Octant_base& other) = default;
    Octant_base(Octant_base&& other) = default;
    Octant_base& operator=(const Octant_base& other) & = default;
    Octant_base& operator=(Octant_base&& other) & = default;
    ~Octant_base() = default;

    Octant_base(const coordinate_type& _x, int _level)
    : key_(key_type(_x, _level))
    {
    }

    Octant_base(const key_type& _k)
    : key_(_k)
    {
    }

  public:
    /** @brief Get octant key*/
    const key_type& key() const noexcept { return key_; }

    /** @brief Octant level relative to root of tree*/
    auto level() const noexcept { return key_.level(); }

    /** @brief Octant level relative to root of tree*/
    auto tree_level() const noexcept { return key_.level(); }

    /** @brief Get octant coordinate (integer) based on tree structure */
    coordinate_type tree_coordinate() const noexcept
    {
        return key_.coordinate();
    }

    friend std::ostream& operator<<(std::ostream& os, const Octant_base& n)
    {
        os << n.key_;
        return os;
    }

    const int& rank() const noexcept { return rank_; }
    int&       rank() noexcept { return rank_; }

  private: //Serialization
    friend class boost::serialization::access;

    template<class Archive>
    void serialize(Archive& ar, const unsigned int version)
    {
        ar& key_;
        ar& rank_;
    }

  protected:
    key_type key_;
    int      rank_ = -1;
};

} //namespace octree
#endif
