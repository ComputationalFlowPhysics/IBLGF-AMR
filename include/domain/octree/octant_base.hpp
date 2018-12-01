#ifndef OCTREE_INCLUDED_NODE_HPP
#define OCTREE_INCLUDED_NODE_HPP

#include <vector>
#include <memory>
#include <cmath>
#include <set>
#include <string>
#include <map>

// IBLGF-specific
#include <global.hpp>
#include <domain/octree/key.hpp>

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

    using key_type             = Key<Dim>;
    using coordinate_type      = typename key_type::coordinate_type;
    using real_coordinate_type = types::coordinate_type<float_type,Dim>;
    using key_index_type       = typename key_type::value_type;

    using tree_type = Tree<Dim, DataType>;
    static constexpr int num_children(){ return pow(2,Dim); };

public:

	Octant_base()                                      = delete;
	Octant_base(const Octant_base& other)              = default;
	Octant_base(Octant_base&& other)                   = default;
	Octant_base& operator=(const Octant_base& other) & = default;
	Octant_base& operator=(Octant_base&& other)      & = default;
	~Octant_base()                                     = default;



    Octant_base(const coordinate_type& _x, int _level, tree_type* _t)
    :key_(key_type(_x,_level)),
     t_(_t){}


    Octant_base(const key_type& _k, tree_type* _t)
    :key_(_k),
     t_(_t)
    { }

public:

    /** @brief Get octant key*/
    key_type key() const noexcept{return key_;}

    /** @brief Get tree pointer*/
    tree_type* tree()const noexcept{return t_;}

    /** @brief Octant level relative to root of tree*/
    auto level() const noexcept{return key_.level();}

    /** @brief Octant level relative to root of tree*/
    auto tree_level() const noexcept{return key_.level();}

    /** @brief Refinement level: level relative to base level */
    auto refinement_level() const noexcept{return tree_level()-t_->base_level();}



    /** @brief Get octant coordinate (integer) based on tree structure */
    coordinate_type tree_coordinate() const noexcept
    {
        return key_.coordinate();
    }

    /** @brief Get octant coordinate based on physical/global domain */
    real_coordinate_type global_coordinate() const noexcept
    {
        real_coordinate_type tmp=this->tree_coordinate();
        tmp/=(std::pow(2,this->level()-t_->base_level()));
        return this->tree()->octant_to_level_coordinate(tmp);
    }


    /** @brief Get child of type Octant_base */
    Octant_base child_base( int _i ) const noexcept
    {
        return Octant_base(this->key_.child(_i), t_);
    }

    /** @brief Get parent of type Octant_base */
    Octant_base parent_base() const noexcept
    {
        return Octant_base(this->key_.parent(),t_);
    }

    /** @brief Get parent of type Octant_base with same coordinate than
     *         current octant.
     * */
    Octant_base equal_coordinate_parent() const noexcept
    {
        return Octant_base(this->key_.equal_coordinate_parent(),t_);
    }


    /** @brief Get neighbor of type Octant_base
     *
     *  @param[in] _offset Offset from current octant in terms of
     *                     tree coordinates, i.e. octants.
     * */
    Octant_base neighbor(const coordinate_type& _offset)
    {
        Octant_base nn(this->key_.neighbor(_offset),tree());
    }


    friend std::ostream& operator<<(std::ostream& os, const Octant_base& n)
    {
        os<<n.key_;
        return os;
    }

    void flag(node_flag _id)noexcept{ id_=_id;  }


protected:
    key_type key_;
    tree_type* t_=nullptr;
    node_flag id_= node_flag::octant;
};



} //namespace octree
#endif
