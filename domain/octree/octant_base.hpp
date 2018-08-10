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

    using key_type = Key<Dim>;
    using coordinate_type=typename key_type::coordinate_type;
    using real_coordinate_type = types::coordinate_type<float_type,Dim>;
    using key_index_type = typename key_type::value_type;

    using tree_type = Tree<Dim, DataType>;
    static constexpr int num_children(){return pow(2,Dim);};

public:

	Octant_base() = delete;
	Octant_base(const Octant_base& other) = default;
	Octant_base(Octant_base&& other) = default;
	Octant_base& operator=(const Octant_base& other) & = default;
	Octant_base& operator=(Octant_base&& other) & = default;
	~Octant_base() = default;



    Octant_base(const coordinate_type& _x, int _level, tree_type* _t)
    :key_(key_type(_x,_level)), 
     t_(_t){}


    Octant_base(const key_type& _k, tree_type* _t)
    :key_(_k),
     t_(_t)
    { }

public:

    key_type key() const noexcept{return key_;}
    tree_type* tree()const noexcept{return t_;}
    auto level() const noexcept{return key_.level();}

    coordinate_type coordinate() const noexcept
    {
        return key_.coordinate();
    }

    real_coordinate_type real_coordinate() const noexcept
    {
        //real_coordinate_type tmp=this->coordinate();
        //return tmp/=(std::pow(2,this->level()-t_->base_level()));
        
        real_coordinate_type tmp=this->coordinate();
        tmp/=(std::pow(2,this->level()-t_->base_level()));
        return this->tree()->octant_to_real_coordinate(tmp);

    }

    coordinate_type fine_level_coordinate() const noexcept
    {
        coordinate_type tmp=this->coordinate();
        return tmp*=(std::pow(2, t_->depth()-this->level()));
    }

    Octant_base child( int _i ) const noexcept
    {
        return Octant_base(this->key_.child(_i), t_);
    }

    Octant_base parent() const noexcept
    {
        return Octant_base(this->key_.parent(),t_);
    }

    Octant_base equal_coordinate_parent() const noexcept
    {
        return Octant_base(this->key_.equal_coordinate_parent(),t_);
    }


    Octant_base neighbor(const coordinate_type& _offset)
    {
        Octant_base nn(this->key_.neighbor(_offset),tree());
        if(!is_hanging() && !is_boundary())
        {
            return nn;
        }
        else if(!is_boundary() )
        {
            return nn.parent();
        }
        else
        {
            throw std::runtime_error("You are at the boundary");
        }
    }


    friend std::ostream& operator<<(std::ostream& os, const Octant_base& n)
    {
        os<<n.key_;
        return os;
    }

    bool is_hanging()const noexcept{return id_==node_flag::hanging;}
    bool is_boundary()  const noexcept{return id_==node_flag::boundary_node;}
    void flag(node_flag _id)noexcept{ id_=_id;  }


protected:
    key_type key_;
    tree_type* t_=nullptr;
    node_flag id_= node_flag::octant;
};



} //namespace octree
#endif 

