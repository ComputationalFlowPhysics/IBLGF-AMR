#ifndef OCTREE_INCLUDED_LINEAR_TREE_HPP
#define OCTREE_INCLUDED_LINEAR_TREE_HPP


#include<vector>
#include<memory>
#include<cmath>
#include<set>
#include<string>
#include<map>
#include<unordered_map>
#include<functional>

#include "tree_utils.hpp"
#include "octant.hpp"
#include "global.hpp"

namespace octree
{


template<int Dim, class DataType>
class Tree
{

public: //memeber types

    static constexpr int dimension = Dim;
    using octant_type = Octant<Dim, DataType>;
    using octant_base_type = Octant_base<Dim,DataType>;
    using real_coordinate_type= typename octant_base_type::real_coordinate_type;
    using key_type = typename octant_base_type::key_type;
    using coordinate_type = typename key_type::coordinate_type;
    using scalar_coordinate_type = typename key_type::scalar_coordinate_type;

    using octant_map_type= std::map<key_type,octant_type>;
    using octant_iterator = MapValueIterator<octant_map_type>;
    using raw_octant_iterator = typename octant_map_type::iterator;

    using coordinate_transform_t=
        std::function<real_coordinate_type(real_coordinate_type)>;

public:
    friend octant_base_type;
    friend octant_type;


public:
    Tree() = default;
    Tree(const Tree& other) = delete;
    Tree(Tree&& other) = default;
    Tree& operator=(const Tree& other) & = delete;
    Tree& operator=(Tree&& other) & = default;
    ~Tree() = default;

    Tree (const coordinate_type& _nCells )
    {
        auto extent=_nCells;
        base_level_=key_type::minimum_level(extent);
        depth_=base_level_;
        const coordinate_type base(0);
        rcIterator<Dim>::apply(base, extent, [&]( const coordinate_type& _p ) {
                this->insert_octant(_p, this->base_level_);
                });
    }

    //Construct given some boxes of unit 1 per box and 
    //a maximal allowable domain extent for the base level
    Tree (const std::vector<coordinate_type>& _points,
          int _base_level)
    {
        this->base_level_=_base_level;
        depth_=base_level_;
        for(auto& p : _points)
        {
            this->insert_octant(p, this->base_level_);
        }

    }

public:
   
    octant_iterator begin_octants()const noexcept {return octants_.begin();}
    octant_iterator end_octants()const noexcept {return octants_.end();}

    octant_iterator begin_octants()noexcept {return octants_.begin();}
    octant_iterator end_octants()noexcept {return octants_.end();}

    auto num_octants()const noexcept { return octants_.size();}

    const int& base_level() const noexcept{return base_level_;}
    int& base_level() noexcept{return base_level_;}
    const int& depth() const noexcept{return depth_;}
    int& depth() noexcept{return depth_;}

    template<class Function>
    void refine(octant_iterator& _l,const Function& _f, bool _recursive=false )
    {
        if(_l->is_hanging())return;
        for(int i=0;i<_l->num_children();++i)
        {
            octant_type child(_l->child(i));
            _f(child);
            auto c=octants_.emplace(child.key(),child);
            c.first->second.flag(node_flag::octant);
        }
        _l=octants_.erase(_l);
        ++depth_;
        if(!_recursive) std::advance(_l,_l->num_children()-1);
    }

 
    void determine_hangingOctants()
    {
        for(auto it=begin_octants();it!=end_octants();++it)
        {
            it->determine_hangingOctants();
        }
    }

    const auto& get_octant_to_real_coordinate() const noexcept
    {
        return octant_to_real_coordinate_;
    }
    auto& get_octant_to_real_coordinate() noexcept
    {
        return octant_to_real_coordinate_;
    }
    template<class T>
    auto octant_to_real_coordinate(T _x)
    {
        return octant_to_real_coordinate_(_x);
    }


private: 

    template<class Node>
    bool has_node(Node& _node ) const
    {
        auto it=octants_.find(_node.key());
        if(it!=octants_.end())
            return true;
        return false;
    }

    auto find_octant(octant_base_type _node)
    {
        return octant_iterator(octants_.find(_node.key()));
    }

    auto find_octant_any_level(octant_base_type _node)  noexcept
    {
        octant_base_type n=_node;
        const auto it=octants_.find(n.key());
        if(it!=octants_.end())
        {
             return octant_iterator(it);
        }
        else
        {
            for(auto i=this->depth();i>=base_level();--i)
            {
                const auto parent =n.equal_coordinate_parent();
                const auto it=octants_.find(parent.key());
                if( it!=octants_.end() )return octant_iterator(it);
                n=parent;
            }
            return octant_iterator(it);
        }
        return octant_iterator(it);
    }


    void insert_octant(const coordinate_type& _x, int _level)
    { 
        const octant_type c( _x, _level , this);
        octants_.emplace(c.key(), c);
    }

    void insert_octant(octant_base_type& _n)
    { 
        octants_.emplace(_n.key(), _n);
    }

    
    static real_coordinate_type unit_transform(coordinate_type _x)
    {
        return _x;
    }



private:
    octant_map_type octants_;
    coordinate_transform_t octant_to_real_coordinate_=&Tree::unit_transform;
    int base_level_=0;
    int depth_=0;

};


} //namespace ocTree
#endif 
