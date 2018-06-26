#ifndef INCLUDED_LINEAR_TREE_HPP
#define INCLUDED_LINEAR_TREE_HPP


#include <vector>
#include <memory>
#include <cmath>
#include <set>
#include <string>
#include <map>
#include "leaf.h"
#include "key.h"
#include "utils.h"

namespace octree
{


template<int Dim, class DataType>
class LinearTree
{

public: //memeber types

    using key_t = Key<Dim>;
    using coordinate_t = typename key_t::coordinate_t;
    using data_t = DataType;
    using leaf_t = Leaf<Dim,data_t>;
    using container_t= std::set<leaf_t>;
    using iterator = typename container_t::iterator;
	using const_iterator = typename container_t::const_iterator;


public:
    LinearTree(){leaf_t::tree_ptr=this;}
	LinearTree(const LinearTree& other) = delete;
	LinearTree(LinearTree&& other) = default;
	LinearTree& operator=(const LinearTree& other) & = delete;
	LinearTree& operator=(LinearTree&& other) & = default;
	~LinearTree() = default;

    LinearTree (const coordinate_t& _extent )
    {
        leaf_t::tree_ptr=this;
        for(std::size_t d=0;d<_extent.size();++d)
        {
            int level= static_cast<int >(std::log2(_extent[d]))+1;
            if(level>base_level_) base_level_=level;
        }

        const coordinate_t base(0);
        rcIterator<Dim>::apply(base, _extent, [&]( const coordinate_t& _p ) {
                this->insert(_p, this->base_level_);
                });
    }


public:

	void insert(const coordinate_t& _x, int _level)
    { 
        nodes_.insert(leaf_t(_x,_level));
    }
	
    auto begin_nodes() const {return nodes_.cbegin();}
    auto begin_nodes() {return nodes_.cbegin();}
    auto end_nodes() const {return nodes_.cend();}
    auto end_nodes() {return nodes_.cend();}

    const int& base_level() const noexcept{return base_level_;}
    int& base_level() noexcept{return base_level_;}

    //TODO: Refine and coarsening can be done much more
    //      efficiently with hints.
    template<class Function>
    void refine (iterator& _l,const Function& _f )
    {
        for(int i=0;i<_l->num_children;++i)
        {
            leaf_t child(_l->child_key(i).id() );
            _f(child);
            nodes_.insert(child);
        }
        _l=nodes_.erase(_l);
    }


    void coarsen( iterator& _l )
    {
        leaf_t parent(_l->parent_key());
        nodes_.insert(parent);
        for(int i=0;i<_l->num_children;++i)
        {
            leaf_t child(parent.child_key(i) );
            auto it=nodes_.find(child);
            _l=nodes_.erase(it);
        }
    }

private:
    container_t nodes_;
    int base_level_=0;

};


} //namespace ocTree
#endif 
