#ifndef OCTREE_INCLUDED_LINEAR_TREE_HPP
#define OCTREE_INCLUDED_LINEAR_TREE_HPP


#include <vector>
#include <memory>
#include <cmath>
#include <set>
#include <string>
#include <map>
#include <unordered_map>
#include <functional>

// IBLGF-specific
#include <global.hpp>
#include <domain/octree/octant.hpp>
#include <domain/octree/tree_utils.hpp>

namespace octree
{


template<int Dim, class DataType>
class Tree
{

public: //memeber types

    static constexpr int dimension = Dim;
    using octant_type              = Octant<Dim, DataType>;
    using octant_base_type         = Octant_base<Dim, DataType>;
    using real_coordinate_type     = typename octant_base_type::real_coordinate_type;
    using key_type                 = typename octant_base_type::key_type;
    using coordinate_type          = typename key_type::coordinate_type;
    using scalar_coordinate_type   = typename key_type::scalar_coordinate_type;

    using octant_ptr_map_type = std::map<key_type,octant_type*>;
    using octant_iterator     = MapValuePtrIterator<octant_ptr_map_type>;
    using raw_octant_iterator = typename octant_ptr_map_type::iterator;

    using dfs_iterator = typename detail::iterator_depth_first<octant_type>;
    using bfs_iterator = typename detail::iterator_breadth_first<octant_type>;

    using coordinate_transform_t =
        std::function<real_coordinate_type(real_coordinate_type)>;

public:
    friend octant_base_type;
    friend octant_type;


public:
    Tree() = default;
    Tree(const Tree& other)              = delete;
    Tree(Tree&& other)                   = default;
    Tree& operator=(const Tree& other) & = delete;
    Tree& operator=(Tree&& other)      & = default;
    ~Tree() = default;

 
    //Construct given some boxes of unit 1 per box and 
    //a maximal allowable domain extent for the base level
    Tree (const std::vector<coordinate_type>& _points, int _base_level)
    {
        this->base_level_ = _base_level;
        depth_ = base_level_+1;


        //Construct interior octants, leaf map && level map
        root_=std::make_shared<octant_type>(
                coordinate_type(0), 0,this);
        for(auto& p : _points)
        {
            auto leaf=this->insert_td(p,this->base_level_);
            leafs_.emplace(leaf->key(), leaf);
        }
        construct_level_maps();
    }


public:
   
    octant_iterator begin_leafs() const noexcept {return leafs_.begin();}
    octant_iterator end_leafs  () const noexcept {return leafs_.end();}

    octant_iterator begin_leafs() noexcept {return leafs_.begin();}
    octant_iterator end_leafs  () noexcept {return leafs_.end();}

    auto num_leafs() const noexcept {return leafs_.size();}

    const int& base_level() const noexcept{return base_level_;}
    int& base_level      () noexcept      {return base_level_;}
    const int& depth     () const noexcept{return depth_;}
    int& depth           () noexcept      {return depth_;}

    octant_type* root()const noexcept{return root_.get();}

    template<class Function>
    void refine(octant_iterator& _l, const Function& _f, bool _recursive = false)
    {
        if(_l->is_hanging())return;
        for(int i=0;i<_l->num_children();++i)
        {
            auto child=_l->refine(i);
            _f(child);
            auto c=leafs_.emplace(child->key(),child);
            c.first->second->flag(node_flag::octant);
            level_maps_[child->level()].emplace(child->key(),child);

        }
        _l=leafs_.erase(_l);
        ++depth_;
        if(!_recursive) std::advance(_l,_l->num_children()-1);
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

public: //traversals

	template<class Function>
	void traverse_dfs(Function f)
	{
		depth_first_search(root(), f);
	}
	
	template<class Function>
	void traverse_bfs(Function f, 
                      int min_level=0, 
                      int max_level=key_type::max_level())
	{
		for (int i=min_level; i<max_level; ++i)
			breadth_first_search(root(), f, i);
	}

private: 

    template<class Node>
    bool has_leaf(Node& _node ) const
    {
        auto it=leafs_.find(_node.key());
        if(it!=leafs_.end())
            return true;
        return false;
    }

    auto find_leafs(octant_base_type _node)
    {
        return octant_iterator(leafs_.find(_node.key()));
    }

    auto find_leaf_any_level(octant_base_type _node)  noexcept
    {
        octant_base_type n=_node;
        const auto it=leafs_.find(n.key());
        if(it!=leafs_.end())
        {
             return octant_iterator(it);
        }
        else
        {
            for(auto i=this->depth();i>=base_level();--i)
            {
                const auto parent =n.equal_coordinate_parent();
                const auto it=leafs_.find(parent.key());
                if( it!=leafs_.end() )return octant_iterator(it);
                n=parent;
            }
            return octant_iterator(it);
        }
        return octant_iterator(it);
    }

private:  //Top down insert strategy

    octant_type* insert_td(const coordinate_type& x, 
                    int level) 
    { 
        return insert_td(key_type(x,level)); 
    }
	octant_type* insert_td(const key_type& k)
    { 
        return insert_impl_top_down(k,root_.get());
    }

    octant_type* insert_impl_top_down(const key_type& k, octant_type* n) const
    {
        if (n->key() == k) return n;
        for (int i=n->num_children()-1; i>=0; --i)
        {
            if (n->children_[i])
            {
                if ( n->children_[i]->key() <= k) 
                {
                    return insert_impl_top_down(k, n->children_[i].get());
                }
            }
            else 
            {
                const auto ck = n->key().child(i);
                if (ck <= k)
                {
                    n->refine(i);
                    return insert_impl_top_down(k, n->children_[i].get());
                }
            }
        }
        throw std::runtime_error(
                "tree.hpp:insert_impl_top_down: Should have exited before!"
                );
    }

    template<class Function>
    void breadth_first_search(octant_type* n, Function& f, int level)
    {
        if (n->level() == level)
        {
            f(n);
            return;
        }
        for (std::size_t i=0; i<n->num_children(); ++i)
        {
            if(n->children_[i]) 
                breadth_first_search(n->children_[i].get(), f, level);
        }
    }


    template<class Function>
    void depth_first_search(octant_type* n, Function& f)
    {
        f(n);
        for (std::size_t i=0; i<n->num_children(); ++i)
        {
            if (n->child(i)) depth_first_search(n->child(i),f);
        }
    }

    void construct_level_maps()
    {
        level_maps_.clear();
        level_maps_.resize(this->depth()+1);
        dfs_iterator it_begin(root()); dfs_iterator it_end;
        for(auto it =it_begin;it!=it_end;++it)
        {
           level_maps_[it->level()].emplace(it->key(),it.ptr());
        }
    }


    
    static real_coordinate_type unit_transform(coordinate_type _x)
    {
        return _x;
    }



private:
    //octant_map_type leafs_; 
    coordinate_transform_t octant_to_real_coordinate_=&Tree::unit_transform;
    int base_level_=0;
    int depth_=0;
    std::shared_ptr<octant_type> root_=nullptr;

    //Convenience containers ... might be too expensive for book keeping
    std::vector<octant_ptr_map_type> level_maps_;
    octant_ptr_map_type leafs_;

};


} //namespace octree
#endif 
