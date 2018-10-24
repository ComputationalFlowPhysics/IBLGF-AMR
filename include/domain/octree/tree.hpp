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


/** @brief Octree
 *
 *  Parallel octree implementation using a full pointer-based implementation.
 */
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
        std::function<real_coordinate_type(real_coordinate_type, int _level)>;

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


    /**
     *  @brief Top-down construction of octree.
     *
     *  Given a vector of points and the correpsonding base level an
     *  octree is top down constructed, the leafs extracted and the level maps
     *  filled. The base level implicitly sets the maximum allowable extent of
     *  the sim.
     *
     *  \param[in] _points     Vector of lower-left corners of octants
     *  \param[in] _base_level Level of _points. Note that this fixed a maximum
     *                         extent, which can be respresented.
     */
    Tree (const std::vector<coordinate_type>& _points, int _base_level)
    {
        this->base_level_ = _base_level;
        depth_ = base_level_ + 1;

        //Construct interior octants, leaf map && level map
        root_ = std::make_shared<octant_type>(coordinate_type(0), 0, this);
        for (auto& p : _points)
        {
            auto leaf = this->insert_td(p, this->base_level_);
            leafs_.emplace(leaf->key(), leaf);
        }
        construct_level_maps();
        construct_neighbor_lists();
    }


public:

    octant_iterator begin_leafs() const noexcept {return leafs_.begin();}
    octant_iterator end_leafs  () const noexcept {return leafs_.end();}

    octant_iterator begin_leafs() noexcept {return leafs_.begin();}
    octant_iterator end_leafs  () noexcept {return leafs_.end();}

    octant_iterator begin(int _level)noexcept
    {
        return level_maps_[_level].begin();
    }
    octant_iterator end(int _level)noexcept
    {
        return level_maps_[_level].end();
    }
    octant_iterator begin(int _level) const noexcept
    {
        return level_maps_[_level].begin();
    }
    octant_iterator end  (int _level) const noexcept
    {
        return level_maps_[_level].end();
    }

    const octant_iterator find(int _level, key_type key) const noexcept
    {
        return level_maps_[_level].find(key);
    }

    octant_iterator find(int _level, key_type key) noexcept
    {
        return level_maps_[_level].find(key);
    }

    octant_type* find_octant(key_type _k)
    {

        auto it = root();
        auto l = it->level();


        while (it->key() != _k)
        {
            auto child_ = it->child(_k.sib_number(l));

            if  (child_ == nullptr)
                return nullptr;

            it = child_;

            l++;
        }

        return it;
    }




    auto num_leafs() const noexcept {return leafs_.size();}

    const int& base_level() const noexcept{return base_level_;}
    int& base_level      () noexcept      {return base_level_;}
    const int& depth     () const noexcept{return depth_;}
    int& depth           () noexcept      {return depth_;}

    octant_type* root()const noexcept{return root_.get();}

    template<class Function>
    void refine(octant_iterator& _l, const Function& _f,
                 bool _recursive = false)
    {
        for(int i=0;i<_l->num_children();++i)
        {
            auto child=_l->refine(i);
            _f(child);
            auto c=leafs_.emplace(child->key(),child);
            c.first->second->flag(node_flag::octant);
            level_maps_[child->level()].emplace(child->key(),child);
        }

        _l = leafs_.erase(_l);
        if(_l->level()+1 > depth_) depth_=_l->level()+1;
        if(!_recursive) std::advance(_l,_l->num_children()-1);
    }

    const auto& get_octant_to_level_coordinate() const noexcept
    {
        return octant_to_real_coordinate_;
    }
    auto& get_octant_to_level_coordinate() noexcept
    {
        return octant_to_real_coordinate_;
    }
    template<class T>
    auto octant_to_level_coordinate(T _x, int _level=0)
    {
        return octant_to_real_coordinate_(_x, _level);
    }

public: //traversals

    /**
     * @brief Recursive depth-first traversal
     *
     * Note: Better to use the corresponding iterators.
     * @param [in] f Function to be applied to dfs-node
     */
	template<class Function>
	void traverse_dfs(Function f)
	{
		depth_first_traverse(root(), f);
	}


    /**
     * @brief Recursive breadth-first traversal
     *
     * Note: Better to use the corresponding iterators.
     * @param [in] f Function to be applied to dfs-node
     * @param [in] min_level Startlevel
     * @param [in] max_level Endlevel
     */
	template<class Function>
	void traverse_bfs(Function f,
                      int min_level=0,
                      int max_level=key_type::max_level())
	{
		for (int i=min_level; i<max_level; ++i)
			breadth_first_traverse(root(), f, i);
	}

private: //find

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

    octant_type* find_leaf(key_type _k)
    {
        auto it=leafs_.find(_k);
        if(it!=leafs_.end())
            return it->second;
        else return nullptr;
    }



private:  //Top down insert strategy

    octant_type* insert_td(const coordinate_type& x, int level)
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

private: //traversal

    template<class Function>
    void breadth_first_traverse(octant_type* n, Function& f, int level)
    {
        if (n->level() == level)
        {
            f(n);
            return;
        }
        for (std::size_t i=0; i<n->num_children(); ++i)
        {
            if(n->children_[i])
                breadth_first_traverse(n->children_[i].get(), f, level);
        }
    }

    template<class Function>
    void depth_first_traverse(octant_type* n, Function& f)
    {
        f(n);
        for (std::size_t i=0; i<n->num_children(); ++i)
        {
            if (n->child(i)) depth_first_traverse(n->child(i),f);
        }
    }

private: // misc
public: // misc

    void construct_level_maps()
    {
        level_maps_.clear();
        level_maps_.resize(key_type::max_level());
        dfs_iterator it_begin(root()); dfs_iterator it_end;
        for(auto it =it_begin;it!=it_end;++it)
        {
           level_maps_[it->level()].emplace(it->key(),it.ptr());
        }
    }

    void construct_neighbor_lists()
    {
        dfs_iterator it_begin(root()); dfs_iterator it_end;

        for(auto it =it_begin;it!=it_end;++it)
        {
            construct_octant_neighbor(it);
        }
    }

    void construct_octant_neighbor(auto it)
    {
        for (int id = 0; id<27; id++)
        {
            int tmp = id;
            auto idx = tmp % 3 -1;
            tmp /=3;
            auto idy = tmp % 3 -1;
            tmp /=3;
            auto idz = tmp % 3 -1;

            auto k = it->key().neighbor({{idx,idy,idz}});
            auto neighbor_i = find_octant(k);
            if (neighbor_i == nullptr)
                continue;

            it->neighbor(id, neighbor_i);
            }
    }



    static coordinate_type unit_transform(coordinate_type _x, int _level)
    {
        return _x;
    }

private:
    /** \brief Coordinate transform from octant coordinate to real coordinates*/
    coordinate_transform_t octant_to_real_coordinate_=&Tree::unit_transform;


    int base_level_=0;                              ///< Base level
    int depth_=0;                                   ///< Tree depth
    std::shared_ptr<octant_type> root_=nullptr;     ///< Tree root
    std::vector<octant_ptr_map_type> level_maps_;   ///< Octants per level
    octant_ptr_map_type leafs_;                     ///< Map of tree leafs

};


} //namespace octree
#endif
