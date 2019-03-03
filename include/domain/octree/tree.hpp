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

    template<class Iterator=bfs_iterator>
    using conditional_iterator = typename detail::ConditionalIterator<Iterator>;

    using block_descriptor_type = typename octant_type::block_descriptor_type;

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
        std::vector<key_type> keys;
        for(auto& p : _points )
        {
            keys.push_back(key_type(p,_base_level));
        }
        this->init(keys, _base_level);
    }

    /**
     *  @brief Top-Down construction of octree.
     *
     *  Basd on a vector of keys
     */
    Tree(  const std::vector<key_type>& _keys, int _base_level)
    {
        this->init(_keys, _base_level);
    }

    Tree(int _base_level)
    {
        this->base_level_ = _base_level;
        depth_ = base_level_ + 1;
        root_ = std::make_shared<octant_type>(
                coordinate_type(0), 0, this);
    }

public:


    template<class Function=std::function<void(octant_type* c)>>
    void init(const std::vector<key_type>& _keys,
              const Function& f=[](octant_type* o){ return; })
    {
        this->init(_keys, this->base_level_,f);
        // Maps construction

        // Ke TODO: check if this is ok
        // this->construct_leaf_maps();
        this->construct_level_maps();
    }

    template<class Function=std::function<void(octant_type* c)>>
    void init(const std::vector<key_type>& _keys, int _base_level,
              const Function& f=[](octant_type* o){ return; })
    {
        this->base_level_ = _base_level;
        depth_ = base_level_ + 1;
        root_ = std::make_shared<octant_type>(coordinate_type(0), 0, this);
        for (auto& k : _keys)
        {
            auto octant = this->insert_td(k);
            f(octant);
            if (octant->level()+1 > depth_) depth_=octant->level()+1;
        }
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
            child->flag_leaf(true);
            _f(child);
            leafs_.emplace(child->key(),child);
            level_maps_[child->level()].emplace(child->key(),child);
        }

        leafs_.erase(_l->key());
        _l ->flag_leaf(false);

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

public: // misc

    void construct_level_maps() noexcept
    {
        level_maps_.clear();
        level_maps_.resize(key_type::max_level());
        dfs_iterator it_begin(root()); dfs_iterator it_end;
        for(auto it =it_begin;it!=it_end;++it)
        {
           level_maps_[it->level()].emplace(it->key(),it.ptr());
        }
    }


public: // neighborlist
    auto  construct_neighbor_lists(bool _global=true)noexcept
    {
        dfs_iterator begin(root()); dfs_iterator end;
        std::set<key_type> res;
        for(auto it =begin;it!=end;++it)
        {
            neighbor_lookup(it.ptr(),res,_global);
        }
        return res;
    }

    template<class Client, class InitFunction>
    void query_neighbor_octants( Client* _c, InitFunction& _f )
    {
        auto key_set=construct_neighbor_lists();
        std::vector<key_type> keys;
        keys.insert(keys.begin(),key_set.begin(), key_set.end());

        auto ranks= _c->rank_query( keys );
        for(std::size_t i = 0; i < ranks.size();++i )
        {
            if(ranks[i]>=0)
            {
                auto nn = this->insert_td(keys[i]);
                std::set<key_type> dummy;
                neighbor_lookup(nn,dummy, false, true );
                _f(nn);
                nn->rank()=ranks[i];
            }
        }
    }

private: // neighborlist
    /** @brief Construction neighbor list based on  local tree */
    void neighbor_lookup( octant_type* it,
                          std::set<key_type>& res,
                          bool _global,
                          bool _update_neighbors=false )
    {
        it->neighbor_clear();
        auto nkeys=it->get_neighbor_keys();
        for(std::size_t i = 0; i< nkeys.size();++i)
        {
            if(nkeys[i].is_end()) continue;
            const auto neighbor_i = this->find_octant(nkeys[i]);
            it->neighbor(i, neighbor_i);

            if(_global && !neighbor_i)
            {
                res.emplace( nkeys[i] );
            }
            if(neighbor_i && _update_neighbors)
            {
                it->neighbor(i, neighbor_i);
                const auto offset=
                    it->tree_coordinate()-neighbor_i->tree_coordinate();
                const auto idx=
                    nearast_neighbor_hood.globalCoordinate_to_index(offset);
                neighbor_i->neighbor(idx,it);
            }
        }
    }


public: // influence list



    /** @brief Construct the influence list of this octrant
     *
     *  @detail: Influence list are the children of the parent of the octant it
     *           without the nearest neighbor.
     */
    auto construct_influence_lists(bool _global= true)
    {
        dfs_iterator it_begin(root()); dfs_iterator it_end;

        std::set<influence_helper> res;
        for(auto it =it_begin;it!=it_end;++it)
        {
            influence_lookup(it.ptr(),res);
        }
        return res;
    }

    template<class Client, class InitFunction>
    void query_influence_octants( Client* _c, InitFunction& _f )
    {
        const auto infl_helper=construct_influence_lists();
        std::vector<key_type> keys;
        for(auto& inf : infl_helper)
        {
            keys.emplace_back(inf.key());
        }

        auto ranks= _c->rank_query( keys );
        int count=0;
        for(auto& inf : infl_helper)
        {
            if(ranks[count]>=0)
            {
                auto nn = this->insert_td(inf.key());
                nn->rank()=ranks[count];
                std::set<influence_helper> dummy;
                influence_lookup(nn,dummy, false );
                inf.set(nn);
                _f(nn);
            }
            ++count;
        }


        //boost::mpi::communicator w;
        //dfs_iterator begin(root()); dfs_iterator end;
        //for(auto it =begin;it!=end;++it)
        //{
        //    for(int i =0;i<189;++i)
        //    {
        //        auto nn=it->neighbor(i);
        //        if(nn && nn->rank()!=w.rank() && nn->rank()>=0)
        //        {
        //            std::cout<<"inf->rank() "<<nn->rank() <<std::endl;
        //        }
        //    }
        //}
    }

private:// influence list

    //FIXME: This is bullshit, lets thinks about a set or indexing the
    //       whole 6^3 field
    struct influence_helper
    {
        influence_helper( key_type _k ): key_(_k){ }
        bool operator< (const influence_helper &other) const
        {
            return key_ < other.key_;
        }


        ///< append to lists
        void update(octant_type* _oc, int _inf_number) const
        {
            influence_.emplace_back(_oc);
            influence_number_.emplace_back(_inf_number);
        }

        //Set the octant belonging to key within the infl of ocant in
        //influence_
        void set(octant_type* oc) const
        {
            for(std::size_t i =0; i<influence_.size();++i)
            {
                influence_[i]->influence(influence_number_[i], oc);
            }
        }

        const auto& key() const {return key_;}
        auto& key() {return key_;}

    private:
        key_type key_; //of octant in questions

        //ocntants influence by key-cotant and influence index
        mutable  std::vector<octant_type*> influence_;
        mutable std::vector<int> influence_number_;
    };

    /** @brief Construct the influence list of this octrant.
     *
     *  @detail: Influence list are the children of the parent of the octant it
     *           without the nearest neighbor.
     *           This list is stored direcly in the octrant
     *  @return: List of keys that have not been in the local tree.
     *           These keys need to checked for existence in the global tree.
     */
    auto influence_lookup(octant_type* it,
                          std::set<influence_helper>& influence_set,
                          bool _global=true)
    {
        it->influence_clear();
        if(!it ||  !it->parent()) return;
        //std::set<infl_helper> blafskjdfdsl sadkjad a
        //std::set<infl

        int infl_id = 0;
        it->influence_number(infl_id);
        const auto coord = it->key().coordinate();

        const auto p = it->parent();
        for (int p_n_id=0;
                 p_n_id<static_cast<int>(p->num_neighbors());
                 ++p_n_id)
        {
            const auto  p_n = p->neighbor(p_n_id);
            if (p_n)
            {
                for(int p_n_child_id=0;
                        p_n_child_id<static_cast<int>(p_n->num_children());
                        ++p_n_child_id)
                {
                    const auto child_key = p_n->key().child(p_n_child_id);
                    const auto p_n_c_coord = child_key.coordinate();
                    if ((std::abs(p_n_c_coord.x() - coord.x())>1) ||
                            (std::abs(p_n_c_coord.y() - coord.y())>1) ||
                            (std::abs(p_n_c_coord.z() - coord.z())>1))
                    {
                        const auto p_n_child = p_n->child(p_n_child_id);
                        if (p_n_child)
                        {
                            it->influence(infl_id, p_n_child);
                        }
                        else if( _global  )
                        {
                            auto inf= influence_set.emplace(child_key);
                            inf.first->update( it, infl_id);
                        }

                        infl_id++;
                    }
                }
            }
        }
        it->influence_number(infl_id);
    }


public: //children and parent queries

    /** @brief Query the ranks for all children */
    template<class Client, class InitFunction>
    void query_children( Client* _c, InitFunction& _f )
    {
        dfs_iterator it_begin(root()); dfs_iterator it_end;

        std::vector<key_type> keys;
        for(auto it =it_begin;it!=it_end;++it)
        {
            const auto it_key = it->key();
            if(it->locally_owned())
            {
                //Check children
                for(int i=0;i<it->num_children();++i)
                {

                    if(!it->child(i))
                    {
                        keys.emplace_back(it_key.child(i));
                    }
                    else if(!it->child(i)->locally_owned())
                    {
                        keys.emplace_back(it_key.child(i));
                    }
                }
            }
        }

        auto ranks= _c->rank_query( keys );
        for(std::size_t i = 0; i < ranks.size();++i )
        {
            if(ranks[i]>=0)
            {
                auto nn = this->insert_td(keys[i]);
                _f(nn);
                nn->rank()=ranks[i];
            }
        }
    }

  /** @brief Query the ranks for all children */
    template<class Client, class InitFunction>
    void query_parents( Client* _c, InitFunction& _f )
    {
        dfs_iterator it_begin(root()); dfs_iterator it_end;

        std::set<key_type> keys_set;
        std::vector<key_type> keys;
        for(auto it =it_begin;it!=it_end;++it)
        {
            const auto it_key = it->key();
            if(it->locally_owned())
            {
                //Check children
                if( !(it->parent()) || (!it->parent()->locally_owned()) )
                {
                    keys_set.insert(it_key.parent());
                }
            }
        }
        std::copy(keys_set.begin(), keys_set.end(), std::back_inserter(keys));
        auto ranks= _c->rank_query( keys );
        for(std::size_t i = 0; i < ranks.size();++i )
        {
            if(ranks[i]>=0)
            {
                auto nn = this->insert_td(keys[i]);
                _f(nn);
                nn->rank()=ranks[i];
            }
        }
    }


    /** @brief Query ranks for all interior octants */
    template<class Client, class InitFunction>
    void query_interior( Client* _c, InitFunction _f)
    {
        dfs_iterator it_begin(root()); dfs_iterator it_end;

        std::vector<key_type> keys;
        for(auto it =it_begin;it!=it_end;++it)
        {
            if( it->rank()<0 )
            {
                keys.emplace_back(it->key());
            }
        }

        auto ranks= _c->rank_query( keys );
        for(std::size_t i = 0; i < ranks.size();++i )
        {
            if(ranks[i]>=0)
            {
                auto nn = this->find_octant(keys[i]);
                _f(nn);
                nn->rank()=ranks[i];
            }
        }
    }

    /** @brief Query ranks for all interior octants */
    template<class Client>
    void query_leaves( Client* _c)
    {
        boost::mpi::communicator  w;

        dfs_iterator it_begin(root()); dfs_iterator it_end;

        std::vector<key_type> keys;
        for(auto it =it_begin;it!=it_end;++it)
        {
            keys.emplace_back(it->key());
        }

        auto leaves= _c->leaf_query( keys );

        int i = 0;
        for(auto it =it_begin;it!=it_end;++it)
        {
            it->flag_leaf((leaves[i++]));
        }
    }


public: //Query ranks of all octants, which are assigned in local tree

    /** @brief Query from server and construct all
     *         maps for neighbors, influence
     *         list, children and interior nodes
     **/
    template<class Client, class InitFunction>
    void construct_maps( Client* _c, InitFunction& _f )
    {
        //Queries
        this->query_neighbor_octants(_c,_f);
        this->query_influence_octants(_c,_f);
        this->query_children(_c,_f);
        this->query_parents(_c,_f);
        this->query_interior(_c, _f);

        //Maps constructions
        this-> construct_level_maps();
    }



public: // leafs maps

    //void construct_flag_leaf()
    //{
    //    dfs_iterator it_begin(root()); dfs_iterator it_end;
    //    for(auto it =it_begin;it!=it_end;++it)
    //    {
    //        it->flag_leaf(it->is_leaf_search());
    //    }
    //}

    auto leaf_map()
    {
        return leafs_;
    }

    void construct_leaf_maps(bool _from_existing_flag=false)
    {
        leafs_.clear();
        dfs_iterator it_begin(root()); dfs_iterator it_end;

        for(auto it =it_begin;it!=it_end;++it)
        {
            if (!_from_existing_flag)
                it->flag_leaf(it->is_leaf_search());

            if(it->is_leaf())
            {
                leafs_.emplace(it->key(), it.ptr());
            }
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


    const block_descriptor_type nearast_neighbor_hood=
        block_descriptor_type(coordinate_type(-1), coordinate_type(3));
};


} //namespace octree
#endif
