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

#ifndef OCTREE_INCLUDED_LINEAR_TREE_HPP
#define OCTREE_INCLUDED_LINEAR_TREE_HPP

#include <vector>
#include <memory>
#include <cmath>
#include <set>
#include <unordered_set>
#include <string>
#include <map>
#include <unordered_map>
#include <functional>

// IBLGF-specific
#include <iblgf/global.hpp>
#include <iblgf/domain/octree/octant.hpp>
#include <iblgf/domain/octree/tree_utils.hpp>

namespace iblgf
{
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
    using octant_type = Octant<Dim, DataType>;
    using octant_base_type = Octant_base<Dim, DataType>;
    using real_coordinate_type =
        typename octant_base_type::real_coordinate_type;
    using key_type = typename octant_base_type::key_type;
    using coordinate_type = typename key_type::coordinate_type;
    using scalar_coordinate_type = typename key_type::scalar_coordinate_type;

    using octant_ptr_map_type = std::map<key_type, octant_type*>;
    using octant_iterator = MapValuePtrIterator<octant_ptr_map_type>;
    using raw_octant_iterator = typename octant_ptr_map_type::iterator;

    using dfs_iterator = typename detail::IteratorDfs<octant_type>;
    using bfs_iterator = typename detail::IteratorBfs<octant_type>;

    template<class Iterator = bfs_iterator>
    using conditional_iterator = typename detail::ConditionalIterator<Iterator>;

    using block_descriptor_type = typename octant_type::block_descriptor_type;

    using coordinate_transform_t =
        std::function<real_coordinate_type(real_coordinate_type, int _level)>;

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
    Tree(const std::vector<coordinate_type>& _points, int _base_level)
    {
        std::vector<key_type> keys;
        for (auto& p : _points) { keys.push_back(key_type(p, _base_level)); }
        this->init(keys, _base_level);
    }

    /**
     *  @brief Top-Down construction of octree.
     *
     *  Basd on a vector of keys
     */
    Tree(const std::vector<key_type>& _keys, int _base_level)
    {
        this->init(_keys, _base_level);
    }

    Tree(int _base_level)
    {
        this->base_level_ = _base_level;
        depth_ = base_level_ + 1;
        root_ = std::make_shared<octant_type>(coordinate_type(0), 0, this);
    }

  public:
    template<class Function = std::function<void(octant_type* c)>>
    void init(
        const std::vector<key_type>& _keys,
        const Function&              f = [](octant_type* o) { return; })
    {
        this->init(_keys, this->base_level_, f);
    }

    template<class Function = std::function<void(octant_type* c)>>
    void init(
        const std::vector<key_type>& _keys, int _base_level,
        const Function& f = [](octant_type* o) { return; })
    {
        this->base_level_ = _base_level;
        depth_ = base_level_ + 1;
        root_ = std::make_shared<octant_type>(coordinate_type(0), 0, this);
        insert_keys(_keys, f);
        //for (auto& k : _keys)
        //{

        //    auto octant = this->insert_td(k);
        //    f(octant);
        //    if (octant->level()+1 > depth_) depth_=octant->level()+1;
        //}
    }

    template<class Function = std::function<void(octant_type* c)>>
    void insert_keys(
        const std::vector<key_type>& _keys,
        const Function&              f = [](octant_type* o) { return; },
        bool                         update_depth = true)
    {
        for (auto& k : _keys)
        {
            auto octant = this->insert_td(k);
            //if (!octant->has_data() || !octant->data_ref().is_allocated())
            f(octant);
        }
    }

  public:
    auto begin() const noexcept { return dfs_iterator(root_.get()); }
    auto end() const noexcept { return dfs_iterator(); }

    octant_iterator begin_leafs() const noexcept { return leafs_.begin(); }
    octant_iterator end_leafs() const noexcept { return leafs_.end(); }

    octant_iterator begin_leafs() noexcept { return leafs_.begin(); }
    octant_iterator end_leafs() noexcept { return leafs_.end(); }

    octant_iterator begin(int _level) noexcept
    {
        return level_maps_[_level].begin();
    }
    octant_iterator end(int _level) noexcept
    {
        return level_maps_[_level].end();
    }
    octant_iterator begin(int _level) const noexcept
    {
        return level_maps_[_level].begin();
    }
    octant_iterator end(int _level) const noexcept
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

    octant_type* find_octant(key_type _k) const
    {
        return find_impl_top_down(_k, root());
        auto it = root();
        auto l = it->level();
        while (it->key() != _k)
        {
            auto child_ = it->child(_k.sib_number(l));
            if (child_ == nullptr) return nullptr;

            it = child_;
            l++;
        }
        return it;
        //dfs_iterator it_begin(root()); dfs_iterator it_end;
        //for(auto it =it_begin;it!=it_end;++it)
        //{
        //    if(it->key()==_k)
        //        return it.ptr();
        //}
        //return find_top_down(_k,root());
        //return nullptr;
    }

    octant_type* find_impl_top_down(const key_type& k, octant_type* n) const
        noexcept
    {
        if (n->key() == k) return n;
        if (n->key().level() == k.level()) return nullptr;
        for (int i = n->num_children() - 1; i >= 0; --i)
        {
            if (n->child(i) && n->child(i)->key() <= k)
                return find_impl_top_down(k, n->child(i));
        }
        return nullptr;
    }

    auto num_leafs() const noexcept { return leafs_.size(); }

    const int& base_level() const noexcept { return base_level_; }
    int&       base_level() noexcept { return base_level_; }
    const int& depth() const noexcept { return depth_; }
    int&       depth() noexcept { return depth_; }

    octant_type* root() const noexcept { return root_.get(); }

    auto unfound_neighbors(octant_type* _l, bool correction_as_neighbors = true)
    {
        std::vector<key_type> keys;

        auto _k = _l->key();
        auto neighbor_keys = _k.get_neighbor_keys();

        for (auto& nk : neighbor_keys)
        {
            if (nk.is_end()) continue;
            if (nk == _k) continue;
            const auto neighbor_i = this->find_octant(nk);
            if (!neighbor_i || !neighbor_i->has_data()) keys.emplace_back(nk);
            else if (!correction_as_neighbors && neighbor_i->is_correction())
                keys.emplace_back(nk);
        }

        return keys;
    }

    template<class Function>
    void insert_correction_neighbor(octant_type* _l, const Function& _f)
    {
        auto keys = unfound_neighbors(_l);

        for (auto& k : keys)
        {
            auto _neighbor = this->insert_td(k);
            if (!_neighbor || !_neighbor->has_data()) _f(_neighbor);
        }
    }

    void deletionReset_2to1(
        octant_type* _l, std::unordered_set<octant_type*>& checklist)
    {
        auto check = checklist.find(_l);

        if (check == checklist.end())
        {
            if (_l->refinement_level() < 0) return;

            auto k = _l->key();
            auto neighbor_keys = k.get_neighbor_keys();
            for (auto nk : neighbor_keys)
            {
                auto neighbor_i = this->find_octant(nk);

                if (!neighbor_i || !neighbor_i->has_data()) continue;

                neighbor_i->aim_deletion(false);
                deletionReset_2to1(neighbor_i->parent(), checklist);
            }

            checklist.emplace(_l);
        }
    }

    bool try_2to1(key_type _k, block_descriptor_type key_bd_box)
    {
        // Dynmaic Programming to rduce repeated checks
        std::unordered_map<key_type, bool> checklist;
        return try_2to1(_k, key_bd_box, checklist);
    }

    bool try_2to1(key_type _k, block_descriptor_type key_bd_box,
        std::unordered_map<key_type, bool>& checklist)
    {
        auto check = checklist.find(_k);

        if (check == checklist.end())
        {
            int  rf_l = _k.level() - base_level_;
            auto neighbor_keys = _k.get_neighbor_keys();

            if (rf_l == 0)
            {
                for (auto nk : neighbor_keys)
                {
                    if (nk == _k) continue;
                    if (!key_bd_box.is_inside(nk.coordinate()))
                    {
                        checklist.emplace(_k, false);
                        return false;
                    }
                }
                checklist.emplace(_k, true);
                return true;
            }
            else if (rf_l > 0)
            {
                for (auto nk : neighbor_keys)
                {
                    if (nk == _k) continue;

                    auto nk_p = nk.parent();
                    if (!try_2to1(nk_p, key_bd_box, checklist))
                    {
                        checklist.emplace(_k, false);
                        return false;
                    }
                }

                checklist.emplace(_k, true);
                return true;
            }
        }
        else
        {
            return check->second;
        }
        return false;
    }

    template<class Function>
    void refine(octant_type* _l, const Function& _f, bool ratio_2to1 = true)
    {
        //Check 2:1 balance constraint
        //bool neighbors_exists=true;
        if (ratio_2to1)
        {
            for (int i = 0; i < _l->nNeighbors(); ++i)
            {
                auto n_i = _l->neighbor(i);
                if (n_i && n_i->has_data()) n_i->aim_deletion(false);
            }

            if (_l->refinement_level() >= 0)
            {
                auto neighbor_keys = unfound_neighbors(_l, false);
                for (auto& k : neighbor_keys)
                {
                    if (k.is_end()) continue;

                    if (_l->refinement_level() == 0)
                    {
                        auto oct = this->insert_td(k);
                        if (!oct->has_data()) _f(oct);

                        oct->flag_leaf(true);
                        oct->flag_correction(false);
                        oct->leaf_boundary() = false;
                        oct->aim_deletion(false);
                    }
                    else
                    {
                        auto parent_key = k.parent();

                        auto pa = this->find_octant(parent_key);
                        if (!pa)
                        {
                            pa = this->insert_td(parent_key);
                            _f(pa);
                        }
                        this->refine(pa, _f, ratio_2to1);
                    }
                }
            }
            else
            {
                //throw
                //std::runtime_error("Cannot satisfy 2:1 refinement requirement for base level ");
            }
        }

        for (int i = 0; i < _l->num_children(); ++i)
        {
            if (_l->child(i) && _l->child(i)->has_data()) continue;
            auto child = _l->refine(i);
            if (!child->has_data()) _f(child);
        }

        for (int i = 0; i < _l->num_children(); ++i)
        {
            auto child = _l->child(i);
            child->flag_leaf(true);
            child->flag_correction(false);
            child->leaf_boundary() = false;
            child->aim_deletion(false);
        }

        _l->flag_leaf(false);
        _l->flag_correction(false);
        _l->leaf_boundary() = false;
        _l->aim_deletion(false);

        if (_l->level() + 2 > depth_) depth_ = _l->level() + 2;
    }

    void delete_oct(octant_type* oct)
    {
        oct->rank() = -1;
        oct->deallocate_data();

        if (oct->is_leaf_search())
        {
            int cnumber = oct->key().child_number();
            oct = oct->parent();

            if (oct) oct->delete_child(cnumber);
        }

        //while (oct->refinement_level()<0 && oct->is_leaf_search())
        //{
        //    oct->rank()=-1;
        //    oct->deallocate_data();

        //    int cnumber=oct->key().child_number();
        //    oct=oct->parent();
        //    if(oct) oct->delete_child(cnumber);
        //}
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
    auto octant_to_level_coordinate(T _x, int _level = 0)
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
    void traverse_bfs(
        Function f, int min_level = 0, int max_level = key_type::max_level())
    {
        for (int i = min_level; i < max_level; ++i)
            breadth_first_traverse(root(), f, i);
    }

  private: //find
    template<class Node>
    bool has_leaf(Node& _node) const
    {
        auto it = leafs_.find(_node.key());
        if (it != leafs_.end()) return true;
        return false;
    }

    auto find_leafs(octant_base_type _node)
    {
        return octant_iterator(leafs_.find(_node.key()));
    }

    octant_type* find_leaf(key_type _k)
    {
        auto it = leafs_.find(_k);
        if (it != leafs_.end()) return it->second;
        else
            return nullptr;
    }

  public: //Top down insert strategy
    octant_type* insert_td(const coordinate_type& x, int level)
    {
        return insert_td(key_type(x, level));
    }
    octant_type* insert_td(const key_type& k)
    {
        return insert_impl_top_down(k, root_.get());
    }

  private: //Top down insert strategy
    octant_type* insert_impl_top_down(const key_type& k, octant_type* n) const
    {
        if (n->key() == k) return n;
        for (int i = n->num_children() - 1; i >= 0; --i)
        {
            if (n->children_[i])
            {
                if (n->children_[i]->key() <= k)
                { return insert_impl_top_down(k, n->children_[i].get()); }
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
            "tree.hpp:insert_impl_top_down: Should have exited before!");
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
        for (std::size_t i = 0; i < n->num_children(); ++i)
        {
            if (n->children_[i])
                breadth_first_traverse(n->children_[i].get(), f, level);
        }
    }

    template<class Function>
    void depth_first_traverse(octant_type* n, Function& f)
    {
        f(n);
        for (std::size_t i = 0; i < n->num_children(); ++i)
        {
            if (n->child(i)) depth_first_traverse(n->child(i), f);
        }
    }

  public: // misc
    void construct_level_maps()
    {
        level_maps_.clear();
        level_maps_.resize(key_type::max_level());
        dfs_iterator it_begin(root());
        dfs_iterator it_end;
        for (auto it = it_begin; it != it_end; ++it)
        {
            if (it.ptr() && it->has_data())
                level_maps_[it->level()].emplace(it->key(), it.ptr());
        }
    }

  public:
    void lookup_local_change(std::set<key_type>& res)
    {
        dfs_iterator begin(root());
        dfs_iterator end;
        for (auto it = begin; it != end; ++it) { res.emplace(it->key()); }
    }

    void lookup_parents(std::set<key_type>& res)
    {
        dfs_iterator begin(root());
        dfs_iterator end;
        for (auto it = begin; it != end; ++it)
        {
            if (!it->locally_owned()) continue;
            res.emplace(it->key().parent());
        }
    }

    void lookup_children(std::set<key_type>& res)
    {
        dfs_iterator begin(root());
        dfs_iterator end;
        for (auto it = begin; it != end; ++it)
        {
            if (!it->locally_owned()) continue;
            auto it_key = it->key();
            for (int i = 0; i < it->num_children(); ++i)
            { res.emplace(it_key.child(i)); }
        }
    }

    void lookup_infls(std::set<key_type>& res)
    {
        dfs_iterator begin(root());
        dfs_iterator end;
        for (auto it = begin; it != end; ++it)
        {
            if (!it->locally_owned()) continue;
            auto n_keys = it->get_infl_keys();
            for (auto& n_k : n_keys)
            {
                if (!n_k.is_end()) res.emplace(n_k);
            }
        }
    }

    void lookup_neighbors(std::set<key_type>& res)
    {
        dfs_iterator begin(root());
        dfs_iterator end;
        for (auto it = begin; it != end; ++it)
        {
            if (!it->locally_owned()) continue;
            auto n_keys = it->get_neighbor_keys();
            for (auto& n_k : n_keys)
            {
                if (!n_k.is_end()) res.emplace(n_k);
            }
        }
    }

    void construct_lists()
    {
        dfs_iterator begin(root());
        dfs_iterator end;
        for (auto it = begin; it != end; ++it)
        {
            if (!it->has_data()) continue;
            neighbor_list_build(it.ptr());
            influence_list_build(it.ptr());
        }
    }

    void neighbor_list_build(octant_type* it)
    {
        boost::mpi::communicator w;

        it->neighbor_clear();
        auto nkeys = it->get_neighbor_keys();

        for (std::size_t i = 0; i < nkeys.size(); ++i)
        {
            const auto neighbor_i = this->find_octant(nkeys[i]);
            if (!neighbor_i || !neighbor_i->has_data())
                it->neighbor(i, nullptr);
            else
                it->neighbor(i, neighbor_i);
        }
    }

    void influence_list_build(octant_type* it)
    {
        it->influence_clear();
        int infl_id = 0;
        it->influence_number(infl_id);
        if (!it->parent()) return;

        auto nkeys = it->get_infl_keys();

        for (std::size_t i = 0; i < nkeys.size(); ++i)
        {
            const auto infl_i = this->find_octant(nkeys[i]);
            if (infl_i && infl_i->has_data()) it->influence(infl_id++, infl_i);
        }
        it->influence_number(infl_id);
    }

  public: //children and parent queries
    template<class Client>
    void query_ranks(Client* _c, std::set<key_type>& res)
    {
        dfs_iterator it_begin(root());
        dfs_iterator it_end;

        std::vector<key_type> keys;
        std::copy(res.begin(), res.end(), std::back_inserter(keys));

        std::sort(
            keys.begin(), keys.end(), [](key_type k1, key_type k2) -> bool {
                return k1.level() > k2.level();
            });
        auto ranks = _c->rank_query(keys);
        for (std::size_t i = 0; i < keys.size(); ++i)
        {
            boost::mpi::communicator world;
            auto                     nn = this->find_octant(keys[i]);
            if (nn && nn->has_data())
            {
                if (ranks[i] > 0) { nn->rank() = ranks[i]; }
                else
                {
                    nn->flag_leaf(false);
                    nn->flag_correction(false);
                    nn->leaf_boundary() = false;
                    nn->aim_deletion(true);

                    this->delete_oct(nn);
                }
            }
            else
            {
                if (ranks[i] > 0)
                {
                    auto nn = this->insert_td(keys[i]);
                    nn->rank() = ranks[i];
                }
            }
        }
    }

    template<class Client>
    void query_masks(Client* _c)
    {
        dfs_iterator it_begin(root());
        dfs_iterator it_end;

        std::vector<key_type> keys;
        for (auto it = it_begin; it != it_end; ++it)
        {
            if (it->has_data()) keys.emplace_back(it->key());
        }

        auto masks = _c->mask_query(keys);

        for (std::size_t i = 0; i < masks.size(); ++i)
        {
            auto nn = this->find_octant(keys[i]);
            if (!nn) throw std::runtime_error("didn't find key for mask query");
            nn->flag_mask((masks[i]));
        }
    }

    template<class Client>
    void query_corrections(Client* _c)
    {
        boost::mpi::communicator w;

        dfs_iterator it_begin(root());
        dfs_iterator it_end;

        std::vector<key_type> keys;
        for (auto it = it_begin; it != it_end; ++it)
        {
            if (it->has_data()) keys.emplace_back(it->key());
        }

        auto corrections = _c->correction_query(keys);

        for (std::size_t i = 0; i < corrections.size(); ++i)
        {
            auto nn = this->find_octant(keys[i]);
            if (!nn)
                throw std::runtime_error(
                    "didn't find key for correction query");
            nn->flag_correction((corrections[i]));
        }
    }

    template<class Client>
    void query_flags(Client* _c)
    {
        boost::mpi::communicator w;

        dfs_iterator it_begin(root());
        dfs_iterator it_end;

        std::vector<key_type> keys;
        for (auto it = it_begin; it != it_end; ++it)
        {
            if (it->has_data()) keys.emplace_back(it->key());
        }

        auto flags = _c->flag_query(keys);

        for (std::size_t i = 0; i < flags.size(); ++i)
        {
            auto nn = this->find_octant(keys[i]);
            if (!nn) throw std::runtime_error("didn't find key for leaf query");
            nn->flags() = flags[i];
        }
    }

  public: //Query ranks of all octants, which are assigned in local tree
    template<class Client>
    void construct_ghosts(Client* _c)
    {
        std::set<key_type> res;
        this->lookup_neighbors(res);
        this->lookup_infls(res);
        this->lookup_children(res);
        this->lookup_parents(res);
        this->lookup_local_change(res);

        this->query_ranks(_c, res);
        this->allocate_ghosts(_c);
    }

    /** @brief Query from server and construct all
     *         maps for neighbors, influence
     *         list, children and interior nodes
     **/
    template<class Client>
    void construct_maps(Client* _c)
    {
        this->construct_ghosts(_c);
        this->construct_lists();
        this->construct_level_maps();
    }

    template<class Client>
    void allocate_ghosts(Client* _c)
    {
        auto _f = [_c, this](octant_type* _o, bool allocate_data) {
            auto level = _o->refinement_level();
            level = level >= 0 ? level : 0;
            auto bbase =
                this->octant_to_level_coordinate(_o->tree_coordinate(), level);

            _o->data_ptr() = std::make_shared<DataType>(
                bbase, _c->domain()->block_extent(), level, allocate_data);
        };

        //Allocate Ghost octants
        dfs_iterator it_begin(root());
        dfs_iterator it_end;
        for (auto it = it_begin; it != it_end; ++it)
        {
            //if (it->locally_owned()) continue;
            if (it->rank() <= 0) continue;
            if (it->has_data() && it->data_ref().is_allocated()) continue;

            bool allocate_data = false;
            for (int i = 0; i < it->num_children(); ++i)
            {
                if (it->child(i) && it->child(i)->has_data() &&
                    it->child(i)->locally_owned())
                    allocate_data = true;
            }
            _f(it.ptr(), allocate_data || it->locally_owned());
        }
    }

  public: // leafs maps
    auto leaf_map() { return leafs_; }

    void construct_leaf_maps(bool _from_existing_flag = false)
    {
        leafs_.clear();
        dfs_iterator it_begin(root());
        dfs_iterator it_end;

        for (auto it = it_begin; it != it_end; ++it)
        {
            //TODO why would it not have ptr()
            if (!it->has_data()) continue;

            if (!_from_existing_flag) it->flag_leaf(it->is_leaf_search());

            if (it->is_leaf()) { leafs_.emplace(it->key(), it.ptr()); }
        }
    }

    static coordinate_type unit_transform(coordinate_type _x, int _level)
    {
        return _x;
    }

  public: //Restart
    void write(std::string _filename) const
    {
        std::ofstream ofs(_filename, std::ios::binary);
        if (!ofs.is_open())
        { throw std::runtime_error("Could not open file: " + _filename); }
        dfs_iterator begin(root());
        dfs_iterator end;
        for (auto it = begin; it != end; ++it)
        {
            const auto id = it->key().id();
            ofs.write(reinterpret_cast<const char*>(&id), sizeof(id));
        }

        for (auto it = begin; it != end; ++it)
        {
            const bool leaf_flag = it->is_leaf();
            ofs.write(
                reinterpret_cast<const char*>(&leaf_flag), sizeof(leaf_flag));
        }
        ofs.close();
    }

    void read(std::string _filename, std::vector<key_type>& keys,
        std::vector<bool>& leafs) const
    {
        std::ifstream ifs(_filename, std::ios::binary);

        if (!ifs.is_open())
        { throw std::runtime_error("Could not open file: " + _filename); }

        ifs.seekg(0, std::ios::end);
        const auto fileSize = ifs.tellg();
        ifs.seekg(0, std::ios::beg);
        const auto size =
            fileSize / (sizeof(typename key_type::value_type) + sizeof(bool));

        for (std::size_t i = 0; i < size; ++i)
        {
            typename key_type::value_type tmp;
            ifs.read(reinterpret_cast<char*>(&tmp), sizeof(tmp));
            keys.emplace_back(key_type(tmp));
        }
        for (std::size_t i = 0; i < size; ++i)
        {
            bool leaf_flag;
            ifs.read(reinterpret_cast<char*>(&leaf_flag), sizeof(leaf_flag));
            leafs.emplace_back(leaf_flag);
        }
        //for(const auto& d: res) std::cout<<d<<std::endl;
        ifs.close();
    }

  private:
    /** \brief Coordinate transform from octant coordinate to real coordinates*/
    coordinate_transform_t octant_to_real_coordinate_ = &Tree::unit_transform;

    int                              base_level_ = 0; ///< Base level
    int                              depth_ = 0;      ///< Tree depth
    std::shared_ptr<octant_type>     root_ = nullptr; ///< Tree root
    std::vector<octant_ptr_map_type> level_maps_;     ///< Octants per level
    octant_ptr_map_type              leafs_;          ///< Map of tree leafs
};

} //namespace octree
} // namespace iblgf
#endif
