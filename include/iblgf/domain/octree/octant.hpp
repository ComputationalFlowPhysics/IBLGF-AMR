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

#ifndef OCTREE_INCLUDED_CELL_HPP
#define OCTREE_INCLUDED_CELL_HPP

#include <vector>
#include <memory>
#include <cmath>
#include <set>
#include <string>
#include <map>

// IBLGF-specific
#include <iblgf/global.hpp>
#include <iblgf/domain/octree/octant_base.hpp>
#include <iblgf/domain/octree/tree_utils.hpp>
#include <iblgf/domain/dataFields/blockDescriptor.hpp>
#include <iblgf/domain/dataFields/datafield.hpp>
#include <iblgf/utilities/crtp.hpp>

namespace iblgf
{
namespace octree
{
template<int Dim, class DataType>
class Tree;

template<class T = void>
class DefaultMixIn
{
};

template<int Dim, class DataType, template<class> class MixIn = DefaultMixIn>
class Octant
: public Octant_base<Dim, DataType>
, MixIn<Octant<Dim, DataType, MixIn>>
{
  public:
    using super_type = Octant_base<Dim, DataType>;
    using octant_base_t = super_type;
    using typename super_type::key_type;
    using typename super_type::coordinate_type;
    using typename super_type::real_coordinate_type;
    using typename super_type::tree_type;
    using octant_iterator = typename tree_type::octant_iterator;

    using block_descriptor_type = typename domain::BlockDescriptor<int, Dim>;
    using octant_datafield_type = typename domain::DataField<Octant*, Dim>;

    using data_type = DataType;

    static constexpr int num_vertices() { return pow(2, Dim); }
    static constexpr int num_faces() { return 2 * Dim; }
    static constexpr int num_edges() { return 2 * num_faces(); }
    static constexpr int nNeighbors()
    {
        return pow(3, Dim);
        ;
    }

  public:
    enum FLAG_LIST
    {
        FlagLeaf,
        FlagCorrection,
        FlagLeafBoundary,
        FlagLast
    };

    enum MASK_LIST
    {
        Mask_FMM_Source,
        Mask_FMM_Target,
        Mask_Last
    };

    static const int fmm_max_idx_ = 30;

    using fmm_mask_type = std::array<std::array<bool, Mask_Last>, fmm_max_idx_>;

    using flag_list_type = std::array<bool, FlagLast>;

    static flag_list_type flag_list_default()
    {
        flag_list_type f;
        f.fill(false);
        return f;
    }

  public:
    friend tree_type;

  public: //Ctors
    Octant() = delete;
    Octant(const Octant& other) = default;
    Octant(Octant&& other) = default;
    Octant& operator=(const Octant& other) & = default;
    Octant& operator=(Octant&& other) & = default;
    ~Octant() = default;

    Octant(const octant_base_t& _n)
    : super_type(_n)
    {
        null_init();
    }

    Octant(const octant_base_t& _n, data_type& _data)
    : super_type(_n)
    , data_(_data)
    {
        null_init();
    }

    Octant(key_type _k, tree_type* _tr)
    : super_type(_k)
    , t_(_tr)
    {
        null_init();
    }

    Octant(const coordinate_type& _x, int _level, tree_type* _tr)
    : super_type(key_type(_x, _level))
    , t_(_tr)
    {
        null_init();
    }

    /** @brief Find leaf that shares a vertex with octant
      *         on same, plus or minus one level
      **/
    Octant* vertex_neighbor(const coordinate_type& _offset)
    {
        // current level
        // FIXME this could be implemented in a faster way
        auto nn = this->key_.neighbor(_offset);
        if (nn == this->key()) return nullptr;
        auto nn_ptr = this->tree()->find_leaf(nn);
        if (nn_ptr != nullptr) { return nn_ptr; }

        // parent level
        const auto parent = this->parent();
        if (parent != nullptr)
        {
            auto p_nn = parent->key().neighbor(_offset);
            if (p_nn == this->key()) return nullptr;
            auto p_ptr = this->tree()->find_leaf(p_nn);
            if (p_ptr) return p_ptr;
        }

        // child level
        const auto child = this->construct_child(0);
        auto       c_nn = child.key().neighbor(_offset);
        if (c_nn == this->key()) return nullptr;
        auto c_ptr = this->tree()->find_leaf(c_nn);
        if (c_ptr) return c_ptr;

        return nullptr;
    }
    void null_init()
    {
        std::fill(neighbor_.begin(), neighbor_.end(), nullptr);
        std::fill(influence_.begin(), influence_.end(), nullptr);

        for (int i = 0; i < fmm_max_idx_; ++i)
            std::fill(fmm_masks_[i].begin(), fmm_masks_[i].end(), false);

        flags_.fill(false);
    }

    auto get_neighbor_keys()
    {
        auto key = this->key();
        return key.get_neighbor_keys();
    }

    auto get_infl_keys()
    {
        auto key = this->key();
        return key.get_infl_keys();
    }

    const flag_list_type& flags() const noexcept { return flags_; }
    flag_list_type&       flags() noexcept { return flags_; }

    const bool& leaf_boundary() const noexcept
    {
        return flags_[FlagLeafBoundary];
    }
    bool& leaf_boundary() noexcept { return flags_[FlagLeafBoundary]; }

    bool is_leaf() const noexcept { return flags_[FlagLeaf]; }
    void flag_leaf(bool flag) noexcept { flags_[FlagLeaf] = flag; }

    bool is_correction() const noexcept { return flags_[FlagCorrection]; }
    void flag_correction(const bool flag) noexcept
    {
        flags_[FlagCorrection] = flag;
    }

    bool physical() const noexcept { return flag_physical_; }
    void physical(bool flag) noexcept { flag_physical_ = flag; }

    bool aim_deletion() const noexcept { return aim_deletion_; }
    void aim_deletion(bool d) noexcept { aim_deletion_ = d; }

    void flag_mask(const fmm_mask_type fmm_flag) noexcept
    {
        fmm_masks_ = fmm_flag;
    }

    bool is_leaf_search(bool require_data = false) const noexcept
    {
        for (int i = 0; i < this->num_children(); ++i)
        {
            if ((require_data && children_[i] && children_[i]->data()) ||
                (!require_data && children_[i]))
                return false;
        }
        return true;
    }

    auto get_vertices() noexcept
    {
        std::vector<decltype(this->tree()->begin_leafs())> res;
        if (this->is_hanging() || this->is_boundary()) return res;

        rcIterator<Dim>::apply(coordinate_type(0), coordinate_type(2),
            [&](const coordinate_type& _p) {
                auto nnn = neighbor(_p);
                if (nnn != this->tree()->end_leafs()) res.emplace_back(nnn);
            });
        return res;
    }

    template<typename octant_t>
    bool inside(octant_t o1, octant_t o2)
    {
        auto k_ = this->key();
        return ((k_ >= o1->key()) && (k_ <= o2->key()));
    }

    template<class Iterator>
    auto compute_index(const Iterator& _it)
    {
        return std::distance(this->tree()->begin_leafs(), _it);
    }
    void index(int _idx) noexcept { idx_ = _idx; }
    int  index() const noexcept { return idx_; }

    auto  data() const noexcept { return data_; }
    auto& data() noexcept { return data_; }

    int  aim_level_change() const noexcept { return aim_level_change_; }
    int& aim_level_change() noexcept { return aim_level_change_; }

    auto deallocate_data() { data_.reset(); }

    Octant* parent() const noexcept { return parent_; }
    Octant* child(int i) const noexcept { return children_[i].get(); }
    void    delete_child(int i) noexcept { children_[i].reset(); }

    void       neighbor_clear() noexcept { neighbor_.fill(nullptr); }
    const auto neighbor_number() const noexcept { return neighbor_.size(); }

    int idx() const noexcept
    {
        const auto cc = this->tree_coordinate();
        return static_cast<int>(
            (this->level() + cc.x() * 25 + cc.y() * 25 * 300 +
                25 * 300 * 300 * cc.z()) %
            boost::mpi::environment::max_tag());
    }

    Octant*  neighbor(int i) const noexcept { return neighbor_[i]; }
    Octant** neighbor_pptr(int i) noexcept { return &neighbor_[i]; }
    void neighbor(int i, Octant* new_neighbor) { neighbor_[i] = new_neighbor; }

    const auto influence_number() const noexcept
    {
        return influence_num; /*influence_.size();*/
    }
    void influence_number(int i) noexcept { influence_num = i; }

    void    influence_clear() noexcept { influence_.fill(nullptr); }
    Octant* influence(int i) const noexcept { return influence_[i]; }
    void    influence(int i, Octant* new_influence)
    {
        influence_[i] = new_influence;
    }

    //bool mask(int i) noexcept{return masks_[i];}
    //bool* mask_ptr(int i) noexcept{return &(masks_[i]);}
    //void mask(int i, bool value)
    //{
    //   masks_[i] = value;
    //}

    fmm_mask_type fmm_mask() noexcept { return fmm_masks_; }

    bool fmm_mask(int fmm_base_level, int i) noexcept
    {
        return fmm_masks_[fmm_base_level][i];
    }

    bool* fmm_mask_ptr(int fmm_base_level, int i) noexcept
    {
        if (fmm_base_level < 0) std::cout << "base level < 0" << std::endl;
        return &(fmm_masks_[fmm_base_level][i]);
    }
    void fmm_mask(int fmm_base_level, int i, bool value)
    {
        fmm_masks_[fmm_base_level][i] = value;
    }

    void        add_load(float_type l) { load_ += l; }
    const auto& load() const noexcept { return load_; }
    auto&       load() noexcept { return load_; }

  public: //mpi info
    bool locally_owned() const noexcept { return comm_.rank() == this->rank(); }
    bool ghost() const noexcept
    {
        return !locally_owned() && this->rank() >= 0;
    }

    bool has_locally_owned_children(int fmm_level, int mask_id) const noexcept
    {
        return has_locally_owned_children(fmm_masks_[fmm_level][mask_id]);
    }

    bool has_locally_owned_children(bool fmm_masks = true) const noexcept
    {
        for (int c = 0; c < this->num_children(); ++c)
        {
            const auto child = this->child(c);
            if (!child || !child->data()) continue;
            if (child->locally_owned() && child->data() &&
                child->data()->is_allocated() && fmm_masks)
            {
                return true;
                break;
            }
        }
        return false;
    }

    std::set<int> unique_child_ranks(int base_level, int mask_id) const noexcept
    {
        return unique_child_ranks(fmm_masks_[base_level][mask_id]);
    }

    std::set<int> unique_child_ranks(bool fmm_masks = true) const noexcept
    {
        std::set<int> unique_ranks;
        for (int c = 0; c < this->num_children(); ++c)
        {
            auto child = this->child(c);
            if (!child || !child->data()) continue;
            if (!child->locally_owned() && fmm_masks)
            { unique_ranks.insert(child->rank()); }
        }
        return unique_ranks;
    }

    std::set<int> unique_infl_ranks()
    {
        std::set<int> unique_inflRanks;

        for (int i = 0; i < this->influence_number(); ++i)
        {
            auto inf = this->influence(i);
            if (inf && inf->data()) { unique_inflRanks.emplace(inf->rank()); }
        }
        std::cout << std::endl;
        return unique_inflRanks;
    }

    std::set<int> unique_neighbor_ranks()
    {
        std::set<int> unique_neighborRanks;
        for (int i = 0; i < this->nNeighbors(); ++i)
        {
            const auto n = this->neighbor(i);
            if (n && n->data())
            {
                std::cout << n->rank() << " " << n->key() << std::endl;
                unique_neighborRanks.emplace(n->rank());
            }
        }
        return unique_neighborRanks;
    }

    std::set<int> unique_infl_neighbor_ranks()
    {
        auto l1 = unique_infl_ranks();
        auto l2 = unique_neighbor_ranks();
        //std::set<int> l12;
        //std::merge(l1.begin(), l1.end(),
        //        l2.begin(), l2.end(),
        //        std::inserter(l12, l12.begin()));

        //l1.merge(unique_neighbor_ranks());

        for (auto r : l1) std::cout << r << " ";
        std::cout << std::endl;

        for (auto r : l2) std::cout << r << " ";
        std::cout << std::endl;

        return l1;
    }

  public: //Access
    /** @brief Get tree pointer*/
    tree_type* tree() const noexcept { return t_; }

    /** @brief Refinement level: level relative to base level */
    auto refinement_level() const noexcept
    {
        return this->tree_level() - t_->base_level();
    }

    /** @brief Get octant coordinate based on physical/global domain */
    real_coordinate_type global_coordinate() const noexcept
    {
        real_coordinate_type tmp = this->tree_coordinate();
        tmp /= (std::pow(2, this->level() - t_->base_level()));
        return this->tree()->octant_to_level_coordinate(tmp);
    }

  public: //Construct
    /** @brief Get child of type Octant_base */
    Octant construct_child(int _i) const noexcept
    {
        return Octant(this->key_.child(_i), t_);
    }

    /** @brief Get parent of type Octant_base */
    Octant construct_parent() const noexcept
    {
        return Octant(this->key_.parent(), t_);
    }

    /** @brief Get parent of type Octant_base with same coordinate than
     *         current octant.
     * */
    Octant construct_equal_coordinate_parent() const noexcept
    {
        return Octant(this->key_.equal_coordinate_parent(), t_);
    }

    /** @brief Get neighbor of type Octant_base
     *  @param[in] _offset Offset from current octant in terms of
     *                     tree coordinates, i.e. octants.
     * */
    Octant construct_neighbor(const coordinate_type& _offset)
    {
        Octant nn(this->key_.neighbor(_offset), tree());
    }
    auto num_neighbors() { return neighbor_.size(); }

  public:
    Octant* refine(unsigned int i)
    {
        if (children_[i]) return children_[i].get();
        children_[i] = std::make_shared<Octant>(this->construct_child(i));
        children_[i]->parent_ = this;
        return children_[i].get();
    }

  private:
    int                                              idx_ = 0;
    float_type                                       load_ = 0;
    boost::mpi::communicator                         comm_;
    std::shared_ptr<data_type>                       data_ = nullptr;
    Octant*                                          parent_ = nullptr;
    std::array<std::shared_ptr<Octant>, pow(2, Dim)> children_ = {{nullptr,
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr}};
    std::array<Octant*, pow(3, Dim)>                 neighbor_ = {nullptr};
    int                                              influence_num = 0;
    std::array<Octant*, 189>                         influence_ = {nullptr};

    bool flag_physical_ = false;

    //bool flag_leaf_=false;
    //bool flag_correction_=false;
    //std::array<bool, Mask_Last + 1> masks_ = {false};

    flag_list_type flags_;
    fmm_mask_type  fmm_masks_;

    tree_type* t_ = nullptr;
    bool       aim_deletion_ = false;
    int        aim_level_change_ = 0;
};

} //namespace octree
} // namespace iblgf
#endif
