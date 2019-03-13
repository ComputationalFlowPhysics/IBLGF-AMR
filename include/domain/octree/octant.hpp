#ifndef OCTREE_INCLUDED_CELL_HPP
#define OCTREE_INCLUDED_CELL_HPP


#include <vector>
#include <memory>
#include <cmath>
#include <set>
#include <string>
#include <map>

// IBLGF-specific
#include <global.hpp>
#include <domain/octree/octant_base.hpp>
#include <domain/octree/tree_utils.hpp>
#include <domain/dataFields/blockDescriptor.hpp>
#include <domain/dataFields/datafield.hpp>
#include <utilities/crtp.hpp>


namespace octree
{

template<int Dim, class DataType>
class Tree;

template<class T=void>
class DefaultMixIn{};

template<int Dim, class DataType, template<class> class MixIn=DefaultMixIn>
class Octant : public Octant_base<Dim,DataType>, MixIn<Octant<Dim, DataType, MixIn>>
{
public:

    using super_type    = Octant_base<Dim, DataType>;
    using octant_base_t = super_type;
    using typename super_type::key_type ;
    using typename super_type::coordinate_type;
    using typename super_type::real_coordinate_type;
    using typename super_type::tree_type;
    using octant_iterator = typename tree_type::octant_iterator;

    using block_descriptor_type = typename domain::BlockDescriptor<int,Dim>;
    using octant_datafield_type = typename  domain::DataField<Octant*, Dim>;

    using data_type = DataType;


    static constexpr int num_vertices(){return pow(2,Dim);}
    static constexpr int num_faces(){return 2*Dim;}
    static constexpr int num_edges(){return 2*num_faces();}
    static constexpr int nNeighbors(){return pow(3,Dim);;}

public:
    friend tree_type;


public: //Ctors

	Octant           ()                       = delete;
	Octant           (const Octant&  other)   = default;
	Octant           (      Octant&& other)   = default;
	Octant& operator=(const Octant&  other) & = default;
	Octant& operator=(      Octant&& other) & = default;
	~Octant          ()                       = default;

    Octant(const octant_base_t& _n) : super_type(_n) {null_init();}

    Octant(const octant_base_t& _n, data_type& _data)
        : super_type(_n), data_(_data) { null_init();}

    Octant(key_type _k, tree_type* _tr)
        : super_type(_k), t_(_tr) { null_init();}

    Octant(const coordinate_type& _x, int _level, tree_type* _tr)
        : super_type(key_type(_x,_level)),t_(_tr) { null_init();}


     /** @brief Find leaf that shares a vertex with octant
      *         on same, plus or minus one level
      **/
    Octant* vertex_neighbor(const coordinate_type& _offset)
    {
        // current level
        // FIXME this could be implemented in a faster way
        auto nn=this->key_.neighbor(_offset);
        if(nn==this->key()) return nullptr;
        auto nn_ptr = this->tree()->find_leaf(nn);
        if (nn_ptr!=nullptr) { return nn_ptr; }

        // parent level
        const auto parent = this->parent();
        if(parent!=nullptr)
        {
            auto p_nn=parent->key().neighbor(_offset);
            if(p_nn==this->key()) return nullptr;
            auto p_ptr = this->tree()->find_leaf(p_nn);
            if(p_ptr) return p_ptr;
        }

        // child level
        const auto child = this->construct_child(0);
        auto c_nn=child.key().neighbor(_offset);
        if(c_nn==this->key()) return nullptr;
        auto c_ptr= this->tree()->find_leaf(c_nn);
        if(c_ptr) return c_ptr;

        return nullptr;
    }
    void null_init()
    {
        std::fill(neighbor_.begin(),neighbor_.end(),nullptr);
        std::fill(masks_.begin(),masks_.end(),false);
        std::fill(influence_.begin(),influence_.end(),nullptr);
    }

    std::array<key_type, nNeighbors()>
    get_neighbor_keys (coordinate_type _offset=coordinate_type(1))
    const noexcept
    {
        const auto key= this->key();
        std::array<key_type,nNeighbors()> res;
        int count=0;
        rcIterator<Dim>::apply(-1*_offset,
                                2*_offset+1,
                               [&](const coordinate_type& _p)
        {
             res[count++] = key.neighbor(_p);;
        });
        return res;
    }



    //TODO: store this bool while constructing
    bool is_leaf()const noexcept{return flag_leaf_;}

    void flag_leaf(const bool flag)noexcept {flag_leaf_ = flag;}

    bool is_leaf_search() const noexcept
    {
        for(int i = 0; i< this->num_children();++i)
        {
            if(children_[i]!=nullptr ) return false;
        }
        return true;
    }

    auto get_vertices() noexcept
    {
        std::vector<decltype(this->tree()->begin_leafs())> res;
        if (this->is_hanging() || this->is_boundary()) return res;

        rcIterator<Dim>::apply(coordinate_type(0),
                               coordinate_type(2),
                               [&](const coordinate_type& _p)
        {
            auto nnn = neighbor(_p);
            if (nnn != this->tree()->end_leafs())
            res.emplace_back(nnn);
        });
        return res;
    }

    template<typename octant_t>
    bool inside(octant_t o1, octant_t o2)
    {
        auto k_ = this->key();
        return ( (k_ >= o1->key()) && (k_<=o2->key()));
    }


    template<class Iterator>
    auto compute_index(const Iterator& _it)
    {
        return std::distance(this->tree()->begin_leafs(), _it);
    }
    void index(int _idx)       noexcept {idx_ = _idx;}
    int  index()         const noexcept {return idx_;}

    auto  data() const noexcept {return data_;}
    auto& data()       noexcept {return data_;}

    Octant* parent()      const noexcept{return parent_;}
    Octant* child (int i) const noexcept{return children_[i].get();}

    void neighbor_clear () noexcept{neighbor_.fill(nullptr);}
    Octant* neighbor (int i) const noexcept{return neighbor_[i];}
    Octant** neighbor_pptr (int i) noexcept{return &neighbor_[i];}
    void neighbor(int i, Octant* new_neighbor)
    {
       neighbor_[i] = new_neighbor;
    }

    const auto influence_number () const noexcept{return influence_.size();}
    void influence_number (int i) noexcept{/*influence_num = i;*/}


    void influence_clear () noexcept{influence_.fill(nullptr);}
    Octant* influence (int i) const noexcept{return influence_[i];}
    void influence(int i, Octant* new_influence)
    {
       influence_[i] = new_influence;
    }

    bool mask(int i) noexcept{return masks_[i];}

    bool* mask_ptr(int i) noexcept{return &(masks_[i]);}

    void mask(int i, bool value)
    {
       masks_[i] = value;
    }

    float_type load()const noexcept
    {
        float_type load=1.0;
        for(int c=0;c<static_cast<int>(influence_.size()) ;++c)
        {
            Octant* inf = this->influence(c);
            if(inf!=nullptr) load=load+1.0;
        }
        for(int c=0;c<static_cast<int>(neighbor_.size());++c)
        {
            Octant* inf = this->neighbor(c);
            if(inf!=nullptr) load=load+1.0;
        }
        if(!is_leaf()) load*=2.0;
        return load;
    }



public: //mpi info

    bool locally_owned() const noexcept { return comm_.rank()==this->rank(); }
    bool ghost() const noexcept { return !locally_owned()&&this->rank()>=0; }

    bool has_locally_owned_children(int mask_id) const noexcept
    {
        for(int c=0;c<this->num_children();++c)
        {
            const auto child = this->child(c);
            if(!child) continue;
            if(child->locally_owned() && child->data() && (masks_[mask_id]))
            {
                return true;
                break;
            }
        }
        return false;
    }

    bool has_locally_owned_children() const noexcept
    {
        for(int c=0;c<this->num_children();++c)
        {
            const auto child = this->child(c);
            if(!child) continue;
            if(child->locally_owned() && child->data())
            {
                return true;
                break;
            }
        }
        return false;
    }

    std::set<int> unique_child_ranks(int mask_id) const noexcept
    {
        std::set<int> unique_ranks;
        for(int c=0;c<this->num_children();++c)
        {
            auto child = this->child(c);
            if(!child) continue;
            if(!child->locally_owned() && (masks_[mask_id]))
            {
                unique_ranks.insert(child->rank());
            }
        }
        return unique_ranks;
    }

    std::set<int> unique_child_ranks() const noexcept
    {
        std::set<int> unique_ranks;
        for(int c=0;c<this->num_children();++c)
        {
            auto child = this->child(c);
            if(!child) continue;
            if(!child->locally_owned())
            {
                unique_ranks.insert(child->rank());
            }
        }
        return unique_ranks;
    }

public: //Access

    /** @brief Get tree pointer*/
    tree_type* tree()const noexcept{return t_;}

    /** @brief Refinement level: level relative to base level */
    auto refinement_level() const noexcept{
        return this->tree_level()-t_->base_level();}

    /** @brief Get octant coordinate based on physical/global domain */
    real_coordinate_type global_coordinate() const noexcept
    {
        real_coordinate_type tmp=this->tree_coordinate();
        tmp/=(std::pow(2,this->level()-t_->base_level()));
        return this->tree()->octant_to_level_coordinate(tmp);
    }

public: //Construct

    /** @brief Get child of type Octant_base */
    Octant construct_child( int _i ) const noexcept
    {
        return Octant(this->key_.child(_i), t_);
    }

    /** @brief Get parent of type Octant_base */
    Octant construct_parent() const noexcept
    {
        return Octant(this->key_.parent(),t_);
    }

    /** @brief Get parent of type Octant_base with same coordinate than
     *         current octant.
     * */
    Octant construct_equal_coordinate_parent() const noexcept
    {
        return Octant(this->key_.equal_coordinate_parent(),t_);
    }

    /** @brief Get neighbor of type Octant_base
     *  @param[in] _offset Offset from current octant in terms of
     *                     tree coordinates, i.e. octants.
     * */
    Octant construct_neighbor(const coordinate_type& _offset)
    {
        Octant nn(this->key_.neighbor(_offset),tree());
    }
    auto num_neighbors(){return neighbor_.size();}


public: //Neighbors

    enum MASK_LIST {
        Mask_FMM_Source,
        Mask_FMM_Target,
        Mask_Last = Mask_FMM_Target };

private:

	Octant* refine(unsigned int i)
	{
        children_[i] = std::make_shared<Octant> (this->construct_child(i));
		children_[i]->parent_ = this;
        return children_[i].get();
	}

private:

    int idx_ = 0;
    boost::mpi::communicator comm_;
    std::shared_ptr<data_type> data_ = nullptr;
    Octant* parent_=nullptr;
    std::array<std::shared_ptr<Octant>,pow(2,Dim)> children_ =
        {{nullptr,nullptr,nullptr,nullptr, nullptr,nullptr,nullptr,nullptr}};
    std::array<Octant*,pow(3,Dim) > neighbor_ = {nullptr};
    int influence_num = 0;
    std::array<Octant*, 189 > influence_= {nullptr};
    bool flag_leaf_=true;
    std::array<bool, Mask_Last + 1> masks_ = {false};
    tree_type* t_=nullptr;
};



} //namespace octree
#endif
