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

public: 
    friend tree_type;


public: //Ctors

	Octant           ()                       = delete;
	Octant           (const Octant&  other)   = default;
	Octant           (      Octant&& other)   = default;
	Octant& operator=(const Octant&  other) & = default;
	Octant& operator=(      Octant&& other) & = default;
	~Octant          ()                       = default;

    Octant(const octant_base_t& _n) : super_type(_n) {}

    Octant(const octant_base_t& _n, data_type& _data)
        : super_type(_n), data_(_data) { }

    Octant(const coordinate_type& _x, int _level, tree_type* _tr)
        : super_type(key_type(_x,_level),_tr) { }


     /** @brief Find cell that shares a vertex with octant 
      *         on same, plus or minus one level 
      **/
    Octant* vertex_neighbor(const coordinate_type& _offset)
    {
        // current level 
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
        const auto child = this->child_base(0);
        auto c_nn=child.key().neighbor(_offset);
        if(c_nn==this->key()) return nullptr;
        auto c_ptr= this->tree()->find_leaf(c_nn);
        if(c_ptr) return c_ptr;

        return nullptr;
    }


    /** @brief Getting neighboorhood region of octant 
     *
     *  @param[in] _lowBuffer How many octants in negative direction
     *  @param[in] _highBufer How many octants in positive direction
     *  @return Vector of neighborhood octants
     */
    std::vector<Octant*> get_neighborhood(const coordinate_type& _lowBuffer,
            const coordinate_type& _highBuffer ) const noexcept
    {

       std::vector<Octant*> res;
       block_descriptor_type  b;
       b.base(this->tree_coordinate() - _lowBuffer);
       b.max( this->tree_coordinate() + _highBuffer);
       int level=this->level();
       b.level() = level;

       for(auto it  = this->tree()->begin(level);
                it != this->tree()->end(level); ++it)
       {
           if(b.is_inside(it->tree_coordinate()) && it->key()!=this->key())
           {
               res.push_back(*it);
           }
       }
       return res;
    }
    
    /** @brief Getting neighboorhood region of octant 
     *
     *  @param[in] _lowBuffer How many octants in negative direction
     *  @param[in] _highBufer How many octants in positive direction
     *  @return Datafield of octants for convient access
     */
    //octant_datafield_type get_neighborhood(const coordinate_type& _lowBuffer,
    //        const coordinate_type& _highBuffer ) const noexcept
    //{

    //   std::vector<Octant*> res;
    //   block_descriptor_type  b;
    //   b.base(this->tree_coordinate() - _lowBuffer);
    //   b.max( this->tree_coordinate() + _highBuffer);
    //   int level=this->level();
    //   b.level() = level;

    //   for(auto it  = this->tree()->begin(level);
    //            it != this->tree()->end(level); ++it)
    //   {
    //       if(b.is_inside(it->tree_coordinate()))
    //       {
    //           res.push_back(*it);
    //       }
    //   }
    //   return res;
    //}

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


private:

	Octant* refine(unsigned int i)
	{
        children_[i] = std::make_shared<Octant> (this->child_base(i));
		children_[i]->parent_ = this;
        return children_[i].get();
	}

private:

    int idx_ = 0;
    std::shared_ptr<data_type> data_ = nullptr;

    Octant* parent_;

    std::array<std::shared_ptr<Octant>,pow(2,Dim)> children_ =
        {{nullptr,nullptr,nullptr,nullptr, nullptr,nullptr,nullptr,nullptr}};
};



} //namespace octree
#endif 

