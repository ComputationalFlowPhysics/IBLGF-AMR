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


namespace octree
{

template<int Dim, class DataType>
class Tree;

template<int Dim, class DataType>
class Octant : public Octant_base<Dim,DataType>
{
public:

    using super_type    = Octant_base<Dim, DataType>;
    using octant_base_t = super_type;
    using typename super_type::key_type ;
    using typename super_type::coordinate_type;
    using typename super_type::real_coordinate_type;
    using typename super_type::tree_type;

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


    // Returns end() if there is no neighbor
    // TODO: make this an optional or std::pair
    auto neighbor(const coordinate_type& _offset)
    {
        octant_base_t nn(octant_base_t::neighbor(_offset));
        return this->tree()->find_leaf_any_level(nn);
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


protected:

	Octant* refine(unsigned int i)
	{
        children_[i] = std::make_shared<Octant> (this->child_base(i));
		children_[i]->parent_ = this;
        return children_[i].get();
	}

protected:

    int idx_ = 0;
    std::shared_ptr<data_type> data_ = nullptr;

    Octant* parent_;

    //TODO:  Should be a unique pointer
    std::array<std::shared_ptr<Octant>,pow(2,Dim)> children_ =
        {{nullptr,nullptr,nullptr,nullptr, nullptr,nullptr,nullptr,nullptr}};
};



} //namespace octree
#endif 

