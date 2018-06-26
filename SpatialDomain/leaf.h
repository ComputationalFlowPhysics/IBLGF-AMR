#ifndef INCLUDED_LEAF_HPP
#define INCLUDED_LEAF_HPP


#include<vector>
#include<memory>
#include<cmath>
#include<set>
#include<string>
#include<map>

#include "key.h"
#include "utils.h"
#include "types.h"

namespace octree
{

template<int Dim, class DataType>
class LinearTree;

template<int Dim, class DataType>
class Leaf : public Key<Dim>
{
    
public:
    using key_t = Key<Dim>;
    using typename key_t::coordinate_t;
    using real_coordinate_t =  types::coordinate_type<types::value_type,Dim>;
    using typename key_t::id_t;
    using data_t= DataType;
    
public:
    static constexpr int num_children=pow(2,Dim);
    static LinearTree<Dim,DataType>* tree_ptr;

public:

	Leaf() = default;
	Leaf(const Leaf& other) = default;
	Leaf(Leaf&& other) = default;
	Leaf& operator=(const Leaf& other) & = default;
	Leaf& operator=(Leaf&& other) & = default;
	~Leaf() = default;

	Leaf(id_t _id) : key_t(_id) {}
	Leaf(key_t _id) : key_t(_id) {}
	Leaf(const coordinate_t& _x, int _level)
    : key_t(_x,_level) { }

	Leaf(const coordinate_t& _x, int _level, data_t _data)
    : key_t(_x,_level), data_(_data) { }

    const data_t& data()const noexcept{return data_;}
    void data(data_t _d)const noexcept{data_=_d;}

    auto  real_coordinate() const noexcept
    {
        real_coordinate_t tmp=this->coordinate();
        tmp/=(std::pow(2,this->level()-tree_ptr->base_level()));
        return tmp;
    }

private:
    mutable data_t data_;
};


template<int Dim, class DataType>
LinearTree<Dim, DataType>* Leaf<Dim, DataType>::tree_ptr=nullptr;

} //namespace octree
#endif 

