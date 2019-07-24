#ifndef INCLUDED_LGF_DOMAIN_NODE_HPP
#define INCLUDED_LGF_DOMAIN_NODE_HPP

#include <vector>
#include <cmath>

namespace domain
{
template<class Container>
class node 
{
public:

    using coordinate_type = typename Container::coordinate_type;
    using real_coordinate_type = typename Container::real_coordinate_type;

public:

    node(Container* _c , std::size_t _index)
        :c_(_c), index_(_index), level_coordinate_(compute_level_coordinate())
    { }

    node(Container* _c , std::size_t _index, const coordinate_type& _level_coordinate)
        :c_(_c), index_(_index), level_coordinate_(_level_coordinate)
    { }

    ~node() =default;
    node() =default;
    node(const node& rhs)=default;
	node& operator=(const node&) & = default;
    node(node&& rhs)=default;
	node& operator=(node&&) & = default;

public: //Access

    template<template<std::size_t>class Field>
    auto& get_field(int _idx) noexcept{return c_->template get<Field>(_idx);}
    template<template<std::size_t>class Field>
    const auto& get_field(int _idx)const noexcept{ return c_->template get<Field>(_idx); }

    template<template<std::size_t>class Field>
    auto& get(int _idx)noexcept
    {
        return c_->template get<Field>(_idx).get(level_coordinate_,_idx);
    }
    template<template<std::size_t>class Field>
    const auto& get(int _idx)const noexcept
    {
        return c_->template get<Field>(_idx).get(level_coordinate_);
    }

    template<class T>
    auto& get(int _idx)noexcept
    {
        return c_->template get<T>(_idx).get(level_coordinate_);
    }
    template<class T>
    const auto& get(int _idx)const noexcept
    {
        return c_->template get<T>(_idx).get(level_coordinate_);
    } 
    
    template<class T>
    const auto& at_offset(const coordinate_type& _offset,int _idx)const noexcept
    {
        return c_->template get<T>(_idx).get(level_coordinate_+_offset);
    }
    template<template<std::size_t>class Field>
    auto& at_offset(int _i, int _j, int _k, int _idx)noexcept
    {
        return c_->template get<Field>(_idx).
            get(level_coordinate_+coordinate_type({_i,_j,_k}));
    }

    template<class T>
    auto& at_offset(int _i, int _j, int _k, int _idx)noexcept
    {
        return c_->template get<T>(_idx).
            get(level_coordinate_+coordinate_type({_i,_j,_k}));
    }

    
    //scalar field overloads:

    template<template<std::size_t>class Field>
    auto& get_field() noexcept{return c_->template get<Field>();}
    template<template<std::size_t>class Field>
    const auto& get_field()const noexcept{ return c_->template get<Field>(); }

    template<template<std::size_t>class Field>
    auto& get()noexcept
    {
        return c_->template get<Field>().get(level_coordinate_);
    }
    template<template<std::size_t>class Field>
    const auto& get()const noexcept
    {
        return c_->template get<Field>().get(level_coordinate_);
    }

    template<class T>
    auto& get()noexcept
    {
        return c_->template get<T>().get(level_coordinate_);
    }
    template<class T>
    const auto& get()const noexcept
    {
        return c_->template get<T>().get(level_coordinate_);
    } 
    
    template<class T>
    const auto& at_offset(const coordinate_type& _offset)const noexcept
    {
        return c_->template get<T>().get(level_coordinate_+_offset);
    }
    template<template<std::size_t>class Field>
    auto& at_offset(int _i, int _j, int _k)noexcept
    {
        return c_->template get<Field>().
            get(level_coordinate_+coordinate_type({_i,_j,_k}));
    }

    template<class T>
    auto& at_offset(int _i, int _j, int _k)noexcept
    {
        return c_->template get<T>().
            get(level_coordinate_+coordinate_type({_i,_j,_k}));
    }

public:

    coordinate_type level_coordinate()const noexcept{return level_coordinate_;}

    real_coordinate_type global_coordinate()const noexcept
    {
        return (level_coordinate_)/std::pow(2,c_->level());
    }

    std::pair<node,bool> 
    neighbor_check(const coordinate_type& _offset) const noexcept
    {
        auto c = level_coordinate_ +_offset ;
        if(c_->bounding_box().is_inside(c))
        {
            return std::make_pair(node(c_,c_->bounding_box().index(c)),true);
        }
        else
            return std::make_pair(node(c_,0),false);
    }
    inline node neighbor(const coordinate_type& _offset) const noexcept
    {
        return c_->node_field_[index_+c_->bounding_box().index_zeroBase(_offset)];
        //return node(c_,index_+c_->bounding_box().index(_offset), level_coordinate_+_offset);
        //return node(c_,c_->bounding_box().index(level_coordinate_ +_offset));
    }



    auto index(){return index_;}

    bool on_blockBorder()
    {
        return c_->on_boundary(level_coordinate_);
    }
    bool on_max_blockBorder()
    {
        return c_->on_max_boundary(level_coordinate_);
    }


private:
    coordinate_type compute_level_coordinate()const noexcept
    {
        coordinate_type res;
        const auto e=c_->bounding_box().extent();
        res[2]= index_/(e[0]*e[1]);
        res[1]=(index_- res[2]*e[0]*e[1])/e[0];
        res[0]=(index_- res[2]*e[0]*e[1] -res[1]*e[0]);
        res+=c_->bounding_box().base();
        return res;
    }

public: //members

    Container* c_;
    std::size_t index_; //index based on real_block_;
    coordinate_type level_coordinate_;
};

}

#endif
