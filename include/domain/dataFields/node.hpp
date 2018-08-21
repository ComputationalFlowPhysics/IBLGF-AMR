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

    node(Container* _c , std::size_t _index)
        :c_(_c), index_(_index)
    { }

    ~node() =default;
    node(const node& rhs)=delete;
	node& operator=(const node&) & = default;
    node(node&& rhs)=default;
	node& operator=(node&&) & = default;

    using coordinate_type = typename Container::coordinate_type;
    using real_coordinate_type = typename Container::real_coordinate_type;

    

public: //Access

    template<template<std::size_t>class Field>
    auto& get_field() noexcept{return c_->template get<Field>();}

    template<template<std::size_t>class Field>
    const auto& get_field()const noexcept{return c_->template get<Field>();}

    template<template<std::size_t>class Field>
    auto& get()noexcept{return c_->template get<Field>()[index_];}

    template<template<std::size_t>class Field>
    const auto& get()const noexcept{return c_->template get<Field>()[index_];}


    template<std::size_t Idx>
    auto& get()noexcept{return c_->template get<Idx>()[index_];}

    template<std::size_t Idx>
    const auto& get()const noexcept{return c_->template get<Idx>()[index_];}

    template<class T>
    auto& get()noexcept{return c_->template get<T>()[index_];}

    template<class T>
    const auto& get()const noexcept{return c_->template get<T>()[index_];}

    coordinate_type level_coordinate()const noexcept
    {
        coordinate_type res;
        const auto e=c_->descriptor().extent();
        res[2]= index_/(e[0]*e[1]);
        res[1]=(index_- res[2]*e[0]*e[1])/e[0];
        res[0]=(index_- res[2]*e[0]*e[1] -res[1]*e[0]);
        res+=c_->descriptor().base();
        return res;
    }
    real_coordinate_type global_coordinate()const noexcept
    {
        return (level_coordinate())/std::pow(2,c_->descriptor().level());
    }

    std::pair<node,bool> 
    neighbor(const coordinate_type& _offset) const noexcept
    {
        auto c = level_coordinate() +_offset ;
        if(c_->descriptor().is_inside(c))
        {
            return std::make_pair(node(c_,c_->get_index(c)),true);
        }
        else
            return std::make_pair(node(c_,0),false);
    }


    auto index(){return index_;}

    bool on_blockBorder()
    {
        return c_->descriptor().on_boundary(level_coordinate());
    }
    bool on_max_blockBorder()
    {
        return c_->descriptor().on_max_boundary(level_coordinate());
    }



public: //members

    Container* c_;
    std::size_t index_;

};

}

#endif
