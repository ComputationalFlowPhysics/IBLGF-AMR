#ifndef INCLUDED_LGF_DOMAIN_DATABLOCK_HPP
#define INCLUDED_LGF_DOMAIN_DATABLOCK_HPP

#include <iostream>
#include <vector>
#include <tuple>

// IBLGF-specific
#include <types.hpp>
#include <domain/dataFields/blockDescriptor.hpp>
#include <domain/dataFields/datafield_utils.hpp>
#include <domain/dataFields/node.hpp>
#include <domain/dataFields/datafield.hpp>



namespace domain
{

template<std::size_t Dim,
         template<class >class NodeType,
         class... DataFieldType>
class DataBlock : public  BlockDescriptor<int, Dim>
{

    static constexpr auto dimension=Dim;

public: //member types
    using node_t = NodeType<DataBlock>;

    using fields_tuple_t = std::tuple<DataFieldType...>;
    using field_type_iterator_t = tuple_utils::TypeIterator<DataFieldType...>;
    using node_field_type =  DataField<node_t,Dim>;
    using buffer_type = typename node_field_type::buffer_d_t;


    using node_itertor = typename std::vector<node_t>::iterator;
    using node_const_iterator = typename std::vector<node_t>::const_iterator;
    using node_reverse_iterator = typename std::vector<node_t>::reverse_iterator;
    using node_const_reverse_iterator = typename std::vector<node_t>::const_reverse_iterator;
    using size_type = types::size_type;

    using block_descriptor_type = BlockDescriptor<int,dimension>;
    using super_type = block_descriptor_type;
    using extent_t = typename block_descriptor_type::extent_t;
    using base_t = typename block_descriptor_type::base_t;

    template<typename T>
    using vector_type = types::vector_type<T, dimension>;

    using coordinate_type = base_t;
    using real_coordinate_type = vector_type<types::float_type>;


public: //Ctors:

    DataBlock()  = default;
    ~DataBlock() = default;
    DataBlock(const DataBlock& rhs)=delete;
	DataBlock& operator=(const DataBlock&) & = default;

    DataBlock(DataBlock&& rhs)=default;
	DataBlock& operator=(DataBlock&&) & = default;

    DataBlock(base_t _base, extent_t _extent, int _level=0, bool init=true)
    :super_type(_base, _extent, _level)
    {
        if(init)
            this->initialize(this->descriptor());
    }

    DataBlock(const block_descriptor_type& _b)
    :super_type(_b)
    {
        this->initialize( _b );
    }

    template<template<std::size_t> class Field>
    void initialize(const block_descriptor_type& _b)
    {
       this->get<Field>().initialize(_b );
    }

    void initialize(const block_descriptor_type& _b)
    {
        tuple_utils::for_each(fields, [this,&_b](auto& field)
        {
            field.initialize(_b);
        });
        this->generate_nodes();
    }



public: //member functions

    template<class T> auto& get(){return std::get<T>(fields);}
    template<class T> const auto& get()const{return std::get<T>(fields);}

    template<class T> auto& get_data(){return std::get<T>(fields).data();}
    template<class T> const auto& get_data()const{return std::get<T>(fields).data();}

    template<class T> auto& get_linalg(){return std::get<T>(fields).linalg();}
    template<class T> const auto& get_linalg()const{return std::get<T>(fields).linalg();}

    template<class T> auto& get_linalg_data(){return std::get<T>(fields).linalg_data();}
    template<class T> const auto& get_linalg_data()const{return std::get<T>(fields).linalg_data();}

    template<class Field>
    auto& get(int _i, int _j, int _k)
    {
        return std::get<Field>(fields).get(_i,_j,_k);
    }
    template< class Field>
    const auto& get(int _i, int _j, int _k)const
    {
        return std::get<Field>(fields).get(_i,_j,_k);
    }

    template<class Field>
    auto& get(const coordinate_type& _c)
    {
        return std::get<Field>(fields).get(_c);
    }
    template<class Field>
    const auto& get(const coordinate_type& _c) const
    {
        return std::get<Field>(fields).get(_c);
    }

    template<class Field>
    auto& get_local(int _i, int _j, int _k)
    {
        return std::get<Field>(fields).get_local(_i,_j,_k);
    }

    template<class Field>
    const auto& get_local(int _i, int _j, int _k)const
    {
        return std::get<Field>(fields).get_local(_i,_j,_k);
    }

    auto& node(int _i, int _j, int _k )noexcept
    {
        return node_field_.get(_i,_j,_k);
    }

    friend std::ostream& operator<<(std::ostream& os, const  DataBlock& c)
    {
        tuple_utils::for_each(c.fields, [&os](auto& field){
                os<<"container field: "<<field.name()<<std::endl;
                });
        return os;
    }

    template<class Function>
    void for_fields(Function&& F  )
    {
        tuple_utils::for_each(fields, F);
    }

    block_descriptor_type descriptor()const noexcept { return *this; }
    block_descriptor_type bounding_box()const noexcept { return bounding_box_; }

    size_type get_index(coordinate_type _coord) const noexcept
    {
        return this->get_flat_index(_coord);
    }

    auto& node_field()noexcept{return node_field_;}
    const auto& node_field()const noexcept{return node_field_;}

    auto nodes_domain_begin()const noexcept{return nodes_domain_.begin();}
    auto nodes_domain_end()const noexcept{return nodes_domain_.end();}
    const auto& nodes_domain()const{return nodes_domain_;}
    auto& nodes_domain(){return nodes_domain_;}

private: //private member helpers

    /** @brief Generate nodes from the field tuple, both domain and nodes incl
     * buffer
     **/
    void generate_nodes()
    {


        bounding_box_=*this;
        buffer_type lbuff(0), rbuff(0);
        for_fields( [&](auto& field){
            for(int d=0; d<Dim;++d)
            {
                if(field.lbuffer()[d]>lbuff[d])
                    lbuff[d]=field.lbuffer()[d];
                if(field.hbuffer()[d]>rbuff[d])
                    rbuff[d]=field.hbuffer()[d];
            }
           bounding_box_.enlarge_to_fit(field.real_block());
        });

        node_field_.lbuffer()=lbuff;
        node_field_.hbuffer()=rbuff;
        node_field_.initialize(*this);
        for( std::size_t i=0;i<node_field_.size();++i)
        {
            node_field_[i] = node_t(this,i);
        }

        //Store most common views in vector of nodes:
        nodes_domain_.clear();
        nodes_domain_.resize(this->nPoints());
        auto dview=node_field_.domain_view();
        int count=0;
        for(auto it=dview.begin();it!=dview.end();++it)
        {
            nodes_domain_[count++]=*it;
        }
    }


private: //Data members

    /** @brief Fields stored in datablock */
    fields_tuple_t fields;

    /** @brief nodes in domain */
    std::vector<node_t>  nodes_domain_;

    /** @brief field of nodes */
    node_field_type node_field_;

    /** @brief bounding box of all fields in the block*/
    super_type bounding_box_;

};


} //namespace cartesian_mesh

#endif

