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

template<int Dim,
         template<class >class NodeType, 
         template<std::size_t> class ...DataFieldType>
class DataBlock : public  BlockDescriptor<int, Dim>
{

public: //member types

    static constexpr int dim(){return Dim;}
    static constexpr int dimension=Dim;

    
    using node_t = NodeType<DataBlock>;

    using fields_tuple_t = std::tuple<DataFieldType<dimension>...>;
    using field_type_iterator_t = tuple_utils::TypeIterator<DataFieldType<dimension>...>;
    using node_field =  DataField<node_t,Dim>;


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

    template<template<std::size_t> class Field>
    auto& get(){return std::get<Field<dimension>>(fields);}
    template<template<std::size_t> class Field>
    const auto& get()const{return std::get<Field<dimension>>(fields);}

    template<class T> auto& get(){return std::get<T>(fields);}
    template<class T> const auto& get()const{return std::get<T>(fields);}

    template<template<std::size_t> class Field>
    auto& get_data(){return std::get<Field<dimension>>(fields).data();}
    template<template<std::size_t> class Field>
    const auto& get_data()const{return std::get<Field<dimension>>(fields).data();}

    template<class T> auto& get_data(){return std::get<T>(fields).data();}
    template<class T> const auto& get_data()const{return std::get<T>(fields).data();}

    template<template<std::size_t> class Field>
    auto& get(int _i, int _j, int _k)
    {
        return std::get<Field<dimension>>(fields).get(_i,_j,_k);
    }

    template<template<std::size_t> class Field>
    const auto& get(int _i, int _j, int _k)const
    {
        return std::get<Field<dimension>>(fields).get(_i,_j,_k);
    }

    template<template<std::size_t> class Field>
    auto& get_local(int _i, int _j, int _k)
    {
        return std::get<Field<dimension>>(fields).get_local(_i,_j,_k);
    }

    template<template<std::size_t> class Field>
    const auto& get_local(int _i, int _j, int _k)const
    {
        return std::get<Field<dimension>>(fields).get_local(_i,_j,_k);
    }


    auto nodes_begin()const noexcept{return nodes_.begin();}
    auto nodes_end()const noexcept{return nodes_.end();}

    const auto& nodes()const{return nodes_;}
    auto& nodes(){return nodes_;}

    auto& get_node(int _i, int _j, int _k )noexcept
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

    //FIXME:
    auto& get_node_field(){return node_field_;}

private: //private member helpers

    /** @brief Generate nodes from the field tuple, both domain and nodes incl
     * buffer
     **/
    void generate_nodes()
    {


        bounding_box_=*this;
        for_fields( [&](auto& field){
           bounding_box_.enlarge_to_fit(field.real_block());});

        nodes_.clear();
        auto size=this->nPoints();
        for(std::size_t i=0; i<size;++i)
        {
            nodes_.emplace_back(this,i );
        }

        //nodes_all_.clear();

        //make sure that there is a least a righ buffer of one:
        //FIXME: This should actually be 
        node_field_.initialize(bounding_box_);
        for( std::size_t i=0;i<node_field_.size();++i)
        {
            node_field_[i] = node_t(this,i);
        }
    }


private: //Data members

    /** @brief Fields stored in datablock */
    fields_tuple_t fields;

    /** @brief nodes in domain */
    std::vector<node_t>  nodes_;

    /** @brief field of nodes */
    node_field node_field_;

    /** @brief bounding box of all fields in the block*/
    super_type bounding_box_;                 



};


} //namespace cartesian_mesh

#endif

