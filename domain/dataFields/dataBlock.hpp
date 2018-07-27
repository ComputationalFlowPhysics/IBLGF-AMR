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



namespace domain
{

template<int Dim,
         template<class >class NodeType, 
         template<std::size_t> class ...DataFieldType>
class DataBlock
{

public: //member types

    static constexpr int dim(){return Dim;}
    static constexpr int dimension=Dim;

    
    using node_t = NodeType<DataBlock>;

    using fields_tuple_t = std::tuple<DataFieldType<dimension>...>;
    using field_type_iterator_t = tuple_utils::TypeIterator<DataFieldType<dimension>...>;


    using node_itertor = typename std::vector<node_t>::iterator;
    using node_const_iterator = typename std::vector<node_t>::const_iterator;
    using node_reverse_iterator = typename std::vector<node_t>::reverse_iterator;
    using node_const_reverse_iterator = typename std::vector<node_t>::const_reverse_iterator;
    using size_type = types::size_type;

    using block_descriptor_type = BlockDescriptor<int,dimension>;
    using extent_t = typename block_descriptor_type::extent_t;
    using base_t = typename block_descriptor_type::base_t;

    template<typename T>
    using vector_type = types::vector_type<T, dimension>;

    using coordinate_type = base_t;
    using real_coordinate_type = vector_type<types::float_type>;


public: //Ctors:

    DataBlock()=default;
    ~DataBlock() =default;
    DataBlock(const DataBlock& rhs)=delete;
	DataBlock& operator=(const DataBlock&) & = default;

    DataBlock(DataBlock&& rhs)=default;
	DataBlock& operator=(DataBlock&&) & = default;

    DataBlock(base_t _base, extent_t _extent, int _level=0)
    :block_(_base, _extent, _level)
    {
        this->initialize(block_);
    }

    DataBlock(const block_descriptor_type& _b)
    :block_(_b)
    {
        this->initialize( _b );
    }


    void initialize(const block_descriptor_type& _b)
    {
        size_type size=1;
        for(int d=0;d<dimension;++d) size*= _b.extent()[d];
        tuple_utils::for_each(fields, [this,&size](auto& field)
        {
            field.resize(size);
        });
        this->generate_nodes();
    }



public: //member functions

    template<template<std::size_t> class Field>
    auto& get(){return std::get<Field<dimension>>(fields);}

    template<template<std::size_t> class Field>
    const auto& get()const{return std::get<Field<dimension>>(fields);}

    template<std::size_t  Idx>
    auto& get(){return std::get<Idx>(fields);}

    template<std::size_t  Idx>
    const auto& get()const{return std::get<Idx>(fields);}

    template<class T>
    auto& get(){return std::get<T>(fields);}
    template<class T>
    const auto& get()const{return std::get<T>(fields);}





    
    auto nodes_begin()const noexcept{return nodes_.begin();}
    auto nodes_end()const noexcept{return nodes_.end();}

    const auto& nodes()const{return nodes_;}
    auto& nodes(){return nodes_;}



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

    const block_descriptor_type& descriptor()const noexcept{return block_;}

    /*
	template<int nDim = 2>
    inline size_type get_index(coordinate_type _coord,
            typename std::enable_if< Dim==nDim  >::type* = 0) const noexcept
    {
        return _coord[0]+_coord[1]*block_.extent()[0];
    }
	template<int nDim = 3>
    */
    inline size_type get_index(coordinate_type _coord) const noexcept
    {
        return _coord[0]+_coord[1]*block_.extent()[0] +
               _coord[2]*block_.extent()[0]*block_.extent()[1];
    }

private: //private member helpers

    void generate_nodes()
    {
        nodes_.clear();
        const auto& f = std::get<0>(fields);
        for(std::size_t i=0; i<f.size();++i)
        {
            nodes_.emplace_back(this,i );
        }
    }

  

private: //Data members

    block_descriptor_type block_;
    fields_tuple_t fields;
    std::vector<node_t>  nodes_;
};


} //namespace cartesian_mesh

#endif

