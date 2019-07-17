#ifndef IBLGF_INCLUDED_OPERATORS_HPP
#define IBLGF_INCLUDED_OPERATORS_HPP

#include <dictionary/dictionary.hpp>
#include <domain/dataFields/datafield.hpp>
#include <types.hpp>


namespace domain
{

struct Operator
{

public:

    Operator(const Operator& other) = default;
    Operator(Operator&& other) = default;
    Operator& operator=(const Operator& other) & = default;
    Operator& operator=(Operator&& other) & = default;
    ~Operator() = default;
    Operator()=default;

public:

    template<class Source, class Dest, class Block>
    static void laplace(Block& block, float_type dx_level) noexcept
    {
        auto& nodes_domain=block.nodes_domain();

        const auto fac=1.0/(dx_level*dx_level);
        for(auto it2=nodes_domain.begin();it2!=nodes_domain.end();++it2 )
        {
            it2->template get<Dest>()= 
                      -6.0* it2->template get<Source>()+
                      it2->template at_offset<Source>(0,0,-1)+
                      it2->template at_offset<Source>(0,0,+1)+
                      it2->template at_offset<Source>(0,-1,0)+
                      it2->template at_offset<Source>(0,+1,0)+
                      it2->template at_offset<Source>(-1,0,0)+
                      it2->template at_offset<Source>(+1,0,0);
            it2->template get<Dest>()*=fac;
        }
    }


    template<class Source, class DestTuple, class Block, 
             typename std::enable_if<
               Source::mesh_type == MeshObject::cell,
            void>::type* = nullptr
            >
    static void gradient(Block& block, float_type dx_level) noexcept
    {
        auto& nodes_domain=block.nodes_domain();
        const auto fac = 1.0/dx_level; 
        for(auto it2=nodes_domain.begin();it2!=nodes_domain.end();++it2 )
        {

            it2->template get<typename std::tuple_element<0,DestTuple>::type>()= 
                      fac*(it2->template get<Source>()-it2->template at_offset<Source>(-1,0,0));
            it2->template get<typename std::tuple_element<1,DestTuple>::type>()= 
                      fac*(it2->template get<Source>()-it2->template at_offset<Source>(0,-1,0));
            it2->template get<typename std::tuple_element<2,DestTuple>::type>()= 
                      fac*(it2->template get<Source>()-it2->template at_offset<Source>(0,0,-1));
        }
    }
    
    template<class SourceTuple, class Dest, class Block,
             typename std::enable_if<
                (Dest::mesh_type   == MeshObject::cell) && 
                (std::tuple_element<0,SourceTuple>::type::mesh_type == MeshObject::face) , 
            void>::type* = nullptr
            >
    static void divergence(Block& block, float_type dx_level) noexcept
    {
        auto& nodes_domain=block.nodes_domain();
        const auto fac = 1.0/dx_level; 
        for(auto it2=nodes_domain.begin();it2!=nodes_domain.end();++it2)
        {

            it2->template get<Dest>()=
                -it2->template get<typename std::tuple_element<0,SourceTuple>::type>()
                -it2->template get<typename std::tuple_element<1,SourceTuple>::type>()
                -it2->template get<typename std::tuple_element<2,SourceTuple>::type>()
                +it2->template at_offset<typename std::tuple_element<0,SourceTuple>::type>(1,0,0)
                +it2->template at_offset<typename std::tuple_element<1,SourceTuple>::type>(0,1,0)
                +it2->template at_offset<typename std::tuple_element<2,SourceTuple>::type>(0,0,1);
            it2->template get<Dest>()*=fac;
        }
    }

    template<class Source, class Dest, class Block,
             typename std::enable_if<
                (Source::type == MeshObject::face) && 
                (Dest::type   == MeshObject::edge), 
            void>::type* = nullptr
            >
    static void curl(Block* block, float_type dx_level) noexcept
    {
        std::cout<<"Curl "<<std::endl;
    }

public:

};
}

#endif
