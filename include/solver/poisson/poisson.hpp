#ifndef IBLGF_INCLUDED_SOLVER_POISSON_HPP
#define IBLGF_INCLUDED_SOLVER_POISSON_HPP

#include <iostream>
#include <vector>
#include <tuple>

#include <algorithm>
#include <list>
#include <vector>
#include <iostream>
#include <random>
#include <cmath>
#include <functional>
#include <cstring>
#include <fftw3.h>

// IBLGF-specific
#include <global.hpp>
#include <simulation.hpp>
#include <domain/domain.hpp>
#include <domain/dataFields/dataBlock.hpp>
#include <domain/dataFields/datafield.hpp>
#include <domain/octree/tree.hpp>
#include <IO/parallel_ostream.hpp>
#include <lgf/lgf.hpp>
#include <fmm/fmm.hpp>

//#include<utilities/convolution.hpp>
#include "../../utilities/convolution.hpp"
#include<utilities/interpolation.hpp>
#include<solver/poisson/poisson.hpp>

namespace solver
{

template<class Simulation>
class PoissonSolver
{

public: //member types
    using simulation_type = Simulation;
    using domain_type = typename Simulation::domain_type;
    using datablock_type = typename domain_type::datablock_t;
    using tree_t = typename domain_type::tree_t;
    using octant_t = typename tree_t::octant_type;
    using block_type = typename datablock_type::block_descriptor_type;
    using convolution_t = fft::Convolution;



    static constexpr int lBuffer=1; ///< Lower left buffer for interpolation
    static constexpr int rBuffer=1; ///< Lower left buffer for interpolation

    PoissonSolver( Simulation* _simulation)
    :
    sim_(_simulation), 
    domain_(&_simulation->domain_),
    conv_(domain_->block_extent()+lBuffer+rBuffer,
         domain_->block_extent()+lBuffer+rBuffer),
    fmm_(domain_->block_extent()[0]+1)
    { 
    }


public:

    /** @brief Solve the poisson equation using lattice Green's functions and 
     *         a block-refined mesh.
     *  @detail Lattice Green's functions are used for solving the poisson 
     *  equation. FFT is used for the level convolution's. 
     *  Interpolation/coarsification is used to project the solutions to fine
     *  and coarse meshes, respectively.
     */
    template<
        template<std::size_t>class Source, 
        template<std::size_t>class Target
            >
    void solve()
    {
        //TODO:
        //make check dimension check on the convolution in debug mode

        // allocate lgf
        // TODO: store this
        std::vector<float_type> lgf;

        const float_type dx_base=domain_->dx_base();

        //Coarsification:
        pcout<<"coarsification "<<std::endl;
        for (int ls = domain_->tree()->depth()-2;
                 ls >= domain_->tree()->base_level(); --ls)
        {
            for (auto it_s  = domain_->begin(ls);
                      it_s != domain_->end(ls); ++it_s)
            {
                this->coarsify<Source>(*it_s);
            }
        }

        //Level-Interactions
        pcout<<"Level interactions "<<std::endl;
        for (int l  = domain_->tree()->base_level();
                 l < domain_->tree()->depth(); ++l)
        {
            for (auto it_t  = domain_->begin(l);
                      it_t != domain_->end(l); ++it_t)
            {

                auto refinement_level = it_t->refinement_level();
                auto dx_level =  dx_base/std::pow(2,refinement_level);
                for (auto it_s  = domain_->begin(l);
                          it_s != domain_->end(l); ++it_s)
                {

                    if( !(it_s->is_leaf()) && !(it_t->is_leaf()) )
                    {
                         continue;
                    }

                    const auto t_base = it_t->data()->template get<Target>().
                                            real_block().base();
                    const auto s_base = it_s->data()->template get<Source>().
                                            real_block().base();
                    
                    // Get extent of Source region
                    const auto s_extent = it_s->data()->template get<Source>().
                                            real_block().extent();
                    const auto shift    = t_base - s_base;
                    
                    // Calculate the dimensions of the LGF to be allocated
                    const auto base_lgf   = shift - (s_extent - 1);
                    const auto extent_lgf = 2 * (s_extent) - 1;
                    
                    // Calculate the LGF
                    lgf_.get_subblock(block_type (base_lgf, extent_lgf), lgf);

                    // Perform convolution
                    conv_.execute_field(lgf, it_s->data()->template get<Source>());
                    
                    // Extract the solution
                    block_type  extractor(s_base, s_extent);
                    conv_.add_solution(extractor,
                                      it_t->data()->template get<Target>(),
                                      dx_level*dx_level);
                }
            }
        }


        // Interpolation
        std::cout<<"Interpolation"<<std::endl;
        for (int lt = domain_->tree()->base_level(); 
                 lt < domain_->tree()->depth(); ++lt)
        {
            for (auto it_t  = domain_->begin(lt);
                      it_t != domain_->end(lt); ++it_t)
            {
                if(it_t->is_leaf()) continue;
                this->interpolate_cc<Target>(*it_t);
            }
        }
    }

    /** @brief Coarsify the source field. 
     *  @detail Given a parent, coarsify the field from its children and
     *  assign it to the parent. Coarsification is an average, ie 2nd order
     *  accurate.
     */
    template<template<std::size_t>class Field >
    void coarsify(octant_t* _parent)
    {
        //Note: only works for an even number of nodes 
        //      Base of all children is even, so now buffer is needed
        auto parent = _parent;
        if(parent->is_leaf())return;

        for (int i = 0; i < parent->num_children(); ++i)
        {
            auto child = parent->child(i);
            auto child_view= child->data()->descriptor();
            if(child==nullptr) continue;

            for (auto kc  = child_view.base()[2];
                    kc <= child_view.max()[2]; kc+=2)
            {
                for (auto jc  = child_view.base()[1];
                        jc  <= child_view.max()[1]; jc+=2)
                {
                    for (auto ic = child_view.base()[0];
                            ic <= child_view.max()[0]; ic+=2)
                    {
                        const float_type avg=1./8*(
                            child->data()->template get<Field>(ic,  jc,  kc)+
                            child->data()->template get<Field>(ic+1,jc,  kc)+
                            child->data()->template get<Field>(ic,  jc+1,kc)+
                            child->data()->template get<Field>(ic+1,jc+1,kc)+
                            child->data()->template get<Field>(ic,  jc,  kc+1)+
                            child->data()->template get<Field>(ic+1,jc,  kc+1)+
                            child->data()->template get<Field>(ic,  jc+1,kc+1)+
                            child->data()->template get<Field>(ic+1,jc+1,kc+1));
                        int ip=ic/2;
                        int jp=jc/2;
                        int kp=kc/2;

                        parent->data()->template get<Field>(ip,jp,kp) =avg;

                        //std::cout<<avg<<std::endl;
                    }
                }
            }
        }
    }

    /** @brief Interplate the target field. 
     *  @detail Given a parent field, interpolate it onto the child meshes.
     *  Interpolation is 2nd order accurate.
     */
    template<template<std::size_t>class Field >
    void interpolate_cc(const octant_t* _b_parent)
    {
        
        //interpolation 
        for (int i = 0; i < _b_parent->num_children(); ++i)
        {
            auto child = _b_parent->child(i);
            if(child==nullptr) continue;
            block_type child_view =  
                child->data()->template get<Field>().real_block();

            auto parent_view = 
                _b_parent->data()->template get<Field>().real_block();

            
            int count=0;
            for (auto kc  = child_view.base()[2];
                      kc <= child_view.max()[2]; ++kc)
            {
                for (auto jc  = child_view.base()[1];
                          jc  <= child_view.max()[1]; ++jc)
                {
                    for (auto ic = child_view.base()[0];
                              ic <= child_view.max()[0]; ++ic)
                    {
                        int min_x= (ic+1)/2-1;
                        int min_y= (jc+1)/2-1;
                        int min_z= (kc+1)/2-1;
                        const float_type x= (ic-0.5)/2.0; 
                        const float_type y= (jc-0.5)/2.0; 
                        const float_type z= (kc-0.5)/2.0;

                        const float_type interp= 
                            interpolation::interpolate(
                                    min_x, min_y, min_z, 
                                    x, y, z, 
                                    _b_parent->data()->template get<Field>(),
                                     count++);
                        child->data()->template get<Field>(ic,jc,kc) += interp;
                    }
                }
            }
        }
    }

private:
    Simulation*                 sim_;       ///< simualtion 
    domain_type*                domain_;    ///< domain
    convolution_t               conv_;       ///< fft convolution
    fmm::Fmm                    fmm_;       ///< fast-multipole 
    lgf::LGF<lgf::Lookup>       lgf_;       ///< Lookup for the LGFs

    parallel_ostream::ParallelOstream pcout; 
    
};

}

#endif // IBLGF_INCLUDED_POISSON_HPP
