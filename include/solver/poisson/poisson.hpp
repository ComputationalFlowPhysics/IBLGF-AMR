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
#include<utilities/cell_center_nli_intrp.hpp>
#include<solver/poisson/poisson.hpp>

namespace solver
{

template<class Simulation>
class PoissonSolver
{

public: //member types
    using simulation_type      = Simulation;
    using domain_type          = typename Simulation::domain_type;
    using datablock_type       = typename domain_type::datablock_t;
    using tree_t               = typename domain_type::tree_t;
    using octant_t             = typename tree_t::octant_type;
    using block_type           = typename datablock_type::block_descriptor_type;
    using convolution_t        = typename fft::Convolution;
    using real_coordinate_type = typename domain_type::real_coordinate_type;
    using coordinate_type      = typename domain_type::coordinate_type;



    static constexpr int lBuffer=1; ///< Lower left buffer for interpolation
    static constexpr int rBuffer=1; ///< Lower left buffer for interpolation

    PoissonSolver( Simulation* _simulation)
    :
    sim_(_simulation),
    domain_(&_simulation->domain_),
    conv_(domain_->block_extent()+lBuffer+rBuffer,
          domain_->block_extent()+lBuffer+rBuffer),
    fmm_(domain_->block_extent()[0]+lBuffer+rBuffer),
    c_cntr_nli_(domain_->block_extent()[0]+lBuffer+rBuffer)
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
        template<std::size_t>class Target,
        template<std::size_t>class fmm_s,
        template<std::size_t>class fmm_t,
        template<std::size_t>class fmm_tmp
            >
    void apply_amr_lgf()
    {
        // allocate lgf
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
            //test for FMM
            fmm_.fmm_for_level<Source, Target, fmm_s, fmm_t, fmm_tmp>(domain_, l, false);
            fmm_.fmm_for_level<Source, Target, fmm_s, fmm_t, fmm_tmp>(domain_, l, true);
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
                //this->interpolate<Target>(*it_t);
                c_cntr_nli_.nli_intrp_node<Target>(it_t);

            }
        }

    }

    template<
        template<std::size_t>class target,
        template<std::size_t>class target_tmp,
        template<std::size_t>class diff_target
    >
    void apply_amr_laplace()
    {

        const float_type dx_base=domain_->dx_base();

        //Coarsification:
        pcout<<"Laplace - coarsification "<<std::endl;
        for (int ls = domain_->tree()->depth()-2;
                 ls >= domain_->tree()->base_level(); --ls)
        {
            for (auto it_s  = domain_->begin(ls);
                      it_s != domain_->end(ls); ++it_s)
            {
                this->coarsify<target>(*it_s);
            }
        }

        //Level-Interactions
        pcout<<"Laplace - level interactions "<<std::endl;
        for (int l  = domain_->tree()->base_level();
                 l < domain_->tree()->depth(); ++l)
        {
            for (auto it  = domain_->begin(l);
                      it != domain_->end(l); ++it)
            {
                const auto s_extent = it->data()->template get<target>().
                                                real_block().extent();
                // copy to tmp
                auto& target_data = it->data()->template get_linalg_data<target>();
                auto& target_data_tmp  = it->data()->template get_linalg_data<target_tmp>();

                //for ( int i =1; i<s_extent[0]-1; ++i){
                //    for ( int j = 1; j<s_extent[1]-1; ++j){
                //        for ( int k = 1; k<s_extent[2]-1; ++k){
                //                target_data_tmp(k,j,i)  = target_data(k,j,i);
                //        }
                //    }
                //}

                target_data_tmp  = target_data * 1.0;
            }

            for (auto it  = domain_->begin(l);
                      it != domain_->end(l); ++it)
            {
                auto refinement_level = it->refinement_level();
                auto dx_level =  dx_base/std::pow(2,refinement_level);


                auto& target_data = it->data()->template get_linalg_data<target>();//.get()->cube_noalias_view();
                auto& target_data_tmp  = it->data()->template get_linalg_data<target_tmp>();//.get()->cube_noalias_view();
                auto& diff_target_data = it->data()->template get_linalg_data<diff_target>();//.get()->cube_noalias_view();

                const auto s_extent = it->data()->template get<target>().
                                                real_block().extent();

                // laplace of it_t data with zero bcs
                if ((it->is_leaf()))
                {
                    //for ( int i =1; i<s_extent[0]-1; ++i){
                    //    for ( int j = 1; j<s_extent[1]-1; ++j){
                    //        for ( int k = 1; k<s_extent[2]-1; ++k){
                    //            // FIXME actually k j i order due to the
                    //            // differences in definition of mem layout
                    //            diff_target_data(k,j,i)  = - 6.0 * target_data_tmp(k,j,i);
                    //            diff_target_data(k,j,i) += target_data_tmp(k,j,i-1);
                    //            diff_target_data(k,j,i) += target_data_tmp(k,j,i+1);
                    //            diff_target_data(k,j,i) += target_data_tmp(k,j-1,i);
                    //            diff_target_data(k,j,i) += target_data_tmp(k,j+1,i);
                    //            diff_target_data(k,j,i) += target_data_tmp(k+1,j,i);
                    //            diff_target_data(k,j,i) += target_data_tmp(k-1,j,i);
                    //        }
                    //    }
                    //}

                    for ( int i =1; i<s_extent[0]-1; ++i){
                        for ( int j = 1; j<s_extent[1]-1; ++j){
                            for ( int k = 1; k<s_extent[2]-1; ++k){
                                // FIXME actually k j i order due to the
                                // differences in definition of mem layout
                                diff_target_data(k,j,i)  = - 6.0 * target_data_tmp(k,j,i);
                            }
                        }
                    }

                    for ( int i = 1; i<s_extent[0]; ++i){
                        for ( int j = 0; j<s_extent[1]; ++j){
                            for ( int k = 0; k<s_extent[2]; ++k){
                                diff_target_data(k,j,i) += target_data_tmp(k,j,i-1);
                            }
                        }
                    }

                    for ( int i = 0; i<s_extent[0]-1; ++i){
                        for ( int j = 0; j<s_extent[1]; ++j){
                            for ( int k = 0; k<s_extent[2]; ++k){
                                diff_target_data(k,j,i) += target_data_tmp(k,j,i+1);
                            }
                        }
                    }

                    for ( int i = 0; i<s_extent[0]; ++i){
                        for ( int j = 1; j<s_extent[1]; ++j){
                            for ( int k = 0; k<s_extent[2]; ++k){
                                diff_target_data(k,j,i) += target_data_tmp(k,j-1,i);
                            }
                        }
                    }

                    for ( int i = 0; i<s_extent[0]; ++i){
                        for ( int j = 0; j<s_extent[1]-1; ++j){
                            for ( int k = 0; k<s_extent[2]; ++k){
                                diff_target_data(k,j,i) += target_data_tmp(k,j+1,i);
                            }
                        }
                    }

                    for ( int i = 0; i<s_extent[0]; ++i){
                        for ( int j = 0; j<s_extent[1]; ++j){
                            for ( int k = 1; k<s_extent[2]; ++k){
                                diff_target_data(k,j,i) += target_data_tmp(k-1,j,i);
                            }
                        }
                    }

                    for ( int i =0; i<s_extent[0]; ++i){
                        for ( int j = 0; j<s_extent[1]; ++j){
                            for ( int k = 0; k<s_extent[2]-1; ++k){
                                diff_target_data(k,j,i) += target_data_tmp(k+1,j,i);
                            }
                        }
                    }

                }


                // laplace of contribution of neighbors
                //int ni;

                //ni = 4;
                //auto n_s = it->neighbor(ni);
                //if (n_s)
                //if (!( !(it->is_leaf()) && !(n_s->is_leaf()) ))
                //{
                //    auto& target_nghb_data = n_s->data()->template get_linalg_data<target_tmp>();
                //    for ( int i =0; i<s_extent[0]; ++i){
                //        for ( int j = 0; j<s_extent[1]; ++j){
                //            diff_target_data(0,j,i) += target_nghb_data(s_extent[2]-3, j, i);
                //            diff_target_data(0,j,i) -= target_nghb_data(s_extent[2]-2, j, i) * 6.0;
                //            diff_target_data(0,j,i) += target_nghb_data(s_extent[2]-1, j, i);

                //            diff_target_data(1,j,i) += target_nghb_data(s_extent[2]-2, j, i);
                //            diff_target_data(1,j,i) -= target_nghb_data(s_extent[2]-1, j, i) * 6.0;

                //            diff_target_data(2,j,i) += target_nghb_data(s_extent[2]-1, j, i);
                //        }
                //    }
                //}

                //ni = 10;
                //n_s = it->neighbor(ni);
                //if (n_s)
                //if (!( !(it->is_leaf()) && !(n_s->is_leaf()) ))
                //{
                //    auto& target_nghb_data = n_s->data()->template get_linalg_data<target_tmp>();
                //    for ( int i =0; i<s_extent[0]; ++i){
                //        for ( int k = 0; k<s_extent[2]; ++k){
                //            diff_target_data(k,0,i) += target_nghb_data(k, s_extent[1]-3, i);
                //            diff_target_data(k,0,i) -= target_nghb_data(k, s_extent[1]-2, i) * 6.0;
                //            diff_target_data(k,0,i) += target_nghb_data(k, s_extent[1]-1, i);

                //            diff_target_data(k,1,i) += target_nghb_data(k, s_extent[1]-2, i);
                //            diff_target_data(k,1,i) -= target_nghb_data(k, s_extent[1]-1, i) * 6.0;

                //            diff_target_data(k,2,i) += target_nghb_data(k, s_extent[1]-1, i);
                //        }
                //    }
                //}

                //ni = 12;
                //n_s = it->neighbor(ni);
                //if (n_s)
                //if (!( !(it->is_leaf()) && !(n_s->is_leaf()) ))
                //{
                //    auto& target_nghb_data = n_s->data()->template get_linalg_data<target_tmp>();
                //    for ( int j = 0; j<s_extent[1]; ++j){
                //        for ( int k = 0; k<s_extent[2]; ++k){
                //            diff_target_data(k,j,0) += target_nghb_data(k, j, s_extent[0]-3);
                //            diff_target_data(k,j,0) -= target_nghb_data(k, j, s_extent[0]-2)*6.0;
                //            diff_target_data(k,j,0) += target_nghb_data(k, j, s_extent[0]-1);

                //            diff_target_data(k,j,1) += target_nghb_data(k, j, s_extent[0]-2);
                //            diff_target_data(k,j,1) -= target_nghb_data(k, j, s_extent[0]-1)*6.0;

                //            diff_target_data(k,j,2) += target_nghb_data(k, j, s_extent[0]-1);
                //        }
                //    }
                //}

                //ni = 14;
                //n_s = it->neighbor(ni);
                //if (n_s)
                //if (!( !(it->is_leaf()) && !(n_s->is_leaf()) ))
                //{
                //    auto& target_nghb_data = n_s->data()->template get_linalg_data<target_tmp>();
                //    for ( int j = 0; j<s_extent[1]; ++j){
                //        for ( int k = 0; k<s_extent[2]; ++k){
                //            diff_target_data(k,j,s_extent[2]-1) += target_nghb_data(k, j, 2);
                //            diff_target_data(k,j,s_extent[2]-1) -= target_nghb_data(k, j, 1) * 6.0;
                //            diff_target_data(k,j,s_extent[2]-1) += target_nghb_data(k, j, 0);

                //            diff_target_data(k,j,s_extent[2]-2) += target_nghb_data(k, j, 1);
                //            diff_target_data(k,j,s_extent[2]-2) -= target_nghb_data(k, j, 0) * 6.0;

                //            diff_target_data(k,j,s_extent[2]-3) += target_nghb_data(k, j, 0);
                //        }
                //    }
                //}


                //ni = 16;
                //n_s = it->neighbor(ni);
                //if (n_s)
                //if (!( !(it->is_leaf()) && !(n_s->is_leaf()) ))
                //{
                //    auto& target_nghb_data = n_s->data()->template get_linalg_data<target_tmp>();
                //    for ( int i =0; i<s_extent[0]; ++i){
                //        for ( int k = 0; k<s_extent[2]; ++k){
                //            diff_target_data(k, s_extent[1]-1, i) += target_nghb_data(k, 2, i);
                //            diff_target_data(k, s_extent[1]-1, i) -= target_nghb_data(k, 1, i) * 6.0;
                //            diff_target_data(k, s_extent[1]-1, i) += target_nghb_data(k, 0, i);

                //            diff_target_data(k, s_extent[1]-2, i) += target_nghb_data(k, 1, i);
                //            diff_target_data(k, s_extent[1]-2, i) -= target_nghb_data(k, 0, i) * 6.0;

                //            diff_target_data(k, s_extent[1]-3, i) += target_nghb_data(k, 0, i);
                //        }
                //    }
                //}

                //ni = 22;
                //n_s = it->neighbor(ni);
                //if (n_s)
                //if (!( !(it->is_leaf()) && !(n_s->is_leaf()) ))
                //{
                //    auto& target_nghb_data = n_s->data()->template get_linalg_data<target_tmp>();

                //    for ( int i =0; i<s_extent[0]; ++i){
                //        for ( int j = 0; j<s_extent[1]; ++j){
                //            diff_target_data(s_extent[2]-1, j, i) += target_nghb_data(2, j, i);
                //            diff_target_data(s_extent[2]-1, j, i) -= target_nghb_data(1, j, i) * 6.0;
                //            diff_target_data(s_extent[2]-1, j, i) += target_nghb_data(0, j, i);

                //            diff_target_data(s_extent[2]-2, j, i) += target_nghb_data(1, j, i);
                //            diff_target_data(s_extent[2]-2, j, i) -= target_nghb_data(0, j, i) * 6.0;

                //            diff_target_data(s_extent[2]-3, j, i) += target_nghb_data(0, j, i);
                //        }
                //    }
                //}

                diff_target_data *= (1/dx_level) * (1/dx_level);
                //std::cout<< target_data_tmp << std::endl;
                //std::cout<< diff_target_data << std::endl;


            }
        }

        // Interpolation
        //std::cout<<"Laplace - interpolation"<<std::endl;
        //for (int lt = domain_->tree()->base_level();
        //         lt < domain_->tree()->depth(); ++lt)
        //{
        //    for (auto it_t  = domain_->begin(lt);
        //              it_t != domain_->end(lt); ++it_t)
        //    {
        //        if(it_t->is_leaf()) continue;
        //        this->interpolate<diff_target>(*it_t);
        //    }
        //}

    }

    template<
        template<std::size_t>class Source,
        template<std::size_t>class Target,
        template<std::size_t>class fmm_s,
        template<std::size_t>class fmm_t,
        template<std::size_t>class Target_fmm,
        template<std::size_t>class fmm_tmp,
        template<std::size_t>class amr_lap_target,
        template<std::size_t>class amr_lap_tmp
    >
    void solve()
    {
        apply_amr_lgf<Source, Target_fmm, fmm_s, fmm_t, fmm_tmp>();

        apply_amr_laplace<Target_fmm, amr_lap_tmp, amr_lap_target>();
    }

    /** @brief Coarsify the source field.
     *  @detail Given a parent, coarsify the field from its children and
     *  assign it to the parent. Coarsification is an average, ie 2nd order
     *  accurate.
     */
    template<template<std::size_t>class Field >
    void coarsify(octant_t* _parent)
    {
        auto parent = _parent;
        const coordinate_type stride(2);
        if(parent->is_leaf())return;

        for (int i = 0; i < parent->num_children(); ++i)
        {
            auto child = parent->child(i);
            auto child_view= child->data()->descriptor();
            if(child==nullptr) continue;

            auto cview =child->data()->node_field().view(child_view,stride);

            cview.iterate([&]( auto& n )
            {
                const float_type avg=1./8*(
                                n.template at_offset<Field>(0,0,0)+
                                n.template at_offset<Field>(1,0,0)+
                                n.template at_offset<Field>(0,1,0)+
                                n.template at_offset<Field>(1,1,0)+
                                n.template at_offset<Field>(0,0,1)+
                                n.template at_offset<Field>(1,0,1)+
                                n.template at_offset<Field>(0,1,1)+
                                n.template at_offset<Field>(1,1,1));

                const auto pcoord=n.level_coordinate()/2;
                parent->data()-> template get<Field>(pcoord) =avg;
            });
        }
    }


    /** @brief Interplate the target field.
     *  @detail Given a parent field, interpolate it onto the child meshes.
     *  Interpolation is 2nd order accurate.
     */
    template<template<std::size_t>class Field >
    void interpolate(const octant_t* _b_parent)
    {
        for (int i = 0; i < _b_parent->num_children(); ++i)
        {
            auto child = _b_parent->child(i);
            if (child==nullptr) continue;
            block_type child_view =
                child->data()->template get<Field>().real_block();
            auto cview =child->data()->node_field().view(child_view);

            cview.iterate([&]( auto& n )
            {
                const auto& coord=n.level_coordinate();
                auto min =(coord+1)/2-1;
                real_coordinate_type x=(coord-0.5)/2.0;

                const float_type interp=
                    interpolation::interpolate(
                            min.x(), min.y(), min.z(),
                            x[0], x[1], x[2],
                            _b_parent->data()->template get<Field>()) ;
                    n.template get<Field>()+=interp;
            });
        }
    }


private:
    Simulation*                         sim_;       ///< simualtion
    domain_type*                        domain_;    ///< domain
    convolution_t                       conv_;      ///< fft convolution
    fmm::Fmm                            fmm_;       ///< fast-multipole
    interpolation::cell_center_nli      c_cntr_nli_;
    lgf::LGF<lgf::Lookup>               lgf_;       ///< Lookup for the LGFs

    parallel_ostream::ParallelOstream pcout;

};

}

#endif // IBLGF_INCLUDED_POISSON_HPP
