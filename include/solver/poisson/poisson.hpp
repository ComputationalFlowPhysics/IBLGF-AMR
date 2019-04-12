#ifndef IBLGF_INCLUDED_SOLVER_POISSON_HPP
#define IBLGF_INCLUDED_SOLVER_POISSON_HPP

#include <iostream>
#include <algorithm>
#include <vector>
#include <cmath>

// IBLGF-specific
#include <global.hpp>
#include <simulation.hpp>
#include <domain/domain.hpp>
#include <IO/parallel_ostream.hpp>

#include "../../utilities/convolution.hpp"
#include<utilities/cell_center_nli_intrp.hpp>

namespace solver
{


using namespace domain;

/** @brief Poisson solver using lattice Green's functions. Convolutions
 *         are computed using FMM and near field interaction are computed
 *         using blockwise fft-convolutions.
 */
template<class Setup>
class PoissonSolver
{

public: //member types

    using simulation_type      = typename Setup::simulation_t;
    using domain_type          = typename simulation_type::domain_type;
    using datablock_type       = typename domain_type::datablock_t;
    using tree_t               = typename domain_type::tree_t;
    using octant_t             = typename tree_t::octant_type;
    using block_type           = typename datablock_type::block_descriptor_type;
    using real_coordinate_type = typename domain_type::real_coordinate_type;
    using coordinate_type      = typename domain_type::coordinate_type;

    //Fields
    using coarse_target_sum = typename Setup::coarse_target_sum;
    using source_tmp = typename Setup::source_tmp;

    //FMM
    using Fmm_t =  typename Setup::Fmm_t;

    static constexpr int lBuffer=1; ///< Lower left buffer for interpolation
    static constexpr int rBuffer=1; ///< Lower left buffer for interpolation

    PoissonSolver(simulation_type* _simulation)
    :
    domain_(_simulation->domain_.get()),
    conv_(domain_->block_extent()+lBuffer+rBuffer,
          domain_->block_extent()+lBuffer+rBuffer),
    fmm_(domain_->block_extent()[0]+lBuffer+rBuffer),
    c_cntr_nli_(domain_->block_extent()[0]+lBuffer+rBuffer)
    {
    }

public:

    /** @brief Solve the poisson equation using lattice Green's functions on
     *         a block-refined mesh for a given Source and Target field.
     *
     *  @detail Lattice Green's functions are used for solving the poisson
     *  equation. FMM is used for the level convolution's and the near field
     *  convolutions are computed using FFT.
     *  Second order interpolation and coarsification operators are used
     *  to project the solutions to fine and coarse meshes, respectively.
     */
    template< class Source, class Target >
    void apply_amr_lgf()
    {

        auto client = domain_->decomposition().client();
        if(!client)return;

        // allocate lgf
        std::vector<float_type> lgf;
        const float_type dx_base=domain_->dx_base();

        // Clean
        for (auto it = domain_->begin();
                it != domain_->end();
                ++it)
        {
            auto& cp2 = it ->data()->template get_linalg_data<source_tmp>();
            cp2 *=0.0;

        }

        // Copy source
        for (auto it  = domain_->begin_leafs();
                it != domain_->end_leafs(); ++it)
            if (it->locally_owned())
            {
                auto& cp1 = it ->data()->template get_linalg_data<Source>();
                auto& cp2 = it ->data()->template get_linalg_data<source_tmp>();

                cp2 = cp1 * 1.0;

            }

        //Coarsification:
        pcout<<"coarsification "<<std::endl;
        for (int ls = domain_->tree()->depth()-2;
                 ls >= domain_->tree()->base_level(); --ls)
        {
            for (auto it_s  = domain_->begin(ls);
                      it_s != domain_->end(ls); ++it_s)
                if (it_s->data())
                {
                    this->coarsify<source_tmp>(*it_s);
                }

            domain_->decomposition().client()->
                template communicate_updownward_add<source_tmp, source_tmp>(ls,true,false);

        }

        //Level-Interactions
        pcout<<"Level interactions "<<std::endl;
        for (int l  = domain_->tree()->base_level()+0;
                l < domain_->tree()->depth(); ++l)
        {
            for (auto it_s  = domain_->begin(l);
                    it_s != domain_->end(l); ++it_s)
                if (it_s->data() && !it_s->locally_owned())
                {
                    auto& cp2 = it_s ->data()->template get_linalg_data<source_tmp>();
                    cp2*=0.0;
                }

            //test for FMM
            //fmm_.template fmm_for_level<source_tmp, Target>(domain_, l, false);
            //fmm_.template fmm_for_level<source_tmp, Target>(domain_, l, true);

            fmm_.template fmm_for_level_test<source_tmp, Target>(domain_, l, false);
            fmm_.template fmm_for_level_test<source_tmp, Target>(domain_, l, true);
            //this->level_convolution_fft<source_tmp, Target>(l);


            //for (auto it  = domain_->begin(l);
            //          it != domain_->end(l); ++it)
            //{
            //    if(it->is_leaf()) continue;

            //    auto& cp1 = it ->data()->template get_linalg_data<Target>();
            //    auto& cp2 = it ->data()->
            //        template get_linalg_data<coarse_target_sum>();

            //    cp2 = cp1 * 1.0;

            //}

            //domain_->decomposition().client()->
            //    template communicate_updownward_assign
            //        <coarse_target_sum, coarse_target_sum>(l,false,false);

            //for (auto it  = domain_->begin(l);
            //          it != domain_->end(l); ++it)
            //{
            //    if(it->is_leaf()) continue;
            //    c_cntr_nli_.nli_intrp_node<
            //                coarse_target_sum, coarse_target_sum
            //                >(it);

            //    int refinement_level = it->refinement_level();
            //    double dx = dx_base/std::pow(2,refinement_level);
            //    c_cntr_nli_.add_source_correction<
            //                            coarse_target_sum, source_tmp
            //                            >(it, dx/2.0);
            //}

        }

        // Interpolation
        pcout<<"Interpolation"<<std::endl;
        for (int lt = domain_->tree()->base_level();
               lt < domain_->tree()->depth(); ++lt)
        {
            domain_->decomposition().client()->
                template communicate_updownward_assign<Target, Target>(lt,false,false);

            for (auto it_t  = domain_->begin(lt);
                      it_t != domain_->end(lt); ++it_t)
            {
                if(it_t->is_leaf() ) continue;
                c_cntr_nli_.nli_intrp_node<Target, Target>(it_t);
            }

        }

    }


    /** @brief Compute level interactions with FFT instead of FMM.  */
    template<
        class Source,
        class Target
        >
    void level_convolution_fft( int level)
    {

        const float_type dx_base=domain_->dx_base();
        for (auto it_t  = domain_->begin(level);
                  it_t != domain_->end(level); ++it_t)
        {
            auto refinement_level = it_t->refinement_level();
            auto dx_level =  dx_base/std::pow(2,refinement_level);
            for (auto it_s  = domain_->begin(level);
                      it_s != domain_->end(level); ++it_s)
            {

                if( !(it_s->is_leaf()) && !(it_t->is_leaf()) )
                { continue; }

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

                // Perform convolution
                conv_.execute_field(block_type (base_lgf, extent_lgf),0,
                        it_s->data()->template get<Source>());

                // Extract the solution
                block_type  extractor(s_base, s_extent);
                conv_.add_solution(extractor,
                                  it_t->data()->template get<Target>(),
                                  dx_level*dx_level);
            }
        }
    }

    /** @brief Compute the laplace operator of the target field and store
     *         it in diff_target.
     */
    template< class target, class diff_target >
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

            domain_->decomposition().client()->
                template communicate_updownward_add<target, target>(ls,true,false);
        }

        //Level-Interactions
        pcout<<"Laplace - level interactions "<<std::endl;
        for (int l  = domain_->tree()->base_level();
                 l < domain_->tree()->depth(); ++l)
        {

            for (auto it  = domain_->begin(l);
                      it != domain_->end(l); ++it)
            {
                auto refinement_level = it->refinement_level();
                auto dx_level =  dx_base/std::pow(2,refinement_level);

                auto& target_data = it->data()->
                    template get_linalg_data<target>();
                auto& diff_target_data = it->data()->
                    template get_linalg_data<diff_target>();

                const auto s_extent = it->data()->template get<target>().
                                                real_block().extent();

                // laplace of it_t data with zero bcs
                if ((it->is_leaf()))
                {
                    for ( int i =1; i<s_extent[0]-1; ++i){
                        for ( int j = 1; j<s_extent[1]-1; ++j){
                            for ( int k = 1; k<s_extent[2]-1; ++k){
                                diff_target_data(i,j,k)  =
                                    target_data(i,j,k) * -6.0;
                                diff_target_data(i,j,k) += target_data(i,j,k-1);
                                diff_target_data(i,j,k) += target_data(i,j,k+1);
                                diff_target_data(i,j,k) += target_data(i,j-1,k);
                                diff_target_data(i,j,k) += target_data(i,j+1,k);
                                diff_target_data(i,j,k) += target_data(i+1,j,k);
                                diff_target_data(i,j,k) += target_data(i-1,j,k);
                            }
                        }
                    }

                }
                diff_target_data *= (1/dx_level) * (1/dx_level);
            }
        }
    }

    template< class Source, class Target >
    void solve()
    {
        apply_amr_lgf<Source, Target>();
    }

    template<class Target, class Laplace>
    void laplace_diff()
    {
        apply_amr_laplace<Target, Laplace>();
    }


    /** @brief Coarsify the source field.
     *  @detail Given a parent, coarsify the field from its children and
     *  assign it to the parent. Coarsification is an average, ie 2nd order
     *  accurate.
     */
    template<class Field >
    void coarsify(octant_t* _parent)
    {
        auto parent = _parent;
        if(parent->is_leaf())return;

        //auto pview =parent->data()->node_field().view(parent->data()->descriptor());
        //pview.iterate([&]( auto& n )
        //        {
        //        n.template get<Field>()=0.0;
        //        });

        for (int i = 0; i < parent->num_children(); ++i)
        {
            auto child = parent->child(i);
            if(child==nullptr) continue;
            auto child_view= child->data()->descriptor();

            auto cview =child->data()->node_field().view(child_view);

            cview.iterate([&]( auto& n )
            {
                const float_type avg=1./8* n.template get<Field>();
                auto pcoord=n.level_coordinate();
                for(std::size_t d=0;d<pcoord.size();++d)
                    pcoord[d]= std::floor(pcoord[d]/2.0);
                parent->data()-> template get<Field>(pcoord) +=avg;
            });
        }
    }


    /** @brief Interplate the target field.
     *  @detail Given a parent field, interpolate it onto the child meshes.
     *  Interpolation is 2nd order accurate.
     */
    template<class Field >
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
    domain_type*                      domain_;    ///< domain
    fft::Convolution                  conv_;      ///< fft convolution
    Fmm_t                             fmm_;       ///< fast-multipole
    interpolation::cell_center_nli    c_cntr_nli_;///< Lagrange Interpolation
    parallel_ostream::ParallelOstream pcout=parallel_ostream::ParallelOstream(1);

};

}

#endif // IBLGF_INCLUDED_POISSON_HPP
