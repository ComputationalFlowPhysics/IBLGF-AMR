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
#include <utilities/cell_center_nli_intrp.hpp>
#include<operators/operators.hpp>

#include <lgf/lgf_gl.hpp>
#include <lgf/lgf_ge.hpp>

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
    using source_tmp        = typename Setup::source_tmp;
    using target_tmp        = typename Setup::target_tmp;
    using correction_tmp    = typename Setup::correction_tmp;

    //FMM
    using Fmm_t     = typename Setup::Fmm_t;
    using lgf_lap_t = typename lgf::LGF_GL<3>;
    using lgf_if_t  = typename lgf::LGF_GE<3>;

    static constexpr int lBuffer=1; ///< Lower left buffer for interpolation
    static constexpr int rBuffer=1; ///< Lower left buffer for interpolation

    PoissonSolver(simulation_type* _simulation)
    :
    domain_(_simulation->domain_.get()),
    fmm_(domain_->block_extent()[0]+lBuffer+rBuffer),
    c_cntr_nli_(domain_->block_extent()[0]+lBuffer+rBuffer)
    {
    }

public:
    template< class Source, class Target >
    void apply_lgf()
    {
        //this->apply_lgf<Source, Target>(&lgf_if_);
        this->apply_lgf<Source, Target>(&lgf_lap_);
    }
    template< class Source, class Target >
    void apply_lgf_IF(float_type _alpha_base)
    {
        lgf_if_.alpha_base_level()=_alpha_base;

        //for (std::size_t entry=0; entry<Source::nFields; ++entry)
        for (std::size_t entry=0; entry<1; ++entry)
            this->apply_if<Source, Target>(&lgf_if_, entry);

    }


    template< class Source, class Target, class Kernel >
    void apply_if(Kernel*  _kernel, std::size_t _field_idx=0)
    {

        auto client = domain_->decomposition().client();
        if(!client)return;

        // Cleaning
        clean_field<source_tmp>();
        clean_field<target_tmp>();

        // Copy source
        copy_leaf<Source, source_tmp>(_field_idx,0,false);

        //Coarsification:
        source_coarsify();

        // For IF, interpolate source to correction buffers

        for (int l  = domain_->tree()->base_level();
                l < domain_->tree()->depth()-1; ++l)
        {

            client->template buffer_exchange<source_tmp>(l);
            // Sync
            domain_->decomposition().client()->
                template communicate_updownward_assign
                    <source_tmp, source_tmp>(l,false,false,-1);

            // Interpolate
            for (auto it  = domain_->begin(l);
                      it != domain_->end(l); ++it)
            {
                if(!it->data() || !it->data()->is_allocated()) continue;
                //TODO make this for face mesh
                c_cntr_nli_.nli_intrp_node< source_tmp, source_tmp >(it, true);
            }


        }

        for (int l  = domain_->tree()->base_level();
                l < domain_->tree()->depth(); ++l)
        {

            for (auto it_s  = domain_->begin(l);
                    it_s != domain_->end(l); ++it_s)
                if (it_s->data() && !it_s->locally_owned())
                {

                    if(!it_s ->data()->is_allocated())continue;
                    auto& cp2 = it_s ->data()->template get_linalg_data<source_tmp>();
                    cp2*=0.0;
                }

            _kernel->change_level(l-domain_->tree()->base_level());

            fmm_.template apply<source_tmp, target_tmp>(domain_, _kernel, l, false, 1.0);
        }

        copy_leaf<target_tmp, Target>(0, _field_idx, true);
    }

     /** @brief Solve the poisson equation using lattice Green's functions on
     *         a block-refined mesh for a given Source and Target field.
     *
     *  @detail Lattice Green's functions are used for solving the poisson
     *  equation. FMM is used for the level convolution's and the near field
     *  convolutions are computed using FFT.
     *  Second order interpolation and coarsification operators are used
     *  to project the solutions to fine and coarse meshes, respectively.
     */
    template< class Source, class Target, class Kernel >
    void apply_lgf(Kernel*  _kernel, std::size_t _field_idx=0)
    {

        auto client = domain_->decomposition().client();
        if(!client)return;

        const float_type dx_base=domain_->dx_base();
        // Cleaning
        clean_field<source_tmp>();
        clean_field<target_tmp>();

        // Copy source
        copy_leaf<Source, source_tmp>(_field_idx, 0, false);

        //Coarsification:
        source_coarsify();

        //Level-Interactions
        for (int l  = domain_->tree()->base_level();
                l < domain_->tree()->depth(); ++l)
        {
            for (auto it_s  = domain_->begin(l);
                    it_s != domain_->end(l); ++it_s)
                if (it_s->data() && !it_s->locally_owned())
                {

                    if(!it_s ->data()->is_allocated())continue;
                    auto& cp2 = it_s ->data()->template get_linalg_data<source_tmp>();
                    cp2*=0.0;
                }

            //
            _kernel->change_level(l-domain_->tree()->base_level());

            //test for FMM
            fmm_.template apply<source_tmp, target_tmp>(domain_, _kernel, l, false, 1.0);
            fmm_.template apply<source_tmp, target_tmp>(domain_, _kernel, l, true, -1.0);

            //Interpolate
            // Sync
            domain_->decomposition().client()->
                template communicate_updownward_assign
                    <target_tmp, target_tmp>(l,false,false,-1);

            for (auto it  = domain_->begin(l);
                    it != domain_->end(l); ++it)
            {
                if(!it->data() || !it->data()->is_allocated()) continue;
                c_cntr_nli_.nli_intrp_node< target_tmp, target_tmp >(it);
            }

            // Correction for LGF
            for (auto it  = domain_->begin(l);
                    it != domain_->end(l); ++it)
            {
                int refinement_level = it->refinement_level();
                double dx_level = dx_base/std::pow(2,refinement_level);

                if(!it->data() || !it->data()->is_allocated()) continue;
                domain::Operator::laplace<target_tmp, correction_tmp>
                ( *(it->data()),dx_level);

            }

            client->template buffer_exchange<correction_tmp>(l);

            for (auto it  = domain_->begin(l);
                    it != domain_->end(l); ++it)
            {
                if(!it->data() || !it->data()->is_allocated()) continue;
                c_cntr_nli_.nli_intrp_node< correction_tmp, source_tmp >(it);
            }

            for (auto it  = domain_->begin(l);
                    it != domain_->end(l); ++it)
            {
                int refinement_level = it->refinement_level();
                double dx = dx_base/std::pow(2,refinement_level);
                c_cntr_nli_.add_source_correction
                <target_tmp, source_tmp>(it, dx/2.0);
            }

        }

        // Copy to Target
        copy_leaf<target_tmp, Target>(0, _field_idx, true);
    }

    template<class field>
    void clean_field()
    {
        for (auto it  = domain_->begin();
                it != domain_->end(); ++it)
        {
            if(!it->data() || !it->data()->is_allocated()) continue;

            for(auto& e: it->data()->template get_data<field>())
                e=0.0;
        }
    }

    template<class from, class to>
    void copy_leaf(std::size_t _field_idx_from=0, std::size_t _field_idx_to=0, bool with_buffer=false)
    {
        for (auto it  = domain_->begin_leafs();
                it != domain_->end_leafs(); ++it)
            if (it->locally_owned())
            {
                auto& lin_data_1 = it->data()->
                    template get_linalg_data<from>(_field_idx_from);
                auto& lin_data_2 = it->data()->
                    template get_linalg_data<to>(_field_idx_to);

                if (with_buffer)
                    xt::noalias(lin_data_2) = lin_data_1 * 1.0;
                else
                    xt::noalias( view(lin_data_2,
                                xt::range(1,-1),  xt::range(1,-1), xt::range(1,-1)) ) =
                        view(lin_data_1, xt::range(1,-1), xt::range(1,-1), xt::range(1,-1));
            }
    }

    void source_coarsify()
    {
        for (int ls = domain_->tree()->depth()-2;
                ls >= domain_->tree()->base_level(); --ls)
        {
            for (auto it_s  = domain_->begin(ls);
                    it_s != domain_->end(ls); ++it_s)
                if (it_s->data())
                {
                    //TODO change coarsify to mesh type dependent
                    this->coarsify<source_tmp>(*it_s);
                }

            domain_->decomposition().client()->
            template communicate_updownward_add<source_tmp, source_tmp>
            (ls,true,false,-1);

        }

    }


    /** @brief Compute level interactions with FFT instead of FMM.  */
    template<
        class Conv,
        class Source,
        class Target
        >
    void level_convolution_fft(const Conv& _conv, int level)
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
                _conv.execute_field(block_type (base_lgf, extent_lgf),0,
                        it_s->data()->template get<Source>());

                // Extract the solution
                block_type  extractor(s_base, s_extent);
                _conv.add_solution(extractor,
                                  it_t->data()->template get<Target>(),
                                  dx_level*dx_level);
            }
        }
    }

    /** @brief Compute the laplace operator of the target field and store
     *         it in diff_target.
     */
    template< class target, class diff_target >
    void apply_laplace()
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
                if (!it_s->data()) continue;
                this->coarsify<target>(*it_s);
            }

            domain_->decomposition().client()->
                template communicate_updownward_add<target, target>(ls,true,false,-1);
        }

        //Level-Interactions
        pcout<<"Laplace - level interactions "<<std::endl;
        for (int l  = domain_->tree()->base_level();
                 l < domain_->tree()->depth(); ++l)
        {

            for (auto it  = domain_->begin(l);
                      it != domain_->end(l); ++it)
            {
                if (!it->data() || !it->locally_owned() || !it ->data()->is_allocated()) continue;
                auto refinement_level = it->refinement_level();
                auto dx_level =  dx_base/std::pow(2,refinement_level);

                auto& diff_target_data = it->data()->
                    template get_linalg_data<diff_target>();

                // laplace of it_t data with zero bcs
                if ((it->is_leaf()))
                {
                    auto& nodes_domain=it->data()->nodes_domain();
                    for(auto it2=nodes_domain.begin();it2!=nodes_domain.end();++it2 )
                    {
                        it2->template get<diff_target>()=
                                  -6.0* it2->template get<target>()+
                                  it2->template at_offset<target>(0,0,-1)+
                                  it2->template at_offset<target>(0,0,+1)+
                                  it2->template at_offset<target>(0,-1,0)+
                                  it2->template at_offset<target>(0,+1,0)+
                                  it2->template at_offset<target>(-1,0,0)+
                                  it2->template at_offset<target>(+1,0,0);
                    }
                }
                diff_target_data *= (1/dx_level) * (1/dx_level);
            }
        }
    }

    template< class Source, class Target >
    void solve()
    {
        apply_lgf<Source, Target>();
    }

    template<class Target, class Laplace>
    void laplace_diff()
    {
        apply_laplace<Target, Laplace>();
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

        for (int i = 0; i < parent->num_children(); ++i)
        {
            auto child = parent->child(i);
            if(child==nullptr || !child->data() || !child->locally_owned()) continue;

            if (child->is_correction()) continue;

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
    Fmm_t                             fmm_;       ///< fast-multipole
    lgf_lap_t                         lgf_lap_;
    lgf_if_t                          lgf_if_;
    interpolation::cell_center_nli    c_cntr_nli_;///< Lagrange Interpolation
    parallel_ostream::ParallelOstream pcout=parallel_ostream::ParallelOstream(1);

};

}

#endif // IBLGF_INCLUDED_POISSON_HPP
