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
#include <utilities/extrapolation_cell_center_nli_intrp.hpp>
#include<operators/operators.hpp>

#include <lgf/lgf_gl.hpp>
#include <lgf/lgf_ge.hpp>

#include <linalg/linalg.hpp>

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
    using corr_lap_tmp    = typename Setup::corr_lap_tmp;
    using source_correction_tmp    = typename Setup::source_correction_tmp;

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
    c_cntr_nli_(domain_->block_extent()[0]+lBuffer+rBuffer),
    extrp_c_cntr_nli_(domain_->block_extent()[0]+lBuffer+rBuffer)
    {
    }

public:
    template< class Source, class Target >
    void apply_lgf()
    {
        //this->apply_lgf<Source, Target>(&lgf_if_);
        for (std::size_t entry=0; entry<Source::nFields; ++entry)
            this->apply_lgf<Source, Target>(&lgf_lap_,entry);
    }
    template< class Source, class Target >
    void apply_lgf_IF(float_type _alpha_base)
    {
        lgf_if_.alpha_base_level()=_alpha_base;

        for (std::size_t entry=0; entry<Source::nFields; ++entry)
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
        source_coarsify(_field_idx, Source::mesh_type);

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
                c_cntr_nli_.nli_intrp_node<source_tmp, source_tmp>
                    (it, Source::mesh_type, _field_idx, true);
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
                    std::fill (cp2.begin(),cp2.end(),0.0);
                }

            _kernel->change_level(l-domain_->tree()->base_level());

            fmm_.template apply<source_tmp, target_tmp>(domain_, _kernel, l, false, 1.0);
            // Interpolate
            // Sync
            domain_->decomposition().client()->
                template communicate_updownward_assign
                    <target_tmp, target_tmp>(l,false,false,-1);

            for (auto it  = domain_->begin(l);
                      it != domain_->end(l); ++it)
            {
                if(!it->data() || !it->data()->is_allocated()) continue;
                c_cntr_nli_.nli_intrp_node<target_tmp, target_tmp>
                    (it, Source::mesh_type, _field_idx, true);
            }

            copy_level<target_tmp, Target>(l, 0, _field_idx, true);
        }

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
    //template< class Source, class Target, class Kernel >
    //void apply_lgf(Kernel*  _kernel, std::size_t _field_idx=0)
    //{

    //    auto client = domain_->decomposition().client();
    //    if(!client)return;

    //    const float_type dx_base=domain_->dx_base();
    //    // Cleaning
    //    clean_field<source_tmp>();
    //    clean_field<target_tmp>();
    //    clean_field<correction_tmp>();

    //    // Copy source
    //    copy_leaf<Source, source_tmp>(_field_idx, 0, false);

    //    //Coarsification:
    //    source_coarsify(_field_idx, Source::mesh_type);

    //    //Level-Interactions
    //    for (int l  = domain_->tree()->base_level();
    //            l < domain_->tree()->depth(); ++l)
    //    {
    //        for (auto it_s  = domain_->begin(l);
    //                it_s != domain_->end(l); ++it_s)
    //            if (it_s->data() && !it_s->locally_owned())
    //            {
    //                if(!it_s ->data()->is_allocated())continue;
    //                auto& cp2 = it_s ->data()->template get_linalg_data<source_tmp>();
    //                std::fill (cp2.begin(),cp2.end(),0.0);
    //            }

    //        //test for FMM
    //        fmm_.template apply<source_tmp, target_tmp>(domain_, _kernel, l, false, 1.0);
    //        copy_level<target_tmp, Target>(l, 0, _field_idx, true);

    //        fmm_.template apply<source_tmp, target_tmp>(domain_, _kernel, l, true, -1.0);

    //        // Interpolate
    //        // Sync
    //        domain_->decomposition().client()->
    //            template communicate_updownward_assign
    //                <target_tmp, target_tmp>(l,false,false,-1);

    //        for (auto it  = domain_->begin(l);
    //                it != domain_->end(l); ++it)
    //        {
    //            if(!it->data() || !it->data()->is_allocated()) continue;
    //            c_cntr_nli_.nli_intrp_node<target_tmp, target_tmp>(it, Source::mesh_type, _field_idx);
    //        }

    //        // Correction for LGF
    //        for (auto it  = domain_->begin(l);
    //                it != domain_->end(l); ++it)
    //        {
    //            int refinement_level = it->refinement_level();
    //            double dx_level = dx_base/std::pow(2,refinement_level);

    //            if(!it->data() || !it->data()->is_allocated()) continue;
    //            domain::Operator::laplace<target_tmp, correction_tmp>
    //            ( *(it->data()),dx_level);

    //        }

    //        client->template buffer_exchange<correction_tmp>(l);

    //        for (auto it  = domain_->begin(l);
    //                it != domain_->end(l); ++it)
    //        {
    //            if(!it->data() || !it->data()->is_allocated()) continue;
    //            c_cntr_nli_.nli_intrp_node< correction_tmp, correction_tmp >(it, Source::mesh_type, _field_idx);
    //        }

    //        for (auto it  = domain_->begin(l);
    //                it != domain_->end(l); ++it)
    //        {
    //            int refinement_level = it->refinement_level();
    //            double dx = dx_base/std::pow(2,refinement_level);
    //            c_cntr_nli_.add_source_correction
    //                <target_tmp, correction_tmp>(it, dx/2.0);
    //        }

    //        for (auto it  = domain_->begin(l+1); it != domain_->end(l+1); ++it)
    //        {
    //            if (it->locally_owned())
    //            {
    //                auto& lin_data_1 = it->data()->
    //                    template get_linalg_data<correction_tmp>(0);
    //                auto& lin_data_2 = it->data()->
    //                    template get_linalg_data<source_tmp>(0);
    //                xt::noalias(lin_data_2) += lin_data_1 * 1.0;
    //            }
    //        }
    //    }

    //    // Copy to Target
    //    // copy_leaf<target_tmp, Target>(0, _field_idx, true);
    //}

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

        boost::mpi::communicator world;
        //if(domain_->is_client())world=world.split(1);
        //else world=world.split(0);

        auto client = domain_->decomposition().client();
        if(!client)return;

        const float_type dx_base=domain_->dx_base();

        // Cleaning
        clean_field<source_tmp>();
        clean_field<target_tmp>();
        clean_field<correction_tmp>();
        clean_field<corr_lap_tmp>();
        clean_field<source_correction_tmp>();

        // Copy source
        copy_leaf<Source, source_tmp>(_field_idx, 0, true);

        if(Source::mesh_type!=MeshObject::cell)
        {
            throw
            std::runtime_error
            ("Coarsification for non-cell centers needs to be implemented. ");
        }

        Timings timing;

        //world.barrier();
        auto t0_all=clock_type::now();
        auto t0_coarsify=clock_type::now();

        //source_coarsify(_field_idx, Source::mesh_type);

        //Coarsification:
        for (int l = domain_->tree()->depth()-2;
                l >= domain_->tree()->base_level(); --l)
        {
            for (auto it_s  = domain_->begin(l);
                    it_s != domain_->end(l); ++it_s)
                {
                    if(!it_s->data() || !it_s->data()->is_allocated()) continue;
                    this->coarsify<source_tmp, source_tmp>(*it_s, 1.0, false, true);
                }

            domain_->decomposition().client()->
                template communicate_updownward_add<source_tmp, source_tmp>
                    (l,true,false,-1);

        }

        for (int l = domain_->tree()->depth()-2;
                l >= domain_->tree()->base_level(); --l)
        {
            client->template buffer_exchange<source_tmp>(l);
            domain_->decomposition().client()->
                template communicate_updownward_assign
                <source_tmp, source_tmp>(l,false,false,-1);

            for (auto it  = domain_->begin(l);
                    it != domain_->end(l); ++it)
            {
                if(!it->data() || !it->data()->is_allocated()) continue;

                const bool correction_buffer_only = true;
                c_cntr_nli_.nli_intrp_node< source_tmp, source_tmp>(it, Source::mesh_type, _field_idx, correction_buffer_only,false);

                this->coarsify<source_tmp, source_correction_tmp>(*it, 1.0, correction_buffer_only, false);
            }

        }

        auto t1_coarsify= clock_type::now();
        timing.coarsification = t1_coarsify-t0_coarsify;

        const auto t0_level_interaction = clock_type::now();

        //Level-Interactions

        for (int l  = domain_->tree()->base_level();
                l < domain_->tree()->depth(); ++l)
        {
            const auto t0_level=clock_type::now();

            for (auto it_s  = domain_->begin(l);
                    it_s != domain_->end(l); ++it_s)
                if (it_s->data() && !it_s->locally_owned())
                {

                    if(!it_s ->data()->is_allocated())continue;
                    auto& cp2 = it_s ->data()->template get_linalg_data<source_tmp>();
                    cp2*=0.0;
                }

            // test for FMM
            const auto t2=clock_type::now();
            fmm_.template apply<source_tmp, target_tmp>(domain_, _kernel, l, false, 1.0);
            // Copy to Target
            for (auto it  = domain_->begin(l);
                    it != domain_->end(l); ++it)
                if (it->locally_owned() && it->is_leaf())
                {
                    it->data()->template get_linalg<Target>(_field_idx).get()->
                    cube_noalias_view() =
                    it->data()->template get_linalg_data<target_tmp>();
                }


            // minus back the half from correction
                // delete correction parents
            for (auto it  = domain_->begin(l);
                    it != domain_->end(l); ++it)
            {
                if(!it->data() || !it->data()->is_allocated()) continue;
                bool correction_parent=false;
                for (std::size_t i=0; i<it->num_children(); ++i)
                    if (it->child(i) && it->child(i)->is_correction())
                        correction_parent = true;

                if (correction_parent)
                {
                    auto& cp2 = it ->data()->template get_linalg_data<source_tmp>();
                    cp2*=0.0;
                }
            }
                // add back correction source
            domain_->decomposition().client()->
                template communicate_updownward_add<source_correction_tmp, source_tmp>
                    (l,true,false,-1);

            // minus middle
            fmm_.template apply<source_tmp, target_tmp>(domain_, _kernel, l, true, -1.0);
            // Interpolate
            domain_->decomposition().client()->
                template communicate_updownward_assign
                    <target_tmp, target_tmp>(l,false,false,-1);

            for (auto it  = domain_->begin(l);
                      it != domain_->end(l); ++it)
            {
                if(!it->data() || !it->data()->is_allocated()) continue;
                c_cntr_nli_.nli_intrp_node< target_tmp, target_tmp >
                    (it, Source::mesh_type, _field_idx, false, false);
            }

            //if (l == domain_->tree()->depth()-1) continue;
            //client->template buffer_exchange<target_tmp>(l+1);

            // Use the temporary value as an approximation for the buffer for
            // the correction term -LP

            // Correction for LGF

            for (auto it  = domain_->begin(l);
                    it != domain_->end(l); ++it)
            {
                int refinement_level = it->refinement_level();
                double dx_level = dx_base/std::pow(2,refinement_level);

                if (!it->data() || !it->data()->is_allocated()) continue;
                //domain::Operator::laplace<target_tmp, correction_tmp>
                //( *(it->data()),dx_level);
                domain::Operator::laplace<target_tmp, corr_lap_tmp>
                ( *(it->data()),dx_level);
            }

            client->template buffer_exchange<corr_lap_tmp>(l);
            domain_->decomposition().client()->
                template communicate_updownward_assign
                <corr_lap_tmp, corr_lap_tmp>(l,false,false,-1);

            for (auto it  = domain_->begin(l);
                    it != domain_->end(l); ++it)
            {
                if(!it->data() || !it->data()->is_allocated()) continue;

                const bool correction_buffer_only = false;
                //c_cntr_nli_.nli_intrp_node< corr_lap_tmp, correction_tmp>(it, Source::mesh_type, _field_idx, correction_buffer_only,false);
            }


            // start here

            //client->template buffer_exchange<Source>(l);

            // FMM for non-leaf

            for (auto it  = domain_->begin(l);
                    it != domain_->end(l); ++it)
            {
                int refinement_level = it->refinement_level();
                double dx = dx_base/std::pow(2,refinement_level);
                c_cntr_nli_.add_source_correction
                    <target_tmp, correction_tmp>(it, dx/2.0);
            }

            for (auto it  = domain_->begin(l+1);
                    it != domain_->end(l+1); ++it)
                if (it->locally_owned())
                {
                    auto& lin_data_1 = it->data()->
                    template get_linalg_data<correction_tmp>(0);
                    auto& lin_data_2 = it->data()->
                    template get_linalg_data<source_tmp>(0);
                    auto& lin_data_3 = it->data()->
                    template get_linalg_data<target_tmp>(0);

                    xt::noalias(lin_data_2) += lin_data_1 * 1.0;
                }
        }

        for (int ls = domain_->tree()->depth()-2;
                ls >= domain_->tree()->base_level(); --ls)
        {
            const auto t0_level=clock_type::now();
            for (auto it_s  = domain_->begin(ls);
                    it_s != domain_->end(ls); ++it_s)
            {
                if(!it_s->data() || !it_s->data()->is_allocated()) continue;
                this->coarsify<correction_tmp, correction_tmp>(*it_s);
            }

            domain_->decomposition().client()->
            template communicate_updownward_add<correction_tmp, correction_tmp>
            (ls,true,false,-1);
        }

        //world.barrier();
        const auto t1_level_interaction=clock_type::now();
        timing.level_interaction=t1_level_interaction-t0_level_interaction
                                  -timing.interpolation;


        const auto t1_all=clock_type::now();
        timing.global=t1_all-t0_all;
    }

    template<class field>
    void clean_field()
    {
        for (auto it  = domain_->begin();
                it != domain_->end(); ++it)
        {
            if(!it->data() || !it->data()->is_allocated()) continue;

            auto& lin_data = it->data()->
                    template get_linalg_data<field>();

            std::fill (lin_data.begin(),lin_data.end(),0.0);
        }
    }

    template<class from, class to>
    void copy_level(int level, std::size_t _field_idx_from=0, std::size_t _field_idx_to=0, bool with_buffer=false)
    {
        for (auto it  = domain_->begin(level);
                it != domain_->end(level); ++it)
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

    void source_coarsify(std::size_t _field_idx, MeshObject mesh_type)
    {
        if(mesh_type!=MeshObject::cell)
        {
            throw
            std::runtime_error("Coarsification for non-cell centers needs to be implemented. ");
        }
        auto client = domain_->decomposition().client();
        if(!client)return;

        for (int ls = domain_->tree()->depth()-2;
                ls >= domain_->tree()->base_level(); --ls)
        {
            //client->template buffer_exchange<source_tmp>(ls+1);

            for (auto it_s  = domain_->begin(ls);
                    it_s != domain_->end(ls); ++it_s)
                {
                    if(!it_s->data() || !it_s->data()->is_allocated()) continue;

                    //c_cntr_nli_.nli_antrp_node
                    //    <source_tmp, source_tmp>(*it_s,mesh_type,_field_idx);

                    this->coarsify<source_tmp,source_tmp>(*it_s);

                }

            domain_->decomposition().client()->
                template communicate_updownward_add<source_tmp, source_tmp>
                    (ls,true,false,-1);
        }
    }


    /** @brief Compute the laplace operator of the target field and store
     *         it in diff_target.
     */
    template< class Target, class DiffTarget >
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
                //if (!it_s->data()) continue;
                if(!it_s->data() || !it_s->data()->is_allocated()) continue;
                this->coarsify<Target, Target>(*it_s);
            }

            domain_->decomposition().client()->
                template communicate_updownward_add<Target, Target>(ls,true,false,-1);
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
                    template get_linalg_data<DiffTarget>();

                // laplace of it_t data with zero bcs
                if ((it->is_leaf()))
                {
                    auto& nodes_domain=it->data()->nodes_domain();
                    for(auto it2=nodes_domain.begin();it2!=nodes_domain.end();++it2 )
                    {
                        it2->template get<DiffTarget>()=
                                  -6.0* it2->template get<Target>()+
                                  it2->template at_offset<Target>(0,0,-1)+
                                  it2->template at_offset<Target>(0,0,+1)+
                                  it2->template at_offset<Target>(0,-1,0)+
                                  it2->template at_offset<Target>(0,+1,0)+
                                  it2->template at_offset<Target>(-1,0,0)+
                                  it2->template at_offset<Target>(+1,0,0);
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
        //apply_laplace<Target, Laplace>();
    }


    /** @brief Coarsify the source field.
     *  @detail Given a parent, coarsify the field from its children and
     *  assign it to the parent. Coarsification is an average, ie 2nd order
     *  accurate.
     */
    //template<class Field >
    //void coarsify(octant_t* _parent, MeshObject mesh_type, std::size_t _field_idx)
    //{
    //    int n = child.shape()[0];

    //    int idx_x = (child_idx & ( 1 << 0 )) >> 0;
    //    int idx_y = (child_idx & ( 1 << 1 )) >> 1;
    //    int idx_z = (child_idx & ( 1 << 2 )) >> 2;

    //    // Relative position 0 -> coincide with child
    //    // Relative position 1 -> half cell off with the child

    //    std::array<int, 3> relative_positions{{1,1,1}};
    //    if (mesh_obj == MeshObject::face)
    //        relative_positions[_field_idx]=0;
    //    else if (mesh_obj == MeshObject::cell)
    //    {
    //    }
    //    else
    //        throw std::runtime_error(
    //                "Wrong type of mesh to be interpolated");

    //    idx_x += relative_positions[0]*max_relative_pos;
    //    idx_y += relative_positions[1]*max_relative_pos;
    //    idx_z += relative_positions[2]*max_relative_pos;

    //    auto parent = _parent;
    //    if(parent->is_leaf())return;

    //    for (int i = 0; i < parent->num_children(); ++i)
    //    {
    //        auto child = parent->child(i);
    //        if(child==nullptr || !child->data() || !child->locally_owned()) continue;
    //        if (child->is_correction()) continue;

    //        auto child_view= child->data()->descriptor();
    //        auto cview =child->data()->node_field().view(child_view);

    //        cview.iterate([&]( auto& n )
    //        {
    //            const float_type avg=1./8* n.template get<Field>();
    //            auto pcoord=n.level_coordinate();
    //            for(std::size_t d=0;d<pcoord.size();++d)
    //                pcoord[d]= std::floor(pcoord[d]/2.0);
    //            parent->data()-> template get<Field>(pcoord) +=avg;
    //        });
    //    }
    //}


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



    template<class Field_c, class Field_p >
    void coarsify(octant_t* _parent, float_type factor=1.0, bool correction_only=false, bool exclude_correction = false)
    {
        auto parent = _parent;

        for (int i = 0; i < parent->num_children(); ++i)
        {
            auto child = parent->child(i);
            if(child==nullptr || !child->data() || !child->locally_owned()) continue;
            if (correction_only && !child->is_correction())
                continue;
            if (exclude_correction && child->is_correction())
                continue;

            auto child_view= child->data()->descriptor();

            auto cview =child->data()->node_field().view(child_view);

            cview.iterate([&]( auto& n )
            {
                const float_type avg=1./8* n.template get<Field_c>();
                auto pcoord=n.level_coordinate();
                for(std::size_t d=0;d<pcoord.size();++d)
                    pcoord[d]= std::floor(pcoord[d]/2.0);
                parent->data()-> template get<Field_p>(pcoord) +=avg*factor;
            });
        }
    }


private:
    domain_type*                      domain_;    ///< domain
    Fmm_t                             fmm_;       ///< fast-multipole
    lgf_lap_t                         lgf_lap_;
    lgf_if_t                          lgf_if_;
    interpolation::cell_center_nli    c_cntr_nli_;///< Lagrange Interpolation
    interpolation::extrapolation_cell_center_nli    extrp_c_cntr_nli_;///< Lagrange Interpolation
    parallel_ostream::ParallelOstream pcout=parallel_ostream::ParallelOstream(1);

    //Timings:
    struct Timings{
        std::vector<mDuration_type>       level;
        mDuration_type                    global=mDuration_type(0);
        mDuration_type                    coarsification=mDuration_type(0);
        mDuration_type                    level_interaction=mDuration_type(0);
        mDuration_type                    interpolation=mDuration_type(0);

        friend std::ostream& operator<<(std::ostream& os, const  Timings& _t)
        {

            os<<"global "<<" coarsification "<<" level_interaction "<<" interpolation"<<std::endl;
            os<<_t.global.count()<<" "<<_t.coarsification.count()<<" "
              <<_t.level_interaction.count()<<" " <<_t.interpolation.count()<<" "
              <<std::endl;
            for(auto& t :  _t.level)
            {
                os<<t.count()<<" ";
            }
            os<<std::endl;
            return os;
        }
    };
};

}

#endif // IBLGF_INCLUDED_POISSON_HPP
