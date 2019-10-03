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


        auto client = domain_->decomposition().client();
        if(!client)return;

        const float_type dx_base=domain_->dx_base();

        // Cleaning
        for (auto it  = domain_->begin();
                it != domain_->end(); ++it)
            if (it->locally_owned())
            {
                for(auto& e: it->data()->template get_data<source_tmp>())
                    e=0.0;

                for(auto& e: it->data()->template get_data<target_tmp>())
                    e=0.0;
            }

        // Copy source
        for (auto it  = domain_->begin_leafs();
                it != domain_->end_leafs(); ++it)
            if (it->locally_owned())
            {
                auto lin_data_1 = it->data()->template get_linalg_data<Source>(_field_idx);
                auto lin_data_2 = it->data()->template get_linalg_data<source_tmp>();
                xt::noalias( view(lin_data_2,
                            xt::range(1,-1),  xt::range(1,-1), xt::range(1,-1)) ) =
                view(lin_data_1, xt::range(1,-1),  xt::range(1,-1), xt::range(1,-1));
            }

        if(Source::mesh_type!=MeshObject::cell)
        {
            throw 
            std::runtime_error
            ("Coarsification for non-cell centers needs to be implemented. ");
        }

        timings_=Timings() ;
        const int nLevels=domain_->nLevels();
        timings_.level.resize(nLevels);
        timings_.fmm_level.resize(nLevels);
        timings_.fmm_level_nl.resize(nLevels);

        auto t0_all=clock_type::now();
        auto t0_coarsify=clock_type::now();

        //source_coarsify(_field_idx, Source::mesh_type);

        //Coarsification:
        for (int ls = domain_->tree()->depth()-2;
                ls >= domain_->tree()->base_level(); --ls)
        {
            const auto t0_level=clock_type::now();
            for (auto it_s  = domain_->begin(ls);
                    it_s != domain_->end(ls); ++it_s)
                {
                    if(!it_s->data() || !it_s->data()->is_allocated()) continue;
                    this->coarsify<source_tmp>(*it_s);

                }

            domain_->decomposition().client()->
                template communicate_updownward_add<source_tmp, source_tmp>
                    (ls,true,false,-1);
            
            const auto t1_level=clock_type::now();
              
            timings_.level[ls-domain_->tree()->base_level()+1]=t1_level-t0_level;
        }
        
        auto t1_coarsify= clock_type::now();
        timings_.coarsification = t1_coarsify-t0_coarsify;

        const auto t0_level_interaction = clock_type::now();

        //Level-Interactions
        for (int l = domain_->tree()->base_level();
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

            _kernel->change_level(l-domain_->tree()->base_level());

            //test for FMM
            fmm_.template apply<source_tmp, target_tmp>(domain_, _kernel, l, false, 1.0);
            timings_.fmm_level[l-domain_->tree()->base_level()]=fmm_.timings();
            fmm_.template apply<source_tmp, target_tmp>(domain_, _kernel, l, true, -1.0);
            timings_.fmm_level_nl[l-domain_->tree()->base_level()]=fmm_.timings();


            domain_->decomposition().client()->
                template communicate_updownward_assign
                    <target_tmp, target_tmp>(l,false,false,-1);


            // Interpolate
            const auto t2=clock_type::now();
            for (auto it  = domain_->begin(l);
                      it != domain_->end(l); ++it)
            {
                if(it->is_leaf() || !it->data() || !it->data()->is_allocated()) continue;

                c_cntr_nli_.nli_intrp_node< target_tmp, target_tmp >
                    (it, Source::mesh_type, _field_idx, false);
            }
            const auto t3=clock_type::now();
            timings_.interpolation+=(t3-t2);

            // Correction
            //if (l == domain_->tree()->depth()-1) continue;
            if (l != domain_->tree()->depth()-1)
            {
                for (auto it  = domain_->begin(l);
                        it != domain_->end(l); ++it)
                {
                    int refinement_level = it->refinement_level();
                    double dx = dx_base/std::pow(2,refinement_level);
                    c_cntr_nli_.add_source_correction
                        <target_tmp, source_tmp>(it, dx/2.0);
                }
            }

            const auto t1_level=clock_type::now();
            mDuration_type tmp=t1_level-t0_level;
            //std::cout<< "Level timing : l="<<l-domain_->tree()->base_level()<<" "<<tmp.count()/1.e3<<std::endl;
            timings_.level[l-domain_->tree()->base_level()]=t1_level-t0_level;
        }

        const auto t1_level_interaction=clock_type::now();
        timings_.level_interaction=t1_level_interaction-t0_level_interaction
                                  -timings_.interpolation;


        const auto t1_all=clock_type::now();
        timings_.global=t1_all-t0_all;

        // Copy to Target
        for (auto it  = domain_->begin_leafs();
                it != domain_->end_leafs(); ++it)
            if (it->locally_owned())
            {
                it->data()->template get_linalg<Target>(_field_idx).get()->
                    cube_noalias_view() =
                            it->data()->template get_linalg_data<target_tmp>();
            }

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

                    this->coarsify<source_tmp>(*it_s);

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
                this->coarsify<Target>(*it_s);
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



    template<class Field >
    void coarsify(octant_t* _parent)
    {
        auto parent = _parent;
        if(parent->is_leaf())return;

        for (int i = 0; i < parent->num_children(); ++i)
        {
            auto child = parent->child(i);
            if(child==nullptr || !child->data() || !child->locally_owned()) continue;
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

   auto& timings() noexcept {return timings_;}
   const auto& timings() const noexcept {return timings_;}

   template<class Out>
   void print_timings( Out& os, Out& os_level )
   {
       int width=20;
       timings_.accumulate(this->domain_->client_communicator());
       const auto pts=this->domain_->get_nPoints();

        os<<std::left
            <<std::setw(15) <<"npts"
            <<std::setw(width) <<"global[s]"
            <<std::setw(width) <<"gbl rate[pts/s]"
            <<std::setw(width) <<"gbl eff[s/pt]"
            <<std::setw(width) <<"coarsing[%]"
            <<std::setw(width) <<"level[%]"
            <<std::setw(width) <<"interp[%]"
        <<std::endl;
       os<<std::left<<std::scientific<<std::setprecision(7)
           <<std::setw(15) <<pts.back()
           <<std::setw(width) <<timings_.global.count()/1.e3
           <<std::setw(width) <<pts.back()/static_cast<float_type>(timings_.global.count()/1.e3)
           <<std::setw(width) <<static_cast<float_type>(timings_.global.count()/1.e3)/pts.back()
           <<std::defaultfloat
           <<std::setw(width) <<100.0*static_cast<float_type>(timings_.coarsification.count())/timings_.global.count()
           <<std::setw(width) <<100.0*static_cast<float_type>(timings_.level_interaction.count())/timings_.global.count()
           <<std::setw(width) <<100.0*static_cast<float_type>(timings_.interpolation.count())/timings_.global.count()
       <<std::endl;

           auto all=static_cast<float_type>(timings_.coarsification.count())/timings_.global.count()+
           static_cast<float_type>(timings_.level_interaction.count())/timings_.global.count()+
           static_cast<float_type>(timings_.interpolation.count())/timings_.global.count();

       int c=0;
       width=15;
       os_level<<std::left<<std::scientific<<std::setprecision(5)
            <<std::setw(10) <<"level"
            <<std::setw(15) <<"npts"
            <<std::setw(width) <<"gbl[s]"
            <<std::setw(width) <<"rate[pts/s]"
            <<std::setw(width) <<"eff[s/pt]"

            <<std::setw(width) <<"fmm gbl "
            <<std::setw(width) <<"anterp "
            <<std::setw(width) <<"Bx "
            <<std::setw(width) <<"fft "
            <<std::setw(width) <<"fft ratio "
            <<std::setw(width) <<"interp "

            <<std::setw(width) <<"fmm_nl gbl "
            <<std::setw(width) <<"anterp "
            <<std::setw(width) <<"Bx "
            <<std::setw(width) <<"fft "
            <<std::setw(width) <<"fft ratio "
            <<std::setw(width) <<"interp "
       <<std::endl;

       for(std::size_t i=0; i<timings_.level.size();++i)
       {

           auto& t=timings_.level[i];
           auto& fmm=timings_.fmm_level[i];
           auto& fmm_nl=timings_.fmm_level_nl[i];

           os_level<<std::setw(10)<<c
             <<std::setw(15)<<pts[c]
             <<std::setw(width)<<static_cast<float_type>(t.count())/1.e3
             <<std::setw(width)<<pts[c]/(static_cast<float_type>(t.count())/1.e3)
             <<std::setw(width)<<(static_cast<float_type>(t.count())/1.e3)/pts[c]

             <<std::setw(width)<<(static_cast<float_type>(fmm.global.count())/1.e3)
             <<std::setw(width)<<(static_cast<float_type>(fmm.anterp.count())/1.e3)
             <<std::setw(width)<<(static_cast<float_type>(fmm.bx.count())/1.e3)
             <<std::setw(width)<<(static_cast<float_type>(fmm.fftw.count())/1.e3)
             <<std::setw(width)<<(static_cast<float_type>(fmm.fftw_count_max)/fmm.fftw_count_min)
             <<std::setw(width)<<(static_cast<float_type>(fmm.interp.count())/1.e3)

             <<std::setw(width)<<(static_cast<float_type>(fmm_nl.global.count())/1.e3)
             <<std::setw(width)<<(static_cast<float_type>(fmm_nl.anterp.count())/1.e3)
             <<std::setw(width)<<(static_cast<float_type>(fmm_nl.bx.count())/1.e3)
             <<std::setw(width)<<(static_cast<float_type>(fmm_nl.fftw.count())/1.e3)
             <<std::setw(width)<<(static_cast<float_type>(fmm_nl.fftw_count_max)/fmm_nl.fftw_count_min)
             <<std::setw(width)<<(static_cast<float_type>(fmm_nl.interp.count())/1.e3)

           <<std::endl;
           ++c;
       }
       os_level<<std::defaultfloat<<std::endl;
       os<<std::defaultfloat<<std::endl;
   }


private:
    domain_type*                      domain_;    ///< domain
    Fmm_t                             fmm_;       ///< fast-multipole
    lgf_lap_t                         lgf_lap_;
    lgf_if_t                          lgf_if_;
    interpolation::cell_center_nli    c_cntr_nli_;///< Lagrange Interpolation
    parallel_ostream::ParallelOstream pcout=parallel_ostream::ParallelOstream(1);

    //Timings:
    struct Timings{
        std::vector<mDuration_type>                level;
        std::vector<typename Fmm_t::Timings>       fmm_level;
        std::vector<typename Fmm_t::Timings>       fmm_level_nl;
        mDuration_type                    global=mDuration_type(0);
        mDuration_type                    coarsification=mDuration_type(0);
        mDuration_type                    level_interaction=mDuration_type(0);
        mDuration_type                    interpolation=mDuration_type(0);

        //TODO
        //Gather all and take max
        void accumulate(boost::mpi::communicator _comm) noexcept
        {
            Timings tlocal=*this;

            std::vector<decltype(tlocal.global.count())> clevel;
            clevel.resize(level.size());
            decltype(tlocal.global.count()) cglobal,ccoarsification,clevel_interaction, cinterpolation;

            
            boost::mpi::all_reduce(_comm,tlocal.global.count(), cglobal,
                    [&](const auto& v0, const auto& v1){return v0>v1? v0  :v1;} );
            boost::mpi::all_reduce(_comm,tlocal.coarsification.count(), ccoarsification,
                    [&](const auto& v0, const auto& v1){return v0>v1? v0  :v1;} );
            boost::mpi::all_reduce(_comm,tlocal.level_interaction.count(), clevel_interaction,
                    [&](const auto& v0, const auto& v1){return v0>v1? v0  :v1;} );
            boost::mpi::all_reduce(_comm,tlocal.interpolation.count(), cinterpolation,
                    [&](const auto& v0, const auto& v1){return v0>v1? v0  :v1;} );

            //For levels:
            for(std::size_t i =0;i<level.size();++i)
            {
                fmm_level[i].accumulate(_comm);
                fmm_level_nl[i].accumulate(_comm);

                boost::mpi::all_reduce(_comm,tlocal.level[i].count(), clevel[i],
                        [&](const auto& v0, const auto& v1){return v0>v1? v0  :v1;} );
                this->level[i]=mDuration_type(clevel[i]);
            }
            this->global=mDuration_type(cglobal);
            this->coarsification=mDuration_type(ccoarsification);
            this->interpolation=mDuration_type(cinterpolation);
            this->level_interaction=mDuration_type(clevel_interaction);
        }

        friend std::ostream& operator<<(std::ostream& os, const  Timings& _t)
        {

            //auto nPts=_t.domain_->get_nPoints();
            //const int nLevels=_t.domain_->nLevels();
            
            const int width=20;
            os<<std::left
                <<std::setw(width) <<"global [ms]"
                <<std::setw(width) <<"coarsification"
                <<std::setw(width) <<"level_interaction "
                <<std::setw(width) <<"interpolation"
            <<std::endl;

            os<<std::left<<std::scientific<<std::setprecision(10)
                <<std::setw(width) <<_t.global.count()
                <<std::setw(width) <<_t.coarsification.count()
                <<std::setw(width) <<_t.level_interaction.count()
                <<std::setw(width) <<_t.interpolation.count()
            <<std::endl;

            os<<"Time per level:"<<std::endl;
            int c=0;
            for(auto& t :  _t.level)
            {
                os<<std::setw(4)<<c++
                  <<std::setw(width)<<t.count()
                <<std::endl;
            }
            os<<std::endl;
            return os;
        } 
    };

    Timings timings_;
};

}

#endif // IBLGF_INCLUDED_POISSON_HPP
