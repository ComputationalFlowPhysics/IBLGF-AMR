#ifndef IBLGF_INCLUDED_IFHERK_SOLVER_HPP
#define IBLGF_INCLUDED_IFHERK_SOLVER_HPP

#include <iostream>
#include <algorithm>
#include <vector>
#include <cmath>
#include <array>

// IBLGF-specific
#include <global.hpp>
#include <simulation.hpp>
#include <domain/domain.hpp>
#include <IO/parallel_ostream.hpp>
#include <solver/poisson/poisson.hpp>
#include <operators/operators.hpp>

namespace solver
{

using namespace domain;

/** @brief Integrating factor 3-stage Runge-Kutta time integration
 * */
template<class Setup>
class Ifherk
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
    using poisson_solver_t     = typename Setup::poisson_solver_t;

    //Fields
    //using coarse_target_sum = typename Setup::coarse_target_sum;
    //using source_tmp = typename Setup::source_tmp;

    //FMM
    using Fmm_t     = typename Setup::Fmm_t;

    using u   = typename Setup::u;
    using stream_f   = typename Setup::stream_f;
    using p   = typename Setup::p;
    using q_i = typename Setup::q_i;
    using r_i = typename Setup::r_i;
    using g_i = typename Setup::g_i;
    using d_i = typename Setup::d_i;

    using cell_aux   = typename Setup::cell_aux;
    using edge_aux   = typename Setup::edge_aux;
    using face_aux   = typename Setup::face_aux;
    using w_1        = typename Setup::w_1;
    using w_2        = typename Setup::w_2;
    using u_i        = typename Setup::u_i;


    static constexpr int lBuffer=1; ///< Lower left buffer for interpolation
    static constexpr int rBuffer=1; ///< Lower left buffer for interpolation

    Ifherk(simulation_type* _simulation)
    :
    simulation_(_simulation),
    domain_(_simulation->domain_.get())
    //psolver(_simulation)
    {
        std::cout<< "IFHERK initialized --- "  << std::endl;
        dx_          = domain_->dx_base();
        cfl_         = _simulation->dictionary()->
            template get_or<float_type>("cfl",0.2);
        dt_          = _simulation->dictionary()->
            template get_or<float_type>("dt",-1.0);
        tot_steps_   = _simulation->dictionary()->
            template get<int>("nBaseLevelTimeSteps");
        cfl_max_     = _simulation->dictionary()->
            template get_or<float_type>("cfl_max",1000);
        Re_          = _simulation->dictionary()->
            template get<float_type>("Re");
        output_freq_ = _simulation->dictionary()->
            template get<float_type>("output_frequency");
        nLevelRefinement_=_simulation->dictionary_->
            template get_or<int>("nLevels",0);
        if (dt_<0)
            dt_ = dx_*cfl_;

        dt_/=pow(2.0,nLevelRefinement_);
        tot_steps_ *= pow(2,nLevelRefinement_);
        output_freq_ *= pow(2,nLevelRefinement_);

        float_type tmp = Re_*dx_*dx_/dt_;

        alpha_[0]=(c_[1]-c_[0])/tmp;
        alpha_[1]=(c_[2]-c_[1])/tmp;
        alpha_[2]=(c_[3]-c_[2])/tmp;
        fname_prefix_="";
    }

public:
    void time_march()
    {
        boost::mpi::communicator world;
        parallel_ostream::ParallelOstream pcout=parallel_ostream::ParallelOstream(world.size()-1);

        T_ = 0.0;
        n_step_=0;

        pcout<<"Time marching with dt = " << dt_ << std::endl;
        pcout<<"                   nsteps = " << tot_steps_ << std::endl;


        write_timestep();
        for (int i=0; i<tot_steps_; ++i)
        {
            if(domain_->is_client())
            {
                mDuration_type ifherk_if(0);
                TIME_CODE( ifherk_if, SINGLE_ARG(
                            time_step();
                            ));
                pcout<<ifherk_if.count()<<std::endl;
            }

            // Add dt to Time
            T_ += dt_;
            n_step_ = round(T_ / dt_);
            pcout<<"T = " << T_ << " -----------------" << std::endl;

            world.barrier();
            if ( n_step_ % output_freq_ == 0)
                write_timestep();
        }

    }
    void write_timestep()
    {
        pcout << "- writing at T = " << T_ << std::endl;
        simulation_->write2(fname(n_step_));
        pcout << "- finishing writing " << std::endl;
    }

    std::string fname(int _n)
    {
        return fname_prefix_+"ifherk_"+std::to_string(_n)+".hdf5";
    }

    template<class Field>
    void up_and_down()
    {
        //claen non leafs
        clean<Field>(true);
        //Coarsification:
        for (std::size_t _field_idx=0; _field_idx<Field::nFields; ++_field_idx)
            psolver.template source_coarsify<Field,Field>(_field_idx, _field_idx, Field::mesh_type);
        // Interpolate to correction buffer
        for (std::size_t _field_idx=0; _field_idx<Field::nFields; ++_field_idx)
            psolver.template intrp_to_correction_buffer<Field, Field>(_field_idx, _field_idx, Field::mesh_type, true, false);
    }

    template<class AdaptField, class CriterionField>
    void adapt()
    {
        auto client = domain_->decomposition().client();

        if (client)
        {
            //claen non leafs
            clean<AdaptField>(true);

            //Coarsification:
            for (std::size_t _field_idx=0; _field_idx<AdaptField::nFields; ++_field_idx)
                psolver.template source_coarsify<AdaptField,AdaptField>(_field_idx, _field_idx, AdaptField::mesh_type);

        }

        auto intrp_list = domain_->template adapt<CriterionField>();

        if (client)
        {

            // Intrp
            for (std::size_t _field_idx=0; _field_idx<AdaptField::nFields; ++_field_idx)
            {

                for (int l = domain_->tree()->depth()-2;
                        l >= domain_->tree()->base_level(); --l)
                {
                    client->template buffer_exchange<AdaptField>(l);

                    domain_->decomposition().client()->
                    template communicate_updownward_assign
                    <AdaptField, AdaptField>(l,false,false,-1,_field_idx);
                }

                for (auto oct:intrp_list)
                {
                    psolver.c_cntr_nli_.template nli_intrp_node<AdaptField, AdaptField>(oct, AdaptField::mesh_type, _field_idx, _field_idx, false, false);
                }
            }
        }
    }


    void time_step()
    {
        // Initialize IFHERK
        // q_1 = u
        boost::mpi::communicator world;
        auto client=domain_->decomposition().client();

        ////claen non leafs
        clean<u>(true);
        up_and_down<u>();

        // Solve stream function to pad base level u->u_pad
        pad_velocity<u, u>();
        copy<u, q_i>();

        // Stage 1
        // ******************************************************************
        pcout<<"Stage 1"<< std::endl;
        clean<g_i>();
        clean<d_i>();
        clean<cell_aux>();
        clean<face_aux>();

        nonlinear<u,g_i>(coeff_a(1,1)*(-dt_));
        copy<q_i, r_i>();
        add<g_i, r_i>();
        lin_sys_solve(alpha_[0]);


        // Stage 2
        // ******************************************************************
        pcout<<"Stage 2"<< std::endl;
        clean<r_i>();
        clean<d_i>();
        clean<cell_aux>();

        //cal wii
        //r_i = q_i + dt(a21 w21)
        //w11 = (1/a11)* dt (g_i - face_aux)

        add<g_i, face_aux>(-1.0);
        copy<face_aux, w_1>(-1.0/dt_/coeff_a(1,1));

        psolver.template apply_lgf_IF<q_i, q_i>(alpha_[0]);
        psolver.template apply_lgf_IF<w_1, w_1>(alpha_[0]);

        add<q_i, r_i>();
        add<w_1, r_i>(dt_*coeff_a(2,1));

        up_and_down<u_i>();
        nonlinear<u_i,g_i>(coeff_a(2,2)*(-dt_));
        add<g_i, r_i>( );

        lin_sys_solve(alpha_[1]);

        // Stage 3
        // ******************************************************************
        pcout<<"Stage 3"<< std::endl;
        clean<d_i>();
        clean<cell_aux>();
        clean<w_2>();

        add<g_i, face_aux>(-1.0);
        copy<face_aux, w_2>(-1.0/dt_/coeff_a(2,2));
        copy<q_i, r_i>();
        add<w_1, r_i>(dt_*coeff_a(3,1));
        add<w_2, r_i>(dt_*coeff_a(3,2));

        psolver.template apply_lgf_IF<r_i, r_i>(alpha_[1]);

        up_and_down<u_i>();
        nonlinear<u_i,g_i>( coeff_a(3,3)*(-dt_) );
        add<g_i, r_i>();

        lin_sys_solve(alpha_[2]);

        // ******************************************************************
        copy<u_i, u>();
        copy<d_i, p>(1.0/coeff_a(3,3)/dt_);
        // ******************************************************************

    }


private:
    void lin_sys_solve(float_type _alpha) noexcept
    {
         divergence<r_i, cell_aux>();
         psolver.template apply_lgf<cell_aux, d_i>();
         gradient<d_i,face_aux>();

         add<face_aux, r_i>(-1.0);
         if (std::fabs(_alpha)>1e-4)
             psolver.template apply_lgf_IF<r_i, u_i>(_alpha);
         else
             copy<r_i,u_i>();
    }

    float_type coeff_a(int i, int j)const noexcept {return a_[i*(i-1)/2+j-1];}

    template <typename F>
    void clean(bool non_leaf_only=false) noexcept
    {
        for (auto it  = domain_->begin();
                  it != domain_->end(); ++it)
        {
            if (!it->data()) continue;
            if (!it ->data()->is_allocated())continue;


            for (std::size_t field_idx=0; field_idx<F::nFields; ++field_idx)
            {
                auto& lin_data = it->data()->
                template get_linalg_data<F>(field_idx);

                int N=it->data()->descriptor().extent()[0];

                xt::noalias( view(lin_data,xt::all(),xt::all(),0)) *= 0.0;
                xt::noalias( view(lin_data,xt::all(),0,xt::all())) *= 0.0;
                xt::noalias( view(lin_data,0,xt::all(),xt::all())) *= 0.0;
                xt::noalias( view(lin_data,N+1,xt::all(),xt::all())) *= 0.0;
                xt::noalias( view(lin_data,xt::all(),N+1,xt::all())) *= 0.0;
                xt::noalias( view(lin_data,xt::all(),xt::all(),N+1)) *= 0.0;

                if (non_leaf_only && it->is_leaf() && it->locally_owned() && !it->is_correction()) continue;
                std::fill(lin_data.begin(),lin_data.end(),0.0);

            }
        }
    }

    template<class Velocity_in, class Velocity_out>
    void pad_velocity()
    {
        auto client=domain_->decomposition().client();
        clean<edge_aux>();

        const auto dx_base = domain_->dx_base();


        for (int l  = domain_->tree()->base_level();
                l < domain_->tree()->depth(); ++l)
        {
            client->template buffer_exchange<Velocity_in>(l);
            for (auto it  = domain_->begin(l);
                    it != domain_->end(l); ++it)
            {
                if(!it->locally_owned() || !it->data()) continue;
                if(it->is_correction()) continue;

                const auto dx_level =  dx_base/std::pow(2,it->refinement_level());
                if (it->is_leaf())
                    domain::Operator::curl<Velocity_in,edge_aux>( *(it->data()),dx_level);
            }
        }

        const int l  = domain_->tree()->base_level();

        clean_leaf_correction_boundary<edge_aux>(l, true);

        psolver.template apply_lgf<edge_aux, stream_f>(true);

        for (auto it  = domain_->begin(l);
                it != domain_->end(l); ++it)
        {
            if(!it->locally_owned() || !it->data()) continue;
            if(!it->is_correction()) continue;

            const auto dx_level =  dx_base/std::pow(2,it->refinement_level());
            domain::Operator::curl_transpose<stream_f,Velocity_out>( *(it->data()),dx_level, -1.0);
        }
        client->template buffer_exchange<Velocity_out>(l);

    }

    template <typename F>
    void clean_leaf_correction_boundary(int l, bool leaf_only_boundary=false) noexcept
    {
        for (auto it  = domain_->begin(l);
                it != domain_->end(l); ++it)
        {
                if(!it->locally_owned()) continue;
                if(!it->data() || !it->data()->is_allocated()) continue;

                if (leaf_only_boundary && it->is_correction())
                {
                    for (std::size_t field_idx=0;
                            field_idx<F::nFields; ++field_idx)
                    {
                        auto& lin_data = it ->data()->
                        template get_linalg_data<F>(field_idx);

                        std::fill(lin_data.begin(),lin_data.end(),0.0);
                    }
                }
        }

        if (l==domain_->tree()->base_level())
        for (auto it  = domain_->begin(l);
                it != domain_->end(l); ++it)
        {
            if(!it->locally_owned()) continue;
            if(!it->data() || !it->data()->is_allocated()) continue;

            for(std::size_t i=0;i< it->num_neighbors();++i)
            {
                auto it2=it->neighbor(i);
                if (!it2 || (leaf_only_boundary && it2->is_correction()))
                {
                    for (std::size_t field_idx=0; field_idx<F::nFields; ++field_idx)
                    {
                        auto& lin_data = it->data()->
                        template get_linalg_data<F>(field_idx);

                        int N=it->data()->descriptor().extent()[0];

                        if (i==4)
                            xt::noalias( view(lin_data,xt::all(),xt::all(),1)) *= 0.0;
                        else if (i==10)
                            xt::noalias( view(lin_data,xt::all(),1,xt::all())) *= 0.0;
                        else if (i==12)
                            xt::noalias( view(lin_data,1,xt::all(),xt::all())) *= 0.0;
                        else if (i==14)
                            xt::noalias( view(lin_data,N,xt::all(),xt::all())) *= 0.0;
                        else if (i==16)
                            xt::noalias( view(lin_data,xt::all(),N,xt::all())) *= 0.0;
                        else if (i==22)
                            xt::noalias( view(lin_data,xt::all(),xt::all(),N)) *= 0.0;
                    }
                }
            }
        }

    }
    //TODO maybe to be put directly intor operators:
    template<class Source, class Target>
    void nonlinear(float_type _scale=1.0) noexcept
    {
        clean<edge_aux>();
        clean<Target>();

        auto client=domain_->decomposition().client();
        const auto dx_base = domain_->dx_base();

        for (int l  = domain_->tree()->base_level();
                l < domain_->tree()->depth(); ++l)
        {

            client->template buffer_exchange<Source>(l);

            for (auto it  = domain_->begin(l);
                    it != domain_->end(l); ++it)
            {
                if(!it->locally_owned() || !it->data()) continue;
                if(it->is_correction()) continue;

                const auto dx_level =  dx_base/std::pow(2,it->refinement_level());
                domain::Operator::curl<Source,edge_aux>( *(it->data()),dx_level);
            }

            client->template buffer_exchange<edge_aux>(l);
            clean_leaf_correction_boundary<edge_aux>(l, true);

            for (auto it  = domain_->begin(l);
                    it != domain_->end(l); ++it)
            {
                if(!it->locally_owned() || !it->data())continue;
                //const auto dx_level =  dx_base/std::pow(2,it->refinement_level());
                if(it->is_correction()) continue;

                domain::Operator::nonlinear<Source,edge_aux,Target>
                    ( *(it->data()));

                for (std::size_t field_idx=0; field_idx<Target::nFields; ++field_idx)
                {
                    auto& lin_data = it->data()->
                        template get_linalg_data<Target>(field_idx);

                    lin_data *= _scale;
                }

            }

            client->template buffer_exchange<Target>(l);
            clean_leaf_correction_boundary<edge_aux>(l, false);
        }
    }

    template<class Source, class Target>
    void divergence() noexcept
    {
        auto client=domain_->decomposition().client();

        for (int l  = domain_->tree()->base_level();
                l < domain_->tree()->depth(); ++l)
        {
            client->template buffer_exchange<Source>(l);
            const auto dx_base = domain_->dx_base();

            for (auto it  = domain_->begin(l);
                    it != domain_->end(l); ++it)
            {
                if(!it->locally_owned() || !it->data()) continue;
                const auto dx_level =  dx_base/std::pow(2.0,it->refinement_level());
                domain::Operator::divergence<Source,Target>( *(it->data()),dx_level);
            }

            //client->template buffer_exchange<Target>(l);
            clean_leaf_correction_boundary<Target>(l, true);
        }
    }
    template<class Source, class Target>
    void gradient(float_type _scale=1.0) noexcept
    {
        for (int l  = domain_->tree()->base_level();
                l < domain_->tree()->depth(); ++l)
        {
            auto client=domain_->decomposition().client();
            //client->template buffer_exchange<Source>(l);
            const auto dx_base = domain_->dx_base();
            for (auto it  = domain_->begin(l);
                    it != domain_->end(l); ++it)
            {
                if(!it->locally_owned() || !it->data()) continue;
                const auto dx_level =  dx_base/std::pow(2.0,it->refinement_level());
                domain::Operator::gradient<Source,Target>( *(it->data()),dx_level);
                for (std::size_t field_idx=0; field_idx<Target::nFields; ++field_idx)
                {
                    auto& lin_data = it->data()->
                    template get_linalg_data<Target>(field_idx);

                    lin_data *= _scale;
                }
            }
            client->template buffer_exchange<Target>(l);
        }
    }

    template <typename From, typename To>
    void add(float_type scale=1.0) noexcept
    {
        static_assert (From::nFields == To::nFields, "number of fields doesn't match when add");
        for (auto it  = domain_->begin();
                  it != domain_->end(); ++it)
        {
            if(!it->locally_owned() || !it->data()) continue;
            for (std::size_t field_idx=0; field_idx<From::nFields; ++field_idx)
            {

                it->data()->template get_linalg<To>(field_idx).get()->
                    cube_noalias_view() +=
                     it->data()->template get_linalg_data<From>(field_idx) * scale;
            }
        }
    }

    template <typename From, typename To>
    void copy(float_type scale=1.0) noexcept
    {
        static_assert (From::nFields == To::nFields, "number of fields doesn't match when copy");

        for (auto it  = domain_->begin();
                  it != domain_->end(); ++it)
        {
            if(!it->locally_owned() || !it->data()) continue;
            for (std::size_t field_idx=0; field_idx<From::nFields; ++field_idx)
            {
                it->data()->template get_linalg<To>(field_idx).get()->
                    cube_noalias_view() =
                     it->data()->template get_linalg_data<From>(field_idx) * scale;
            }

        }
    }



private:
    simulation_type*  simulation_;
    domain_type* domain_;    ///< domain
    poisson_solver_t psolver;

    float_type T_;
    float_type dt_, dx_;
    float_type Re_;
    float_type cfl_max_, cfl_;
    int output_freq_;
    int tot_steps_;
    int n_step_;
    int nLevelRefinement_;

    std::string fname_prefix_;
    vector_type<float_type, 6> a_{{1.0/3, -1.0, 2.0, 0.0, 0.75, 0.25}};
    vector_type<float_type, 4> c_{{0.0, 1.0/3, 1.0, 1.0}};
    vector_type<float_type, 3> alpha_{{0.0,0.0,0.0}};
    parallel_ostream::ParallelOstream pcout=parallel_ostream::ParallelOstream(1);

};

}

#endif // IBLGF_INCLUDED_POISSON_HPP
