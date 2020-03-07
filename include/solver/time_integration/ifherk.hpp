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
#include <utilities/misc_math_functions.hpp>

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

    using correction   = typename Setup::correction;

    static constexpr int lBuffer=1; ///< Lower left buffer for interpolation
    static constexpr int rBuffer=1; ///< Lower left buffer for interpolation

    Ifherk(simulation_type* _simulation)
    :
    simulation_(_simulation),
    domain_(_simulation->domain_.get()),
    psolver(_simulation)
    {
        // parameters --------------------------------------------------------

        dx_base_          = domain_->dx_base();
        max_ref_level_    = _simulation->dictionary()->template get<float_type>("nLevels");
        cfl_              = _simulation->dictionary()->template get_or<float_type>("cfl",0.2);
        dt_base_          = _simulation->dictionary()->template get_or<float_type>("dt",-1.0);
        tot_base_steps_   = _simulation->dictionary()->template get<int>("nBaseLevelTimeSteps");
        Re_               = _simulation->dictionary()->template get<float_type>("Re");
        output_base_freq_ = _simulation->dictionary()->template get<float_type>("output_frequency");
        cfl_max_          = _simulation->dictionary()->template get_or<float_type>("cfl_max",1000);

        if (dt_base_<0)
            dt_base_ = dx_base_*cfl_;

        // adaptivity --------------------------------------------------------
        adapt_freq_       = _simulation->dictionary()->template get_or<float_type>("adapt_frequency", 1);
        T_max_            = tot_base_steps_*dt_base_;
        update_marching_parameters();

        // restart -----------------------------------------------------------
        write_restart_=_simulation->dictionary()->template get_or<bool>("write_restart",true);

        if (write_restart_)
            restart_base_freq_ = _simulation->dictionary()->template get<float_type>("restart_write_frequency");

        // IF constants ------------------------------------------------------
       fname_prefix_="";

        // miscs -------------------------------------------------------------
    }

public:
    void update_marching_parameters()
    {
        nLevelRefinement_ = domain_->tree()->depth()-domain_->tree()->base_level()-1;
        dt_               = dt_base_/math::pow2(nLevelRefinement_);

        float_type tmp = Re_*dx_base_*dx_base_/dt_;
        alpha_[0]=(c_[1]-c_[0])/tmp;
        alpha_[1]=(c_[2]-c_[1])/tmp;
        alpha_[2]=(c_[3]-c_[2])/tmp;
 }
    void time_march(bool use_restart=false)
    {
        use_restart_=use_restart;
        boost::mpi::communicator world;
        parallel_ostream::ParallelOstream pcout=parallel_ostream::ParallelOstream(world.size()-1);

        pcout<<"Time marching ------------------------------------------------ "<< std::endl;
        // --------------------------------------------------------------------
        if (use_restart_)
        {
            Dictionary info_d(simulation_->restart_load_dir()+"/restart_info");
            T_=info_d.template get<float_type>("T");
            source_max_=info_d.template get<float_type>("source_max");
        }
        else
        {
            T_ = 0.0;
        }

        // ----------------------------------- start -------------------------

        clean_up_initial_velocity();
        write_timestep();

        int adapt_count=0;
        while(T_<=T_max_+1e-10)
        {

            // balance load
            if ( (adapt_count-1) % adapt_freq_ ==0)
            {
                domain_->decomposition().template balance<u,p>();
            }

            // -------------------------------------------------------------
            // time marching
            if(domain_->is_client())
            {
                mDuration_type ifherk_if(0);
                TIME_CODE( ifherk_if, SINGLE_ARG(
                            time_step();
                            ));
                pcout<<ifherk_if.count()<<std::endl;
            }


            // -------------------------------------------------------------
            // adapt

            // clean up the block boundary of cell_aux for smoother adaptation

            if(domain_->is_client())
                clean<cell_aux>(true, 2);

            if (T_<=1e-5)
                this->template update_source_max<cell_aux>();

            if ( adapt_count % adapt_freq_ ==0)
            {
                if(domain_->is_client())
                {
                    up_and_down<u>();
                    pad_velocity<u, u>();
                }
                this->template adapt<u, cell_aux>(false);

            }
            adapt_count++;


            // -------------------------------------------------------------
            // update stats & output
            update_marching_parameters();

            T_ += dt_;
            float_type tmp_n=T_/dt_base_*math::pow2(max_ref_level_);
            int tmp_int_n=int(tmp_n+0.5);

            if ( write_restart_ && ( std::fabs(tmp_int_n - tmp_n)<1e-4 ) && (tmp_int_n%restart_base_freq_==0) )
            {
                restart_n_last_=tmp_int_n;
                write_restart();
            }

            if ( ( std::fabs(tmp_int_n - tmp_n)<1e-4 ) && (tmp_int_n%output_base_freq_==0) )
            {
                n_step_= tmp_int_n;
                write_timestep();
            }

            world.barrier();
            if (domain_->is_server())
            {
                std::cout<<"T = " << T_<<", n = "<< tmp_int_n << " -----------------" << std::endl;
                std::cout<<"Total number of leaf octants: "<<domain_->num_leafs()<<std::endl;
            }

        }

    }
    void clean_up_initial_velocity()
    {
        if(domain_->is_client())
        {

            up_and_down<u>();
            auto client = domain_->decomposition().client();
            clean<edge_aux>();
            clean<stream_f>();
            for (int l  = domain_->tree()->base_level();
                    l < domain_->tree()->depth(); ++l)
            {
                client->template buffer_exchange<u>(l);
                for (auto it  = domain_->begin(l);
                        it != domain_->end(l); ++it)
                {
                    if(!it->locally_owned() || it->is_correction()) continue;

                    const auto dx_level =  dx_base_/math::pow2(it->refinement_level());
                    domain::Operator::curl<u,edge_aux>( *(it->data()),dx_level);
                }
            }
            //clean_leaf_correction_boundary<edge_aux>(domain_->tree()->base_level(), true,2);

            clean<u>();
            psolver.template apply_lgf<edge_aux, stream_f>();
            for (int l  = domain_->tree()->base_level();
                    l < domain_->tree()->depth(); ++l)
            {
                for (auto it  = domain_->begin(l);
                        it != domain_->end(l); ++it)
                {
                    if(!it->locally_owned() ) continue;

                    const auto dx_level =  dx_base_/math::pow2(it->refinement_level());
                    domain::Operator::curl_transpose<stream_f,u>( *(it->data()),dx_level, -1.0);
                }
                client->template buffer_exchange<u>(l);
            }
        }

    }

    template<class Field>
    void update_source_max()
    {
        float_type max_local=0.0;
        for (auto it  = domain_->begin();
                  it != domain_->end(); ++it)
        {
            if (!it->locally_owned()) continue;
            float_type tmp=
                    domain::Operator::maxabs<Field>(*(it->data()));

            if (tmp>max_local)
                max_local=tmp;
        }

        boost::mpi::all_reduce(comm_,max_local,source_max_,[&](const auto& v0,
                    const auto& v1){return v0>v1? v0  :v1;} );
    }

    void write_restart()
    {
        boost::mpi::communicator world;

        world.barrier();
        if (domain_->is_server() && write_restart_)
        {
            std::cout<<"restart: backup" << std::endl;
            simulation_->copy_restart();
        }
        world.barrier();

        pcout<<"restart: write" << std::endl;
        simulation_->write2("", true);

        if (domain_->is_server())
        {
            simulation_->write_tree();
        }
        write_info();
    }

    void write_timestep()
    {
        boost::mpi::communicator world;
        pcout << "- writing at T = " << T_ << ", n = "<< n_step_ << std::endl;
        simulation_->write2(fname(n_step_));
        //simulation_->domain()->tree()->write("tree_restart.bin");
        world.barrier();
        //simulation_->domain()->tree()->read("tree_restart.bin");
        pcout << "- output writing finished -" << std::endl;
    }

    void write_info()
    {
        if (domain_->is_server())
        {
            std::ofstream ofs(simulation_->restart_write_dir()+"/restart_info", std::ofstream::out);
            if(!ofs.is_open())
            {
                throw std::runtime_error("Could not open file for info write " );
            }

            ofs.precision(20);
            ofs<<"T = " << T_ << ";" << std::endl;
            ofs<<"source_max = " << source_max_ << ";" << std::endl;
            ofs<<"restart_n_last = " << restart_n_last_ << ";" << std::endl;

            ofs.close();
        }
    }

    std::string fname(int _n)
    {
        return fname_prefix_+"ifherk_"+std::to_string(_n)+".hdf5";
    }

    // ----------------------------------------------------------------------
    template<class Field>
    void up_and_down()
    {
        //claen non leafs
        clean<Field>(true);
        this->up<Field>();
        this->down_to_correction<Field>();
    }

    template<class Field>
    void up()
    {
        //Coarsification:
        for (std::size_t _field_idx=0; _field_idx<Field::nFields; ++_field_idx)
            psolver.template source_coarsify<Field,Field>(_field_idx, _field_idx, Field::mesh_type, false, false, false, true);
    }

    template<class Field>
    void down_to_correction()
    {
       // Interpolate to correction buffer
        for (std::size_t _field_idx=0; _field_idx<Field::nFields; ++_field_idx)
            psolver.template intrp_to_correction_buffer<Field, Field>(_field_idx, _field_idx, Field::mesh_type, true, false, true);
    }

    template<class AdaptField, class CriterionField>
    void adapt(bool coarsify_field=true)
    {
        boost::mpi::communicator world;
        auto client = domain_->decomposition().client();

        //adaptation neglect the boundary oscillations
        clean_leaf_correction_boundary<cell_aux>(domain_->tree()->base_level(),true);

        world.barrier();

        if (coarsify_field)
        {
            pcout<< "Adapt - coarsify"  << std::endl;
            if (client)
            {
                //claen non leafs
                clean<AdaptField>(true);

                //Coarsification:
                for (std::size_t _field_idx=0; _field_idx<AdaptField::nFields; ++_field_idx)
                    psolver.template source_coarsify<AdaptField,AdaptField>(_field_idx, _field_idx, AdaptField::mesh_type);

            }
        }

        world.barrier();
        pcout<< "Adapt - communication"  << std::endl;
        auto intrp_list = domain_->template adapt<CriterionField>(source_max_);

        world.barrier();
        pcout<< "Adapt - intrp"  << std::endl;
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

                for (auto& oct:intrp_list)
                {
                    if (!oct || !oct->data()) continue;
                    psolver.c_cntr_nli().template nli_intrp_node<AdaptField, AdaptField>(oct, AdaptField::mesh_type, _field_idx, _field_idx, false, false);
                }
            }
        }

        //test correction --------------------------------------------------
        //for (std::size_t _field_idx=0; _field_idx<correction::nFields; ++_field_idx)
        //{

        //    for (int l = domain_->tree()->depth()-1;
        //            l >= domain_->tree()->base_level(); --l)
        //    {
        //        for (auto it=domain_->begin(l); it!=domain_->end(l); ++it)
        //        {
        //            if (!it->data() || !it->data()->is_allocated()) continue;
        //            auto& lin_data = it->data()->
        //                template get_linalg_data<correction>(_field_idx);
        //            if (it->is_correction())
        //                std::fill(lin_data.begin(),lin_data.end(),-1000.0);
        //            else
        //                std::fill(lin_data.begin(),lin_data.end(),0.0);
        //        }
        //    }
        //}

        // for (std::size_t _field_idx=0; _field_idx<correction::nFields; ++_field_idx)
        //{

        //    for (int l = domain_->tree()->depth()-2;
        //            l >= domain_->tree()->base_level(); --l)
        //    {
        //        for (auto it=domain_->begin(l); it!=domain_->end(l); ++it)
        //        {

        //            for(int c=0;c<it->num_children();++c)
        //            {
        //                const auto child = it->child(c);
        //                if(!child || !child->data() || !child->locally_owned() || !child->is_correction() ) continue;

        //                auto& lin_data = child->data()->
        //                    template get_linalg_data<correction>(_field_idx);

        //                std::fill(lin_data.begin(),lin_data.end(),-1000.0);
        //            }
        //        }
        //    }

        //    for (std::size_t _field_idx=0; _field_idx<correction::nFields; ++_field_idx)
        //        psolver.template source_coarsify<correction,correction>(_field_idx, _field_idx, correction::mesh_type, true, false);
        //}

        world.barrier();
        pcout<< "Adapt - done"  << std::endl;
    }


    void time_step()
    {
        // Initialize IFHERK
        // q_1 = u
        boost::mpi::communicator world;
        auto client=domain_->decomposition().client();

        ////claen non leafs
        up_and_down<u>();

        // Solve stream function to pad base level u->u_pad
        stage_idx_=0;
        mDuration_type t_pad(0);
        TIME_CODE( t_pad, SINGLE_ARG(
                    pad_velocity<u, u>();
                    ));
        pcout<< "pad u      in "<<t_pad.count() << std::endl;


        copy<u, q_i>();

        // Stage 1
        // ******************************************************************
        pcout<<"Stage 1"<< std::endl;
        stage_idx_=1;
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
        stage_idx_=2;
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
        stage_idx_=3;
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
    float_type coeff_a(int i, int j)const noexcept {return a_[i*(i-1)/2+j-1];}


    void lin_sys_solve(float_type _alpha) noexcept
    {
        auto client=domain_->decomposition().client();

        divergence<r_i, cell_aux>();

        mDuration_type t_lgf(0);
        TIME_CODE( t_lgf, SINGLE_ARG(
                    psolver.template apply_lgf<cell_aux, d_i>();
                    ));
        pcout<< "LGF solved in "<<t_lgf.count() << std::endl;

        gradient<d_i,face_aux>();
        add<face_aux, r_i>(-1.0);
        if (std::fabs(_alpha)>1e-4)
        {
            mDuration_type t_if(0);
            TIME_CODE( t_if, SINGLE_ARG(
                        psolver.template apply_lgf_IF<r_i, u_i>(_alpha);
                        ));
            pcout<< "IF  solved in "<<t_if.count() << std::endl;
        }
        else
            copy<r_i,u_i>();
    }


    template <typename F>
    void clean(bool non_leaf_only=false, int clean_width=1) noexcept
    {
        for (auto it  = domain_->begin();
                  it != domain_->end(); ++it)
        {
            if (!it->data()) continue;
            if (!it->data()->is_allocated())continue;

            for (std::size_t field_idx=0; field_idx<F::nFields; ++field_idx)
            {
                auto& lin_data = it->data()->template get_linalg_data<F>(field_idx);

                if (non_leaf_only && it->is_leaf() && it->locally_owned() )
                {
                    int N=it->data()->descriptor().extent()[0];

                    view(lin_data,xt::all(),xt::all(),xt::range(0,clean_width))  *= 0.0;
                    view(lin_data,xt::all(),xt::range(0,clean_width),xt::all())  *= 0.0;
                    view(lin_data,xt::range(0,clean_width),xt::all(),xt::all())  *= 0.0;
                    view(lin_data,xt::range(N+2-clean_width,N+3),xt::all(),xt::all())  *= 0.0;
                    view(lin_data,xt::all(),xt::range(N+2-clean_width,N+3),xt::all())  *= 0.0;
                    view(lin_data,xt::all(),xt::all(),xt::range(N+2-clean_width,N+3))  *= 0.0;
                }
                else
                {

                    //TODO whether to clean base_level correction?
                    std::fill(lin_data.begin(),lin_data.end(),0.0);
                }

            }
        }
    }


    template<class Velocity_in, class Velocity_out>
    void pad_velocity(bool _exchange_buffer=true)
    {
        auto client=domain_->decomposition().client();

        clean<edge_aux>();
        clean<stream_f>();

        auto dx_base = domain_->dx_base();

        for (int l  = domain_->tree()->base_level();
                l < domain_->tree()->depth(); ++l)
        {
            if (_exchange_buffer)
                client->template buffer_exchange<Velocity_in>(l);

            for (auto it  = domain_->begin(l);
                    it != domain_->end(l); ++it)
            {
                if(!it->locally_owned() || !it->data()) continue;
                if(it->is_correction()) continue;
                if(!it->is_leaf()) continue;

                const auto dx_level =  dx_base/math::pow2(it->refinement_level());
                if (it->is_leaf())
                    domain::Operator::curl<Velocity_in,edge_aux>( *(it->data()),dx_level);
            }
        }

        int l  = domain_->tree()->base_level();
        clean_leaf_correction_boundary<edge_aux>(l, true,2);
        //clean_leaf_correction_boundary<edge_aux>(l, false,2+stage_idx_);
        psolver.template apply_lgf<edge_aux, stream_f>(true);

        for (auto it  = domain_->begin(l);
                it != domain_->end(l); ++it)
        {
            if(!it->locally_owned() || !it->data()) continue;
            if(!it->is_correction()) continue;

            const auto dx_level =  dx_base/math::pow2(it->refinement_level());
            domain::Operator::curl_transpose<stream_f,Velocity_out>( *(it->data()),dx_level, -1.0);
        }

        //client->template buffer_exchange<Velocity_out>(l);

    }


    template <typename F>
    void clean_leaf_correction_boundary(int l, bool leaf_only_boundary=false, int clean_width=1) noexcept
    {
        for (auto it  = domain_->begin(l);
                it != domain_->end(l); ++it)
        {
            if(!it->locally_owned())
            {
                if(!it->data() || !it->data()->is_allocated()) continue;
                for (std::size_t field_idx=0;
                        field_idx<F::nFields; ++field_idx)
                {
                    auto& lin_data = it ->data()->
                        template get_linalg_data<F>(field_idx);

                    std::fill(lin_data.begin(),lin_data.end(),0.0);
                }
            }
        }

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

        //std::cout<< "-------------------------------------------------------"<<std::endl;
        if (l==domain_->tree()->base_level())
        for (auto it  = domain_->begin(l);
                it != domain_->end(l); ++it)
        {
            if(!it->locally_owned()) continue;
            if(!it->data() || !it->data()->is_allocated()) continue;
            //std::cout<<it->key()<<std::endl;

            for(std::size_t i=0;i< it->num_neighbors();++i)
            {
                auto it2=it->neighbor(i);
                //if (it2)
                //    std::cout<<i<<it2->key()<<std::endl;
                if ((!it2 || !it2->data()) || (leaf_only_boundary && it2->is_correction()))
                {
                    for (std::size_t field_idx=0; field_idx<F::nFields; ++field_idx)
                    {
                        auto& lin_data = it->data()->
                            template get_linalg_data<F>(field_idx);

                        int N=it->data()->descriptor().extent()[0];

                        // somehow we delete the outer 2 planes
                        if (i==4)
                            view(lin_data,xt::all(),xt::all(),xt::range(0,clean_width))  *= 0.0;
                        else if (i==10)
                            view(lin_data,xt::all(),xt::range(0,clean_width),xt::all())  *= 0.0;
                        else if (i==12)
                            view(lin_data,xt::range(0,clean_width),xt::all(),xt::all())  *= 0.0;
                        else if (i==14)
                            view(lin_data,xt::range(N+2-clean_width,N+3),xt::all(),xt::all())  *= 0.0;
                        else if (i==16)
                            view(lin_data,xt::all(),xt::range(N+2-clean_width,N+3),xt::all())  *= 0.0;
                        else if (i==22)
                            view(lin_data,xt::all(),xt::all(),xt::range(N+2-clean_width,N+3))  *= 0.0;
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
                //if(it->is_correction()) continue;

                const auto dx_level =  dx_base/math::pow2(it->refinement_level());
                domain::Operator::curl<Source,edge_aux>( *(it->data()),dx_level);
            }

            client->template buffer_exchange<edge_aux>(l);
            clean_leaf_correction_boundary<edge_aux>(l, false,2+stage_idx_);

            for (auto it  = domain_->begin(l);
                    it != domain_->end(l); ++it)
            {
                if(!it->locally_owned() || !it->data())continue;
                //if(it->is_correction()) continue;

                domain::Operator::nonlinear<Source,edge_aux,Target>
                    ( *(it->data()));

                for (std::size_t field_idx=0; field_idx<Target::nFields; ++field_idx)
                {
                    auto& lin_data = it->data()->
                        template get_linalg_data<Target>(field_idx);

                    lin_data *= _scale;
                }

            }

            //client->template buffer_exchange<Target>(l);
            clean_leaf_correction_boundary<Target>(l, false,3+stage_idx_);
        }
    }


    template<class Source, class Target>
    void divergence() noexcept
    {
        auto client=domain_->decomposition().client();

        up_and_down<Source>();

        for (int l  = domain_->tree()->base_level();
                l < domain_->tree()->depth(); ++l)
        {
            client->template buffer_exchange<Source>(l);
            const auto dx_base = domain_->dx_base();

            for (auto it  = domain_->begin(l);
                    it != domain_->end(l); ++it)
            {
                if(!it->locally_owned() || !it->data()) continue;
                const auto dx_level =  dx_base/math::pow2(it->refinement_level());
                domain::Operator::divergence<Source,Target>( *(it->data()),dx_level);
            }

            //client->template buffer_exchange<Target>(l);
            //clean_leaf_correction_boundary<Target>(l, true, 1+stage_idx_);
            clean_leaf_correction_boundary<Target>(l, false,4+stage_idx_);
        }
    }


    template<class Source, class Target>
    void gradient(float_type _scale=1.0) noexcept
    {
        //up_and_down<Source>();

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
                const auto dx_level =  dx_base/math::pow2(it->refinement_level());
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

    float_type T_, T_max_;
    float_type dt_base_, dt_, dx_base_;
    float_type Re_;
    float_type cfl_max_, cfl_;
    float_type source_max_;
    int max_ref_level_=0;
    int output_base_freq_;
    int adapt_freq_;
    int tot_base_steps_;
    int n_step_=0;
    int restart_n_last_=0;
    int nLevelRefinement_;
    int stage_idx_=0;

    bool use_restart_=false;
    bool write_restart_=false;
    int  restart_base_freq_;

    std::string fname_prefix_;
    vector_type<float_type, 6> a_{{1.0/3, -1.0, 2.0, 0.0, 0.75, 0.25}};
    vector_type<float_type, 4> c_{{0.0, 1.0/3, 1.0, 1.0}};
    vector_type<float_type, 3> alpha_{{0.0,0.0,0.0}};
    parallel_ostream::ParallelOstream pcout=parallel_ostream::ParallelOstream(1);
    boost::mpi::communicator            comm_;
};

}

#endif // IBLGF_INCLUDED_POISSON_HPP
