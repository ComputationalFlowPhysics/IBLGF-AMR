//      ▄▄▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄▄▄   ▄            ▄▄▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄▄▄▄
//     ▐░░░░░░░░░░░▌▐░░░░░░░░░░▌ ▐░▌          ▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌
//      ▀▀▀▀█░█▀▀▀▀ ▐░█▀▀▀▀▀▀▀█░▌▐░▌          ▐░█▀▀▀▀▀▀▀▀▀ ▐░█▀▀▀▀▀▀▀▀▀
//          ▐░▌     ▐░▌       ▐░▌▐░▌          ▐░▌          ▐░▌
//          ▐░▌     ▐░█▄▄▄▄▄▄▄█░▌▐░▌          ▐░▌ ▄▄▄▄▄▄▄▄ ▐░█▄▄▄▄▄▄▄▄▄
//          ▐░▌     ▐░░░░░░░░░░▌ ▐░▌          ▐░▌▐░░░░░░░░▌▐░░░░░░░░░░░▌
//          ▐░▌     ▐░█▀▀▀▀▀▀▀█░▌▐░▌          ▐░▌ ▀▀▀▀▀▀█░▌▐░█▀▀▀▀▀▀▀▀▀
//          ▐░▌     ▐░▌       ▐░▌▐░▌          ▐░▌       ▐░▌▐░▌
//      ▄▄▄▄█░█▄▄▄▄ ▐░█▄▄▄▄▄▄▄█░▌▐░█▄▄▄▄▄▄▄▄▄ ▐░█▄▄▄▄▄▄▄█░▌▐░▌
//     ▐░░░░░░░░░░░▌▐░░░░░░░░░░▌ ▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░▌
//      ▀▀▀▀▀▀▀▀▀▀▀  ▀▀▀▀▀▀▀▀▀▀   ▀▀▀▀▀▀▀▀▀▀▀  ▀▀▀▀▀▀▀▀▀▀▀  ▀

#ifndef IBLGF_INCLUDED_IFHERK_SOLVER_HPP
#define IBLGF_INCLUDED_IFHERK_SOLVER_HPP

#include <iostream>
#include <algorithm>
#include <vector>
#include <cmath>
#include <array>

// IBLGF-specific
#include <iblgf/global.hpp>
#include <iblgf/simulation.hpp>
#include <iblgf/domain/domain.hpp>
#include <iblgf/IO/parallel_ostream.hpp>
#include <iblgf/solver/poisson/poisson.hpp>
#include <iblgf/solver/linsys/linsys.hpp>
#include <iblgf/operators/operators.hpp>
#include <iblgf/utilities/misc_math_functions.hpp>

namespace iblgf
{
namespace solver
{
using namespace domain;

/** @brief Integrating factor 3-stage Runge-Kutta time integration
 * */
template<class Setup>
class Ifherk
{
  public: //member types
    using simulation_type = typename Setup::simulation_t;
    using domain_type = typename simulation_type::domain_type;
    using datablock_type = typename domain_type::datablock_t;
    using tree_t = typename domain_type::tree_t;
    using octant_t = typename tree_t::octant_type;
    using MASK_TYPE = typename octant_t::MASK_TYPE;
    using block_type = typename datablock_type::block_descriptor_type;
    using real_coordinate_type = typename domain_type::real_coordinate_type;
    using coordinate_type = typename domain_type::coordinate_type;
    using poisson_solver_t = typename Setup::poisson_solver_t;
    using linsys_solver_t = typename Setup::linsys_solver_t;

    using ib_t = typename domain_type::ib_t;
    using force_type = typename ib_t::force_type;

    //FMM
    using Fmm_t = typename Setup::Fmm_t;

    using test_type = typename Setup::test_type;

    using u_type = typename Setup::u_type;
    using stream_f_type = typename Setup::stream_f_type;
    using p_type = typename Setup::p_type;
    using q_i_type = typename Setup::q_i_type;
    using r_i_type = typename Setup::r_i_type;
    using g_i_type = typename Setup::g_i_type;
    using d_i_type = typename Setup::d_i_type;

    using cell_aux_type = typename Setup::cell_aux_type;
    using edge_aux_type = typename Setup::edge_aux_type;
    using face_aux_type = typename Setup::face_aux_type;
    using face_aux2_type = typename Setup::face_aux2_type;
    using correction_tmp_type = typename Setup::correction_tmp_type;
    using w_1_type = typename Setup::w_1_type;
    using w_2_type = typename Setup::w_2_type;
    using u_i_type = typename Setup::u_i_type;

    static constexpr int lBuffer = 1; ///< Lower left buffer for interpolation
    static constexpr int rBuffer = 1; ///< Lower left buffer for interpolation
    Ifherk(simulation_type* _simulation)
    : simulation_(_simulation)
    , domain_(_simulation->domain_.get())
    , psolver(_simulation)
    , lsolver(_simulation)
    {
        // parameters --------------------------------------------------------

        dx_base_ = domain_->dx_base();
        max_ref_level_ =
            _simulation->dictionary()->template get<float_type>("nLevels");
        cfl_ =
            _simulation->dictionary()->template get_or<float_type>("cfl", 0.2);
        dt_base_ =
            _simulation->dictionary()->template get_or<float_type>("dt", -1.0);
        tot_base_steps_ =
            _simulation->dictionary()->template get<int>("nBaseLevelTimeSteps");
        Re_ = _simulation->dictionary()->template get<float_type>("Re");
        output_base_freq_ = _simulation->dictionary()->template get<float_type>(
            "output_frequency");
        cfl_max_ = _simulation->dictionary()->template get_or<float_type>(
            "cfl_max", 1000);
        updating_source_max_ = _simulation->dictionary()->template get_or<bool>(
            "updating_source_max", true);
        all_time_max_ = _simulation->dictionary()->template get_or<bool>(
            "all_time_max", true);

        use_adaptation_correction = _simulation->dictionary()->template get_or<bool>(
            "use_adaptation_correction", true);

        b_f_mag = _simulation->dictionary()->template get_or<float_type>(
            "b_f_mag", 0.0);

        b_f_eps = _simulation->dictionary()->template get_or<float_type>(
            "b_f_eps", 1e-3);

        use_filter = _simulation->dictionary()->template get_or<bool>(
            "use_filter", true);

        cg_threshold_ = simulation_->dictionary_->template get_or<float_type>("cg_threshold",1e-3);
        cg_max_itr_ = simulation_->dictionary_->template get_or<int>("cg_max_itr", 40);
        write_stress_ = simulation_->dictionary_->template get_or<bool>("write_stress", false);
        write_noca_ = simulation_->dictionary_->template get_or<bool>("write_noca", false);

        if (dt_base_ < 0) dt_base_ = dx_base_ * cfl_;

        // adaptivity --------------------------------------------------------
        adapt_freq_ = _simulation->dictionary()->template get_or<float_type>(
            "adapt_frequency", 1);
        T_max_ = tot_base_steps_ * dt_base_;
        update_marching_parameters();

        // support of IF in every dirrection is about 3.2 corresponding to 1e-5
        // with coefficient alpha = 1
        // so need update
        max_vel_refresh_ = floor(14/(3.3/(Re_*dx_base_*dx_base_/dt_)));
        pcout<<"maximum steps allowed without vel refresh = " << max_vel_refresh_ <<std::endl;

        // restart -----------------------------------------------------------
        write_restart_ = _simulation->dictionary()->template get_or<bool>(
            "write_restart", true);

        if (write_restart_)
            restart_base_freq_ =
                _simulation->dictionary()->template get<float_type>(
                    "restart_write_frequency");

        // IF constants ------------------------------------------------------
        fname_prefix_ = "";

        // miscs -------------------------------------------------------------
        if (write_noca_)
        {
            omega_cross_x.resize(domain_->dimension());
            std::fill(omega_cross_x.begin(), omega_cross_x.end(), 0.0);

            auto force_vec = this->template Force_Noca<u_type>();
        }
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
        use_restart_ = use_restart;
        boost::mpi::communicator          world;
        parallel_ostream::ParallelOstream pcout =
            parallel_ostream::ParallelOstream(world.size() - 1);

        pcout
            << "Time marching ------------------------------------------------ "
            << std::endl;
        // --------------------------------------------------------------------
        if (use_restart_)
        {
            just_restarted_= true;
            Dictionary info_d(simulation_->restart_load_dir()+"/restart_info");
            T_=info_d.template get<float_type>("T");
            adapt_count_=info_d.template get<int>("adapt_count");
            T_last_vel_refresh_=info_d.template get_or<float_type>("T_last_vel_refresh", 0.0);
            source_max_[0]=info_d.template get<float_type>("cell_aux_max");
            source_max_[1]=info_d.template get<float_type>("u_max");
            pcout<<"Restart info ------------------------------------------------ "<< std::endl;
            pcout<<"T = "<< T_<< std::endl;
            pcout<<"adapt_count = "<< adapt_count_<< std::endl;
            pcout<<"cell aux max = "<< source_max_[0]<< std::endl;
            pcout<<"u max = "<< source_max_[1]<< std::endl;
            pcout<<"T_last_vel_refresh = "<< T_last_vel_refresh_<< std::endl;
            if(domain_->is_client())
            {
                //pad_velocity<u_type, u_type>(true);
            }
            write_timestep();
        }
        else
        {
            T_ = 0.0;
            adapt_count_=0;

            write_timestep();
        }

        // ----------------------------------- start -------------------------

        while(T_<T_max_-1e-10)
        {


            // -------------------------------------------------------------
            // adapt

            // clean up the block boundary of cell_aux_type for smoother adaptation

            if(domain_->is_client())
            {
                clean<cell_aux_type>(true, 2);
                clean<edge_aux_type>(true, 1);
                clean<correction_tmp_type>(true, 2);
            }
            else
            {
                //const auto& lb = domain_->level_blocks();
		/*std::vector<int> lb;
		domain_->level_blocks(lb);*/
                auto lb = domain_->level_blocks();
                std::cout<<"Blocks on each level: ";

                for (int c: lb)
                    std::cout<< c << " ";
                std::cout<<std::endl;

            }

            // copy flag correction to flag old correction
            for (auto it  = domain_->begin();
                    it != domain_->end(); ++it)
            {
                it->flag_old_correction(false);
            }

            for (auto it  = domain_->begin(domain_->tree()->base_level());
                    it != domain_->end(domain_->tree()->base_level()); ++it)
            {
                it->flag_old_correction(it->is_correction());
            }

            int c=0;

            for (auto it = domain_->begin(); it != domain_->end(); ++it)
            {
                if (!it->locally_owned()) continue;
                if (it->is_ib() || it->is_extended_ib())
                {
                    auto& lin_data =
                        it->data_r(test_type::tag(), 0).linalg_data();
                    std::fill(lin_data.begin(), lin_data.end(), 2.0);
                    c+=1;
                }

            }
            boost::mpi::communicator world;
            int c_all;
            boost::mpi::all_reduce(
                world, c, c_all, std::plus<int>());
            pcout<< "block = " << c_all << std::endl;


            if ( adapt_count_ % adapt_freq_ ==0 && adapt_count_ != 0)
            {
                if (adapt_count_==0 || updating_source_max_)
                {
                    this->template update_source_max<cell_aux_type>(0);
                    this->template update_source_max<edge_aux_type>(1);
                }

                //if(domain_->is_client())
                //{
                //    up_and_down<u>();
                //    pad_velocity<u, u>();
                //}
                if (!just_restarted_) {
                    this->adapt(false);
                    // adapt_corr_time_step();
                }
                just_restarted_=false;

            }

            // balance load
            if ( adapt_count_ % adapt_freq_ ==0)
            {
                clean<u_type>(true);
                domain_->decomposition().template balance<u_type,p_type>();
                //domain_->decomposition().template balance<u_type,p_type>();
            }

            adapt_count_++;

            // -------------------------------------------------------------
            // time marching

            mDuration_type ifherk_if(0);
            TIME_CODE( ifherk_if, SINGLE_ARG(
                        time_step();
                        ));
            pcout<<ifherk_if.count()<<std::endl;
            
            // -------------------------------------------------------------
            // update stats & output

            T_ += dt_;
            float_type tmp_n = T_ / dt_base_ * math::pow2(max_ref_level_);
            int        tmp_int_n = int(tmp_n + 0.5);

            if (write_restart_ && (std::fabs(tmp_int_n - tmp_n) < 1e-4) &&
                (tmp_int_n % restart_base_freq_ == 0))
            {
                restart_n_last_ = tmp_int_n;
                write_restart();
            }

            if ((std::fabs(tmp_int_n - tmp_n) < 1e-4) &&
                (tmp_int_n % output_base_freq_ == 0))
            {
                //test
                //

                //clean<test_type>();
                //auto f = [](std::size_t idx, float_type t, auto coord = {0, 0, 0}){return 1.0;};
                //domain::Operator::add_field_expression<test_type>(domain_, f, T_, 1.0);
                //clean_leaf_correction_boundary<test_type>(domain_->tree()->base_level(),true,2);

                n_step_ = tmp_int_n;
                write_timestep();
                // only update dt after 1 output so it wouldn't do 3 5 7 9 ...
                // and skip all outputs
                update_marching_parameters();
            }

            write_stats(tmp_n);

        }
    }
    void clean_up_initial_velocity()
    {
        if (domain_->is_client())
        {
            up_and_down<u_type>();
            auto client = domain_->decomposition().client();
            clean<edge_aux_type>();
            clean<stream_f_type>();
            for (int l = domain_->tree()->base_level();
                 l < domain_->tree()->depth(); ++l)
            {
                client->template buffer_exchange<u_type>(l);
                for (auto it = domain_->begin(l); it != domain_->end(l); ++it)
                {
                    if (!it->locally_owned() || it->is_correction()) continue;

                    const auto dx_level =
                        dx_base_ / math::pow2(it->refinement_level());
                    domain::Operator::curl<u_type, edge_aux_type>(
                        it->data(), dx_level);
                }
            }
            //clean_leaf_correction_boundary<edge_aux_type>(domain_->tree()->base_level(), true,2);

            clean<u_type>();
            psolver.template apply_lgf<edge_aux_type, stream_f_type>();
            for (int l = domain_->tree()->base_level();
                 l < domain_->tree()->depth(); ++l)
            {
                for (auto it = domain_->begin(l); it != domain_->end(l); ++it)
                {
                    if (!it->locally_owned()) continue;

                    const auto dx_level =
                        dx_base_ / math::pow2(it->refinement_level());
                    domain::Operator::curl_transpose<stream_f_type, u_type>(
                        it->data(), dx_level, -1.0);
                }
                client->template buffer_exchange<u_type>(l);
            }
        }
    }

    template<class Field>
    void update_source_max(int idx)
    {
        float_type max_local = 0.0;
        for (auto it = domain_->begin(); it != domain_->end(); ++it)
        {
            if (!it->locally_owned()) continue;
            float_type tmp = domain::Operator::blockRootMeanSquare<Field>(it->data());

            if (tmp > max_local) max_local = tmp;
        }

        float_type new_maximum=0.0;
        boost::mpi::all_reduce(
            comm_, max_local, new_maximum, boost::mpi::maximum<float_type>());

        //source_max_[idx] = std::max(source_max_[idx], new_maximum);
        if (all_time_max_)
            source_max_[idx] = std::max(source_max_[idx], new_maximum);
        else
            source_max_[idx] = 0.5*( source_max_[idx] + new_maximum );

        if (source_max_[idx]< 1e-2)
            source_max_[idx] = 1e-2;

        pcout << "source max = "<< source_max_[idx] << std::endl;
    }

    void write_restart()
    {
        boost::mpi::communicator world;

        world.barrier();
        if (domain_->is_server() && write_restart_)
        {
            std::cout << "restart: backup" << std::endl;
            simulation_->copy_restart();
        }
        world.barrier();

        pcout << "restart: write" << std::endl;
        simulation_->write("", true);

        write_info();
        world.barrier();
    }

    void write_stats(int tmp_n)
    {
        boost::mpi::communicator world;
        world.barrier();

        // - Numeber of cells -----------------------------------------------
        int c_allc_global;
        int c_allc = domain_->num_allocations();
        boost::mpi::all_reduce(
                world, c_allc, c_allc_global, std::plus<int>());

        if (domain_->is_server())
        {
            std::cout<<"T = " << T_<<", n = "<< tmp_n << " -----------------" << std::endl;
            auto lb = domain_->level_blocks();
            std::cout<<"Blocks on each level: ";

            for (int c: lb)
                std::cout<< c << " ";
            std::cout<<std::endl;

            std::cout<<"Total number of leaf octants: "<<domain_->num_leafs()<<std::endl;
            std::cout<<"Total number of leaf + correction octants: "<<domain_->num_corrections()+domain_->num_leafs()<<std::endl;
            std::cout<<"Total number of allocated octants: "<<c_allc_global<<std::endl;
            std::cout<<" -----------------" << std::endl;
        }


        // - Forcing ------------------------------------------------
        auto& ib = domain_->ib();
        ib.clean_non_local();
	real_coordinate_type tmp_coord(0.0);

        force_type sum_f(ib.force().size(), tmp_coord);
        if (ib.size() > 0)
        {
            boost::mpi::all_reduce(world, &ib.force(0), ib.size(), &sum_f[0],
                std::plus<real_coordinate_type>());
        }
        if (domain_->is_server())
        {
            std::vector<float_type> f(domain_->dimension(), 0.);
            if (ib.size() > 0)
            {
                for (std::size_t d = 0; d < domain_->dimension(); ++d)
                    for (std::size_t i = 0; i < ib.size(); ++i)
                        f[d] += sum_f[i][d] * 1.0 / coeff_a(3, 3) / dt_ *
                                ib.force_scale();
                //f[d]+=sum_f[i][d] * 1.0 / dt_ * ib.force_scale();

                std::cout<<"ib  size: "<<ib.size()<<std::endl;
                std::cout << "Forcing = ";

                for (std::size_t d = 0; d < domain_->dimension(); ++d)
                    std::cout << f[d] << " ";
                std::cout << std::endl;

                std::cout << " -----------------" << std::endl;
            }

            std::ofstream outfile;
            int width=20;

            outfile.open("fstats.txt", std::ios_base::app); // append instead of overwrite
            outfile <<std::setw(width) << tmp_n <<std::setw(width)<<std::scientific<<std::setprecision(9);
            outfile <<std::setw(width) << T_ <<std::setw(width)<<std::scientific<<std::setprecision(9);
            for (auto& element:f)
            {
                outfile<<element<<std::setw(width);
            }
            outfile<<std::endl;
        }



        force_type All_sum_f(ib.force().size(), tmp_coord);
        force_type All_sum_f_glob(ib.force().size(), tmp_coord);
        this->template ComputeForcing<u_type, p_type, u_i_type, d_i_type>(All_sum_f);

        if (ib.size() > 0)
        {
            for (std::size_t d = 0; d < All_sum_f[0].size(); ++d)
            {
                for (std::size_t i = 0; i < All_sum_f.size(); ++i)
                {
                    if (world.rank() != domain_->ib().rank(i))
                        All_sum_f[i][d] = 0.0;
                }
            }

            boost::mpi::all_reduce(world,
                &All_sum_f[0], All_sum_f.size(), &All_sum_f_glob[0],
                std::plus<real_coordinate_type>());
        }

        if (domain_->is_server())
        {
            std::vector<float_type> f(ib_t::force_dim, 0.);
            if (ib.size() > 0)
            {
                for (std::size_t d = 0; d < ib_t::force_dim; ++d)
                    for (std::size_t i = 0; i < ib.size(); ++i)
                        f[d] += All_sum_f_glob[i][d] * ib.force_scale();
                //f[d]+=sum_f[i][d] * 1.0 / dt_ * ib.force_scale();

                std::cout << "ib  size: " << ib.size() << std::endl;
                std::cout << "New Forcing = ";

                for (std::size_t d = 0; d < domain_->dimension(); ++d)
                    std::cout << -f[d] << " ";
                std::cout << std::endl;

                std::cout << " -----------------" << std::endl;
            }

            std::ofstream outfile;
            int           width = 20;

            outfile.open("fstats_from_cg.txt",
                std::ios_base::app); // append instead of overwrite
            outfile << std::setw(width) << tmp_n << std::setw(width)
                    << std::scientific << std::setprecision(9);
            outfile << std::setw(width) << T_ << std::setw(width)
                    << std::scientific << std::setprecision(9);
            for (int  i = 0 ; i < domain_->dimension(); i++) { outfile << -f[i] << std::setw(width); }
            outfile << std::endl;
            outfile.close();
            std::cout << "finished writing new forcing" << std::endl;


            if (write_stress_)
            {
                outfile.open("fstats_stress_x.txt",
                    std::ios_base::app); // append instead of overwrite
                outfile << std::setw(width) << tmp_n << std::setw(width)
                        << std::scientific << std::setprecision(9);
                outfile << std::setw(width) << T_ << std::setw(width)
                        << std::scientific << std::setprecision(9);
                for (int i = 0; i < ib.size(); i++) { outfile << All_sum_f_glob[i][0] << std::setw(width); }
                outfile << std::endl;
                outfile.close();

                outfile.open("fstats_stress_y.txt",
                    std::ios_base::app); // append instead of overwrite
                outfile << std::setw(width) << tmp_n << std::setw(width)
                        << std::scientific << std::setprecision(9);
                outfile << std::setw(width) << T_ << std::setw(width)
                        << std::scientific << std::setprecision(9);
                for (int i = 0; i < ib.size(); i++) { outfile << All_sum_f_glob[i][1] << std::setw(width); }
                outfile << std::endl;
                outfile.close();
                std::cout << "finished writing new stress" << std::endl;
            }
        }
        if (write_noca_)
        {
            auto force_vec = this->template Force_Noca<u_type>();

            if (domain_->is_server())
            {
                std::cout << "NoCA Forcing = ";
                for (std::size_t d = 0; d < domain_->dimension(); ++d)
                    std::cout << force_vec[d] << " ";
                std::cout << std::endl;
                std::cout << " -----------------" << std::endl;

                std::ofstream outfile;
                int           width = 20;

                outfile.open("fstats_noca.txt",
                    std::ios_base::app); // append instead of overwrite
                outfile << std::setw(width) << tmp_n << std::setw(width)
                        << std::scientific << std::setprecision(9);
                outfile << std::setw(width) << T_ << std::setw(width)
                        << std::scientific << std::setprecision(9);
                for (int  i = 0 ; i < domain_->dimension(); i++) { outfile << force_vec[i] << std::setw(width); }
                outfile << std::endl;
                outfile.close();
            }
        }

        world.barrier();
    }


    template<class Velocity_in>
    void curl()
    {
        auto client=domain_->decomposition().client();

        if (!client) return;

        //up_and_down<Velocity_in>();
        clean<Velocity_in>(true);
        //this->up<Velocity_in>(false);
        up_and_down<Velocity_in>();
        clean<edge_aux_type>();

        auto dx_base = domain_->dx_base();

        for (int l  = domain_->tree()->base_level();
                l < domain_->tree()->depth(); ++l)
        {
            client->template buffer_exchange<Velocity_in>(l);

            for (auto it  = domain_->begin(l);
                    it != domain_->end(l); ++it)
            {
                if(!it->locally_owned() || !it->has_data()) continue;
                if(it->is_correction()) continue;
                //if(!it->is_leaf()) continue;

                const auto dx_level =  dx_base/math::pow2(it->refinement_level());
                //if (it->is_leaf())
                domain::Operator::curl<Velocity_in,edge_aux_type>( it->data(),dx_level);
            }
        }

        //clean<Velocity_out>();
        clean_leaf_correction_boundary<edge_aux_type>(domain_->tree()->base_level(), true, 2);
    }

    template<class Velocity_in>
    std::vector<float_type> Force_Noca() {
        curl<Velocity_in>();
        std::vector<float_type> omega_cross_x_glob(domain_->dimension(), 0.);
        std::vector<float_type> omega_cross_x_tmp(domain_->dimension(), 0.);
        if (!domain_->is_server()) {
            auto center = (domain_->bounding_box().max() -
                domain_->bounding_box().min() + 1) / 2.0 +
                domain_->bounding_box().min();
            //std::vector<float_type> omega_cross_x_tmp(domain_->dimension(), 0.);
            for (auto it = domain_->begin(); it != domain_->end(); ++it) {
                if (!it->locally_owned()) continue;
                if (!it->is_leaf()) continue;
                if (it->is_correction()) continue;
                auto scaling = std::pow(2, it->refinement_level());
                const auto dx_level = dx_base_ / math::pow2(it->refinement_level());

                for (auto& n : it->data()) {
                    const auto& coord = n.level_coordinate();
                    float_type x = static_cast<float_type>
                        (coord[0]) * dx_level;
                    float_type y = static_cast<float_type>
                        (coord[1]) * dx_level;

                    if ((x*x + y*y) < 0.25) continue;

                    omega_cross_x_tmp[0] +=  y * n(edge_aux_type::tag(), 0) * std::pow(dx_level, domain_->dimension());
                    omega_cross_x_tmp[1] += -x * n(edge_aux_type::tag(), 0) * std::pow(dx_level, domain_->dimension());

                }
                
            }

            
        }

        boost::mpi::communicator world;
        boost::mpi::all_reduce(world, &omega_cross_x_tmp[0], omega_cross_x_tmp.size(), &omega_cross_x_glob[0], std::plus<float_type>());

        std::vector<float_type> force_vec(domain_->dimension(), 0.);

        for (int i = 0; i < domain_->dimension(); i++) {
            pcout << omega_cross_x_glob[i] << " ";
            force_vec[i] = (omega_cross_x_glob[i] - omega_cross_x[i]) / dt_;
            omega_cross_x[i] = omega_cross_x_glob[i];
        }

        pcout << std::endl;

        world.barrier();

        return force_vec;
    }

    template<class Velocity_in>
    void clean_center_velocity() {

        auto f = simulation_->frame_vel();
        if (!domain_->is_server()) {
            auto center = (domain_->bounding_box().max() -
                domain_->bounding_box().min() + 1) / 2.0 +
                domain_->bounding_box().min();
            //std::vector<float_type> omega_cross_x_tmp(domain_->dimension(), 0.);
            for (auto it = domain_->begin(); it != domain_->end(); ++it) {
                if (!it->locally_owned()) continue;
                if (!it->is_leaf()) continue;
                if (it->is_correction()) continue;
                auto scaling = std::pow(2, it->refinement_level());
                const auto dx_level = dx_base_ / math::pow2(it->refinement_level());

                for (auto& n : it->data()) {
                    const auto& coord = n.level_coordinate();
                    float_type x = static_cast<float_type>
                        (coord[0]) * dx_level;
                    float_type y = static_cast<float_type>
                        (coord[1]) * dx_level;

                    auto coord_real = n.global_coordinate() * dx_base_;

                    float_type x_v = x + 0.5*dx_level;
                    float_type y_u = y + 0.5*dx_level;

                    if (std::sqrt(x_v*x_v + y*y) < (0.5)) {
                        
                        n(Velocity_in::tag(), 1) = f(1, T_stage_, coord_real);
                    }
                    if (std::sqrt(x*x + y_u*y_u) < (0.5)) {
                        n(Velocity_in::tag(), 0) = f(0, T_stage_, coord_real);
                    }

                }
                
            }

            
        }
    }


    template<class Source_face, class Source_cell, class Target_face, class Target_cell>
    void ComputeForcing(force_type& force_target) noexcept
    {
        if (domain_->is_server())
            return;
        auto client = domain_->decomposition().client();

        boost::mpi::communicator world;

        //pad_velocity<Source_face, Source_face>(true);

        clean<face_aux2_type>();
        clean<edge_aux_type>();
        clean<w_1_type>();
        clean<w_2_type>();

        //clean<r_i_T_type>(); //use r_i as the result of applying Jcobian in the first block
        //clean<cell_aux_T_type>(); //use cell aux_type to be the second block
        clean<Target_face>();
        clean<Target_cell>();
        real_coordinate_type tmp_coord(0.0);
        auto forcing_tmp = force_target;
        std::fill(forcing_tmp.begin(), forcing_tmp.end(),
            tmp_coord);
        std::fill(force_target.begin(), force_target.end(),
            tmp_coord); //use forcing tmp to store the last block,
            //use forcing_old to store the forcing at previous Newton iteration
        //computing wii in Andre's paper
        real_coordinate_type tmp_coord1(1.0);
        auto forcing_1 = force_target;
        std::fill(forcing_1.begin(), forcing_1.end(),
            tmp_coord1);

        lsolver.template smearing<w_1_type>(forcing_1, true);
        computeWii<w_1_type>();
        //finish computing Wii


        
        laplacian<Source_face, face_aux2_type>();

        //clean_center_velocity<face_aux2_type>();

        clean<r_i_type>();
        clean<g_i_type>();
        gradient<Source_cell, r_i_type>(1.0);

        //clean_center_velocity<r_i_type>();

        //pcout << "Computed Laplacian " << std::endl;

        nonlinear<Source_face, g_i_type>();

        //pcout << "Computed Nonlinear Jac " << std::endl;

        add<g_i_type, Target_face>(1);

        add<face_aux2_type, Target_face>(-1.0 / Re_);

        add<r_i_type, Target_face>(1.0);

        lsolver.template projection<Target_face>(forcing_tmp);

        auto r = forcing_tmp;

        //force_target = forcing_tmp;

        float_type r2_old = dotVec(r, r);

        if (r2_old < 1e-12) {
            if (world.rank() == 1) {
                std::cout << "r0 small, exiting" << std::endl;
            }
            return;
        }

        auto p = r;

        for (int i = 0; i < cg_max_itr_; i++) {
            clean<r_i_type>();
            lsolver.template smearing<r_i_type>(p, false);
            auto Ap = r;
            cleanVec(Ap,false);
            lsolver.template projection<r_i_type>(Ap);
            r2_old = dotVec(r, r);
            float_type pAp = dotVec(p, Ap);
            float_type alpha = r2_old/pAp;
            //force_target += alpha*p;
            addVec(force_target, p, 1.0, alpha);
            //r -= alpha*Ap;
            addVec(r, Ap, 1.0, -alpha);
            float_type r2_new = dotVec(r, r);
            float_type f2 = dotVec(force_target, force_target);
            if (world.rank() == 1)
            {
                std::cout << "r2/f2 = " << r2_new / f2 << " f2 = " << f2 << std::endl;
            }

            if (std::sqrt(r2_new/f2) < cg_threshold_) {
		if (!use_filter) {
		    return;
		}
		else {
                clean<r_i_type>();
                lsolver.template smearing<r_i_type>(force_target, false);
                product<r_i_type, w_1_type, w_2_type>();
                auto Ap = r;
                cleanVec(Ap,false);
                lsolver.template projection<w_2_type>(Ap);

                force_target = Ap;
                
                return;
		}
            }
            float_type beta = r2_new/r2_old;
            //p = r+beta*p;
            addVec(p, r, beta, 1.0);

        }        
    }


    template<class VecType>
    void addVec(VecType& a, VecType& b, float_type w1, float_type w2)
    {
        float_type s = 0;
        for (std::size_t i=0; i<a.size(); ++i)
        {
            if (domain_->ib().rank(i)!=comm_.rank()) {
                for (std::size_t d=0; d<a[0].size(); ++d) {
                    a[i][d] = 0;
                }
                continue;
            }

            for (std::size_t d=0; d<a[0].size(); ++d)
                a[i][d] =a[i][d]*w1 + b[i][d]*w2;
        }
    }

    template<class VecType>
    void cleanVec(VecType& a, bool nonloc = true)
    {
        
        for (std::size_t i=0; i<a.size(); ++i)
        {
            if (domain_->ib().rank(i)!=comm_.rank()) {
                for (std::size_t d=0; d<a[0].size(); ++d) {
                    a[i][d] = 0;
                }
                continue;
            }

            if (!nonloc)
            {
                for (std::size_t d = 0; d < a[0].size(); ++d) { a[i][d] = 0; }
            }
        }
    }

    template<class VecType>
    float_type dotVec(VecType& a, VecType& b)
    {
        float_type s = 0;
        for (std::size_t i=0; i<a.size(); ++i)
        {
            if (domain_->ib().rank(i)!=comm_.rank())
                continue;

            for (std::size_t d=0; d<a[0].size(); ++d)
                s+=a[i][d]*b[i][d];
        }

        float_type s_global=0.0;
        boost::mpi::all_reduce(domain_->client_communicator(), s,
                s_global, std::plus<float_type>());
        return s_global;
    }

    void write_timestep()
    {
        boost::mpi::communicator world;
        world.barrier();
        pcout << "- writing at T = " << T_ << ", n = " << n_step_ << std::endl;
        //simulation_->write(fname(n_step_));
        simulation_->writeWithTime(fname(n_step_), T_, dt_);
        //simulation_->domain()->tree()->write("tree_restart.bin");
        world.barrier();
        //simulation_->domain()->tree()->read("tree_restart.bin");
        pcout << "- output writing finished -" << std::endl;
    }

    void write_info()
    {
        if (domain_->is_server())
        {
            std::ofstream ofs(
                simulation_->restart_write_dir() + "/restart_info",
                std::ofstream::out);
            if (!ofs.is_open())
            {
                throw std::runtime_error("Could not open file for info write ");
            }

            ofs.precision(20);
            ofs<<"T = " << T_ << ";" << std::endl;
            ofs<<"adapt_count = " << adapt_count_ << ";" << std::endl;
            ofs<<"cell_aux_max = " << source_max_[0] << ";" << std::endl;
            ofs<<"u_max = " << source_max_[1] << ";" << std::endl;
            ofs<<"restart_n_last = " << restart_n_last_ << ";" << std::endl;
            ofs<<"T_last_vel_refresh = " << T_last_vel_refresh_ << ";" << std::endl;

            ofs.close();
        }
    }

    std::string fname(int _n)
    {
        return fname_prefix_+std::to_string(_n);
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
    void up(bool leaf_boundary_only=false)
    {
        //Coarsification:
        for (std::size_t _field_idx=0; _field_idx<Field::nFields(); ++_field_idx)
            psolver.template source_coarsify<Field,Field>(_field_idx, _field_idx, Field::mesh_type(), false, false, false, leaf_boundary_only);
    }

    template<class Field>
    void down_to_correction()
    {
        // Interpolate to correction buffer
        for (std::size_t _field_idx = 0; _field_idx < Field::nFields();
             ++_field_idx)
            psolver.template intrp_to_correction_buffer<Field, Field>(
                _field_idx, _field_idx, Field::mesh_type(), true, false);
    }

    void adapt(bool coarsify_field=true, bool check_source_max=true)
    {
        boost::mpi::communicator world;
        // std::cout << "Adaptation started" << std::endl;
        auto                     client = domain_->decomposition().client();

        if (check_source_max && (source_max_[0]<1e-10 || source_max_[1]<1e-10)) return;

        //adaptation neglect the boundary oscillations
        clean_leaf_correction_boundary<cell_aux_type>(domain_->tree()->base_level(),true,2);
        clean_leaf_correction_boundary<edge_aux_type>(domain_->tree()->base_level(),true,2);

        world.barrier();

        if (coarsify_field)
        {
            pcout << "Adapt - coarsify" << std::endl;
            if (client)
            {
                //claen non leafs
                clean<u_type>(true);
                this->up<u_type>(false);
                ////Coarsification:
                //for (std::size_t _field_idx=0; _field_idx<u::nFields; ++_field_idx)
                //    psolver.template source_coarsify<u_type,u_type>(_field_idx, _field_idx, u::mesh_type);
            }
        }

        world.barrier();
        pcout<< "Adapt - communication"  << std::endl;
        auto intrp_list = domain_->adapt(source_max_, base_mesh_update_);

        world.barrier();
        pcout << "Adapt - intrp" << std::endl;
        if (client)
        {
            // Intrp
            for (std::size_t _field_idx=0; _field_idx<u_type::nFields(); ++_field_idx)
            {
                for (int l = domain_->tree()->depth() - 2;
                     l >= domain_->tree()->base_level(); --l) // finest level is l=depth-1 and we only 
                {
                    client->template buffer_exchange<u_type>(l);

                    domain_->decomposition().client()->
                    template communicate_updownward_assign
                    <u_type, u_type>(l,false,false,-1,_field_idx);
                }

                for (auto& oct : intrp_list)
                {
                    if (!oct || !oct->has_data()) continue;
                    psolver.c_cntr_nli().template nli_intrp_node<u_type, u_type>(oct, u_type::mesh_type(), _field_idx, _field_idx, false, false);
                }
            }
        }
        world.barrier();
        pcout << "Adapt - done" << std::endl;
    }

    void time_step()
    {
        // Initialize IFHERK
        // q_1 = u_type
        boost::mpi::communicator world;
        auto                     client = domain_->decomposition().client();

        T_stage_ = T_;

        ////claen non leafs

        stage_idx_=0;

        // Solve stream function to refresh base level velocity
        mDuration_type t_pad(0);
        TIME_CODE( t_pad, SINGLE_ARG(
                    pcout<< "base level mesh update = "<<base_mesh_update_<< std::endl;
                    if (    base_mesh_update_ ||
                            ((T_-T_last_vel_refresh_)/(Re_*dx_base_*dx_base_) * 3.3>7))
                    {
                        pcout<< "pad_velocity, last T_vel_refresh = "<<T_last_vel_refresh_<< std::endl;
                        T_last_vel_refresh_=T_;
                        
                        if (!domain_->is_client())
                            return;
                        pad_velocity<u_type, u_type>(true);
                        // pad_velocity<u_mean_type, u_mean_type>(true);
                        adapt_corr_time_step();
                    }
                    else
                    {
                        if (!domain_->is_client())
                            return;
                        up_and_down<u_type>();
                        // up_and_down<u_mean_type>();
                    }
                    ));
        base_mesh_update_=false;
        pcout<< "pad u      in "<<t_pad.count() << std::endl;

        copy<u_type, q_i_type>();

        // Stage 1
        // ******************************************************************
        pcout << "Stage 1" << std::endl;
        T_stage_ = T_ + dt_*c_[0];
        stage_idx_ = 1;
        clean<g_i_type>();
        clean<d_i_type>();
        clean<cell_aux_type>();
        clean<face_aux_type>();

        nonlinear<u_type, g_i_type>(coeff_a(1, 1) * (-dt_));
        copy<q_i_type, r_i_type>();
        add<g_i_type, r_i_type>();
        lin_sys_with_ib_solve(alpha_[0]);

        //clean_center_velocity<u_i_type>();


        // Stage 2
        // ******************************************************************
        pcout << "Stage 2" << std::endl;
        T_stage_ = T_ + dt_*c_[1];
        stage_idx_ = 2;
        clean<r_i_type>();
        clean<d_i_type>();
        clean<cell_aux_type>();

        //cal wii
        //r_i_type = q_i_type + dt(a21 w21)
        //w11 = (1/a11)* dt (g_i_type - face_aux_type)

        add<g_i_type, face_aux_type>(-1.0);
        copy<face_aux_type, w_1_type>(-1.0 / dt_ / coeff_a(1, 1));

        psolver.template apply_lgf_IF<q_i_type, q_i_type>(alpha_[0]);
        psolver.template apply_lgf_IF<w_1_type, w_1_type>(alpha_[0]);

        add<q_i_type, r_i_type>();
        add<w_1_type, r_i_type>(dt_ * coeff_a(2, 1));

        up_and_down<u_i_type>();
        nonlinear<u_i_type, g_i_type>(coeff_a(2, 2) * (-dt_));
        add<g_i_type, r_i_type>();

        lin_sys_with_ib_solve(alpha_[1]);

        //clean_center_velocity<u_i_type>();

        // Stage 3
        // ******************************************************************
        pcout << "Stage 3" << std::endl;
        T_stage_ = T_ + dt_*c_[2];
        stage_idx_ = 3;
        clean<d_i_type>();
        clean<cell_aux_type>();
        clean<w_2_type>();

        add<g_i_type, face_aux_type>(-1.0);
        copy<face_aux_type, w_2_type>(-1.0 / dt_ / coeff_a(2, 2));
        copy<q_i_type, r_i_type>();
        add<w_1_type, r_i_type>(dt_ * coeff_a(3, 1));
        add<w_2_type, r_i_type>(dt_ * coeff_a(3, 2));

        psolver.template apply_lgf_IF<r_i_type, r_i_type>(alpha_[1]);

        up_and_down<u_i_type>();
        nonlinear<u_i_type, g_i_type>(coeff_a(3, 3) * (-dt_));
        add<g_i_type, r_i_type>();

        lin_sys_with_ib_solve(alpha_[2]);

        //clean_center_velocity<u_i_type>();

        // ******************************************************************
        copy<u_i_type, u_type>();
        copy<d_i_type, p_type>(1.0 / coeff_a(3, 3) / dt_);
        // ******************************************************************
        
    }





    void adapt_corr_time_step()
    {
        if (!use_adaptation_correction) return;
        // Initialize IFHERK
        // q_1 = u_type
        boost::mpi::communicator world;
        auto                     client = domain_->decomposition().client();

        T_stage_ = T_;

        ////claen non leafs

        stage_idx_=0;

        // Solve stream function to refresh base level velocity
        mDuration_type t_pad(0);
        TIME_CODE( t_pad, SINGLE_ARG(
                    pcout<< "adapt_corr_time_step()" << std::endl;
                    
                    if (!domain_->is_client())
                        return;
                    up_and_down<u_type>();
                    ));
        //base_mesh_update_=false;
        //pcout<< "pad u      in "<<t_pad.count() << std::endl;

        copy<u_type, q_i_type>();

        // Stage 1
        // ******************************************************************
        pcout << "Stage 1" << std::endl;
        T_stage_ = T_;
        stage_idx_ = 1;
        clean<g_i_type>();
        clean<d_i_type>();
        clean<cell_aux_type>();
        clean<face_aux_type>();

        //nonlinear<u_type, g_i_type>(0.0);
        copy<q_i_type, r_i_type>();
        //add<g_i_type, r_i_type>();
        lin_sys_with_ib_solve(0.0, false);


        // Stage 2
        // ******************************************************************
        pcout << "Stage 2" << std::endl;
        T_stage_ = T_;
        stage_idx_ = 2;
        clean<r_i_type>();
        clean<d_i_type>();
        clean<cell_aux_type>();

        //cal wii
        //r_i_type = q_i_type + dt(a21 w21)
        //w11 = (1/a11)* dt (g_i_type - face_aux_type)

        //add<g_i_type, face_aux_type>(-1.0);
        //copy<face_aux_type, w_1_type>(0.0);

        //psolver.template apply_lgf_IF<q_i_type, q_i_type>(0.0);
        //psolver.template apply_lgf_IF<w_1_type, w_1_type>(0.0);

        add<q_i_type, r_i_type>();
        //add<w_1_type, r_i_type>(0.0);

        up_and_down<u_i_type>();
        //nonlinear<u_i_type, g_i_type>(0.0);
        //add<g_i_type, r_i_type>();

        lin_sys_with_ib_solve(0.0, false);

        // Stage 3
        // ******************************************************************
        pcout << "Stage 3" << std::endl;
        T_stage_ = T_;
        stage_idx_ = 3;
        clean<d_i_type>();
        clean<cell_aux_type>();
        clean<w_2_type>();

        //add<g_i_type, face_aux_type>(-1.0);
        //copy<face_aux_type, w_2_type>(0.0);
        copy<q_i_type, r_i_type>();
        //add<w_1_type, r_i_type>(0.0);
        //add<w_2_type, r_i_type>(0.0);

        psolver.template apply_lgf_IF<r_i_type, r_i_type>(0.0);

        up_and_down<u_i_type>();
        //nonlinear<u_i_type, g_i_type>(0.0);
        //add<g_i_type, r_i_type>();

        lin_sys_with_ib_solve(0.0, false);
        // copy<u_mean_type,u_mean_type>((adapt_count_-1.0)/(adapt_count_));
        // add<u_i_type, u_mean_type>(1.0/(adapt_count_));
        // ******************************************************************
        copy<u_i_type, u_type>();
        copy<d_i_type, p_type>(0.0);
        // ******************************************************************
    }

    template <typename F>
    void clean(bool non_leaf_only=false, int clean_width=1) noexcept
    {
        for (auto it = domain_->begin(); it != domain_->end(); ++it)
        {
            if (!it->has_data()) continue;
            if (!it->data().is_allocated()) continue;

            for (std::size_t field_idx = 0; field_idx < F::nFields(); ++field_idx)
            {
                auto& lin_data = it->data_r(F::tag(), field_idx).linalg_data();

                if (non_leaf_only && it->is_leaf() && it->locally_owned())
                {
                    int N = it->data().descriptor().extent()[0];
		    if(domain_->dimension() == 3) {
                    view(lin_data, xt::all(), xt::all(),
                        xt::range(0, clean_width)) *= 0.0;
                    view(lin_data, xt::all(), xt::range(0, clean_width),
                        xt::all()) *= 0.0;
                    view(lin_data, xt::range(0, clean_width), xt::all(),
                        xt::all()) *= 0.0;
                    view(lin_data, xt::range(N + 2 - clean_width, N + 3),
                        xt::all(), xt::all()) *= 0.0;
                    view(lin_data, xt::all(),
                        xt::range(N + 2 - clean_width, N + 3), xt::all()) *=
                        0.0;
                    view(lin_data, xt::all(), xt::all(),
                        xt::range(N + 2 - clean_width, N + 3)) *= 0.0;
		    }
		    else {
                    view(lin_data, xt::all(), xt::range(0, clean_width)) *= 0.0;
                    view(lin_data, xt::range(0, clean_width), xt::all()) *= 0.0;
                    view(lin_data, xt::range(N + 2 - clean_width, N + 3),xt::all()) *= 0.0;
                    view(lin_data, xt::all(),xt::range(N + 2 - clean_width, N + 3)) *=0.0;
		    }
                }
                else
                {
                    //TODO whether to clean base_level correction?
                    std::fill(lin_data.begin(), lin_data.end(), 0.0);
                }
            }
        }
    }

    template <typename F>
    void clean_leaf_correction_boundary(int l, bool leaf_only_boundary=false, int clean_width=1) noexcept
    {
        for (auto it = domain_->begin(l); it != domain_->end(l); ++it)
        {
            if (!it->locally_owned())
            {
                if (!it->has_data() || !it->data().is_allocated()) continue;
                for (std::size_t field_idx = 0; field_idx < F::nFields();
                     ++field_idx)
                {
                    auto& lin_data =
                        it->data_r(F::tag(), field_idx).linalg_data();
                    std::fill(lin_data.begin(), lin_data.end(), 0.0);
                }
            }
        }

        for (auto it = domain_->begin(l); it != domain_->end(l); ++it)
        {
            if (!it->locally_owned()) continue;
            if (!it->has_data() || !it->data().is_allocated()) continue;

            if (leaf_only_boundary && (it->is_correction() || it->is_old_correction() ))
            {
                for (std::size_t field_idx = 0; field_idx < F::nFields();
                     ++field_idx)
                {
                    auto& lin_data =
                        it->data_r(F::tag(), field_idx).linalg_data();
                    std::fill(lin_data.begin(), lin_data.end(), 0.0);
                }
            }
        }


        //---------------
        if (l==domain_->tree()->base_level())
        for (auto it  = domain_->begin(l);
                it != domain_->end(l); ++it)
        {
            if(!it->locally_owned()) continue;
            if(!it->has_data() || !it->data().is_allocated()) continue;
            //std::cout<<it->key()<<std::endl;

            for(std::size_t i=0;i< it->num_neighbors();++i)
            {
                auto it2=it->neighbor(i);
                if ((!it2 || !it2->has_data()) || (leaf_only_boundary && (it2->is_correction() || it2->is_old_correction() )))
                {
                    for (std::size_t field_idx=0; field_idx<F::nFields(); ++field_idx)
                    {
                        domain::Operator::smooth2zero<F>( it->data(), i);
                    }
                }
            }
        }
    }
    template <class vel_in, class vel_out>
    void pad_access(bool refresh_correction_only=true)
    {
        pad_velocity<vel_in, vel_out>(refresh_correction_only);
    }


private:
    float_type coeff_a(int i, int j)const noexcept {return a_[i*(i-1)/2+j-1];}

    void lin_sys_solve(float_type _alpha) noexcept
    {
        auto client=domain_->decomposition().client();

        divergence<r_i_type, cell_aux_type>();

        domain_->client_communicator().barrier();
        mDuration_type t_lgf(0);
        TIME_CODE( t_lgf, SINGLE_ARG(
                    psolver.template apply_lgf<cell_aux_type, d_i_type>();
                    ));
        pcout<< "LGF solved in "<<t_lgf.count() << std::endl;

        gradient<d_i_type,face_aux_type>();
        add<face_aux_type, r_i_type>(-1.0);
        if (std::fabs(_alpha)>1e-4)
        {
            mDuration_type t_if(0);
            domain_->client_communicator().barrier();
            TIME_CODE( t_if, SINGLE_ARG(
                        psolver.template apply_lgf_IF<r_i_type, u_i_type>(_alpha);
                        ));
            pcout<< "IF  solved in "<<t_if.count() << std::endl;
        }
        else
            copy<r_i_type,u_i_type>();
    }


    void lin_sys_with_ib_solve(float_type _alpha, bool write_prev_force = true) noexcept
    {
        auto client=domain_->decomposition().client();

        divergence<r_i_type, cell_aux_type>();

        domain_->client_communicator().barrier();
        mDuration_type t_lgf(0);
        TIME_CODE( t_lgf, SINGLE_ARG(
                    psolver.template apply_lgf<cell_aux_type, d_i_type>();
                    ));
        domain_->client_communicator().barrier();
        pcout<< "LGF solved in "<<t_lgf.count() << std::endl;

        copy<r_i_type, face_aux2_type>();
        gradient<d_i_type,face_aux_type>();
        add<face_aux_type, face_aux2_type>(-1.0);

        // IB
        if (std::fabs(_alpha)>1e-14)
            psolver.template apply_lgf_IF<face_aux2_type, face_aux2_type>(_alpha, MASK_TYPE::IB2xIB);

        domain_->client_communicator().barrier();
        pcout<< "IB IF solved "<<std::endl;
        mDuration_type t_ib(0);
        domain_->ib().force() = domain_->ib().force_prev(stage_idx_);
        //domain_->ib().scales(coeff_a(stage_idx_, stage_idx_));
        TIME_CODE( t_ib, SINGLE_ARG(
                    lsolver.template ib_solve<face_aux2_type>(_alpha, T_stage_);
                    ));

        if (write_prev_force) domain_->ib().force_prev(stage_idx_) = domain_->ib().force();
        //domain_->ib().scales(1.0/coeff_a(stage_idx_, stage_idx_));

        pcout<< "IB  solved in "<<t_ib.count() << std::endl;

        // new presure field
        lsolver.template pressure_correction<d_i_type>();
        gradient<d_i_type, face_aux_type>();

        lsolver.template smearing<face_aux_type>(domain_->ib().force(), false);
        add<face_aux_type, r_i_type>(-1.0);

        if (std::fabs(_alpha)>1e-14)
        {
            mDuration_type t_if(0);
            domain_->client_communicator().barrier();
            TIME_CODE( t_if, SINGLE_ARG(
                        psolver.template apply_lgf_IF<r_i_type, u_i_type>(_alpha);
                        ));
            pcout<< "IF  solved in "<<t_if.count() << std::endl;
        }
        else
            copy<r_i_type,u_i_type>();

        // test -------------------------------------
        //force_type tmp(domain_->ib().force().size(), (0.,0.,0.));
        //lsolver.template projection<u_i_type>(tmp);
        //domain_->ib().communicator().compute_indices();
        //domain_->ib().communicator().communicate(true, tmp);
        //if (comm_.rank()==1)
        //{
        //    lsolver.printvec(tmp, "u");
        //}


    }



    template<class Velocity_in, class Velocity_out>
    void pad_velocity(bool refresh_correction_only=true)
    {
        auto client=domain_->decomposition().client();

        //up_and_down<Velocity_in>();
        clean<Velocity_in>(true);
        this->up<Velocity_in>(false);
        clean<edge_aux_type>();
        clean<stream_f_type>();

        auto dx_base = domain_->dx_base();

        for (int l  = domain_->tree()->base_level();
                l < domain_->tree()->depth(); ++l)
        {
            client->template buffer_exchange<Velocity_in>(l);

            for (auto it  = domain_->begin(l);
                    it != domain_->end(l); ++it)
            {
                if(!it->locally_owned() || !it->has_data()) continue;
                if(it->is_correction()) continue;
                //if(!it->is_leaf()) continue;

                const auto dx_level =  dx_base/math::pow2(it->refinement_level());
                //if (it->is_leaf())
                domain::Operator::curl<Velocity_in,edge_aux_type>( it->data(),dx_level);
            }
        }

        //clean<Velocity_out>();
        clean_leaf_correction_boundary<edge_aux_type>(domain_->tree()->base_level(), true, 2);
        //clean_leaf_correction_boundary<edge_aux_type>(l, false,2+stage_idx_);
        psolver.template apply_lgf<edge_aux_type, stream_f_type>(MASK_TYPE::STREAM);

        int l_max = refresh_correction_only ?
        domain_->tree()->base_level()+1 : domain_->tree()->depth();
        for (int l  = domain_->tree()->base_level();
                l < l_max; ++l)
        {
            for (auto it  = domain_->begin(l);
                    it != domain_->end(l); ++it)
            {
                if(!it->locally_owned() || !it->has_data()) continue;
                //if(!it->is_correction() && refresh_correction_only) continue;

                const auto dx_level =  dx_base/math::pow2(it->refinement_level());
                    domain::Operator::curl_transpose<stream_f_type,Velocity_out>( it->data(),dx_level, -1.0);
            }
        }

        this->down_to_correction<Velocity_out>();
    }



    //TODO maybe to be put directly intor operators:
    template<class Source, class Target>
    void nonlinear(float_type _scale = 1.0) noexcept
    {
        clean<edge_aux_type>();
        clean<Target>();
        clean<face_aux_type>();

        auto       client = domain_->decomposition().client();
        const auto dx_base = domain_->dx_base();

        //up_and_down<Source>();

        for (int l = domain_->tree()->base_level();
             l < domain_->tree()->depth(); ++l)
        {

            client->template buffer_exchange<Source>(l);

            for (auto it = domain_->begin(l); it != domain_->end(l); ++it)
            {
                if (!it->locally_owned() || !it->has_data()) continue;

                const auto dx_level =
                    dx_base / math::pow2(it->refinement_level());
                domain::Operator::curl<Source, edge_aux_type>(
                    it->data(), dx_level);
            }
        }

        // clean_leaf_correction_boundary<edge_aux_type>(domain_->tree()->base_level(), true, 2);
        // add background velocity
        copy<Source, face_aux_type>();
        domain::Operator::add_field_expression<face_aux_type>(domain_, simulation_->frame_vel(), T_stage_, -1.0);

        for (int l = domain_->tree()->base_level();
             l < domain_->tree()->depth(); ++l)
        {
            client->template buffer_exchange<edge_aux_type>(l);
            // client->template buffer_exchange<face_aux_type>(l);
            clean_leaf_correction_boundary<edge_aux_type>(l, false, 2);

            for (auto it = domain_->begin(l); it != domain_->end(l); ++it)
            {
                if (!it->locally_owned() || !it->has_data()) continue;

                domain::Operator::nonlinear<face_aux_type, edge_aux_type, Target>(
                    it->data());

                for (std::size_t field_idx = 0; field_idx < Target::nFields();
                     ++field_idx)
                {
                    auto& lin_data =
                        it->data_r(Target::tag(), field_idx).linalg_data();
                    lin_data *= _scale; //scale with time step size
                }
            }

            //client->template buffer_exchange<Target>(l);
            //clean_leaf_correction_boundary<Target>(l, true,3);
        }

        if (std::abs(b_f_mag) > 1e-5) {
            add_body_force<Target>(_scale);
        }

        
    }

    template<class target>
    void add_body_force(float_type scale) noexcept {
        //float_type eps = 1e-3;
        auto dx_base = domain_->dx_base();
        for (auto it = domain_->begin(); it != domain_->end(); ++it)
		{

			if (!it->locally_owned()) continue;

			auto dx_level = dx_base / std::pow(2, it->refinement_level());
			auto scaling = std::pow(2, it->refinement_level());

			for (auto& node : it->data())
			{

				const auto& coord = node.level_coordinate();

				
				float_type x = static_cast<float_type>
					(coord[0]) * dx_level;
				float_type y = static_cast<float_type>
					(coord[1]) * dx_level;
				//z = static_cast<float_type>
				//(coord[2]-center[2]*scaling+0.5)*dx_level;

				//node(edge_aux,0) = vor(x,y-0.5*vort_sep,0)+ vor(x,y+0.5*vort_sep,0);
				node(target::tag(), 1) += -scale * b_f_mag * y / (y*y + b_f_eps);

            }

		}
    }

    template<class Source, class Target>
    void divergence() noexcept
    {
        auto client = domain_->decomposition().client();

        up_and_down<Source>();

        for (int l = domain_->tree()->base_level();
             l < domain_->tree()->depth(); ++l)
        {
            client->template buffer_exchange<Source>(l);
            const auto dx_base = domain_->dx_base();

            for (auto it = domain_->begin(l); it != domain_->end(l); ++it)
            {
                if (!it->locally_owned() || !it->has_data()) continue;
                const auto dx_level =
                    dx_base / math::pow2(it->refinement_level());
                domain::Operator::divergence<Source, Target>(
                    it->data(), dx_level);
            }

            //client->template buffer_exchange<Target>(l);
            clean_leaf_correction_boundary<Target>(l, true, 2);
            //clean_leaf_correction_boundary<Target>(l, false,4+stage_idx_);
        }
    }

    template<class Source, class Target>
    void gradient(float_type _scale = 1.0) noexcept
    {
        //up_and_down<Source>();
        domain::Operator::domainClean<Target>(domain_);

        for (int l = domain_->tree()->base_level();
             l < domain_->tree()->depth(); ++l)
        {
            auto client = domain_->decomposition().client();
            //client->template buffer_exchange<Source>(l);
            const auto dx_base = domain_->dx_base();
            for (auto it = domain_->begin(l); it != domain_->end(l); ++it)
            {
                if (!it->locally_owned() || !it->has_data()) continue;
                const auto dx_level =
                    dx_base / math::pow2(it->refinement_level());
                domain::Operator::gradient<Source, Target>(
                    it->data(), dx_level);
                for (std::size_t field_idx = 0; field_idx < Target::nFields();
                     ++field_idx)
                {
                    auto& lin_data =
                        it->data_r(Target::tag(), field_idx).linalg_data();

                    lin_data *= _scale;
                }
            }
            client->template buffer_exchange<Target>(l);
        }
    }


    template<class Source, class Target>
    void laplacian() noexcept
    {
        auto client = domain_->decomposition().client();

        domain::Operator::domainClean<Target>(domain_);

        clean<edge_aux_type>();

        up_and_down<Source>();

        /*for (int l = domain_->tree()->base_level();
             l < domain_->tree()->depth(); ++l)
        {
            client->template buffer_exchange<Source>(l);
            const auto dx_base = domain_->dx_base();

            for (auto it = domain_->begin(l); it != domain_->end(l); ++it)
            {
                if (!it->locally_owned() || !it->has_data()) continue;
                const auto dx_level =
                    dx_base / math::pow2(it->refinement_level());
                domain::Operator::laplace<Source, Target>(it->data(), dx_level);
            }

            //client->template buffer_exchange<Target>(l);
            //clean_leaf_correction_boundary<Target>(l, true, 2);
            //clean_leaf_correction_boundary<Target>(l, false,4+stage_idx_);
        }*/


        for (int l = domain_->tree()->base_level();
             l < domain_->tree()->depth(); ++l)
        {
            client->template buffer_exchange<Source>(l);
            const auto dx_base = domain_->dx_base();

            for (auto it = domain_->begin(l); it != domain_->end(l); ++it)
            {
                if (!it->locally_owned() || !it->has_data()) continue;
                if (!it->is_leaf() && !it->is_correction()) continue;
                const auto dx_level =
                    dx_base / math::pow2(it->refinement_level());
                domain::Operator::curl<Source, edge_aux_type>(it->data(), dx_level);
            }

            //client->template buffer_exchange<Target>(l);
            //clean_leaf_correction_boundary<Target>(l, true, 2);
            //clean_leaf_correction_boundary<Target>(l, false,4+stage_idx_);
        }

        this->up<edge_aux_type>();

        for (int l = domain_->tree()->base_level();
             l < domain_->tree()->depth(); ++l)
        {
            client->template buffer_exchange<edge_aux_type>(l);
            const auto dx_base = domain_->dx_base();

            for (auto it = domain_->begin(l); it != domain_->end(l); ++it)
            {
                if (!it->locally_owned() || !it->has_data()) continue;
                if (!it->is_leaf() && !it->is_correction()) continue;
                const auto dx_level =
                    dx_base / math::pow2(it->refinement_level());
                domain::Operator::curl_transpose<edge_aux_type, Target>(it->data(), dx_level, -1.0);
            }

            //client->template buffer_exchange<Target>(l);
            //clean_leaf_correction_boundary<Target>(l, true, 2);
            //clean_leaf_correction_boundary<Target>(l, false,4+stage_idx_);
        }

        //clean_leaf_correction_boundary<Target>(domain_->tree()->base_level(), true, 2);

        //clean<Source>(true);
        //clean<Target>(true);
    }

    template<typename From, typename To>
    void add(float_type scale = 1.0) noexcept
    {
        static_assert(From::nFields() == To::nFields(),
            "number of fields doesn't match when add");
        for (auto it = domain_->begin(); it != domain_->end(); ++it)
        {
            if (!it->locally_owned() || !it->has_data()) continue;
            for (std::size_t field_idx = 0; field_idx < From::nFields();
                 ++field_idx)
            {
                it->data_r(To::tag(), field_idx)
                    .linalg()
                    .get()
                    ->cube_noalias_view() +=
                    it->data_r(From::tag(), field_idx).linalg_data() * scale;
            }
        }
    }

    template<typename Field>
    void computeWii() noexcept
    {
        auto dx_base = domain_->dx_base();
        for (auto it = domain_->begin(); it != domain_->end(); ++it)
		{

			if (!it->locally_owned()) continue;

			auto dx_level = dx_base / std::pow(2, it->refinement_level());
			auto scaling = std::pow(2, it->refinement_level());

            for (std::size_t field_idx = 0; field_idx < Field::nFields();
                 ++field_idx)
            {
                for (auto& n:it->data().node_field()) {
                    float_type val = n(Field::tag(), field_idx);

                    if (std::fabs(val) > 1e-4) {
                        n(Field::tag(), field_idx) = 1/val;
                    }
                }
            }
		}
    }

    template<typename From1, typename From2, typename To>
    void product() noexcept
    {
        static_assert(From1::nFields() == To::nFields(),
            "number of fields doesn't match when copy");

        static_assert(From2::nFields() == To::nFields(),
            "number of fields doesn't match when copy");

        for (auto it = domain_->begin(); it != domain_->end(); ++it)
        {
            if (!it->locally_owned() || !it->has_data()) continue;
            for (std::size_t field_idx = 0; field_idx < From1::nFields();
                 ++field_idx)
            {
                for (auto& n:it->data().node_field())
                    n(To::tag(), field_idx) = n(From1::tag(), field_idx) * n(From2::tag(), field_idx);
            }
        }
    }

    template<typename From, typename To>
    void copy(float_type scale = 1.0) noexcept
    {
        static_assert(From::nFields() == To::nFields(),
            "number of fields doesn't match when copy");

        for (auto it = domain_->begin(); it != domain_->end(); ++it)
        {
            if (!it->locally_owned() || !it->has_data()) continue;
            for (std::size_t field_idx = 0; field_idx < From::nFields();
                 ++field_idx)
            {
                for (auto& n:it->data().node_field())
                    n(To::tag(), field_idx) = n(From::tag(), field_idx) * scale;
            }
        }
    }

  private:
    simulation_type* simulation_;
    domain_type*     domain_; ///< domain
    poisson_solver_t psolver;
    linsys_solver_t lsolver;


    bool base_mesh_update_=false;

    float_type T_, T_stage_, T_max_;
    float_type dt_base_, dt_, dx_base_;
    float_type Re_;
    float_type cfl_max_, cfl_;
    std::vector<float_type> source_max_{0.0, 0.0};

    float_type cg_threshold_;
    int  cg_max_itr_;

    float_type T_last_vel_refresh_=0.0;

    float_type b_f_mag, b_f_eps;

    std::vector<float_type> omega_cross_x;

    int max_vel_refresh_=1;
    int max_ref_level_=0;
    int output_base_freq_;
    int adapt_freq_;
    int tot_base_steps_;
    int n_step_ = 0;
    int restart_n_last_ = 0;
    int nLevelRefinement_;
    int stage_idx_ = 0;

    bool use_filter = true; //if use filter when computing force

    bool use_restart_=false;
    bool just_restarted_=false;
    bool write_restart_=false;
    bool updating_source_max_ = false;
    bool write_stress_= false;
    bool write_noca_= false;
    bool all_time_max_;
    bool use_adaptation_correction;
    int restart_base_freq_;
    int adapt_count_;

    std::string                       fname_prefix_;
    vector_type<float_type, 6>        a_{{1.0 / 3, -1.0, 2.0, 0.0, 0.75, 0.25}};
    vector_type<float_type, 4>        c_{{0.0, 1.0 / 3, 1.0, 1.0}};
    //vector_type<float_type, 6>        a_{{1.0 / 2, sqrt(3)/3, (3-sqrt(3))/3, (3+sqrt(3))/6, -sqrt(3)/3, (3+sqrt(3))/6}};
    //vector_type<float_type, 4>        c_{{0.0, 0.5, 1.0, 1.0}};
    vector_type<float_type, 3>        alpha_{{0.0, 0.0, 0.0}};
    parallel_ostream::ParallelOstream pcout =
        parallel_ostream::ParallelOstream(1);
    boost::mpi::communicator comm_;

};

} // namespace solver
} // namespace iblgf

#endif // IBLGF_INCLUDED_POISSON_HPP
