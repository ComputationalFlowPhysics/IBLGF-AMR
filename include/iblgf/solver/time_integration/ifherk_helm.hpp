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

#ifndef IBLGF_INCLUDED_IFHERK_HELM_SOLVER_HPP
#define IBLGF_INCLUDED_IFHERK_HELM_SOLVER_HPP

#include <iostream>
#include <algorithm>
#include <vector>
#include <cmath>
#include <array>
#include <time.h>
#include <random>

// IBLGF-specific
#include <iblgf/global.hpp>
#include <iblgf/simulation.hpp>
#include <iblgf/domain/domain.hpp>
#include <iblgf/IO/parallel_ostream.hpp>
#include <iblgf/solver/poisson/poisson.hpp>
#include <iblgf/solver/linsys/linsys_helm.hpp>
#include <iblgf/operators/operators.hpp>
#include <iblgf/utilities/misc_math_functions.hpp>
#include <iblgf/solver/time_integration/HelmholtzFFT.hpp>

namespace iblgf
{
namespace solver
{
using namespace domain;

/** @brief Integrating factor 3-stage Runge-Kutta time integration
 * */
template<class Setup>
class Ifherk_HELM
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
    using point_force_type = typename ib_t::point_force_type;

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
    using u_i_real_type = typename Setup::u_i_real_type;
    using vort_i_real_type = typename Setup::vort_i_real_type;
    using r_i_real_type = typename Setup::r_i_real_type;
    using face_aux_real_type = typename Setup::face_aux_real_type;


    static constexpr int lBuffer = 1; ///< Lower left buffer for interpolation
    static constexpr int rBuffer = 1; ///< Lower left buffer for interpolation
    static constexpr std::size_t N_modes = Setup::N_modes; //number of complex modes
    static constexpr std::size_t padded_dim = N_modes*3;
    static constexpr std::size_t nonzero_dim = N_modes*2-1; //minus one to take out the zeroth mode (counted twice if multiply by two directly)
    Ifherk_HELM(simulation_type* _simulation)
    : simulation_(_simulation)
    , domain_(_simulation->domain_.get())
    , psolver(_simulation, N_modes - 1) //minus one to exclude the mode at zero frequency number, will be added back in the Poisson Solver
    , lsolver(_simulation)
    , r2cFunc(padded_dim, nonzero_dim, (domain_->block_extent()[0]+lBuffer+rBuffer), (domain_->block_extent()[1]+lBuffer+rBuffer))
    , c2rFunc(padded_dim, nonzero_dim, (domain_->block_extent()[0]+lBuffer+rBuffer), (domain_->block_extent()[1]+lBuffer+rBuffer))
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
        customized_ic = _simulation->dictionary()->template get_or<bool>("use_init_tree", false);
        c_z = _simulation->dictionary()->template get_or<float_type>(
            "L_z", 1);

        adapt_Fourier = _simulation->dictionary()->template get_or<bool>(
            "adapt_Fourier", true);
        /*const int l_max = domain_->tree()->depth();
        const int l_min = domain_->tree()->base_level();
        const int nLevels = l_max - l_min;
        c_z = dx_base_*N_modes*2/std::pow(2.0, nLevels - 1);*/

        pcout << "c_z is " << c_z << std::endl;

        perturb_nonlin = _simulation->dictionary()->template get_or<float_type>(
            "perturb_nonlin", 0.0);

        additional_modes = _simulation->dictionary()->template get_or<int>("add_modes", N_modes - 1);

        if (dt_base_ < 0) dt_base_ = dx_base_ * cfl_;

        // adaptivity --------------------------------------------------------
        adapt_freq_ = _simulation->dictionary()->template get_or<float_type>(
            "adapt_frequency", 1);
        T_max_ = tot_base_steps_ * dt_base_;
        update_marching_parameters();

        // support of IF in every dirrection is about 3.2 corresponding to 1e-5
        // with coefficient alpha = 1
        // so need update
        max_vel_refresh_ =
            floor(14 / (3.3 / (Re_ * dx_base_ * dx_base_ / dt_)));
        pcout << "maximum steps allowed without vel refresh = "
              << max_vel_refresh_ << std::endl;

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
    }

  public:
    void update_marching_parameters()
    {
        nLevelRefinement_ =
            domain_->tree()->depth() - domain_->tree()->base_level() - 1;
        dt_ = dt_base_ / math::pow2(nLevelRefinement_);

        float_type tmp = Re_ * dx_base_ * dx_base_ / dt_;
        alpha_[0] = (c_[1] - c_[0]) / tmp;
        alpha_[1] = (c_[2] - c_[1]) / tmp;
        alpha_[2] = (c_[3] - c_[2]) / tmp;
    }
    void time_march(bool use_restart = false)
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
            just_restarted_ = true;
            customized_ic   = false;
            Dictionary info_d(
                simulation_->restart_load_dir() + "/restart_info");
            T_ = info_d.template get<float_type>("T");
            adapt_count_ = info_d.template get<int>("adapt_count");
            T_last_vel_refresh_ =
                info_d.template get_or<float_type>("T_last_vel_refresh", 0.0);
            source_max_[0] = info_d.template get<float_type>("cell_aux_max");
            source_max_[1] = info_d.template get<float_type>("u_max");
            pcout
                << "Restart info ------------------------------------------------ "
                << std::endl;
            pcout << "T = " << T_ << std::endl;
            pcout << "adapt_count = " << adapt_count_ << std::endl;
            pcout << "cell aux max = " << source_max_[0] << std::endl;
            pcout << "u max = " << source_max_[1] << std::endl;
            pcout << "T_last_vel_refresh = " << T_last_vel_refresh_
                  << std::endl;
            if (domain_->is_client())
            {
                //pad_velocity<u_type, u_type>(true);
            }
        }
        else
        {
            T_ = 0.0;
            adapt_count_ = 0;

            write_timestep();
        }

        // ----------------------------------- start -------------------------

        while (T_ < T_max_ - 1e-10)
        {
            // -------------------------------------------------------------
            // adapt

            // clean up the block boundary of cell_aux_type for smoother adaptation

            if (domain_->is_client())
            {
                clean<cell_aux_type>(true, 2);
                clean<edge_aux_type>(true, 1);
                clean<correction_tmp_type>(true, 2);
                if (adapt_Fourier)
                {
                    clean_Fourier_modes_all<cell_aux_type>();
                    clean_Fourier_modes_all<edge_aux_type>();
                    //clean_Fourier_modes_all<correction_tmp_type>();
                }
            }
            else
            {
                //const auto& lb = domain_->level_blocks();
                /*std::vector<int> lb;
		        domain_->level_blocks(lb);*/
                auto lb = domain_->level_blocks();
                std::cout << "Blocks on each level: ";

                for (int c : lb) std::cout << c << " ";
                std::cout << std::endl;
            }

            // copy flag correction to flag old correction
            for (auto it = domain_->begin(); it != domain_->end(); ++it)
            {
                it->flag_old_correction(false);
            }

            for (auto it = domain_->begin(domain_->tree()->base_level());
                 it != domain_->end(domain_->tree()->base_level()); ++it)
            {
                it->flag_old_correction(it->is_correction());
            }

            int c = 0;

            for (auto it = domain_->begin(); it != domain_->end(); ++it)
            {
                if (!it->locally_owned()) continue;
                if (it->is_ib() || it->is_extended_ib())
                {
                    auto& lin_data =
                        it->data_r(test_type::tag(), 0).linalg_data();
                    std::fill(lin_data.begin(), lin_data.end(), 2.0);
                    c += 1;
                }
            }
            boost::mpi::communicator world;
            int                      c_all;
            boost::mpi::all_reduce(world, c, c_all, std::plus<int>());
            pcout << "block = " << c_all << std::endl;

            if (adapt_count_ % adapt_freq_ == 0)
            {
                if (adapt_count_ == 0 || updating_source_max_)
                {
                    this->template update_source_max<cell_aux_type>(0);
                    this->template update_source_max<edge_aux_type>(1);
                }

                //if(domain_->is_client())
                //{
                //    up_and_down<u>();
                //    pad_velocity<u, u>();
                //}
                if (!just_restarted_ && !customized_ic) this->adapt(false);
                just_restarted_ = false;
                customized_ic   = false;
            }

            // balance load
            if (adapt_count_ % adapt_freq_ == 0)
            {
                clean<u_type>(true);
                //domain_->decomposition().template balance<u_type,p_type>();
            }

            adapt_count_++;

            // -------------------------------------------------------------
            // time marching

            mDuration_type ifherk_if(0);
            TIME_CODE(ifherk_if, SINGLE_ARG(time_step();));
            pcout << "RK3 solved in " << ifherk_if.count() << std::endl;

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
                    domain::Operator::curl_helmholtz_complex<u_type, edge_aux_type>(it->data(),
                        dx_level, N_modes, c_z);
                }
            }
            //clean_leaf_correction_boundary<edge_aux_type>(domain_->tree()->base_level(), true,2);

            clean<u_type>();
            psolver.template apply_lgf_and_helm<edge_aux_type, stream_f_type>(N_modes, 3);
            for (int l = domain_->tree()->base_level();
                 l < domain_->tree()->depth(); ++l)
            {
                for (auto it = domain_->begin(l); it != domain_->end(l); ++it)
                {
                    if (!it->locally_owned()) continue;

                    const auto dx_level =
                        dx_base_ / math::pow2(it->refinement_level());
                    domain::Operator::curl_transpose_helmholtz_complex<stream_f_type, u_type>(
                        it->data(), dx_level, N_modes, c_z, -1.0);
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
            float_type tmp =
                domain::Operator::blockRootMeanSquare<Field>(it->data());

            if (tmp > max_local) max_local = tmp;
        }

        float_type new_maximum = 0.0;
        boost::mpi::all_reduce(comm_, max_local, new_maximum,
            boost::mpi::maximum<float_type>());

        //source_max_[idx] = std::max(source_max_[idx], new_maximum);
        if (all_time_max_)
            source_max_[idx] = std::max(source_max_[idx], new_maximum);
        else
            source_max_[idx] = 0.5 * (source_max_[idx] + new_maximum);

        if (source_max_[idx] < 1e-2) source_max_[idx] = 1e-2;

        pcout << "source max = " << source_max_[idx] << std::endl;
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
        boost::mpi::all_reduce(world, c_allc, c_allc_global, std::plus<int>());

        if (domain_->is_server())
        {
            std::cout << "T = " << T_ << ", n = " << tmp_n
                      << " -----------------" << std::endl;
            auto lb = domain_->level_blocks();
            std::cout << "Blocks on each level: ";

            for (int c : lb) std::cout << c << " ";
            std::cout << std::endl;

            std::cout << "Total number of leaf octants: "
                      << domain_->num_leafs() << std::endl;
            std::cout << "Total number of leaf + correction octants: "
                      << domain_->num_corrections() + domain_->num_leafs()
                      << std::endl;
            std::cout << "Total number of allocated octants: " << c_allc_global
                      << std::endl;
            std::cout << " -----------------" << std::endl;
        }

        // - Forcing ------------------------------------------------
        auto& ib = domain_->ib();
        ib.clean_non_local();
        point_force_type tmp_force(0.0);

        force_type sum_f(ib.force().size(), tmp_force);
        if (ib.size() > 0)
        {
            boost::mpi::all_reduce(world, &ib.force(0), ib.size(), &sum_f[0],
                std::plus<point_force_type>());
        }
        if (domain_->is_server())
        {
            std::vector<float_type> f(ib_t::force_dim, 0.);
            if (ib.size() > 0)
            {
                for (std::size_t d = 0; d < ib_t::force_dim; ++d)
                    for (std::size_t i = 0; i < ib.size(); ++i)
                        f[d] += sum_f[i][d] * 1.0 / coeff_a(3, 3) / dt_ *
                                ib.force_scale();
                //f[d]+=sum_f[i][d] * 1.0 / dt_ * ib.force_scale();

                std::cout << "ib  size: " << ib.size() << std::endl;
                std::cout << "Forcing = ";

                for (std::size_t d = 0; d < 3; ++d)
                    std::cout << f[d*2*N_modes] << " ";
                std::cout << std::endl;

                std::cout << " -----------------" << std::endl;
            }

            std::ofstream outfile;
            int           width = 20;

            outfile.open("fstats.txt",
                std::ios_base::app); // append instead of overwrite
            outfile << std::setw(width) << tmp_n << std::setw(width)
                    << std::scientific << std::setprecision(9);
            outfile << std::setw(width) << T_ << std::setw(width)
                    << std::scientific << std::setprecision(9);
            for (int  i = 0 ; i < 3; i++) { outfile << f[i*2*N_modes] << std::setw(width); }
            outfile << std::endl;
            outfile.close();
            std::cout << "finished writing forcing" << std::endl;
        }

        world.barrier();
    }

    void write_timestep()
    {
        boost::mpi::communicator world;
        world.barrier();
        pcout << "- writing at T = " << T_ << ", n = " << n_step_ << std::endl;
        simulation_->write(fname(n_step_));
        //simulation_->domain()->tree()->write("tree_restart.bin");
        world.barrier();
        //simulation_->domain()->tree()->read("tree_restart.bin");
        pcout << "- output writing finished -" << std::endl;
    }

    void write_info()
    {
        if (domain_->is_server())
        {
            std::ofstream ofs(simulation_->restart_write_dir() +
                                  "/restart_info",
                std::ofstream::out);
            if (!ofs.is_open())
            {
                throw std::runtime_error("Could not open file for info write ");
            }

            ofs.precision(20);
            ofs << "T = " << T_ << ";" << std::endl;
            ofs << "adapt_count = " << adapt_count_ << ";" << std::endl;
            ofs << "cell_aux_max = " << source_max_[0] << ";" << std::endl;
            ofs << "u_max = " << source_max_[1] << ";" << std::endl;
            ofs << "restart_n_last = " << restart_n_last_ << ";" << std::endl;
            ofs << "T_last_vel_refresh = " << T_last_vel_refresh_ << ";"
                << std::endl;

            ofs.close();
        }
    }

    std::string fname(int _n) { return fname_prefix_ + std::to_string(_n); }

    // ----------------------------------------------------------------------
    template<class Field>
    void up_and_down()
    {
        //claen non leafs
        auto t0 = clock_type::now();
        clean<Field>(true);
        auto t1 = clock_type::now();
        mDuration_type ms_int = t1 - t0;
        pcout << "Cleaning in up and down in " << ms_int.count() << std::endl;
        this->up<Field>();
        auto t2 = clock_type::now();
        mDuration_type ms_up = t2 - t1;
        pcout << "Upward interpolation in up and down in " << ms_up.count() << std::endl;
        this->down_to_correction<Field>();
        auto t3 = clock_type::now();
        mDuration_type ms_down = t3 - t2;
        pcout << "Downward interpolation in up and down in " << ms_down.count() << std::endl;
    }

    template<class Field>
    void up(bool leaf_boundary_only = false)
    {
        //Coarsification:
        /*for (std::size_t _field_idx = 0; _field_idx < Field::nFields();
             ++_field_idx)
            psolver.template source_coarsify_refined<Field, Field>(_field_idx,
                _field_idx, Field::mesh_type(), false, false, false,
                leaf_boundary_only);*/
        psolver.template source_coarsify_all_comp<Field, Field>(Field::mesh_type(), false, false, false,
                leaf_boundary_only);
    }

    template<class Field>
    void down_to_correction()
    {
        // Interpolate to correction buffer
        /*for (std::size_t _field_idx = 0; _field_idx < Field::nFields();
             ++_field_idx)
            psolver.template intrp_to_correction_buffer<Field, Field>(
                _field_idx, _field_idx, Field::mesh_type(), true, false);*/
        psolver.template intrp_to_correction_buffer_all_comp<Field, Field>(
            Field::mesh_type(), true, false);
    }

    void adapt(bool coarsify_field = true)
    {
        boost::mpi::communicator world;
        auto                     client = domain_->decomposition().client();

        if (source_max_[0] < 1e-10 || source_max_[1] < 1e-10) return;

        //adaptation neglect the boundary oscillations
        clean_leaf_correction_boundary<cell_aux_type>(
            domain_->tree()->base_level(), true, 2);
        clean_leaf_correction_boundary<edge_aux_type>(
            domain_->tree()->base_level(), true, 2);

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
        pcout << "Adapt - communication" << std::endl;
        auto intrp_list = domain_->adapt(source_max_, base_mesh_update_);

        world.barrier();
        pcout << "Adapt - intrp" << std::endl;
        if (client)
        {
            // Intrp
            for (std::size_t _field_idx = 0; _field_idx < u_type::nFields();
                 ++_field_idx)
            {
                for (int l = domain_->tree()->depth() - 2;
                     l >= domain_->tree()->base_level(); --l)
                {
                    client->template buffer_exchange<u_type>(l);
                    //std::cout << world.rank() << " after buffer exchange" << std::endl;
                    domain_->decomposition()
                        .client()
                        ->template communicate_updownward_assign<u_type,
                            u_type>(l, false, false, -1, _field_idx);
                    //std::cout << world.rank() << " after communicate_updownward_assign" << std::endl;
                }
                //std::cout << world.rank() << " after first block" << std::endl;
                for (auto& oct : intrp_list)
                {
                    if (!oct || !oct->has_data()) continue;
                    psolver.c_cntr_nli()
                        .template nli_intrp_node<u_type, u_type>(oct,
                            u_type::mesh_type(), _field_idx, _field_idx, false,
                            false);
                }
                //std::cout << world.rank() << " after interpolation" << std::endl;
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

        stage_idx_ = 0;

        // Solve stream function to refresh base level velocity
        mDuration_type t_pad(0);
        TIME_CODE(t_pad, SINGLE_ARG(
                             pcout << "base level mesh update = "
                                   << base_mesh_update_ << std::endl;
                             if (base_mesh_update_ ||
                                 ((T_ - T_last_vel_refresh_) /
                                         (Re_ * dx_base_ * dx_base_) * 3.3 >
                                     7))
                             {
                                 pcout << "pad_velocity, last T_vel_refresh = "
                                       << T_last_vel_refresh_ << std::endl;
                                 T_last_vel_refresh_ = T_;
                                 if (!domain_->is_client()) return;
                                 pad_velocity<u_type, u_type>(true);
                             } else
                             {
                                 if (!domain_->is_client()) return;
                                 up_and_down<u_type>();
                             }));
        base_mesh_update_ = false;
        pcout << "pad u      in " << t_pad.count() << std::endl;
        if (adapt_Fourier) clean_Fourier_modes_all<u_type>();
        copy<u_type, q_i_type>();

        // Stage 1
        // ******************************************************************
        pcout << "Stage 1" << std::endl;
        auto t0 = clock_type::now();

        
        if (adapt_Fourier) {
            clean_Fourier_modes_all<u_i_type>();
            clean_Fourier_modes_all<r_i_type>();
            clean_Fourier_modes_all<q_i_type>();
        }
        auto t1 = clock_type::now();
        mDuration_type ms_int = t1 - t0;
        pcout << "Cleaning Fourier coeff in " << ms_int.count() << std::endl;
        T_stage_ = T_ + dt_ * c_[0];
        stage_idx_ = 1;
        clean<g_i_type>();
        clean<d_i_type>();
        clean<cell_aux_type>();
        clean<face_aux_type>();
        auto t2 = clock_type::now();
        mDuration_type ms_int_clean = t2 - t1;
        pcout << "Cleaning fields in " << ms_int_clean.count() << std::endl;

        mDuration_type nonlinear1(0);
        TIME_CODE(nonlinear1, SINGLE_ARG(nonlinear<u_type, g_i_type>(coeff_a(1, 1) * (-dt_));));
        pcout << "nonlinear term solved in " << nonlinear1.count() << std::endl;

        auto t3 = clock_type::now();       
        //nonlinear<u_type, g_i_type>(coeff_a(1, 1) * (-dt_));
        copy<q_i_type, r_i_type>();
        add<g_i_type, r_i_type>();

        auto t4 = clock_type::now();
        mDuration_type ms_int_add_copy = t4 - t3;
        pcout << "Add and copy fields in " << ms_int_add_copy.count() << std::endl;

        mDuration_type linsys1(0);
        TIME_CODE(linsys1, SINGLE_ARG(lin_sys_with_ib_solve(alpha_[0]);));
        pcout << "linsys solved in " << linsys1.count() << std::endl;
        //lin_sys_with_ib_solve(alpha_[0]);

        // Stage 2
        // ******************************************************************
        pcout << "Stage 2" << std::endl;
        if (adapt_Fourier) {
            clean_Fourier_modes_all<u_i_type>();
            clean_Fourier_modes_all<q_i_type>();
            clean_Fourier_modes_all<g_i_type>();
        }
        T_stage_ = T_ + dt_ * c_[1];
        stage_idx_ = 2;
        clean<r_i_type>();
        clean<d_i_type>();
        clean<cell_aux_type>();

        //cal wii
        //r_i_type = q_i_type + dt(a21 w21)
        //w11 = (1/a11)* dt (g_i_type - face_aux_type)

        add<g_i_type, face_aux_type>(-1.0);
        copy<face_aux_type, w_1_type>(-1.0 / dt_ / coeff_a(1, 1));

        //psolver.template apply_lgf_IF<q_i_type, q_i_type>(alpha_[0]);
        //psolver.template apply_lgf_IF<w_1_type, w_1_type>(alpha_[0]);

        psolver.template apply_helm_if<q_i_type, q_i_type>(alpha_[0], N_modes, c_z);
        psolver.template apply_helm_if<w_1_type, w_1_type>(alpha_[0], N_modes, c_z);

        add<q_i_type, r_i_type>();
        add<w_1_type, r_i_type>(dt_ * coeff_a(2, 1));

        auto t5 = clock_type::now();

        up_and_down<u_i_type>();

        auto t6 = clock_type::now();
        mDuration_type ms_up_and_down = t6 - t5;
        pcout << "Up and down in " << ms_up_and_down.count() << std::endl;
        
        mDuration_type nonlinear2(0);
        TIME_CODE(nonlinear2, SINGLE_ARG(nonlinear<u_i_type, g_i_type>(coeff_a(2, 2) * (-dt_));));
        pcout << "nonlinear term solved in " << nonlinear2.count() << std::endl;
        //nonlinear<u_i_type, g_i_type>(coeff_a(2, 2) * (-dt_));
        add<g_i_type, r_i_type>();

        mDuration_type linsys2(0);
        TIME_CODE(linsys2, SINGLE_ARG(lin_sys_with_ib_solve(alpha_[1]);));
        pcout << "linsys solved in " << linsys2.count() << std::endl;

        //lin_sys_with_ib_solve(alpha_[1]);

        // Stage 3
        // ******************************************************************
        pcout << "Stage 3" << std::endl;
        if (adapt_Fourier) {
            clean_Fourier_modes_all<u_i_type>();
            clean_Fourier_modes_all<r_i_type>();
            clean_Fourier_modes_all<q_i_type>();
            clean_Fourier_modes_all<g_i_type>();
        }
        T_stage_ = T_ + dt_ * c_[2];
        stage_idx_ = 3;
        clean<d_i_type>();
        clean<cell_aux_type>();
        clean<w_2_type>();

        add<g_i_type, face_aux_type>(-1.0);
        copy<face_aux_type, w_2_type>(-1.0 / dt_ / coeff_a(2, 2));
        copy<q_i_type, r_i_type>();
        add<w_1_type, r_i_type>(dt_ * coeff_a(3, 1));
        add<w_2_type, r_i_type>(dt_ * coeff_a(3, 2));

        psolver.template apply_helm_if<r_i_type, r_i_type>(alpha_[1], N_modes, c_z);

        up_and_down<u_i_type>();

        mDuration_type nonlinear3(0);
        TIME_CODE(nonlinear3, SINGLE_ARG(nonlinear<u_i_type, g_i_type>(coeff_a(3, 3) * (-dt_));));
        pcout << "nonlinear term solved in " << nonlinear3.count() << std::endl;

        //nonlinear<u_i_type, g_i_type>(coeff_a(3, 3) * (-dt_));
        add<g_i_type, r_i_type>();

        mDuration_type linsys3(0);
        TIME_CODE(linsys3, SINGLE_ARG(lin_sys_with_ib_solve(alpha_[2]);));
        pcout << "linsys solved in " << linsys3.count() << std::endl;

        //lin_sys_with_ib_solve(alpha_[2]);

        // ******************************************************************
        copy<u_i_type, u_type>();
        copy<d_i_type, p_type>(1.0 / coeff_a(3, 3) / dt_);
        // ******************************************************************
    }

    template<typename F>
    void clean(bool non_leaf_only = false, int clean_width = 1) noexcept
    {
        for (auto it = domain_->begin(); it != domain_->end(); ++it)
        {
            if (!it->has_data()) continue;
            if (!it->data().is_allocated()) continue;

            for (std::size_t field_idx = 0; field_idx < F::nFields();
                 ++field_idx)
            {
                auto& lin_data = it->data_r(F::tag(), field_idx).linalg_data();

                if (non_leaf_only && it->is_leaf() && it->locally_owned())
                {
                    int N = it->data().descriptor().extent()[0];
                    if (domain_->dimension() == 3)
                    {
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
                    else
                    {
                        view(lin_data, xt::all(), xt::range(0, clean_width)) *=
                            0.0;
                        view(lin_data, xt::range(0, clean_width), xt::all()) *=
                            0.0;
                        view(lin_data, xt::range(N + 2 - clean_width, N + 3),
                            xt::all()) *= 0.0;
                        view(lin_data, xt::all(),
                            xt::range(N + 2 - clean_width, N + 3)) *= 0.0;
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

    template<typename F>
    void clean_leaf_correction_boundary(int l, bool leaf_only_boundary = false,
        int clean_width = 1) noexcept
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

            if (leaf_only_boundary &&
                (it->is_correction() || it->is_old_correction()))
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
        if (l == domain_->tree()->base_level())
            for (auto it = domain_->begin(l); it != domain_->end(l); ++it)
            {
                if (!it->locally_owned()) continue;
                if (!it->has_data() || !it->data().is_allocated()) continue;
                //std::cout<<it->key()<<std::endl;

                for (std::size_t i = 0; i < it->num_neighbors(); ++i)
                {
                    auto it2 = it->neighbor(i);
                    if ((!it2 || !it2->has_data()) ||
                        (leaf_only_boundary &&
                            (it2->is_correction() || it2->is_old_correction())))
                    {
                        /*for (std::size_t field_idx = 0;
                             field_idx < F::nFields(); ++field_idx)
                        {*/
                            domain::Operator::smooth2zero<F>(it->data(), i);
                        //}
                    }
                }
            }
    }

    template<typename F>
    void clean_Fourier_modes(int l) noexcept
    {
        int N_max = F::nFields();

        int residue = N_max % (2 * N_modes);

        if (residue != 0)
            throw std::runtime_error(
                "Number of elements are not multiple of 2*N_modes");

        int NComp = N_max / (2 * N_modes);

        int levelDiff = domain_->tree()->depth() - l - 1;
        int levelFactor = std::pow(2, levelDiff);
        int LevelModes = N_modes * 2 / levelFactor;

        for (auto it = domain_->begin(l); it != domain_->end(l); ++it)
        {
            if (!it->locally_owned()) continue;
            if (!it->has_data() || !it->data().is_allocated()) continue;

            for (std::size_t field_idx = LevelModes; field_idx < N_modes * 2;
                 ++field_idx)
            {
                for (int i = 0; i < NComp; i++)
                {
                    auto& lin_data =
                        it->data_r(F::tag(), field_idx + N_modes * 2 * i)
                            .linalg_data();
                    lin_data *= 0;
                }
            }
        }
    }

    template<typename F>
    void clean_Fourier_modes_all() noexcept
    {
        int N_max = F::nFields();

        int residue = N_max % (2 * N_modes);

        if (residue != 0)
            throw std::runtime_error(
                "Number of elements are not multiple of 2*N_modes");

        int NComp = N_max / (2 * N_modes);

        for (int l = domain_->tree()->base_level();
             l < domain_->tree()->depth(); ++l)
        {
            int levelDiff = domain_->tree()->depth() - l - 1;
            int levelFactor = std::pow(2, levelDiff);
            int LevelModes = N_modes * 2 / levelFactor;

            for (auto it = domain_->begin(l); it != domain_->end(l); ++it)
            {
                if (!it->locally_owned()) continue;
                if (!it->has_data() || !it->data().is_allocated()) continue;

                for (std::size_t field_idx = LevelModes;
                     field_idx < N_modes * 2; ++field_idx)
                {
                    for (int i = 0; i < NComp; i++)
                    {
                        auto& lin_data =
                            it->data_r(F::tag(), field_idx + N_modes * 2 * i)
                                .linalg_data();
                        lin_data *= 0;
                    }
                }
            }
        }
    }

  private:
    float_type coeff_a(int i, int j) const noexcept
    {
        return a_[i * (i - 1) / 2 + j - 1];
    }

    void lin_sys_solve(float_type _alpha) noexcept
    {
        auto client = domain_->decomposition().client();

        divergence<r_i_type, cell_aux_type>();

        domain_->client_communicator().barrier();
        mDuration_type t_lgf(0);
        TIME_CODE(t_lgf,
            SINGLE_ARG(psolver.template apply_lgf_and_helm<cell_aux_type, d_i_type>(N_modes);));
        pcout << "LGF solved in " << t_lgf.count() << std::endl;

        gradient<d_i_type, face_aux_type>();
        add<face_aux_type, r_i_type>(-1.0);
        if (std::fabs(_alpha) > 1e-12)
        {
            mDuration_type t_if(0);
            domain_->client_communicator().barrier();
            TIME_CODE(t_if,
                SINGLE_ARG(psolver.template apply_helm_if<r_i_type, u_i_type>(
                    _alpha, N_modes, c_z);));
            pcout << "IF  solved in " << t_if.count() << std::endl;
        }
        else
            copy<r_i_type, u_i_type>();
    }

    void lin_sys_with_ib_solve(float_type _alpha) noexcept
    {
        auto client = domain_->decomposition().client();

        auto t0 = clock_type::now();

        divergence<r_i_type, cell_aux_type>();

        auto t1 = clock_type::now();
        mDuration_type ms_div = t1 - t0;
        pcout << "Linsys Divergence in " << ms_div.count() << std::endl;

        domain_->client_communicator().barrier();
        mDuration_type t_lgf(0);
        TIME_CODE(t_lgf,
            SINGLE_ARG(psolver.template apply_lgf_and_helm<cell_aux_type, d_i_type>(N_modes);));
        domain_->client_communicator().barrier();
        pcout << "LGF solved in " << t_lgf.count() << std::endl;

        auto t2 = clock_type::now();

        copy<r_i_type, face_aux2_type>();
        gradient<d_i_type, face_aux_type>();
        add<face_aux_type, face_aux2_type>(-1.0);

        auto t3 = clock_type::now();
        mDuration_type ms_grad = t3 - t2;
        pcout << "Linsys copy grad add in " << ms_grad.count() << std::endl;

        // IB
        if (std::fabs(_alpha) > 1e-12)
            psolver.template apply_helm_if<face_aux2_type, face_aux2_type>(
                _alpha, N_modes, c_z, 3, MASK_TYPE::IB2xIB);

        domain_->client_communicator().barrier();
        pcout << "IB IF solved " << std::endl;
        mDuration_type t_ib(0);
        domain_->ib().force() = domain_->ib().force_prev(stage_idx_);
        //domain_->ib().scales(coeff_a(stage_idx_, stage_idx_));
        if (adapt_Fourier)
        {
            clean_Fourier_modes_all<face_aux2_type>();
            clean_Fourier_modes_all<edge_aux_type>();
            //clean_Fourier_modes_all<correction_tmp_type>();
        }
        TIME_CODE(t_ib, SINGLE_ARG(lsolver.template ib_solve<face_aux2_type>(
                            _alpha, T_stage_);));

        domain_->ib().force_prev(stage_idx_) = domain_->ib().force();
        //domain_->ib().scales(1.0/coeff_a(stage_idx_, stage_idx_));

        pcout << "IB  solved in " << t_ib.count() << std::endl;

        // new presure field
        auto t4 = clock_type::now();
        lsolver.template pressure_correction<d_i_type>();
        auto t5 = clock_type::now();
        mDuration_type ms_pressure = t5 - t4;
        pcout << "Linsys pressure_corrected in " << ms_pressure.count() << std::endl;

        gradient<d_i_type, face_aux_type>();

        auto t6 = clock_type::now();

        lsolver.template smearing<face_aux_type>(domain_->ib().force(), false);

        auto t7 = clock_type::now();
        mDuration_type ms_smearing = t7 - t6;
        pcout << "Linsys smearing in " << ms_smearing.count() << std::endl;
        add<face_aux_type, r_i_type>(-1.0);

        if (std::fabs(_alpha) > 1e-12)
        {
            mDuration_type t_if(0);
            domain_->client_communicator().barrier();
            TIME_CODE(t_if,
                SINGLE_ARG(psolver.template apply_helm_if<r_i_type, u_i_type>(
                    _alpha, N_modes, c_z);));
            pcout << "IF  solved in " << t_if.count() << std::endl;
        }
        else
            copy<r_i_type, u_i_type>();

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
    void pad_velocity(bool refresh_correction_only = true)
    {
        auto client = domain_->decomposition().client();

        //up_and_down<Velocity_in>();
        clean<Velocity_in>(true);
        this->up<Velocity_in>(false);
        clean<edge_aux_type>();
        clean<stream_f_type>();

        auto dx_base = domain_->dx_base();

        for (int l = domain_->tree()->base_level();
             l < domain_->tree()->depth(); ++l)
        {
            client->template buffer_exchange<Velocity_in>(l);

            for (auto it = domain_->begin(l); it != domain_->end(l); ++it)
            {
                if (!it->locally_owned() || !it->has_data()) continue;
                if (it->is_correction()) continue;
                //if(!it->is_leaf()) continue;

                const auto dx_level =
                    dx_base / math::pow2(it->refinement_level());
                //if (it->is_leaf())
                domain::Operator::curl_helmholtz_complex<Velocity_in, edge_aux_type>(it->data(),
                    dx_level, N_modes, c_z);
            }
        }

        //clean<Velocity_out>();
        clean_leaf_correction_boundary<edge_aux_type>(
            domain_->tree()->base_level(), true, 2);
        //clean_leaf_correction_boundary<edge_aux_type>(l, false,2+stage_idx_);
        psolver.template apply_lgf_and_helm<edge_aux_type, stream_f_type>(N_modes, 3, 
            MASK_TYPE::STREAM);

        int l_max = refresh_correction_only ? domain_->tree()->base_level() + 1
                                            : domain_->tree()->depth();
        for (int l = domain_->tree()->base_level(); l < l_max; ++l)
        {
            for (auto it = domain_->begin(l); it != domain_->end(l); ++it)
            {
                if (!it->locally_owned() || !it->has_data() || !it->data().is_allocated()) continue;
                //if(!it->is_correction() && refresh_correction_only) continue;

                const auto dx_level =
                    dx_base / math::pow2(it->refinement_level());
                domain::Operator::curl_transpose_helmholtz_complex<stream_f_type, Velocity_out>(
                    it->data(), dx_level, N_modes, c_z, -1.0);
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
        clean<face_aux_real_type>();
        clean<face_aux_type>();
        clean<u_i_real_type>();
        clean<vort_i_real_type>();

        auto       client = domain_->decomposition().client();
        const auto dx_base = domain_->dx_base();

        //const float_type dx_fine = dx_base/std::pow(2.0, max_ref_level_)/1.5; //dx at z (homogeneous) direction, is different from others. Also consider the 3/2 rule so that dx decreased by 1.5
        const float_type dx_fine = c_z/static_cast<float_type>(padded_dim); //dx at z (homogeneous) direction, is different from others. Also consider the 3/2 rule so that dx decreased by 1.5


        auto t0 = clock_type::now();
        //clean Fourier coefficents that should be zero
        if (adapt_Fourier)
        {
            for (int l = domain_->tree()->base_level();
                 l < domain_->tree()->depth(); ++l)
            {
                clean_Fourier_modes<Source>(l);

                /*int levelDiff = domain_->tree()->depth() - l - 1;
            int levelFactor = std::pow(2,levelDiff);
            int LevelModes = N_modes*2/levelFactor;
            for (auto it = domain_->begin(l); it != domain_->end(l); ++it)
            {
                if (!it->locally_owned() || !it->has_data() || !it->data().is_allocated()) continue;

                for (std::size_t field_idx = LevelModes; field_idx < N_modes*2;
                     ++field_idx)
                {
                    auto& lin_data =
                        it->data_r(Source::tag(), field_idx).linalg_data();
                    lin_data *= 0;
                    auto& lin_data1 =
                        it->data_r(Source::tag(), field_idx+N_modes*2).linalg_data();
                    lin_data1 *= 0;
                    auto& lin_data2 =
                        it->data_r(Source::tag(), field_idx+N_modes*4).linalg_data();
                    lin_data2 *= 0;
                }
                
            }*/
            }
        }

        for (int l = domain_->tree()->base_level();
             l < domain_->tree()->depth(); ++l)
        {
            client->template buffer_exchange<Source>(l);

            for (auto it = domain_->begin(l); it != domain_->end(l); ++it)
            {
                if (!it->locally_owned() || !it->has_data() || !it->data().is_allocated()) continue;

                const auto dx_level =
                    dx_base / math::pow2(it->refinement_level());

                int totalComp = Source::nFields();
                int dim_0 = domain_->block_extent()[0]+lBuffer+rBuffer;
                int dim_1 = domain_->block_extent()[1]+lBuffer+rBuffer;

                int vec_size = dim_0*dim_1*N_modes*3;
                domain::Operator::FourierTransformC2R<Source, u_i_real_type>(
                    it, N_modes, padded_dim, vec_size, nonzero_dim, dim_0, dim_1, c2rFunc);
                /*domain::Operator::curl_helmholtz<u_i_real_type, vort_i_real_type>(it->data(),
                    dx_level, N_modes, dx_fine);*/
                domain::Operator::curl_helmholtz_complex<Source, edge_aux_type>(it->data(),
                    dx_level, N_modes, c_z);
                domain::Operator::FourierTransformC2R<edge_aux_type, vort_i_real_type>(
                    it, N_modes, padded_dim, vec_size, nonzero_dim, dim_0, dim_1, c2rFunc);
                
            }
        }

        auto t1 = clock_type::now();

        mDuration_type ms_int = t1 - t0;
        pcout << "Fourier transform with curl_helmholtz solved in " << ms_int.count() << std::endl;

        //clean_leaf_correction_boundary<edge_aux_type>(domain_->tree()->base_level(), true, 2);
        // add background velocity
        copy<u_i_real_type, face_aux_real_type>();
        domain::Operator::add_field_expression_nonlinear_helmholtz<face_aux_real_type>(
            domain_, N_modes, simulation_->frame_vel(), T_stage_, -1.0);

        copy<Source, face_aux_type>();
        domain::Operator::add_field_expression_complex_helmholtz<face_aux_type>(
            domain_, N_modes, simulation_->frame_vel(), T_stage_, -1.0);

        auto t2 = clock_type::now();
        ms_int = t2 - t1;
        pcout << "Added field expression in " << ms_int.count() << std::endl;

        for (int l = domain_->tree()->base_level();
             l < domain_->tree()->depth(); ++l)
        {
            client->template buffer_exchange<vort_i_real_type>(l);
            clean_leaf_correction_boundary<vort_i_real_type>(l, false, 2);

            

            for (auto it = domain_->begin(l); it != domain_->end(l); ++it)
            {
                if (!it->locally_owned() || !it->has_data() || !it->data().is_allocated()) continue;

                domain::Operator::nonlinear_helmholtz<face_aux_real_type, vort_i_real_type,
                    r_i_real_type>(it->data(), N_modes);

                for (std::size_t field_idx = 0; field_idx < r_i_real_type::nFields();
                     ++field_idx)
                {
                    auto& lin_data =
                        it->data_r(r_i_real_type::tag(), field_idx).linalg_data();
                    lin_data *= _scale;
                }
            }

            //client->template buffer_exchange<Target>(l);
            //clean_leaf_correction_boundary<Target>(l, true,3);
        }

        auto t3 = clock_type::now();
        ms_int = t3 - t2;
        pcout << "Nonlinear term in real variable solved in " << ms_int.count() << std::endl;
        //transform back
        for (int l = domain_->tree()->base_level();
             l < domain_->tree()->depth(); ++l)
        {
            //client->template buffer_exchange<Source>(l);

            for (auto it = domain_->begin(l); it != domain_->end(l); ++it)
            {
                if (!it->locally_owned() || !it->has_data()) continue;

                const auto dx_level =
                    dx_base / math::pow2(it->refinement_level());

                int totalComp = Source::nFields();
                int dim_0 = domain_->block_extent()[0]+lBuffer+rBuffer;
                int dim_1 = domain_->block_extent()[1]+lBuffer+rBuffer;

                int vec_size = dim_0*dim_1*N_modes*3*3;
                domain::Operator::FourierTransformR2C<r_i_real_type, Target>(
                    it, N_modes, padded_dim, vec_size, nonzero_dim, dim_0, dim_1, r2cFunc, (1 + additional_modes));

                //int vec_size = dim_0*dim_1*N_modes*3*3;
                //also transform vorticity for refinement
                /*domain::Operator::FourierTransformR2C<vort_i_real_type, edge_aux_type>(
                    it, N_modes, padded_dim, vec_size, nonzero_dim, dim_0, dim_1, r2cFunc, (1 + additional_modes));*/

                /*std::vector<float_type> tmp_vec(vec_size, 0.0);
                for (int i = 0; i < N_modes*3*3; i++) {
                    auto& lin_data_ = it->data_r(r_i_real_type::tag(), i);
                    for (int j = 0; j < dim_0*dim_1; j++) {
                        
                        int idx = j * N_modes * 3 * 3 + i;
                        tmp_vec[idx] = tmp_val;
                    }
                    
                }

                r2cFunc.copy_field(tmp_vec);
                r2cFunc.execute();
                std::vector<std::complex<float_type>> output_vel;
                r2cFunc.output_field(output_vel);


                for (int i = 0; i < padded_dim; i++) {
                    auto& lin_data_real = it->data_r(Target::tag(), i*2);
                    auto& lin_data_imag = it->data_r(Target::tag(), i*2+1);
                    for (int j = 0; j < dim_0*dim_1; j++) {
                        int idx = j * padded_dim + i;
                        lin_data_real[j] = output_vel[idx].real();
                        lin_data_imag[j] = output_vel[idx].imag();
                    }   
                }
                */
            }
        }

        auto t4 = clock_type::now();
        ms_int = t4 - t3;
        pcout << "R2C transform solved in " << ms_int.count() << std::endl;
        addPerturb<Target>();
    }



    template<class Target>
    void addPerturb() noexcept
    {
        if (perturb_nonlin > 1e-10)
        {
            auto       client = domain_->decomposition().client();
            const auto dx_base = domain_->dx_base();
            for (int l = domain_->tree()->base_level();
                 l < domain_->tree()->depth(); ++l)
            {
                srand(time(0));
                for (auto it = domain_->begin(l); it != domain_->end(l); ++it)
                {
                    if (!it->locally_owned() || !it->has_data()) continue;

                    for (std::size_t field_idx = 0;
                         field_idx < Target::nFields(); ++field_idx)
                    {
                        for (auto& n : it->data().node_field())
                        {
                            float_type rand_num =
                                static_cast<float_type>(std::rand()) /
                                    static_cast<float_type>(RAND_MAX) -
                                0.5;
                            auto coord = n.global_coordinate() * dx_base;
                            float_type x = coord[0];
                            float_type y = coord[1];
                            float_type Gaussian = std::exp(-x*x - y*y);
                            n(Target::tag(), field_idx) +=
                                rand_num * perturb_nonlin * Gaussian;
                        }
                    }
                }
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

                if (!adapt_Fourier)
                {
                    domain::Operator::divergence_helmholtz_complex<Source,
                        Target>(it->data(), dx_level, N_modes, c_z);
                }
                else
                {
                    int ref_level_up = domain_->tree()->depth() - l - 1;
                    domain::Operator::divergence_helmholtz_complex_refined<Source,
                        Target>(it->data(), dx_level, N_modes, ref_level_up,
                        c_z);
                }
                /*domain::Operator::divergence_helmholtz_complex<Source, Target>(it->data(),
                    dx_level, N_modes, c_z);*/
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
                if (!adapt_Fourier)
                {
                    domain::Operator::gradient_helmholtz_complex<Source,
                        Target>(it->data(), dx_level, N_modes, c_z);
                }
                else
                {
                    int ref_level_up = domain_->tree()->depth() - l - 1;
                    domain::Operator::gradient_helmholtz_complex_refined<Source,
                        Target>(it->data(), dx_level, N_modes, ref_level_up,
                        c_z);
                }
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
                for (auto& n : it->data().node_field())
                    n(To::tag(), field_idx) = n(From::tag(), field_idx) * scale;
            }
        }
    }

  private:
    simulation_type* simulation_;
    domain_type*     domain_; ///< domain
    poisson_solver_t psolver;
    linsys_solver_t  lsolver;
    fft::helm_dfft_r2c r2cFunc;
    fft::helm_dfft_c2r c2rFunc;
    float_type c_z; //for the period in the homogeneous direction
    float_type perturb_nonlin = 0.0;

    int additional_modes = 0;

    bool base_mesh_update_ = false;

    float_type              T_, T_stage_, T_max_;
    float_type              dt_base_, dt_, dx_base_;
    float_type              Re_;
    float_type              cfl_max_, cfl_;
    std::vector<float_type> source_max_{0.0, 0.0};

    float_type T_last_vel_refresh_ = 0.0;

    int max_vel_refresh_ = 1;
    int max_ref_level_ = 0;
    int output_base_freq_;
    int adapt_freq_;
    int tot_base_steps_;
    int n_step_ = 0;
    int restart_n_last_ = 0;
    int nLevelRefinement_;
    int stage_idx_ = 0;

    bool use_restart_ = false;
    bool just_restarted_ = false;
    bool customized_ic = false; // if initial cond it provided, need to treat adapt process like just restarted
    bool write_restart_ = false;
    bool updating_source_max_ = false;
    bool all_time_max_;
    bool adapt_Fourier;
    int  restart_base_freq_;
    int  adapt_count_;

    std::string                fname_prefix_;
    vector_type<float_type, 6> a_{{1.0 / 3, -1.0, 2.0, 0.0, 0.75, 0.25}};
    vector_type<float_type, 4> c_{{0.0, 1.0 / 3, 1.0, 1.0}};
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
