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

#ifndef IBLGF_INCLUDED_NEWTON_HPP
#define IBLGF_INCLUDED_NEWTON_HPP

#include <iostream>
#include <algorithm>
#include <vector>
#include <cmath>
#include <array>
#include <set>

// IBLGF-specific
#include <iblgf/global.hpp>
#include <iblgf/simulation.hpp>
#include <iblgf/domain/domain.hpp>
#include <iblgf/IO/parallel_ostream.hpp>
#include <iblgf/solver/poisson/poisson.hpp>
#include <iblgf/solver/linsys/linsys.hpp>
#include <iblgf/operators/operators.hpp>
#include <iblgf/utilities/misc_math_functions.hpp>

#include <boost/serialization/map.hpp>

namespace iblgf
{
namespace solver
{
using namespace domain;

/** @brief Integrating factor 3-stage Runge-Kutta time integration
 * */
template<class Setup>
class NewtonIteration
{
  public: //member types
    using simulation_type = typename Setup::simulation_t;
    using domain_type = typename simulation_type::domain_type;
    using interpolation_type = typename interpolation::cell_center_nli<domain_type>;
    using datablock_type = typename domain_type::datablock_t;
    using tree_t = typename domain_type::tree_t;
    using key_type = typename tree_t::key_type;
    using octant_t = typename tree_t::octant_type;
    using octant_base_t = typename octant_t::octant_base_t;
    using MASK_TYPE = typename octant_t::MASK_TYPE;
    using block_type = typename datablock_type::block_descriptor_type;
    using real_coordinate_type = typename domain_type::real_coordinate_type;
    using coordinate_type = typename domain_type::coordinate_type;
    using poisson_solver_t = typename Setup::poisson_solver_t;
    using linsys_solver_t = typename Setup::linsys_solver_t;
    using lgf_lap_t = typename poisson_solver_t::lgf_lap_t;

    using ib_t = typename domain_type::ib_t;
    using force_type = typename ib_t::force_type;

    //FMM
    using Fmm_t = typename Setup::Fmm_t;

    using test_type = typename Setup::test_type;

    using u_type = typename Setup::u_type;
    using p_type = typename Setup::p_type;
    using du_i_type = typename Setup::du_i_type;
    using dp_i_type = typename Setup::dp_i_type;
    using fu_i_type = typename Setup::fu_i_type; //store first block of f(x) in Newton method
    using fp_i_type = typename Setup::fp_i_type; //store second block of f(x) in Newton method
    using stream_f_type = typename Setup::stream_f_type;
    //using p_type = typename Setup::p_type;
    //using q_i_type = typename Setup::q_i_type;
    //using r_i_type = typename Setup::r_i_type;
    //using cell_aux2_type = typename Setup::cell_aux2_type;
    using g_i_type = typename Setup::g_i_type;
    //using d_i_type = typename Setup::d_i_type;

    using u_i_bc_type = typename Setup::u_i_bc_type;
    using nonlinear_tmp_type = typename Setup::nonlinear_tmp_type;
    using face_aux_tmp_type = typename Setup::face_aux_tmp_type;
    using laplacian_face_type = typename Setup::laplacian_face_type;
    //using d_i_type = typename Setup::d_i_type;

    //using R_1_type = typename Setup::R_1_type; //R_1 is the first block in the RHS in the Newton iteration
    using nonlinear_tmp1_type = typename Setup::nonlinear_tmp1_type; //temporarily store the nonlinear term

    // initialize some tmp fields for adjoint of Jacobian
    //using r_i_T_type = typename Setup::r_i_T_type;
    //using cell_aux_T_type = typename Setup::cell_aux_T_type;
    //using u_i_T_type = typename Setup::u_i_T_type;

    // initialize tmp fields for calculations
    using r_i_tmp_type = typename Setup::r_i_tmp_type;
    using cell_aux_tmp_type = typename Setup::cell_aux_tmp_type;

    // index fields
    using idx_u_type = typename Setup::idx_u_type;
    using idx_p_type = typename Setup::idx_p_type;
    using idx_w_type = typename Setup::idx_w_type;
    using idx_u_g_type = typename Setup::idx_u_g_type;
    using idx_p_g_type = typename Setup::idx_p_g_type;
    using idx_w_g_type = typename Setup::idx_w_g_type;

    //variable fields for conjugate gradient
    /*using conj_r_face_aux_type = typename Setup::conj_r_face_aux_type;
    using conj_r_cell_aux_type = typename Setup::conj_r_cell_aux_type;
    using conj_p_face_aux_type = typename Setup::conj_p_face_aux_type;
    using conj_p_cell_aux_type = typename Setup::conj_p_cell_aux_type;
    using conj_Ap_face_aux_type = typename Setup::conj_Ap_face_aux_type;
    using conj_Ap_cell_aux_type = typename Setup::conj_Ap_cell_aux_type;
    using conj_Ax_face_aux_type = typename Setup::conj_Ax_face_aux_type;
    using conj_Ax_cell_aux_type = typename Setup::conj_Ax_cell_aux_type;
    //needed for BCGStab
    using conj_rh_face_aux_type = typename Setup::conj_rh_face_aux_type;
    using conj_rh_cell_aux_type = typename Setup::conj_rh_cell_aux_type;
    //using conj_As_face_aux_type = typename Setup::conj_As_face_aux_type;
    //using conj_As_cell_aux_type = typename Setup::conj_As_cell_aux_type;
    using conj_s_face_aux_type = typename Setup::conj_s_face_aux_type;
    using conj_s_cell_aux_type = typename Setup::conj_s_cell_aux_type;*/

    //fields for evaluating Jacobian
    using cell_aux_type = typename Setup::cell_aux_type;
    using edge_aux_type = typename Setup::edge_aux_type;
    using face_aux_type = typename Setup::face_aux_type;
    //using face_aux2_type = typename Setup::face_aux2_type;
    using correction_tmp_type = typename Setup::correction_tmp_type;
    //using u_i_type = typename Setup::u_i_type;

    static constexpr int lBuffer = 1; ///< Lower left buffer for interpolation
    static constexpr int rBuffer = 1; ///< Lower left buffer for interpolation
    NewtonIteration(simulation_type* _simulation)
    : simulation_(_simulation)
    , domain_(_simulation->domain_.get())
    , psolver(_simulation)
    , lsolver(_simulation)
    , c_cntr_nli_(domain_->block_extent()[0]+lBuffer+rBuffer, _simulation->intrp_order()) 
    //get the interpolation matrix
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

        cg_threshold_ = simulation_->dictionary_->template get_or<float_type>("cg_threshold",1e-3);
        Newton_threshold_ = simulation_->dictionary_->template get_or<float_type>("Newton_threshold",1e-3);
        cg_max_itr_ = simulation_->dictionary_->template get_or<int>("cg_max_itr", 40);

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

        real_coordinate_type tmp_coord(0.0);
        forcing_tmp.resize(domain_->ib().size());
        std::fill(forcing_tmp.begin(), forcing_tmp.end(), tmp_coord);
        forcing_old.resize(domain_->ib().size());
        std::fill(forcing_old.begin(), forcing_old.end(), tmp_coord);
        forcing_idx.resize(domain_->ib().size());
        std::fill(forcing_idx.begin(), forcing_idx.end(), tmp_coord);
        forcing_idx_g.resize(domain_->ib().size());
        std::fill(forcing_idx_g.begin(), forcing_idx_g.end(), tmp_coord);

        // Get FMM Info

        Curl_factor = _simulation->dictionary()->template get_or<float_type>(
            "Curl_factor", 1000);

        use_FMM = _simulation->dictionary()->template get_or<bool>(
            "use_FMM_in_Jac", false);
        N_sep = _simulation->dictionary()->template get_or<int>(
            "FMM_sep", 14); //definition of well separated in FMM

        FMM_bin.clear();

        if (use_FMM) {
            auto extent = domain_->bounding_box().max() - domain_->bounding_box().min();
            int max_extent = extent[0];
            if (extent[0] < extent[1]) {
                max_extent = extent[1];
            }

            float_type max_extent_by_N_sep = static_cast<float_type>(max_extent) / static_cast<float_type>(N_sep);
            int max_itr_num = std::ceil(std::log2(max_extent_by_N_sep)/log2(3)) + 2; 

            for (int i = 0; i < max_itr_num;i++) {
                //int max_loc = N_sep * (std::pow(2.0, (i + 1)) - 1);
                //int max_loc = N_sep * (std::pow(3.0, (i + 1)) - 3) / 2;//sum_1^N 3^k = (3^(N+1) - 3)/2
                int max_loc = N_sep * (std::pow(3.0, (i + 1)) - 1) / 2; //i^th circle is 3^(i+1)
                //int sep_loc = std::pow(2.0, i);
                //int sep_loc = std::pow(3.0, i);
                int sep_loc = 1;
                for(int n = 0; n < i;n++) {
                    sep_loc *= 3;
                }
                FMM_bin[max_loc] = sep_loc;
            }

        } 
        

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
                if (!just_restarted_) this->adapt(false);
                just_restarted_ = false;
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
            pcout << ifherk_if.count() << std::endl;

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
                    domain::Operator::curl<u_type, edge_aux_type>(it->data(),
                        dx_level);
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

                std::cout << "ib  size: " << ib.size() << std::endl;
                std::cout << "Forcing = ";

                for (std::size_t d = 0; d < domain_->dimension(); ++d)
                    std::cout << f[d] << " ";
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
            for (auto& element : f) { outfile << element << std::setw(width); }
            outfile << std::endl;
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
        clean<Field>(true);
        this->up<Field>();
        this->down_to_correction<Field>();
    }

    template<class Field>
    void up(bool leaf_boundary_only = false)
    {
        //Coarsification:
        for (std::size_t _field_idx = 0; _field_idx < Field::nFields();
             ++_field_idx)
            psolver.template source_coarsify<Field, Field>(_field_idx,
                _field_idx, Field::mesh_type(), false, false, false,
                leaf_boundary_only);
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

                    domain_->decomposition()
                        .client()
                        ->template communicate_updownward_assign<u_type,
                            u_type>(l, false, false, -1, _field_idx);
                }

                for (auto& oct : intrp_list)
                {
                    if (!oct || !oct->has_data()) continue;
                    psolver.c_cntr_nli()
                        .template nli_intrp_node<u_type, u_type>(oct,
                            u_type::mesh_type(), _field_idx, _field_idx, false,
                            false);
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

        //clean<R_1_type>();
        //copy<u_type, u_i_type>();
        auto forcing_df = forcing_old;
        NewtonRHS<u_type, p_type, fu_i_type, fp_i_type>(forcing_old, forcing_tmp);
        //Solve_Jacobian<fu_i_type, fp_i_type, du_i_type, dp_i_type>(forcing_tmp, forcing_df);
        //BCG_Stab<fu_i_type, fp_i_type, du_i_type, dp_i_type>(forcing_tmp, forcing_df);
        //add<nonlinear_tmp1_type, R_1_type>();
        AddAll<du_i_type, dp_i_type, u_type, p_type>(forcing_df, forcing_old, -1.0);

        float_type res = this->template dotAll<fu_i_type, fp_i_type, fu_i_type, fp_i_type>(forcing_tmp, forcing_tmp);
        float_type f2 = this->template dotAll<u_type, p_type, u_type, p_type>(forcing_old, forcing_old);
        if (comm_.rank()==1)
            std::cout<< "Newton residue square = "<< res/f2<<std::endl;;
        if (sqrt(res/f2)<Newton_threshold_)
            return;
        //add<nonlinear_tmp1_type, R_1_type>(-1.0);
        //Solve_Jacobian();
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

                if (non_leaf_only && (it->is_leaf() || (it->is_correction() && it->refinement_level() == 0)) && it->locally_owned())
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
        /*if (l == domain_->tree()->base_level())
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
                        for (std::size_t field_idx = 0;
                             field_idx < F::nFields(); ++field_idx)
                        {
                            domain::Operator::smooth2zero<F>(it->data(), i);
                        }
                    }
                }
            }*/

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
                        for (std::size_t field_idx = 0;
                             field_idx < F::nFields(); ++field_idx)
                        {
                            auto& lin_data =
                                it->data_r(F::tag(), field_idx).linalg_data();

                            int N = it->data().descriptor().extent()[0];
                            if (i == 1)
                                view(lin_data, xt::all(),
                                    xt::range(0, clean_width)) *= 0.0;
                            else if (i == 3)
                                view(lin_data, xt::range(0, clean_width),
                                    xt::all()) *= 0.0;
                            else if (i == 5)
                                view(lin_data,
                                    xt::range(N + 2 - clean_width, N + 3),
                                    xt::all()) *= 0.0;
                            else if (i == 7)
                                view(lin_data, xt::all(),
                                    xt::range(N + 2 - clean_width, N + 3)) *=
                                    0.0;
                        }
                    }
                }
            }
    }

    /*template<class U_tar, class P_tar>
    void computing_IB_forcing(force_type& forcing_vec) {

    }*/

    void Assigning_idx() {
        //currently only implemented for uniform grid with no LGF yet
        boost::mpi::communicator world;
        world.barrier();
        int counter = 0;
        if (world.rank() == 0) {
            max_idx_from_prev_prc = 0;
            counter = 0;
            world.send(1, 0, counter);
            return;
        }

        int base_level = domain_->tree()->base_level();
        
        
        for (auto it = domain_->begin(base_level); it != domain_->end(base_level); ++it)
        {
            if (!it->locally_owned() || !it->has_data()) continue;
            if (it->is_leaf() && !it->is_correction())
            {
                for (std::size_t field_idx = 0;
                     field_idx < idx_u_type::nFields(); ++field_idx)
                {
                    for (auto& n : it->data())
                    {
                        counter++;
                        n(idx_u_type::tag(), field_idx) =
                            static_cast<float_type>(counter) + 0.5;
                    }
                }
            }
            else if (it->is_correction()) {
                //only setting the leaf points that is next to the leaf to be active
                int N = it->data().descriptor().extent()[0];
                bool tmp[N][N] = {false};
                for (int i = 0; i < it->num_neighbors();i++) {
                    auto it2 = it->neighbor(i);
                    if (!it2 || !it2->is_leaf() || it2->is_correction())
                    {
                        continue;
                    }
                    else {
                        if (i == 0) {
                            tmp[0][0] = true;
                        }
                        if (i == 1) {
                            for (int j = 0; j < N; j++) {
                                tmp[j][0] = true;
                            }
                        }
                        if (i == 2) {
                            tmp[N-1][0] = true;
                        }
                        if (i == 3) {
                            for (int j = 0; j < N; j++) {
                                tmp[0][j] = true;
                            }
                        }
                        if (i == 5) {
                            for (int j = 0; j < N; j++) {
                                tmp[N-1][j] = true;
                            }
                        }
                        if (i == 6) {
                            tmp[0][N-1] = true;
                        }
                        if (i == 7) {
                            for (int j = 0; j < N; j++) {
                                tmp[j][N-1] = true;
                            }
                        }
                        if (i == 8) {
                            tmp[N-1][N-1] = true;
                        }
                    }

                }
                for (std::size_t field_idx = 0;
                     field_idx < idx_u_type::nFields(); ++field_idx)
                {
                    auto& lin_data =
                        it->data_r(idx_u_type::tag(), field_idx).linalg_data();
                    for (int i = 0; i < N; i++)
                    {
                        for (int j = 0; j < N; j++)
                        {
                            if (tmp[i][j])
                            {
                                counter++;
                                view(lin_data, i+1, j+1) =
                                    static_cast<float_type>(counter) + 0.5;
                            }
                            else
                            {
                                view(lin_data, i+1, j+1) = -1;
                            }
                        }
                    }
                }
            }
            else
            {
                for (std::size_t field_idx = 0;
                     field_idx < idx_u_type::nFields(); ++field_idx)
                {
                    for (auto& n : it->data())
                    {
                        //counter++;
                        n(idx_u_type::tag(), field_idx) = -1;
                    }
                }
            }

            //if (it->is_correction()) continue;
        }


        for (auto it = domain_->begin(base_level); it != domain_->end(base_level); ++it)
        {
            if (!it->locally_owned() || !it->has_data()) continue;
            /*if (!it->is_leaf()) continue;
            if (it->is_correction()) continue;
            for (std::size_t field_idx = 0; field_idx < idx_p_type::nFields();
                 ++field_idx)
            {
                for (auto& n : it->data()) {
                    counter++;
                    n(idx_p_type::tag(), field_idx) = static_cast<float_type>(counter) + 0.5;
                }
            }*/
            if (it->is_leaf() && !it->is_correction())
            {
                for (std::size_t field_idx = 0;
                     field_idx < idx_p_type::nFields(); ++field_idx)
                {
                    for (auto& n : it->data())
                    {
                        counter++;
                        n(idx_p_type::tag(), field_idx) =
                            static_cast<float_type>(counter) + 0.5;
                    }
                }
            }
            else if (it->is_correction()) {
                //only setting the leaf points that is next to the leaf to be active
                /*int N = it->data().descriptor().extent()[0];
                bool tmp[N][N] = {false};
                for (int i = 0; i < it->num_neighbors();i++) {
                    auto it2 = it->neighbor(i);
                    if (!it2 || !it2->is_leaf() || it2->is_correction())
                    {
                        continue;
                    }
                    else {
                        if (i == 5) {
                            for (int j = 0; j < N; j++) {
                                tmp[N-1][j] = true;
                            }
                        }
                        if (i == 7) {
                            for (int j = 0; j < N; j++) {
                                tmp[j][N-1] = true;
                            }
                        }
                        continue;
                    }

                }

                auto& lin_data = it->data_r(idx_p_type::tag(), 0).linalg_data();
                for (int i = 0; i < N; i++)
                {
                    for (int j = 0; j < N; j++)
                    {
                        if (tmp[i][j])
                        {
                            counter++;
                            view(lin_data, i + 1, j + 1) =
                                static_cast<float_type>(counter) + 0.5;
                        }
                        else
                        {
                            view(lin_data, i + 1, j + 1) = -1;
                        }
                    }
                }*/
            }
            else
            {
                for (std::size_t field_idx = 0;
                     field_idx < idx_p_type::nFields(); ++field_idx)
                {
                    for (auto& n : it->data())
                    {
                        //counter++;
                        n(idx_p_type::tag(), field_idx) = -1;
                    }
                }
            }
        }

        //also get idx for w (vorticity)
        for (auto it = domain_->begin(base_level); it != domain_->end(base_level); ++it)
        {
            if (!it->locally_owned() || !it->has_data()) continue;
            if (it->is_leaf() && !it->is_correction())
            {
                for (std::size_t field_idx = 0;
                     field_idx < idx_w_type::nFields(); ++field_idx)
                {
                    for (auto& n : it->data())
                    {
                        counter++;
                        n(idx_w_type::tag(), field_idx) =
                            static_cast<float_type>(counter) + 0.5;
                    }
                }
            }
            else if (it->is_correction()) {
                //only setting the leaf points that is next to the leaf to be active
                int N = it->data().descriptor().extent()[0];
                bool tmp[N][N] = {false};
                for (int i = 0; i < it->num_neighbors();i++) {
                    auto it2 = it->neighbor(i);
                    if (!it2 || !it2->is_leaf() || it2->is_correction())
                    {
                        continue;
                    }
                    else {
                        if (i == 0) {
                            tmp[0][0] = true;
                        }
                        if (i == 1) {
                            for (int j = 0; j < N; j++) {
                                tmp[j][0] = true;
                            }
                        }
                        /*if (i == 2) {
                            tmp[N-1][0] = true;
                        }*/
                        if (i == 3) {
                            for (int j = 0; j < N; j++) {
                                tmp[0][j] = true;
                            }
                        }
                        /*if (i == 5) {
                            for (int j = 0; j < N; j++) {
                                tmp[N-1][j] = true;
                            }
                        }
                        if (i == 6) {
                            tmp[0][N-1] = true;
                        }
                        if (i == 7) {
                            for (int j = 0; j < N; j++) {
                                tmp[j][N-1] = true;
                            }
                        }
                        if (i == 8) {
                            tmp[N-1][N-1] = true;
                        }*/
                    }

                }
                for (std::size_t field_idx = 0;
                     field_idx < idx_w_type::nFields(); ++field_idx)
                {
                    auto& lin_data =
                        it->data_r(idx_w_type::tag(), field_idx).linalg_data();
                    for (int i = 0; i < N; i++)
                    {
                        for (int j = 0; j < N; j++)
                        {
                            if (tmp[i][j])
                            {
                                counter++;
                                view(lin_data, i+1, j+1) =
                                    static_cast<float_type>(counter) + 0.5;
                            }
                            else
                            {
                                view(lin_data, i+1, j+1) = -1;
                            }
                        }
                    }
                }
            }
            else
            {
                for (std::size_t field_idx = 0;
                     field_idx < idx_w_type::nFields(); ++field_idx)
                {
                    for (auto& n : it->data())
                    {
                        //counter++;
                        n(idx_w_type::tag(), field_idx) = -1;
                    }
                }
            }
        }

        for (int l = base_level - 1; l >= 0; l--)
        {
            for (auto it = domain_->begin(l); it != domain_->end(l); ++it)
            {
                //if (!it) continue;
                if (!it->locally_owned() || !it->has_data()) continue;
                //if (it->is_leaf() && !it->is_correction())
                //{
                for (std::size_t field_idx = 0;
                     field_idx < idx_w_type::nFields(); ++field_idx)
                {
                    int N = it->data().descriptor().extent()[0];

                    auto& lin_data =
                        it->data_r(idx_w_type::tag(), field_idx).linalg_data();
                    for (int i = 0; i < (N+2); i++)
                    {
                        for (int j = 0; j < (N+2); j++)
                        {
                            counter++;
                            view(lin_data, i, j) =
                                static_cast<float_type>(counter) + 0.5;
                        }
                    }


                    /*for (auto& n : it->data())
                    {
                        counter++;
                        n(idx_w_type::tag(), field_idx) =
                            static_cast<float_type>(counter) + 0.5;
                    }*/
                }
                //}
            }
        }

        for (std::size_t i=0; i<forcing_idx.size(); ++i)
        {
            if (domain_->ib().rank(i)!=comm_.rank()) {
                forcing_idx[i]=-1;
                continue;
            }

            for (std::size_t d=0; d<forcing_idx[0].size(); ++d) {
                counter++;
                forcing_idx[i][d] = static_cast<float_type>(counter) + 0.5;
            }
        }
        max_local_idx = counter;
        domain_->client_communicator().barrier();
        
        if (world.rank() != 0)                  world.recv(world.rank()-1, world.rank() - 1, max_idx_from_prev_prc);
        if (world.rank() != (world.size() - 1)) world.send(world.rank()+1, world.rank(), (counter + max_idx_from_prev_prc));
        for (int i = 1; i < world.size();i++) {
            if (world.rank() == i) std::cout << "rank " << world.rank() << " counter + max idx is " << (counter + max_idx_from_prev_prc) << " max idx from prev prc " << max_idx_from_prev_prc << std::endl;
            domain_->client_communicator().barrier();
        }

        //Also get global idx
        for (auto it = domain_->begin(base_level); it != domain_->end(base_level); ++it)
        {
            if (!it->locally_owned() || !it->has_data()) continue;
            if (it->is_leaf() || it->is_correction())
            {
                for (std::size_t field_idx = 0;
                     field_idx < idx_u_g_type::nFields(); ++field_idx)
                {
                    for (auto& n : it->data())
                    {
                        if (n(idx_u_type::tag(), field_idx) > 0) {
                            n(idx_u_g_type::tag(), field_idx) = n(idx_u_type::tag(), field_idx)+max_idx_from_prev_prc;
                        }
                        else {
                            n(idx_u_g_type::tag(), field_idx) = -1;
                        }
                    }
                }
            }
            else
            {
                for (std::size_t field_idx = 0;
                     field_idx < idx_u_g_type::nFields(); ++field_idx)
                {
                    for (auto& n : it->data())
                    {
                        //counter++;
                        n(idx_u_g_type::tag(), field_idx) = -1;
                    }
                }
            }

            //if (it->is_correction()) continue;
        }
        for (auto it = domain_->begin(base_level); it != domain_->end(base_level); ++it)
        {
            if (!it->locally_owned() || !it->has_data()) continue;
            if (it->is_leaf() || it->is_correction())
            {
                for (std::size_t field_idx = 0;
                     field_idx < idx_p_g_type::nFields(); ++field_idx)
                {
                    for (auto& n : it->data())
                    {
                        if (n(idx_p_type::tag(), field_idx) > 0) {
                            n(idx_p_g_type::tag(), field_idx) = n(idx_p_type::tag(), field_idx)+max_idx_from_prev_prc;
                        }
                        else {
                            n(idx_p_g_type::tag(), field_idx) = -1;
                        }
                        //n(idx_p_g_type::tag(), field_idx) = n(idx_p_type::tag(), field_idx)+max_idx_from_prev_prc;
                    }
                }
            }
            else
            {
                for (std::size_t field_idx = 0;
                     field_idx < idx_p_g_type::nFields(); ++field_idx)
                {
                    for (auto& n : it->data())
                    {
                        //counter++;
                        n(idx_p_g_type::tag(), field_idx) = -1;
                    }
                }
            }
        }

        for (auto it = domain_->begin(base_level); it != domain_->end(base_level); ++it)
        {
            if (!it->locally_owned() || !it->has_data()) continue;
            if (it->is_leaf() || it->is_correction())
            {
                for (std::size_t field_idx = 0;
                     field_idx < idx_w_g_type::nFields(); ++field_idx)
                {
                    for (auto& n : it->data())
                    {
                        if (n(idx_w_type::tag(), field_idx) > 0) {
                            n(idx_w_g_type::tag(), field_idx) = n(idx_w_type::tag(), field_idx)+max_idx_from_prev_prc;
                        }
                        else {
                            n(idx_w_g_type::tag(), field_idx) = -1;
                        }
                        //n(idx_p_g_type::tag(), field_idx) = n(idx_p_type::tag(), field_idx)+max_idx_from_prev_prc;
                    }
                }
            }
            else
            {
                for (std::size_t field_idx = 0;
                     field_idx < idx_w_g_type::nFields(); ++field_idx)
                {
                    for (auto& n : it->data())
                    {
                        //counter++;
                        n(idx_w_g_type::tag(), field_idx) = -1;
                    }
                }
            }
        }

        /*for (int l = base_level - 1; l >= 0; l--)
        {
            for (auto it = domain_->begin(l); it != domain_->end(l); ++it)
            {
                //if (!it) continue;
                if (!it->locally_owned() || !it->has_data()) continue;
                //if (it->is_leaf() && !it->is_correction())
                //{
                for (std::size_t field_idx = 0;
                     field_idx < idx_w_type::nFields(); ++field_idx)
                {
                    for (auto& n : it->data())
                    {
                        n(idx_w_g_type::tag(), field_idx) =
                            n(idx_w_type::tag(), field_idx) +
                            max_idx_from_prev_prc;
                    }
                }
                //}
            }
        }*/

        

        for (int l = base_level - 1; l >= 0; l--)
        {
            for (auto it = domain_->begin(l); it != domain_->end(l); ++it)
            {
                //if (!it) continue;
                if (!it->locally_owned() || !it->has_data()) continue;
                //if (it->is_leaf() && !it->is_correction())
                //{
                int N = it->data().descriptor().extent()[0];
                for (std::size_t field_idx = 0;
                     field_idx < idx_w_type::nFields(); ++field_idx)
                {

                    auto& lin_data1 =
                        it->data_r(idx_w_g_type::tag(), field_idx).linalg_data();
                    auto& lin_data2 =
                        it->data_r(idx_w_type::tag(), field_idx).linalg_data();
                    for (int i = 0; i < (N+2); i++)
                    {
                        for (int j = 0; j < (N+2); j++)
                        {
                            view(lin_data1, i, j) =
                                view(lin_data2, i, j) + max_idx_from_prev_prc;
                        }
                    }


                    /*for (auto& n : it->data())
                    {
                        counter++;
                        n(idx_w_type::tag(), field_idx) =
                            static_cast<float_type>(counter) + 0.5;
                    }*/
                }
                //}
            }
        }
        for (std::size_t i=0; i<forcing_idx_g.size(); ++i)
        {
            if (domain_->ib().rank(i)!=comm_.rank()) {
                forcing_idx_g[i]=-1;
                continue;
            }

            for (std::size_t d=0; d<forcing_idx_g[0].size(); ++d) {
                //counter++;
                forcing_idx_g[i][d] = forcing_idx[i][d] + max_idx_from_prev_prc;
            }
        }
        domain_->client_communicator().barrier();
    }

    void constructing_laplacian() {
        boost::mpi::communicator world;
        world.barrier();

        if (world.rank() == 0) {
            return;
        }

        
        
        if (max_local_idx == 0) {
            std::cout << "idx not initialized, please call Assigning_idx()" << std::endl;
        }

        const auto dx_base = domain_->dx_base();

        mat.resizing_row(max_local_idx+1);

        int base_level = domain_->tree()->base_level();
        if (domain_->is_client())
        {
            auto client = domain_->decomposition().client();

            client->template buffer_exchange<idx_u_type>(base_level);
            client->template buffer_exchange<idx_u_g_type>(base_level);
        }
        for (auto it = domain_->begin(base_level); it != domain_->end(base_level); ++it)
        {
            if (!it->locally_owned() || !it->has_data()) continue;
            if (!it->is_leaf()) continue;
            if (it->is_correction()) continue;
            for (std::size_t field_idx = 0; field_idx < idx_u_type::nFields();
                 ++field_idx)
            {
                for (auto& n : it->data()) {
                    int cur_idx = n(idx_u_type::tag(), field_idx);
                    int glo_idx = n(idx_u_g_type::tag(), field_idx);
                    mat.add_element(cur_idx, glo_idx, -4.0/dx_base/dx_base);
                    glo_idx = n.at_offset(idx_u_g_type::tag(), 0, 1, field_idx);
                    mat.add_element(cur_idx, glo_idx, 1.0/dx_base/dx_base);
                    glo_idx = n.at_offset(idx_u_g_type::tag(), 1, 0, field_idx);
                    mat.add_element(cur_idx, glo_idx, 1.0/dx_base/dx_base);
                    glo_idx = n.at_offset(idx_u_g_type::tag(), 0, -1, field_idx);
                    mat.add_element(cur_idx, glo_idx, 1.0/dx_base/dx_base);
                    glo_idx = n.at_offset(idx_u_g_type::tag(), -1, 0, field_idx);
                    mat.add_element(cur_idx, glo_idx, 1.0/dx_base/dx_base);
                }
            }
        }
        if (domain_->is_client())
        {
            auto client = domain_->decomposition().client();

            client->template buffer_exchange<idx_p_type>(base_level);
            client->template buffer_exchange<idx_p_g_type>(base_level);
        }
        for (auto it = domain_->begin(base_level); it != domain_->end(base_level); ++it)
        {
            if (!it->locally_owned() || !it->has_data()) continue;
            if (!it->is_leaf()) continue;
            if (it->is_correction()) continue;
            for (std::size_t field_idx = 0; field_idx < idx_p_type::nFields();
                 ++field_idx)
            {
                for (auto& n : it->data()) {
                    int cur_idx = n(idx_p_type::tag(), field_idx);
                    int glo_idx = n(idx_p_g_type::tag(), field_idx);
                    mat.add_element(cur_idx, glo_idx, -4.0/dx_base/dx_base);
                    glo_idx = n.at_offset(idx_p_g_type::tag(), 0, 1, field_idx);
                    mat.add_element(cur_idx, glo_idx, 1.0/dx_base/dx_base);
                    glo_idx = n.at_offset(idx_p_g_type::tag(), 1, 0, field_idx);
                    mat.add_element(cur_idx, glo_idx, 1.0/dx_base/dx_base);
                    glo_idx = n.at_offset(idx_p_g_type::tag(), 0, -1, field_idx);
                    mat.add_element(cur_idx, glo_idx, 1.0/dx_base/dx_base);
                    glo_idx = n.at_offset(idx_p_g_type::tag(), -1, 0, field_idx);
                    mat.add_element(cur_idx, glo_idx, 1.0/dx_base/dx_base);
                }
            }
        }
        /*for (std::size_t i=0; i<forcing_idx.size(); ++i)
        {
            if (domain_->ib().rank(i)!=comm_.rank()) {
                forcing_idx[i]=-1;
                continue;
            }

            for (std::size_t d=0; d<forcing_idx[0].size(); ++d) {
                counter++;
                forcing_idx[i][d] = static_cast<float_type>(counter) + 0.5;
            }
        }*/
        domain_->client_communicator().barrier();
    }

    template<class Face, class Cell, class val_type>
    void Grid2CSR(val_type* b) {
        Grid2CSR<Face, Cell, edge_aux_type>(b, this->forcing_tmp);
    }

    template<class Face, class Cell, class Edge, class val_type>
    void Grid2CSR(val_type* b, force_type& forcing_vec, bool set_corr_zero = true) {
        boost::mpi::communicator world;

        if (world.rank() == 0) {
            return;
        }

        domain_->client_communicator().barrier();

        
        
        if (max_local_idx == 0) {
            std::cout << "idx not initialized, please call Assigning_idx()" << std::endl;
        }

        //mat.resizing_row(max_local_idx+1);

        int base_level = domain_->tree()->base_level();
        if (domain_->is_client())
        {
            auto client = domain_->decomposition().client();

            client->template buffer_exchange<idx_u_type>(base_level);
        }
        for (auto it = domain_->begin(base_level); it != domain_->end(base_level); ++it)
        {
            if (!it->locally_owned() || !it->has_data()) continue;
            if (!it->is_leaf()) continue;
            if (!it->is_correction())
            {
                for (std::size_t field_idx = 0;
                     field_idx < idx_u_type::nFields(); ++field_idx)
                {
                    for (auto& n : it->data())
                    {
                        int p_idx_w = n.at_offset(idx_p_type::tag(), -1, 0, 0);
                        int p_idx_e = n.at_offset(idx_p_type::tag(), 1, 0, 0);
                        int p_idx_n = n.at_offset(idx_p_type::tag(), 0, 1, 0);
                        int p_idx_s = n.at_offset(idx_p_type::tag(), 0, -1, 0);
                        if (p_idx_w < 0 && field_idx == 0 && set_corr_zero)
                        {
                            int cur_idx = n(idx_u_type::tag(), 0);
                            b[cur_idx - 1] = 0;
                            continue;
                        }
                        if (p_idx_s < 0 && field_idx == 1 && set_corr_zero)
                        {
                            int cur_idx = n(idx_u_type::tag(), 1);
                            b[cur_idx - 1] = 0;
                            continue;
                        }
                        int cur_idx = n(idx_u_type::tag(), field_idx);
                        b[cur_idx - 1] = n(Face::tag(), field_idx);
                    }
                }
            }
            else if (it->is_correction() && set_corr_zero)
            {
                for (std::size_t field_idx = 0;
                     field_idx < idx_u_type::nFields(); ++field_idx)
                {
                    for (auto& n : it->data())
                    {
                        int cur_idx = n(idx_u_type::tag(), field_idx);
                        if (n(idx_u_type::tag(), field_idx) < 0) { continue; }
                        b[cur_idx - 1] = 0;
                    }
                }
            }
            else {
                for (std::size_t field_idx = 0;
                     field_idx < idx_u_type::nFields(); ++field_idx)
                {
                    for (auto& n : it->data())
                    {
                        int cur_idx = n(idx_u_type::tag(), field_idx);
                        if (n(idx_u_type::tag(), field_idx) < 0) { continue; }
                        b[cur_idx - 1] = n(Face::tag(), field_idx);
                    }
                }
            }
        }
        if (domain_->is_client())
        {
            auto client = domain_->decomposition().client();

            client->template buffer_exchange<idx_p_type>(base_level);
        }
        for (auto it = domain_->begin(base_level); it != domain_->end(base_level); ++it)
        {
            if (!it->locally_owned() || !it->has_data()) continue;
            if (!it->is_leaf()) continue;
            if (!it->is_correction()) {
            for (std::size_t field_idx = 0; field_idx < idx_p_type::nFields();
                 ++field_idx)
            {
                for (auto& n : it->data()) {
                    int cur_idx = n(idx_p_type::tag(), field_idx);
                    b[cur_idx - 1] = n(Cell::tag(), field_idx);
                }
            }
            }
            else if (it->is_correction() && set_corr_zero)
            {
                for (std::size_t field_idx = 0;
                     field_idx < idx_p_type::nFields(); ++field_idx)
                {
                    for (auto& n : it->data())
                    {
                        int cur_idx = n(idx_p_type::tag(), field_idx);
                        if (n(idx_p_type::tag(), field_idx) < 0) { continue; }
                        b[cur_idx - 1] = 0;
                    }
                }
            }
            else {
                for (std::size_t field_idx = 0;
                     field_idx < idx_p_type::nFields(); ++field_idx)
                {
                    for (auto& n : it->data())
                    {
                        int cur_idx = n(idx_p_type::tag(), field_idx);
                        if (n(idx_p_type::tag(), field_idx) < 0) { continue; }
                        b[cur_idx - 1] = n(Cell::tag(), field_idx);
                    }
                }
            }
            
        }

        if (domain_->is_client())
        {
            auto client = domain_->decomposition().client();

            client->template buffer_exchange<idx_w_type>(base_level);
        }
        for (auto it = domain_->begin(base_level); it != domain_->end(base_level); ++it)
        {
            if (!it->locally_owned() || !it->has_data()) continue;
            if (!it->is_leaf()) continue;
            if (!it->is_correction()) {
            for (std::size_t field_idx = 0; field_idx < idx_w_type::nFields();
                 ++field_idx)
            {
                for (auto& n : it->data()) {
                    int cur_idx = n(idx_w_type::tag(), field_idx);
                    b[cur_idx - 1] = n(Edge::tag(), field_idx);
                }
            }
            }
            else if (it->is_correction() && set_corr_zero)
            {
                for (std::size_t field_idx = 0;
                     field_idx < idx_w_type::nFields(); ++field_idx)
                {
                    for (auto& n : it->data())
                    {
                        int cur_idx = n(idx_w_type::tag(), field_idx);
                        if (n(idx_w_type::tag(), field_idx) < 0) { continue; }
                        b[cur_idx - 1] = 0;
                    }
                }
            }
            else {
                for (std::size_t field_idx = 0;
                     field_idx < idx_w_type::nFields(); ++field_idx)
                {
                    for (auto& n : it->data())
                    {
                        int cur_idx = n(idx_w_type::tag(), field_idx);
                        if (n(idx_w_type::tag(), field_idx) < 0) { continue; }
                        b[cur_idx - 1] = n(Edge::tag(), field_idx);
                    }
                }
            }
            
        }

        

        /*for (int l = base_level - 1; l >= 0; l--)
        {
            for (auto it = domain_->begin(l); it != domain_->end(l); ++it)
            {
                //if (!it) continue;
                if (!it->locally_owned() || !it->has_data()) continue;
                //if (it->is_leaf() && !it->is_correction())
                //{
                for (std::size_t field_idx = 0;
                     field_idx < idx_w_type::nFields(); ++field_idx)
                {
                    for (auto& n : it->data())
                    {
                        int cur_idx = n(idx_w_type::tag(), field_idx);
                        b[cur_idx - 1] = n(Edge::tag(), field_idx);
                    }
                }
                //}
            }
        }*/

        for (int l = base_level - 1; l >= 0; l--)
        {
            for (auto it = domain_->begin(l); it != domain_->end(l); ++it)
            {
                //if (!it) continue;
                if (!it->locally_owned() || !it->has_data()) continue;
                //if (it->is_leaf() && !it->is_correction())
                //{
                for (std::size_t field_idx = 0;
                     field_idx < idx_w_type::nFields(); ++field_idx)
                {
                    int N = it->data().descriptor().extent()[0];

                    auto& lin_data =
                        it->data_r(idx_w_type::tag(), field_idx).linalg_data();
                    auto& lin_data_tar =
                        it->data_r(Edge::tag(), field_idx).linalg_data();
                    for (int i = 0; i < (N+2); i++)
                    {
                        for (int j = 0; j < (N+2); j++)
                        {
                            int cur_idx = lin_data.at(i,j);
                            if (!set_corr_zero) b[cur_idx - 1] = lin_data_tar.at(i,j);
                            else  {
                                b[cur_idx - 1] = 0;
                            }
                        }
                    }


                    /*for (auto& n : it->data())
                    {
                        counter++;
                        n(idx_w_type::tag(), field_idx) =
                            static_cast<float_type>(counter) + 0.5;
                    }*/
                }
                //}
            }
        }

        for (std::size_t i=0; i<forcing_idx.size(); ++i)
        {
            if (domain_->ib().rank(i)!=comm_.rank()) {
                continue;
            }

            for (std::size_t d=0; d<forcing_idx[0].size(); ++d) {
                int cur_idx = forcing_idx[i][d];
                b[cur_idx - 1] = forcing_vec[i][d];
            }
        }
        domain_->client_communicator().barrier();
    }


    template<class val_type>
    void CSR2CSR_correction(val_type* source, val_type* target, float_type factor = 1.0) {
        boost::mpi::communicator world;

        if (world.rank() == 0) {
            return;
        }

        domain_->client_communicator().barrier();

        
        
        if (max_local_idx == 0) {
            std::cout << "idx not initialized, please call Assigning_idx()" << std::endl;
        }

        //mat.resizing_row(max_local_idx+1);

        int base_level = domain_->tree()->base_level();
        if (domain_->is_client())
        {
            auto client = domain_->decomposition().client();

            client->template buffer_exchange<idx_u_type>(base_level);
        }
        for (auto it = domain_->begin(base_level); it != domain_->end(base_level); ++it)
        {
            if (!it->locally_owned() || !it->has_data()) continue;
            if (!it->is_leaf()) continue;
            if (it->is_correction())
            {
                for (std::size_t field_idx = 0;
                     field_idx < idx_u_type::nFields(); ++field_idx)
                {
                    for (auto& n : it->data())
                    {
                        int cur_idx = n(idx_u_type::tag(), field_idx);
                        if (n(idx_u_type::tag(), field_idx) < 0) { continue; }
                        target[cur_idx - 1] = source[cur_idx - 1] * factor;
                    }
                }
            }
        }
        /*if (domain_->is_client())
        {
            auto client = domain_->decomposition().client();

            client->template buffer_exchange<idx_p_type>(base_level);
        }
        for (auto it = domain_->begin(base_level); it != domain_->end(base_level); ++it)
        {
            if (!it->locally_owned() || !it->has_data()) continue;
            if (!it->is_leaf()) continue;
            if (it->is_correction())
            {
                for (std::size_t field_idx = 0;
                     field_idx < idx_p_type::nFields(); ++field_idx)
                {
                    for (auto& n : it->data())
                    {
                        int cur_idx = n(idx_p_type::tag(), field_idx);
                        if (n(idx_p_type::tag(), field_idx) < 0) { continue; }
                        target[cur_idx - 1] = source[cur_idx - 1] * factor;
                    }
                }
            }
            
        }*/
        domain_->client_communicator().barrier();
    }

    template<class Face, class Cell, class val_type>
    void CSR2Grid(val_type* b) {
        CSR2Grid<Face, Cell, edge_aux_type>(b, this->forcing_tmp);
    }

    template<class Face, class Cell, class Edge, class val_type>
    void CSR2Grid(val_type* b, force_type& forcing_vec) {
        boost::mpi::communicator world;

        if (world.rank() == 0) {
            return;
        }

        domain_->client_communicator().barrier();

        
        
        if (max_local_idx == 0) {
            std::cout << "idx not initialized, please call Assigning_idx()" << std::endl;
        }

        //mat.resizing_row(max_local_idx+1);

        int base_level = domain_->tree()->base_level();
        if (domain_->is_client())
        {
            auto client = domain_->decomposition().client();

            client->template buffer_exchange<idx_u_type>(base_level);
        }
        for (auto it = domain_->begin(base_level); it != domain_->end(base_level); ++it)
        {
            if (!it->locally_owned() || !it->has_data()) continue;
            if (!it->is_leaf()) continue;
            //if (it->is_correction()) continue;
            for (std::size_t field_idx = 0; field_idx < idx_u_type::nFields();
                 ++field_idx)
            {
                for (auto& n : it->data()) {
                    int cur_idx = n(idx_u_type::tag(), field_idx);
                    if (cur_idx > 0) n(Face::tag(), field_idx) = b[cur_idx-1];
                }
            }
        }
        if (domain_->is_client())
        {
            auto client = domain_->decomposition().client();

            client->template buffer_exchange<idx_p_type>(base_level);
        }
        for (auto it = domain_->begin(base_level); it != domain_->end(base_level); ++it)
        {
            if (!it->locally_owned() || !it->has_data()) continue;
            if (!it->is_leaf()) continue;
            //if (it->is_correction()) continue;
            for (std::size_t field_idx = 0; field_idx < idx_p_type::nFields();
                 ++field_idx)
            {
                for (auto& n : it->data()) {
                    int cur_idx = n(idx_p_type::tag(), field_idx);
                    if (cur_idx > 0) n(Cell::tag(), field_idx) = b[cur_idx-1];
                }
            }
        }

        if (domain_->is_client())
        {
            auto client = domain_->decomposition().client();

            client->template buffer_exchange<idx_w_type>(base_level);
        }
        for (auto it = domain_->begin(base_level); it != domain_->end(base_level); ++it)
        {
            if (!it->locally_owned() || !it->has_data()) continue;
            if (!it->is_leaf()) continue;
            //if (it->is_correction()) continue;
            for (std::size_t field_idx = 0; field_idx < idx_w_type::nFields();
                 ++field_idx)
            {
                for (auto& n : it->data()) {
                    int cur_idx = n(idx_w_type::tag(), field_idx);
                    if (cur_idx > 0) n(Edge::tag(), field_idx) = b[cur_idx-1];
                }
            }
        }

        /*for (int l = base_level - 1; l >= 0; l--)
        {
            for (auto it = domain_->begin(l); it != domain_->end(l); ++it)
            {
                //if (!it) continue;
                if (!it->locally_owned() || !it->has_data()) continue;
                //if (it->is_leaf() && !it->is_correction())
                //{
                for (std::size_t field_idx = 0;
                     field_idx < idx_w_type::nFields(); ++field_idx)
                {
                    for (auto& n : it->data())
                    {
                        int cur_idx = n(idx_w_type::tag(), field_idx);
                        n(Edge::tag(), field_idx) = b[cur_idx - 1];
                    }
                }
            }
        }*/

        for (int l = base_level - 1; l >= 0; l--)
        {
            for (auto it = domain_->begin(l); it != domain_->end(l); ++it)
            {
                //if (!it) continue;
                if (!it->locally_owned() || !it->has_data()) continue;
                //if (it->is_leaf() && !it->is_correction())
                //{
                for (std::size_t field_idx = 0;
                     field_idx < idx_w_type::nFields(); ++field_idx)
                {
                    int N = it->data().descriptor().extent()[0];

                    auto& lin_data =
                        it->data_r(idx_w_type::tag(), field_idx).linalg_data();
                    auto& lin_data_tar =
                        it->data_r(Edge::tag(), field_idx).linalg_data();
                    for (int i = 0; i < (N+2); i++)
                    {
                        for (int j = 0; j < (N+2); j++)
                        {
                            int cur_idx = lin_data.at(i,j);
                            lin_data_tar.at(i,j) = b[cur_idx - 1];
                        }
                    }


                    /*for (auto& n : it->data())
                    {
                        counter++;
                        n(idx_w_type::tag(), field_idx) =
                            static_cast<float_type>(counter) + 0.5;
                    }*/
                }
                //}
            }
        }
        /*for (std::size_t i=0; i<forcing_idx.size(); ++i)
        {
            if (domain_->ib().rank(i)!=comm_.rank()) {
                forcing_idx[i]=-1;
                continue;
            }

            for (std::size_t d=0; d<forcing_idx[0].size(); ++d) {
                counter++;
                forcing_idx[i][d] = static_cast<float_type>(counter) + 0.5;
            }
        }*/
        for (std::size_t i=0; i<forcing_idx.size(); ++i)
        {
            if (domain_->ib().rank(i)!=comm_.rank()) {
                continue;
            }

            for (std::size_t d=0; d<forcing_idx[0].size(); ++d) {
                int cur_idx = forcing_idx[i][d];
                if (cur_idx < 0) {
                    std::cout << "IB forcing idx not consistent" << std::endl;
                }
                forcing_vec[i][d] = b[cur_idx - 1];
            }
        }
        domain_->client_communicator().barrier();
    }

    template<class Edge1, class Edge2>
    void compute_error_nonleaf(std::string _output_prefix = "", bool write_output=false) {
        boost::mpi::communicator world;
        
        const auto dx_base = domain_->dx_base();
        int base_level = domain_->tree()->base_level();

        
        if (world.rank() != 0)
        {
            for (int l = base_level - 1; l >= 0; l--)
            {
                float_type sum_val =  0.0;
                float_type max_val = -1.0;

                std::ofstream myfile;
                if (write_output) myfile.open(_output_prefix + "rank" + std::to_string(world.rank()) + "level" + std::to_string(l) + "err.txt");

                for (auto it = domain_->begin(l); it != domain_->end(l); ++it)
                {
                    float_type dx_level =
                        dx_base * std::pow(2.0, base_level - l);
                    //if (!it) continue;
                    if (!it->locally_owned() || !it->has_data()) continue;
                    //if (it->is_leaf() && !it->is_correction())
                    //{
                    for (std::size_t field_idx = 0;
                         field_idx < idx_w_type::nFields(); ++field_idx)
                    {
                        for (auto& n : it->data())
                        {
                            auto c = n.level_coordinate();
                            int x_c = c.x();
                            int y_c = c.y();

                            if (write_output) myfile << x_c << " " << y_c << " ";
                            float_type val1 = n(Edge1::tag(), field_idx);
                            float_type val2 = n(Edge2::tag(), field_idx);

                            float_type diff_w = val1 - val2;
                            if (write_output) myfile << diff_w << std::endl;

                            float_type diff = std::abs(val1 - val2);
                            sum_val += diff * diff * dx_level*dx_level;
                            if (max_val < diff) { max_val = diff; }
                        }
                    }
                }

                if (write_output) myfile.close();
                float_type max_g;
                float_type sum_g;
                boost::mpi::all_reduce(domain_->client_communicator(), sum_val, sum_g, std::plus<float_type>());

                boost::mpi::all_reduce(domain_->client_communicator(), max_val, max_g, boost::mpi::maximum<float_type>());

                if (world.rank() == 1)
                {
                    std::cout << _output_prefix
                              << "L2 error of upward interpolation from level" << l << " is "
                              << std::sqrt(sum_g) << std::endl;
                    std::cout << _output_prefix
                              << "L_inf error of upward interpolation from level" << l << " is "
                              << max_g << std::endl;
                }
            }
        }
        

        

    }

    void printing_mat(int n, int rank = 1) {
        boost::mpi::communicator world;

        if (world.rank() == rank) mat.print_row(n);
    }

    int num_start() {
        return max_idx_from_prev_prc+1;
    }
    int num_end() {
        return max_local_idx + max_idx_from_prev_prc;
    }
    int total_dim() {
        boost::mpi::communicator world;
        int tot_dim_tmp = max_local_idx + max_idx_from_prev_prc;
        boost::mpi::broadcast(world, tot_dim_tmp, (world.size()-1));
        return tot_dim_tmp;
        
    }


    template<class U_old>
    void construct_linear_mat() {
        boost::mpi::communicator world;
        
        Jac.clean();
        Jac.resizing_row(max_local_idx+1);
        construct_upward_intrp();
        /*if (!use_FMM) construction_BCMat();
        else construction_BCMat_FMM();*/
        construction_BCMat();
        construction_laplacian_u();
        construction_DN_u<U_old>();
        construction_Div();
        construction_Grad();
        construction_Curl();
        construction_Projection();
        construction_Smearing();
        
        if (world.rank() == 0) {
            return;
        }
        Jac = boundary_u + L;
        Jac.add_vec(DN, -1.0);
        Jac.add_vec(Div, -1.0);
        Jac.add_vec(Grad);
        Jac.add_vec(Curl);
        Jac.add_vec(project);
        Jac.add_vec(smearing);
        Jac.add_vec(upward_intrp);
    }

    void construction_laplacian_u() {
        //construction of laplacian for u during stability, resolvent, and Newton iteration
        boost::mpi::communicator world;
        world.barrier();

        if (world.rank() == 0) {
            return;
        }

        if (world.rank() == 1) {
            std::cout << "Constructing Laplacian_u matrix" << std::endl;
        }
       
        if (max_local_idx == 0) {
            std::cout << "idx not initialized, please call Assigning_idx()" << std::endl;
        }

        const auto dx_base = domain_->dx_base();

        L.resizing_row(max_local_idx+1);

        int base_level = domain_->tree()->base_level();
        if (domain_->is_client())
        {
            auto client = domain_->decomposition().client();

            client->template buffer_exchange<idx_u_type>(base_level);
            client->template buffer_exchange<idx_p_type>(base_level);
            client->template buffer_exchange<idx_u_g_type>(base_level);
        }
        for (auto it = domain_->begin(base_level); it != domain_->end(base_level); ++it)
        {
            if (!it->locally_owned() || !it->has_data()) continue;
            if (!it->is_leaf()) continue;
            if (it->is_correction()) continue;
            for (std::size_t field_idx = 0; field_idx < idx_u_type::nFields();
                 ++field_idx)
            {
                for (auto& n : it->data()) {
                    int p_idx_w = n.at_offset(idx_p_type::tag(), -1, 0, 0);
                    int p_idx_e = n.at_offset(idx_p_type::tag(),  1, 0, 0);
                    int p_idx_n = n.at_offset(idx_p_type::tag(),  0, 1, 0);
                    int p_idx_s = n.at_offset(idx_p_type::tag(), 0, -1, 0);
                    if (p_idx_w < 0 && field_idx == 0){
                        int cur_idx = n(idx_u_type::tag(), 0);
                        int glo_idx = n(idx_u_g_type::tag(), 0);
                        L.add_element(cur_idx, glo_idx, 1.0/dx_base);
                        glo_idx = n.at_offset(idx_u_g_type::tag(), -1, 0, 0);
                        L.add_element(cur_idx, glo_idx, -1.0/dx_base);
                        glo_idx = n.at_offset(idx_u_g_type::tag(), -1, 0, 1);
                        L.add_element(cur_idx, glo_idx, -1.0/dx_base);
                        glo_idx = n.at_offset(idx_u_g_type::tag(), -1, 1, 1);
                        L.add_element(cur_idx, glo_idx, 1.0/dx_base);
                        continue;
                    }
                    if (p_idx_s < 0 && field_idx == 1){
                        int cur_idx = n(idx_u_type::tag(), 1);
                        int glo_idx = n(idx_u_g_type::tag(), 1);
                        L.add_element(cur_idx, glo_idx, 1.0/dx_base);
                        glo_idx = n.at_offset(idx_u_g_type::tag(), 0, -1, 1);
                        L.add_element(cur_idx, glo_idx, -1.0/dx_base);
                        glo_idx = n.at_offset(idx_u_g_type::tag(), 0, -1, 0);
                        L.add_element(cur_idx, glo_idx, -1.0/dx_base);
                        glo_idx = n.at_offset(idx_u_g_type::tag(), 1, -1, 0);
                        L.add_element(cur_idx, glo_idx, 1.0/dx_base);
                        continue;
                    }
                    int cur_idx = n(idx_u_type::tag(), field_idx);
                    int glo_idx = n(idx_u_g_type::tag(), field_idx);
                    L.add_element(cur_idx, glo_idx, -4.0/dx_base/dx_base/Re_);
                    glo_idx = n.at_offset(idx_u_g_type::tag(), 0, 1, field_idx);
                    L.add_element(cur_idx, glo_idx, 1.0/dx_base/dx_base/Re_);
                    glo_idx = n.at_offset(idx_u_g_type::tag(), 1, 0, field_idx);
                    L.add_element(cur_idx, glo_idx, 1.0/dx_base/dx_base/Re_);
                    glo_idx = n.at_offset(idx_u_g_type::tag(), 0, -1, field_idx);
                    L.add_element(cur_idx, glo_idx, 1.0/dx_base/dx_base/Re_);
                    glo_idx = n.at_offset(idx_u_g_type::tag(), -1, 0, field_idx);
                    L.add_element(cur_idx, glo_idx, 1.0/dx_base/dx_base/Re_);
                }
            }
        }
        domain_->client_communicator().barrier();
    }

    template<class U_old>
    void construction_DN_u(float_type t = 1) {
        boost::mpi::communicator world;
        world.barrier();

        if (world.rank() == 0) {
            return;
        }

        if (world.rank() == 1) {
            std::cout << "Constructing DN_u matrix" << std::endl;
        }
       
        if (max_local_idx == 0) {
            std::cout << "idx not initialized, please call Assigning_idx()" << std::endl;
        }

        const auto dx_base = domain_->dx_base();

        DN.resizing_row(max_local_idx+1);

        int base_level = domain_->tree()->base_level();
        if (domain_->is_client())
        {
            auto client = domain_->decomposition().client();

            client->template buffer_exchange<idx_u_type>(base_level);
            client->template buffer_exchange<idx_u_g_type>(base_level);
            client->template buffer_exchange<idx_w_g_type>(base_level);
            client->template buffer_exchange<U_old>(base_level);
            clean<edge_aux_type>();
        }
        for (auto it = domain_->begin(base_level); it != domain_->end(base_level); ++it)
        {
            float_type dx = dx_base;
            if (!it->locally_owned() || !it->has_data()) continue;
            if (!it->is_leaf()) continue;
            if (it->is_correction()) continue;

            for (auto& n : it->data())
            {
                
                

                int cur_idx_0 = n(idx_u_type::tag(), 0);
                int cur_idx_1 = n(idx_u_type::tag(), 1);

                int glob_idx_0 = n(idx_u_g_type::tag(), 0);
                int glob_idx_1 = n(idx_u_g_type::tag(), 1);

                int glob_idx_0_1 = n.at_offset(idx_u_g_type::tag(), 0, -1, 0);
                int glob_idx_0_2 = n.at_offset(idx_u_g_type::tag(), 0,  1, 0);
                int glob_idx_0_3 = n.at_offset(idx_u_g_type::tag(), 1,  0, 0);
                int glob_idx_0_4 = n.at_offset(idx_u_g_type::tag(), 1, -1, 0);

                int glob_idx_1_1 = n.at_offset(idx_u_g_type::tag(), -1, 0, 1);
                int glob_idx_1_2 = n.at_offset(idx_u_g_type::tag(),  0, 1, 1);
                int glob_idx_1_3 = n.at_offset(idx_u_g_type::tag(), -1, 1, 1);
                int glob_idx_1_4 = n.at_offset(idx_u_g_type::tag(),  1, 0, 1);

                int glob_idx_00 = n(idx_w_g_type::tag(), 0);
                int glob_idx_01 = n.at_offset(idx_w_g_type::tag(),  0, 1, 0);
                int glob_idx_10 = n.at_offset(idx_w_g_type::tag(),  1, 0, 0);

                /*float_type v_s_00 = n(U_old::tag, 1);
                float_type v_s_10 = n.at_offset(U_old::tag, -1, 0, 1);
                float_type v_s_01 = n.at_offset(U_old::tag,  0, 1, 1);
                float_type v_s_11 = n.at_offset(U_old::tag, -1, 1, 1);

                float_type u_s_00 = n(U_old::tag, 0);
                float_type u_s_01 = n.at_offset(U_old::tag, 0, -1, 0);
                float_type u_s_10 = n.at_offset(U_old::tag, 1,  0, 0);
                float_type u_s_11 = n.at_offset(U_old::tag, 1, -1, 0);*/

                auto coord = n.global_coordinate() * dx_base;
                auto field_func = simulation_->frame_vel();
                float_type u_field_vel = field_func(0, t, coord)*(-1.0);
                float_type v_field_vel = field_func(1, t, coord)*(-1.0);

                float_type v_s_0010 = -(n(U_old::tag(), 1)                   + n.at_offset(U_old::tag(), -1, 0, 1))*0.25 - v_field_vel * 0.5;
                float_type v_s_0111 = -(n.at_offset(U_old::tag(),  0, 1, 1)  + n.at_offset(U_old::tag(), -1, 1, 1))*0.25 - v_field_vel * 0.5;
                float_type u_s_0001 =  (n(U_old::tag(), 0)                   + n.at_offset(U_old::tag(), 0, -1, 0))*0.25 + u_field_vel * 0.5;
                float_type u_s_1011 =  (n.at_offset(U_old::tag(),  1, 0, 0)  + n.at_offset(U_old::tag(), 1, -1, 0))*0.25 + u_field_vel * 0.5;

                //-n.at_offset(edge, 0, 0, 0) *(n.at_offset(face, 0, 0, 1) + n.at_offset(face, -1, 0, 1))
                /*DN.add_element(cur_idx_0, glob_idx_1,    v_s_0010/dx);
                DN.add_element(cur_idx_0, glob_idx_1_1, -v_s_0010/dx);
                //-n.at_offset(edge, 0, 1, 0) *(n.at_offset(face, 0, 1, 1) + n.at_offset(face, -1, 1, 1))
                DN.add_element(cur_idx_0, glob_idx_1_2,  v_s_0111/dx);
                DN.add_element(cur_idx_0, glob_idx_1_3, -v_s_0111/dx);

                //-n.at_offset(edge, 0, 0, 0) *(n.at_offset(face, 0, 0, 1) + n.at_offset(face, -1, 0, 1))
                DN.add_element(cur_idx_0, glob_idx_0,   -v_s_0010/dx);
                DN.add_element(cur_idx_0, glob_idx_0_1,  v_s_0010/dx);
                //-n.at_offset(edge, 0, 1, 0) *(n.at_offset(face, 0, 1, 1) + n.at_offset(face, -1, 1, 1))
                DN.add_element(cur_idx_0, glob_idx_0_2, -v_s_0111/dx);
                DN.add_element(cur_idx_0, glob_idx_0,    v_s_0111/dx);

                // n.at_offset(edge, 0, 0, 0) *(n.at_offset(face, 0, 0, 0) + n.at_offset(face, 0, -1, 0))
                DN.add_element(cur_idx_1, glob_idx_1,    u_s_0001/dx);
                DN.add_element(cur_idx_1, glob_idx_1_1, -u_s_0001/dx);
                // n.at_offset(edge, 1, 0, 0) *(n.at_offset(face, 1, 0, 0) + n.at_offset(face, 1, -1, 0))
                DN.add_element(cur_idx_1, glob_idx_1_4,  u_s_1011/dx);
                DN.add_element(cur_idx_1, glob_idx_1,   -u_s_1011/dx);

                // n.at_offset(edge, 0, 0, 0) *(n.at_offset(face, 0, 0, 0) + n.at_offset(face, 0, -1, 0))
                DN.add_element(cur_idx_1, glob_idx_0,   -u_s_0001/dx);
                DN.add_element(cur_idx_1, glob_idx_0_1,  u_s_0001/dx);
                // n.at_offset(edge, 1, 0, 0) *(n.at_offset(face, 1, 0, 0) + n.at_offset(face, 1, -1, 0))
                DN.add_element(cur_idx_1, glob_idx_0_3, -u_s_1011/dx);
                DN.add_element(cur_idx_1, glob_idx_0_4,  u_s_1011/dx);*/

                int p_idx_w = n.at_offset(idx_p_type::tag(), -1, 0, 0);
                int p_idx_e = n.at_offset(idx_p_type::tag(), 1, 0, 0);
                int p_idx_n = n.at_offset(idx_p_type::tag(), 0, 1, 0);
                int p_idx_s = n.at_offset(idx_p_type::tag(), 0, -1, 0);

                /*if (p_idx_w > 0)
                {
                    DN.add_element(cur_idx_0, glob_idx_00, v_s_0010);
                    DN.add_element(cur_idx_0, glob_idx_01, v_s_0111);
                }
                if (p_idx_s > 0)
                {
                    DN.add_element(cur_idx_1, glob_idx_00, u_s_0001);
                    DN.add_element(cur_idx_1, glob_idx_10, u_s_1011);
                }*/

                if (p_idx_w > 0)
                {
                    //-n.at_offset(edge, 0, 0, 0) *(n.at_offset(face, 0, 0, 1) + n.at_offset(face, -1, 0, 1))
                    DN.add_element(cur_idx_0, glob_idx_1, v_s_0010 / dx);
                    DN.add_element(cur_idx_0, glob_idx_1_1, -v_s_0010 / dx);
                    //-n.at_offset(edge, 0, 1, 0) *(n.at_offset(face, 0, 1, 1) + n.at_offset(face, -1, 1, 1))
                    DN.add_element(cur_idx_0, glob_idx_1_2, v_s_0111 / dx);
                    DN.add_element(cur_idx_0, glob_idx_1_3, -v_s_0111 / dx);

                    //-n.at_offset(edge, 0, 0, 0) *(n.at_offset(face, 0, 0, 1) + n.at_offset(face, -1, 0, 1))
                    DN.add_element(cur_idx_0, glob_idx_0, -v_s_0010 / dx);
                    DN.add_element(cur_idx_0, glob_idx_0_1, v_s_0010 / dx);
                    //-n.at_offset(edge, 0, 1, 0) *(n.at_offset(face, 0, 1, 1) + n.at_offset(face, -1, 1, 1))
                    DN.add_element(cur_idx_0, glob_idx_0_2, -v_s_0111 / dx);
                    DN.add_element(cur_idx_0, glob_idx_0, v_s_0111 / dx);
                }
                if (p_idx_s > 0)
                {
                    // n.at_offset(edge, 0, 0, 0) *(n.at_offset(face, 0, 0, 0) + n.at_offset(face, 0, -1, 0))
                    DN.add_element(cur_idx_1, glob_idx_1, u_s_0001 / dx);
                    DN.add_element(cur_idx_1, glob_idx_1_1, -u_s_0001 / dx);
                    // n.at_offset(edge, 1, 0, 0) *(n.at_offset(face, 1, 0, 0) + n.at_offset(face, 1, -1, 0))
                    DN.add_element(cur_idx_1, glob_idx_1_4, u_s_1011 / dx);
                    DN.add_element(cur_idx_1, glob_idx_1, -u_s_1011 / dx);

                    // n.at_offset(edge, 0, 0, 0) *(n.at_offset(face, 0, 0, 0) + n.at_offset(face, 0, -1, 0))
                    DN.add_element(cur_idx_1, glob_idx_0, -u_s_0001 / dx);
                    DN.add_element(cur_idx_1, glob_idx_0_1, u_s_0001 / dx);
                    // n.at_offset(edge, 1, 0, 0) *(n.at_offset(face, 1, 0, 0) + n.at_offset(face, 1, -1, 0))
                    DN.add_element(cur_idx_1, glob_idx_0_3, -u_s_1011 / dx);
                    DN.add_element(cur_idx_1, glob_idx_0_4, u_s_1011 / dx);
                }

                /*DN.add_element(cur_idx_0, glob_idx_00, v_s_0010);
                DN.add_element(cur_idx_0, glob_idx_01, v_s_0111);

                DN.add_element(cur_idx_1, glob_idx_00, u_s_0001);
                DN.add_element(cur_idx_1, glob_idx_10, u_s_1011);*/
            }
        }
        domain_->client_communicator().barrier();

        curl<U_old, edge_aux_type>();
        if (domain_->is_client())
        {
            auto client = domain_->decomposition().client();
            client->template buffer_exchange<edge_aux_type>(base_level);
        }

        for (auto it = domain_->begin(base_level); it != domain_->end(base_level); ++it)
        {
            if (!it->locally_owned() || !it->has_data()) continue;
            if (!it->is_leaf()) continue;
            if (it->is_correction()) continue;

            for (auto& n : it->data())
            {
                
                
                int cur_idx_0 = n(idx_u_type::tag(), 0);
                int cur_idx_1 = n(idx_u_type::tag(), 1);

                int glob_idx_0_00 = n.at_offset(idx_u_g_type::tag(), 0,  0, 0);
                int glob_idx_0_01 = n.at_offset(idx_u_g_type::tag(), 0, -1, 0);
                int glob_idx_0_10 = n.at_offset(idx_u_g_type::tag(), 1,  0, 0);
                int glob_idx_0_11 = n.at_offset(idx_u_g_type::tag(), 1, -1, 0);

                int glob_idx_1_00 = n.at_offset(idx_u_g_type::tag(),  0, 0, 1);
                int glob_idx_1_01 = n.at_offset(idx_u_g_type::tag(),  0, 1, 1);
                int glob_idx_1_10 = n.at_offset(idx_u_g_type::tag(), -1, 0, 1);
                int glob_idx_1_11 = n.at_offset(idx_u_g_type::tag(), -1, 1, 1);

                float_type om_00 = n.at_offset(edge_aux_type::tag(), 0, 0, 0)*0.25;
                float_type om_01 = n.at_offset(edge_aux_type::tag(), 0, 1, 0)*0.25;
                float_type om_10 = n.at_offset(edge_aux_type::tag(), 1, 0, 0)*0.25;

                int p_idx_w = n.at_offset(idx_p_type::tag(), -1, 0, 0);
                int p_idx_e = n.at_offset(idx_p_type::tag(), 1, 0, 0);
                int p_idx_n = n.at_offset(idx_p_type::tag(), 0, 1, 0);
                int p_idx_s = n.at_offset(idx_p_type::tag(), 0, -1, 0);

                if (p_idx_w > 0)
                {
                    DN.add_element(cur_idx_0, glob_idx_1_00, -om_00);
                    DN.add_element(cur_idx_0, glob_idx_1_10, -om_00);
                    DN.add_element(cur_idx_0, glob_idx_1_01, -om_01);
                    DN.add_element(cur_idx_0, glob_idx_1_11, -om_01);
                }
                if (p_idx_s > 0)
                {
                    DN.add_element(cur_idx_1, glob_idx_0_00, om_00);
                    DN.add_element(cur_idx_1, glob_idx_0_01, om_00);
                    DN.add_element(cur_idx_1, glob_idx_0_10, om_10);
                    DN.add_element(cur_idx_1, glob_idx_0_11, om_10);
                }

                /*DN.add_element(cur_idx_0, glob_idx_1_00, -om_00);
                DN.add_element(cur_idx_0, glob_idx_1_10, -om_00);
                DN.add_element(cur_idx_0, glob_idx_1_01, -om_01);
                DN.add_element(cur_idx_0, glob_idx_1_11, -om_01);

                DN.add_element(cur_idx_1, glob_idx_0_00,  om_00);
                DN.add_element(cur_idx_1, glob_idx_0_01,  om_00);
                DN.add_element(cur_idx_1, glob_idx_0_10,  om_10);
                DN.add_element(cur_idx_1, glob_idx_0_11,  om_10);*/
            }
        }
        domain_->client_communicator().barrier();
    }

    void construction_Grad() {
        //construction of Gradient
        boost::mpi::communicator world;
        world.barrier();


        

        if (world.rank() == 0) {
            return;
        }

        if (world.rank() == 1) {
            std::cout << "Constructing Grad matrix" << std::endl;
        }
       
        if (max_local_idx == 0) {
            std::cout << "idx not initialized, please call Assigning_idx()" << std::endl;
        }

        const auto dx_base = domain_->dx_base();

        Grad.resizing_row(max_local_idx+1);

        int base_level = domain_->tree()->base_level();
        if (domain_->is_client())
        {
            auto client = domain_->decomposition().client();

            //client->template buffer_exchange<idx_u_type>(base_level);
            client->template buffer_exchange<idx_u_g_type>(base_level);

            //client->template buffer_exchange<idx_p_type>(base_level);
            client->template buffer_exchange<idx_p_g_type>(base_level);
        }
        /*for (auto it = domain_->begin(base_level);
             it != domain_->end(base_level); ++it)
        {
            if (!it->locally_owned() || !it->has_data()) continue;
            if (!it->is_leaf()) continue;
            if (it->is_correction()) continue;

            for (auto& n : it->data())
            {
                int cur_idx_0 = n(idx_u_type::tag(), 0);
                int cur_idx_1 = n(idx_u_type::tag(), 1);

                int glo_idx = n(idx_p_g_type::tag(), 0);
                Grad.add_element(cur_idx_0, glo_idx, 1.0 / dx_base);
                //glo_idx = n(idx_p_g_type::tag(), 1);
                Grad.add_element(cur_idx_1, glo_idx, 1.0 / dx_base);

                glo_idx = n.at_offset(idx_p_g_type::tag(), -1, 0, 0);
                Grad.add_element(cur_idx_0, glo_idx, -1.0 / dx_base);

                glo_idx = n.at_offset(idx_p_g_type::tag(), 0, -1, 0);
                Grad.add_element(cur_idx_1, glo_idx, -1.0 / dx_base);
            }
        }*/

        int p_left_bot_idx = -1;
        int p_left_bot_ldx = -1; //locall index
        int u_left_bot_idx = -1;
        int v_left_bot_idx = -1;

        for (auto it = domain_->begin(base_level);
             it != domain_->end(base_level); ++it)
        {
            if (!it->locally_owned() || !it->has_data()) continue;
            if (!it->is_leaf()) continue;
            if (it->is_correction())
            {
                continue;
                /*for (auto& n : it->data())
                {
                    int cur_idx_0 = n(idx_u_type::tag(), 0);
                    int cur_idx_1 = n(idx_u_type::tag(), 1);

                    int glo_idx = n(idx_p_g_type::tag(), 0);
                    int glo_idx_10 = n.at_offset(idx_p_g_type::tag(), -1, 0, 0);
                    int glo_idx_01 = n.at_offset(idx_p_g_type::tag(), 0, -1, 0);
                    if (glo_idx < 0) continue;

                    if (glo_idx_10 > 0) {
                        Grad.add_element(cur_idx_0, glo_idx, 1.0 / dx_base);
                        Grad.add_element(cur_idx_0, glo_idx_10, -1.0 / dx_base);
                    }
                    if (glo_idx_01 > 0) {
                        Grad.add_element(cur_idx_1, glo_idx, 1.0 / dx_base);
                        Grad.add_element(cur_idx_1, glo_idx_01, -1.0 / dx_base);
                    }
                }*/
            }
            else
            {
                /*for (auto& n : it->data())
                {
                    int cur_idx_0 = n(idx_u_type::tag(), 0);
                    int cur_idx_1 = n(idx_u_type::tag(), 1);

                    int glo_idx = n(idx_p_g_type::tag(), 0);
                    Grad.add_element(cur_idx_0, glo_idx, 1.0 / dx_base);
                    //glo_idx = n(idx_p_g_type::tag(), 1);
                    Grad.add_element(cur_idx_1, glo_idx, 1.0 / dx_base);

                    glo_idx = n.at_offset(idx_p_g_type::tag(), -1, 0, 0);
                    Grad.add_element(cur_idx_0, glo_idx, -1.0 / dx_base);

                    glo_idx = n.at_offset(idx_p_g_type::tag(), 0, -1, 0);
                    Grad.add_element(cur_idx_1, glo_idx, -1.0 / dx_base);
                }*/
                for (auto& n : it->data())
                {
                    int cur_idx_0 = n(idx_u_type::tag(), 0);
                    int cur_idx_1 = n(idx_u_type::tag(), 1);

                    int glo_idx_0  = n(idx_p_g_type::tag(), 0);
                    int glo_idx_10 = n.at_offset(idx_p_g_type::tag(), -1, 0, 0);
                    int glo_idx_01 = n.at_offset(idx_p_g_type::tag(), 0, -1, 0);

                    /*if (glo_idx_10 < 0 && glo_idx_01 < 0) {
                        p_left_bot_idx = glo_idx_0;
                        p_left_bot_ldx = n(idx_p_type::tag(), 0);
                        u_left_bot_idx = n.at_offset(idx_u_type::tag(), 1, 0, 0);
                        v_left_bot_idx = n.at_offset(idx_u_type::tag(), 0, 1, 1);
                    }*/

                    if (glo_idx_10 > 0) {
                        Grad.add_element(cur_idx_0, glo_idx_0, 1.0 / dx_base);
                        Grad.add_element(cur_idx_0, glo_idx_10, -1.0 / dx_base);
                    }
                    if (glo_idx_01 > 0) {
                        Grad.add_element(cur_idx_1, glo_idx_0, 1.0 / dx_base);
                        Grad.add_element(cur_idx_1, glo_idx_01, -1.0 / dx_base);
                    }
                }
            }
        }

        /*if (p_left_bot_idx > 0) {
            Grad.add_element(u_left_bot_idx, p_left_bot_idx, 1.0 / dx_base);
            Grad.add_element(v_left_bot_idx, p_left_bot_idx, 1.0 / dx_base);
        }*/

        Grad.clean_entry(1e-15);

        domain_->client_communicator().barrier();
    }

    void construction_Div() {
        //construction of Gradient
        boost::mpi::communicator world;
        world.barrier();

        if (world.rank() == 0) {
            return;
        }

        if (world.rank() == 1) {
            std::cout << "Constructing div matrix" << std::endl;
        }
       
        if (max_local_idx == 0) {
            std::cout << "idx not initialized, please call Assigning_idx()" << std::endl;
        }

        const auto dx_base = domain_->dx_base();

        Div.resizing_row(max_local_idx+1);

        int base_level = domain_->tree()->base_level();
        if (domain_->is_client())
        {
            auto client = domain_->decomposition().client();

            //client->template buffer_exchange<idx_u_type>(base_level);
            client->template buffer_exchange<idx_u_g_type>(base_level);

            //client->template buffer_exchange<idx_p_type>(base_level);
            client->template buffer_exchange<idx_p_g_type>(base_level);
        }
        /*for (auto it = domain_->begin(base_level);
             it != domain_->end(base_level); ++it)
        {
            if (!it->locally_owned() || !it->has_data()) continue;
            if (!it->is_leaf()) continue;
            //if (it->is_correction()) continue;

            for (auto& n : it->data())
            {
                int cur_idx = n(idx_p_type::tag(), 0);
                if (cur_idx < 0) continue;
                int glo_idx_0 = n(idx_u_g_type::tag(), 0);
                int glo_idx_1 = n(idx_u_g_type::tag(), 1);

                
                Div.add_element(cur_idx, glo_idx_0, -1.0 / dx_base);
                Div.add_element(cur_idx, glo_idx_1, -1.0 / dx_base);

                glo_idx_0 = n.at_offset(idx_u_g_type::tag(), 1, 0, 0);
                Div.add_element(cur_idx, glo_idx_0, 1.0 / dx_base);

                glo_idx_1 = n.at_offset(idx_u_g_type::tag(), 0, 1, 1);
                Div.add_element(cur_idx, glo_idx_1, 1.0 / dx_base);
            }
        }*/
        for (auto it = domain_->begin(base_level);
             it != domain_->end(base_level); ++it)
        {
            if (!it->locally_owned() || !it->has_data()) continue;
            if (!it->is_leaf()) continue;
            if (it->is_correction())
            {
                for (auto& n : it->data())
                {
                    int cur_idx = n(idx_p_type::tag(), 0);
                    if (cur_idx < 0) continue;
                    int glo_idx_0_0 = n(idx_u_g_type::tag(), 0);
                    int glo_idx_1_0 = n(idx_u_g_type::tag(), 1);

                    int glo_idx_0_1 = n.at_offset(idx_u_g_type::tag(), -1, 0, 0);
                    int glo_idx_1_1 = n.at_offset(idx_u_g_type::tag(), 0, -1, 1);

                    if (glo_idx_0_0 < 0 || glo_idx_0_1 < 0 || glo_idx_1_0 < 0 ||
                        glo_idx_1_1 < 0)
                    {
                        continue;
                    }

                    Div.add_element(cur_idx, glo_idx_0_0, -1.0 / dx_base);
                    Div.add_element(cur_idx, glo_idx_1_0, -1.0 / dx_base);
                    Div.add_element(cur_idx, glo_idx_0_1, 1.0 / dx_base);
                    Div.add_element(cur_idx, glo_idx_1_1, 1.0 / dx_base);
                }
                //continue;
            }
            else
            {
                for (auto& n : it->data())
                {
                    int cur_idx = n(idx_p_type::tag(), 0);

                    int glo_p_idx_10 = n.at_offset(idx_p_g_type::tag(), -1,  0,  0);
                    int glo_p_idx_01 = n.at_offset(idx_p_g_type::tag(),  0, -1,  0);

                    if (glo_p_idx_10 < 0 && glo_p_idx_01 < 0) {
                        //does not enforce divergence free on one corner, instead force the pressure at that point to be zero
                        //here we chose the bottom left one for convenience during development and verification, 
                        //in the future, need to change to a unique point
                        int glo_p_idx = n(idx_p_g_type::tag(), 0);
                        Div.add_element(cur_idx, glo_p_idx,1.0);
                        continue;
                    }
                    if (cur_idx < 0) continue;
                    int glo_idx_0 = n(idx_u_g_type::tag(), 0);
                    int glo_idx_1 = n(idx_u_g_type::tag(), 1);

                    Div.add_element(cur_idx, glo_idx_0, -1.0 / dx_base);
                    Div.add_element(cur_idx, glo_idx_1, -1.0 / dx_base);

                    glo_idx_0 = n.at_offset(idx_u_g_type::tag(), 1, 0, 0);
                    Div.add_element(cur_idx, glo_idx_0, 1.0 / dx_base);

                    glo_idx_1 = n.at_offset(idx_u_g_type::tag(), 0, 1, 1);
                    Div.add_element(cur_idx, glo_idx_1, 1.0 / dx_base);
                }
            }
        }
        domain_->client_communicator().barrier();
    }

    void construction_Curl() {
        //construction of Gradient
        boost::mpi::communicator world;
        world.barrier();

        if (world.rank() == 0) {
            return;
        }

        if (world.rank() == 1) {
            std::cout << "Constructing div matrix" << std::endl;
        }
       
        if (max_local_idx == 0) {
            std::cout << "idx not initialized, please call Assigning_idx()" << std::endl;
        }

        const auto dx_base = domain_->dx_base();

        Curl.resizing_row(max_local_idx+1);

        int base_level = domain_->tree()->base_level();
        if (domain_->is_client())
        {
            auto client = domain_->decomposition().client();

            client->template buffer_exchange<idx_u_g_type>(base_level);
            client->template buffer_exchange<idx_w_g_type>(base_level);
        }
        for (auto it = domain_->begin(base_level);
             it != domain_->end(base_level); ++it)
        {
            if (!it->locally_owned() || !it->has_data()) continue;
            if (!it->is_leaf()) continue;
            /*if (!it->is_correction())
            {
                for (auto& n : it->data())
                {
                    int cur_idx = n(idx_w_type::tag(), 0);
                    int glo_idx = n(idx_w_g_type::tag(), 0);
                    if (cur_idx < 0) continue;
                    int glo_idx_0_0 = n(idx_u_g_type::tag(), 0);
                    int glo_idx_1_0 = n(idx_u_g_type::tag(), 1);

                    int glo_idx_0_1 = n.at_offset(idx_u_g_type::tag(), 0, -1, 0);
                    int glo_idx_1_1 = n.at_offset(idx_u_g_type::tag(), -1, 0, 1);

                    Curl.add_element(cur_idx, glo_idx_0_0, -1.0 / dx_base);
                    Curl.add_element(cur_idx, glo_idx_1_0, 1.0 / dx_base);
                    Curl.add_element(cur_idx, glo_idx_0_1, 1.0 / dx_base);
                    Curl.add_element(cur_idx, glo_idx_1_1, -1.0 / dx_base);
                    Curl.add_element(cur_idx, glo_idx, -1.0);
                }
                //continue;
            }*/
            for (auto& n : it->data())
            {
                int cur_idx = n(idx_w_type::tag(), 0);
                int glo_idx = n(idx_w_g_type::tag(), 0);
                if (cur_idx < 0) continue;
                int glo_idx_0_0 = n(idx_u_g_type::tag(), 0);
                int glo_idx_1_0 = n(idx_u_g_type::tag(), 1);

                int glo_idx_0_1 = n.at_offset(idx_u_g_type::tag(), 0, -1, 0);
                int glo_idx_1_1 = n.at_offset(idx_u_g_type::tag(), -1, 0, 1);

                Curl.add_element(cur_idx, glo_idx_0_0, -1.0 / dx_base);
                Curl.add_element(cur_idx, glo_idx_1_0, 1.0 / dx_base);
                Curl.add_element(cur_idx, glo_idx_0_1, 1.0 / dx_base);
                Curl.add_element(cur_idx, glo_idx_1_1, -1.0 / dx_base);
                Curl.add_element(cur_idx, glo_idx, -1.0);

                /*Curl.add_element(cur_idx, glo_idx_0_0, -1.0 / dx_base * Curl_factor);
                Curl.add_element(cur_idx, glo_idx_1_0, 1.0 / dx_base * Curl_factor);
                Curl.add_element(cur_idx, glo_idx_0_1, 1.0 / dx_base * Curl_factor);
                Curl.add_element(cur_idx, glo_idx_1_1, -1.0 / dx_base * Curl_factor);
                Curl.add_element(cur_idx, glo_idx, -1.0 * Curl_factor);*/ //divide by dx to make condition number bigger


            }
        }
        Curl.scale_entries(Curl_factor);
        domain_->client_communicator().barrier();
    }


    void construction_Smearing(float_type factor = 1.0) {
        //construction of Gradient
        boost::mpi::communicator world;
        world.barrier();

        if (world.rank() == 0) {
            return;
        }

        if (world.rank() == 1) {
            std::cout << "Constructing smearing matrix" << std::endl;
        }
       
        if (max_local_idx == 0) {
            std::cout << "idx not initialized, please call Assigning_idx()" << std::endl;
        }

        const auto dx_base = domain_->dx_base();

        smearing.resizing_row(max_local_idx+1);

        int base_level = domain_->tree()->base_level();

        domain_->ib().communicator().compute_indices();

        force_type tmp_f_idx_g = forcing_idx_g;
        for (std::size_t i=0; i<tmp_f_idx_g.size(); ++i)
        {

            for (std::size_t d=0; d<tmp_f_idx_g[0].size(); ++d) {
                if (tmp_f_idx_g[i][d] < 0) tmp_f_idx_g[i][d] = 0;
            }
        }
        domain_->client_communicator().barrier();

        real_coordinate_type tmp_coord(0.0);
        force_type forcing_idx_all(domain_->ib().size(), tmp_coord);
        if (domain_->ib().size() > 0)
        {
            boost::mpi::all_reduce(domain_->client_communicator(), &tmp_f_idx_g[0], domain_->ib().size(), &forcing_idx_all[0],
                std::plus<real_coordinate_type>());
        }

        /*for (int i = 0; i < domain_->ib().size();i++) {
            if (boost)
        }*/
        
        //domain_->ib().communicator().communicate(true, forcing_idx_g);

        if (domain_->is_client())
        {
            auto client = domain_->decomposition().client();

            //client->template buffer_exchange<idx_u_type>(base_level);
            client->template buffer_exchange<idx_u_g_type>(base_level);

            //client->template buffer_exchange<idx_p_type>(base_level);
            //client->template buffer_exchange<idx_p_g_type>(base_level);
        }
        for (std::size_t i=0; i<domain_->ib().size(); ++i)
        {
            std::size_t oct_i=0;
            for (auto it: domain_->ib().influence_list(i))
            {
                if (!it->locally_owned()) continue;
                auto ib_coord = domain_->ib().scaled_coordinate(i, it->refinement_level());

                auto block = domain_->ib().influence_pts(i, oct_i);

                for (auto& node : block)
                {
                    auto n_coord = node.level_coordinate();
                    auto dist = n_coord - ib_coord;

                    for (std::size_t field_idx = 0; field_idx < idx_u_type::nFields();
                         field_idx++)
                    {
                        decltype(ib_coord) off(0.5);
                        off[field_idx] = 0.0; // face data location

                        auto ddf = domain_->ib().delta_func();

                        float_type val = ddf(dist + off) * factor;
                        if (std::abs(val) < 1e-12) {
                            //std::cout << "uninfluenced node found with val " << val << std::endl;
                            continue;
                        }
                        int u_loc = node(idx_u_type::tag(),field_idx);
                        //int f_glob = forcing_idx_g[i][field_idx];
                        int f_glob = forcing_idx_all[i][field_idx];
                        smearing.add_element(u_loc, f_glob, val);
                        /*node(u, field_idx) +=
                            f[field_idx] * ddf(dist + off) * factor;*/
                    }
                }

                /*domain::Operator::ib_smearing<U>
                    (ib_coord, f[i], domain_->ib().influence_pts(i, oct_i), domain_->ib().delta_func());*/
                oct_i+=1;
            }
        }
        domain_->client_communicator().barrier();
    }

    void construction_Projection()
    {
        //construction of Gradient
        boost::mpi::communicator world;
        world.barrier();

        //if (world.rank() == 0) { return; }

        if (world.rank() == 1)
        {
            std::cout << "Constructing projection matrix" << std::endl;
        }

        if (max_local_idx == 0 && world.rank() != 0)
        {
            std::cout << "idx not initialized, please call Assigning_idx()"
                      << std::endl;
        }

        const auto dx_base = domain_->dx_base();

        if (world.rank() != 0) project.resizing_row(max_local_idx + 1);
        int base_level = domain_->tree()->base_level();

        //int base_level = domain_->tree()->base_level();

        //domain_->ib().communicator().compute_indices();
        //domain_->ib().communicator().communicate(true, forcing_idx_g);

        if (domain_->is_client())
        {
            auto client = domain_->decomposition().client();

            //client->template buffer_exchange<idx_u_type>(base_level);
            client->template buffer_exchange<idx_u_g_type>(base_level);

            //client->template buffer_exchange<idx_p_type>(base_level);
            //client->template buffer_exchange<idx_p_g_type>(base_level);
        }
        for (std::size_t i = 0; i < domain_->ib().size(); ++i)
        {
            std::vector<int> ib_rank_vec;
            int loc_ib_rank = domain_->ib().rank(i);
            boost::mpi::gather(world, loc_ib_rank, ib_rank_vec, 0);
            int ac_rank = -1;
            if (world.rank() == 0)
            {
                for (int j = 0; j < ib_rank_vec.size(); j++)
                {
                    if (ac_rank > 0 && ib_rank_vec[j] > 0 &&
                        ib_rank_vec[j] != ac_rank)
                    {
                        std::cout
                            << "IB_rank wrong with two different positive ranks "
                            << ac_rank << " and " << ib_rank_vec[j]
                            << std::endl;
                    }
                    if (ib_rank_vec[j] > 0) { ac_rank = ib_rank_vec[j]; }
                }
                if (ac_rank < 0)
                {
                    std::cout << "No processors have the right IB rank"
                              << std::endl;
                }
            }

            int ib_rank = ac_rank;
            boost::mpi::broadcast(world, ib_rank, 0);
            if (world.rank() == 0) continue;
            for (std::size_t field_idx = 0; field_idx < idx_u_g_type::nFields();
                 field_idx++)
            {
                //if (world.rank() == 1) std::cout << "comm_ rank " << comm_.rank();
                int f_loc = forcing_idx[i][field_idx];
                if (domain_->ib().rank(i) != comm_.rank())
                {
                    //std::cout << "Rank " << world.rank() << " ib rank " << domain_->ib().rank(i) << std::endl;
                    
                    //compute and send to the target processor
                    std::map<int, float_type> tmp_row;
                    std::size_t oct_i = 0;

                    for (auto it : domain_->ib().influence_list(i))
                    {
                        if (!it->locally_owned()) continue;
                        auto ib_coord = domain_->ib().scaled_coordinate(i,
                            it->refinement_level());

                        auto block = domain_->ib().influence_pts(i, oct_i);

                        for (auto& node : block)
                        {
                            auto n_coord = node.level_coordinate();
                            auto dist = n_coord - ib_coord;

                            decltype(ib_coord) off(0.5);
                            off[field_idx] = 0.0; // face data location

                            auto ddf = domain_->ib().delta_func();

                            float_type val = ddf(dist + off);
                            if (std::abs(val) < 1e-12) continue;
                            int u_glob = node(idx_u_g_type::tag(), field_idx);
                            //int f_loc = forcing_idx[i][field_idx];
                            this->map_add_element(u_glob, val, tmp_row);
                            /*node(u, field_idx) +=
                            f[field_idx] * ddf(dist + off) * factor;*/
                            
                        }

                        oct_i += 1;
                    }
                    //std::cout << "Rank " << world.rank() << "start sending" << std::endl;
                    world.send(ib_rank, world.rank(), tmp_row);
                } 
                else
                {
                    //std::cout << "Rank " << world.rank() << " ib rank " << domain_->ib().rank(i) << std::endl;
                    std::size_t oct_i = 0;

                    for (auto it : domain_->ib().influence_list(i))
                    {
                        if (!it->locally_owned()) continue;
                        auto ib_coord = domain_->ib().scaled_coordinate(i,
                            it->refinement_level());

                        auto block = domain_->ib().influence_pts(i, oct_i);

                        for (auto& node : block)
                        {
                            auto n_coord = node.level_coordinate();
                            auto dist = n_coord - ib_coord;

                            decltype(ib_coord) off(0.5);
                            off[field_idx] = 0.0; // face data location

                            auto ddf = domain_->ib().delta_func();

                            float_type val = ddf(dist + off);
                            if (std::abs(val) < 1e-12) continue;
                            int u_glob = node(idx_u_g_type::tag(), field_idx);
                            
                            project.add_element(f_loc, u_glob, val);
                            /*node(u, field_idx) +=
                            f[field_idx] * ddf(dist + off) * factor;*/
                        }

                        oct_i += 1;
                    }

                    for (int k = 1; k < world.size(); k++)
                    {
                        if (k == world.rank()) continue;
                        std::map<int, float_type> res;
                        world.recv(k, k, res);
                        for (const auto& [key, val] : res)
                        {
                            project.add_element(f_loc, key, val);
                        }
                        //std::cout << "received " << k << std::endl;
                    }
                    //std::cout << "after recieving" << std::endl;
                }
                domain_->client_communicator().barrier();
            }
        }
        world.barrier();
    }

    void construction_BCMat()
    {
        //construction of Gradient
        boost::mpi::communicator world;
        world.barrier();

        /*if (world.rank() == 0) {
            return;
        }*/

        if (world.rank() == 1)
        {
            std::cout << "Constructing BC matrix" << std::endl;
        }

        if (max_local_idx == 0 && world.rank() != 0)
        {
            std::cout << "idx not initialized, please call Assigning_idx()"
                      << std::endl;
        }

        const auto dx_base = domain_->dx_base();
        int        base_level = domain_->tree()->base_level();

        boundary_u.resizing_row(max_local_idx + 1);

        int counter = 0; //count the number of bc pts in this processor

        if (world.rank() != 0)
        {
            for (auto it = domain_->begin(base_level);
                 it != domain_->end(base_level); ++it)
            {
                if (!it->locally_owned() || !it->has_data()) continue;
                if (!it->is_leaf()) continue;
                if (!it->is_correction()) continue;

                for (auto& n : it->data())
                {
                    int cur_idx_0 = n(idx_u_type::tag(), 0);
                    int cur_idx_1 = n(idx_u_type::tag(), 1);

                    if (cur_idx_0 > 0) { counter++; }
                }
            }
        }

        world.barrier();

        std::vector<int> tasks_vec;

        tasks_vec.resize(world.size());

        tasks_vec[0] = 0;

        if (world.rank() == 0)
        {
            for (int i = 1; i < world.size(); i++)
            {
                int tmp_counter;
                world.recv(i, i, tmp_counter);
                tasks_vec[i] = tmp_counter;
            }
        }
        else
        {
            world.send(0, world.rank(), counter);
        }

        world.barrier();

        boost::mpi::broadcast(world, tasks_vec, 0);

        if (world.rank() == 0)
        {
            std::cout << "Number of tasks received is " << tasks_vec.size()
                      << std::endl;
            int sum = 0;
            for (int i = 0; i < tasks_vec.size(); i++) { sum += tasks_vec[i]; }
            std::cout << "Number of node points need to compute " << sum
                      << std::endl;
        }

        world.barrier();
        /*if (world.rank() == 0) {
            return;
        }*/

        //int base_level = domain_->tree()->base_level();

        //domain_->ib().communicator().compute_indices();
        //domain_->ib().communicator().communicate(true, forcing_idx_g);

        if (domain_->is_client())
        {
            auto client = domain_->decomposition().client();

            //client->template buffer_exchange<idx_u_type>(base_level);
            client->template buffer_exchange<idx_u_g_type>(base_level);

            //client->template buffer_exchange<idx_p_type>(base_level);
            //client->template buffer_exchange<idx_p_g_type>(base_level);
        }

        for (int i = 1; i < tasks_vec.size(); i++)
        {
            int root_rank = domain_->client_communicator().rank();

            boost::mpi::broadcast(world, root_rank, i);
            if (world.rank() == 0) { continue; }
            if (world.rank() == i)
            {
                std::cout << "Rank " << i << " root rank " << root_rank
                          << std::endl;
                for (auto it = domain_->begin(base_level);
                     it != domain_->end(base_level); ++it)
                {
                    if (!it->locally_owned() || !it->has_data()) continue;
                    if (!it->is_leaf()) continue;
                    if (!it->is_correction()) continue;

                    for (auto& n : it->data())
                    {
                        int cur_idx_0 = n(idx_u_type::tag(), 0);
                        int cur_idx_1 = n(idx_u_type::tag(), 1);

                        if (cur_idx_0 > 0)
                        {
                            //int cur_idx_0 = n(idx_u_type::tag(), 0);
                            //int cur_idx_1 = n(idx_u_type::tag(), 1);
                            int glob_idx_0 = n(idx_u_g_type::tag(), 0);
                            int glob_idx_1 = n(idx_u_g_type::tag(), 1);

                            //function that inquires all the idx from other processors
                            const auto& coord_cur = n.level_coordinate();

                            std::map<int, float_type> row_u;
                            std::map<int, float_type> row_v;

                            if (use_FMM)
                            {
                                auto lb_c = it->data_r(idx_w_g_type::tag())
                                                .real_block()
                                                .base();
                                get_BC_blk_from_client_FMM(lb_c, coord_cur, root_rank,
                                    row_u, row_v);
                            }
                            else
                            {
                                get_BC_idx_from_client(coord_cur, root_rank,
                                    row_u, row_v);
                            }

                            for (const auto& [key, val] : row_u)
                            {
                                boundary_u.add_element(cur_idx_0, key, val);
                            }
                            for (const auto& [key, val] : row_v)
                            {
                                boundary_u.add_element(cur_idx_1, key, val);
                            }

                            //add the negative of identity here
                            boundary_u.add_element(cur_idx_0, glob_idx_0, -1.0);
                            boundary_u.add_element(cur_idx_1, glob_idx_1, -1.0);
                        }
                    }
                }
            }
            else
            {
                for (int j = 0; j < tasks_vec[i]; j++)
                {
                    std::map<int, float_type> row_u;
                    std::map<int, float_type> row_v;
                    /*get_BC_idx_from_client(coordinate_type({0, 0}),
                                root_rank, row_u,
                                row_v);*/

                    if (use_FMM)
                    {
                        get_BC_blk_from_client_FMM(coordinate_type({0, 0}),
                            coordinate_type({0, 0}), root_rank, row_u, row_v);
                    }
                    else
                    {
                        get_BC_idx_from_client(coordinate_type({0, 0}),
                            root_rank, row_u, row_v);
                    }
                }
            }
            domain_->client_communicator().barrier();
        }

        domain_->client_communicator().barrier();
    }

    void get_BC_idx_from_client(coordinate_type c, int root, std::map<int, float_type>& row_u, std::map<int, float_type>& row_v) {
        //Send and recv 
        //send coordinates to other processes

        //clear the destination matrix rows
        row_u.clear();
        row_v.clear();

        coordinate_type c_loc = c;
        boost::mpi::broadcast(domain_->client_communicator(), c_loc, root);
        //std::cout << root << " " <<domain_->client_communicator().rank() << " c is " << c_loc[0] << " " << c_loc[1] << std::endl;
        std::map<int, float_type> loc_smat_u; //local resulting matrix to get BC
        std::map<int, float_type> loc_smat_v; //local resulting matrix to get BC

        int base_level = domain_->tree()->base_level();
        const auto dx_base = domain_->dx_base();
        if (domain_->is_client())
        {
            auto client = domain_->decomposition().client();

            //client->template buffer_exchange<idx_u_type>(base_level);
            client->template buffer_exchange<idx_u_g_type>(base_level);

            //client->template buffer_exchange<idx_p_type>(base_level);
            //client->template buffer_exchange<idx_p_g_type>(base_level);
        }
        for (auto it = domain_->begin(base_level);
             it != domain_->end(base_level); ++it)
        {
            if (!it->locally_owned() || !it->has_data()) continue;
            if (!it->is_leaf()) continue;
            if (it->is_correction()) continue;

            for (auto& n : it->data())
            {
                int u_idx_0 = n(idx_u_g_type::tag(), 0);
                int u_idx_0_1 = n.at_offset(idx_u_g_type::tag(), 0, -1, 0);
                int u_idx_1 = n(idx_u_g_type::tag(), 1);
                int u_idx_1_1 = n.at_offset(idx_u_g_type::tag(), -1, 0, 1);

                const auto& coord_loc = n.level_coordinate();

                int x_c = c_loc[0] - coord_loc[0];
                if (x_c != (c_loc[0] - coord_loc[0]))
                {
                    std::cout << "Coordinate type not INT" << std::endl;
                }

                int        y_c = c_loc[1] - coord_loc[1];

                int factor = 1.0;

                if (use_FMM)
                {
                    int max_idx = std::abs(x_c);
                    if (std::abs(x_c) < std::abs(y_c))
                    {
                        max_idx = std::abs(y_c);
                    }
                    int prev_bin = 0;
                    int stride = 1;
                    int center_idx = 0;
                    for (const auto& [key, val] : FMM_bin)
                    {
                        if (max_idx < key) {
                            //this is the bin to stay in and check if the point need to be included
                            stride = val;
                            center_idx = (stride - 1)/2;
                            if (center_idx*2 != (stride - 1)) {
                                throw std::runtime_error("stride is not odd so not power of 3");
                            }
                            break;
                        }
                        prev_bin = key;
                    }
                    int rel_idx_x = prev_bin + std::abs(x_c);
                    int rel_idx_y = prev_bin + std::abs(y_c);
                    if ((rel_idx_x % stride != center_idx) || (rel_idx_y % stride != center_idx)) {
                        continue;
                    }
                    else {
                        factor = stride*stride;
                    }
                }
                float_type lgf_val_00 =
                    lgf_lap_.derived().get(coordinate_type({x_c, y_c}));
                float_type lgf_val_01 =
                    lgf_lap_.derived().get(coordinate_type({x_c, (y_c + 1)}));
                float_type lgf_val_10 =
                    lgf_lap_.derived().get(coordinate_type({(x_c + 1), y_c}));

                float_type u_weight = (lgf_val_01 - lgf_val_00)*factor;
                float_type v_weight = (lgf_val_00 - lgf_val_10)*factor;

                //u = -1/dx * C^T L^inv (dx^2) 1/dx C
                //so all dx cancels

                this->map_add_element(u_idx_0, u_weight, loc_smat_u);
                this->map_add_element(u_idx_0_1, -u_weight, loc_smat_u);
                this->map_add_element(u_idx_1, -u_weight, loc_smat_u);
                this->map_add_element(u_idx_1_1, u_weight, loc_smat_u);

                this->map_add_element(u_idx_0, v_weight, loc_smat_v);
                this->map_add_element(u_idx_0_1, -v_weight, loc_smat_v);
                this->map_add_element(u_idx_1, -v_weight, loc_smat_v);
                this->map_add_element(u_idx_1_1, v_weight, loc_smat_v);
            }
        }

        domain_->client_communicator().barrier();
        std::vector<std::map<int, float_type>> dest_mat_u;
        std::vector<std::map<int, float_type>> dest_mat_v;
        boost::mpi::gather(domain_->client_communicator(), loc_smat_u, dest_mat_u, root);
        boost::mpi::gather(domain_->client_communicator(), loc_smat_v, dest_mat_v, root);

        if (domain_->client_communicator().rank() != root) {
            return;
        }

        for (int i = 0; i < dest_mat_u.size();i++) {
            //accumulate vectors from other processors
            map_add_map(dest_mat_u[i], row_u);
        }

        for (int i = 0; i < dest_mat_v.size();i++) {
            //accumulate vectors from other processors
            map_add_map(dest_mat_v[i], row_v);
        }
    }

    void get_BC_blk_from_client_FMM(coordinate_type lb_c_r, coordinate_type c, int root, 
        std::map<int, float_type>& row_u, std::map<int, float_type>& row_v)
    {
        //Send and recv
        //send coordinates to other processes

        //clear the destination matrix rows
        boost::mpi::communicator world;
        row_u.clear();
        row_v.clear();

        coordinate_type base = domain_->bounding_box().base();
        coordinate_type c_loc = c - base;
        coordinate_type lb_c= lb_c_r - base;

        boost::mpi::broadcast(domain_->client_communicator(), c_loc, root);
        boost::mpi::broadcast(domain_->client_communicator(), lb_c, root);
        //std::cout << root << " " <<domain_->client_communicator().rank() << " c is " << c_loc[0] << " " << c_loc[1] << std::endl;
        std::map<int, float_type> loc_smat_u; //local resulting matrix to get BC
        std::map<int, float_type> loc_smat_v; //local resulting matrix to get BC

        int        base_level = domain_->tree()->base_level();
        const auto dx_base = domain_->dx_base();

        for (int l = base_level; l >= 1; l--)
        {
            for (auto it = domain_->begin(l);
                 it != domain_->end(l); ++it)
            {
                if (!it->locally_owned() || !it->has_data()) continue;
                if (!it->is_leaf() && l == base_level) continue;
                if (it->is_correction() && l == base_level) continue;

                //check if this block should be included
                bool include_oct = false;
                int N = domain_->block_extent()[0];
                coordinate_type add_one;
                add_one[0] = 1;
                add_one[1] = 1;

                int n_range_p = std::pow(2, base_level - l + 1) * N;
                int n_range_c = std::pow(2, base_level - l) * N;
                
                auto cur_real_base = (it->data_r(idx_w_g_type::tag()).real_block().base() - base + add_one) * std::pow(2, base_level - l);
                auto prt_real_base = (it->parent()->data_r(idx_w_g_type::tag()).real_block().base() - base + add_one) * std::pow(2, base_level - l + 1);
                auto tar_real_base = lb_c + add_one;

                //check if parents are neighbors
                auto dist = tar_real_base - prt_real_base;
                if (dist[0] >= -n_range_p && dist[0] < n_range_p * 2 &&
                    dist[1] >= -n_range_p && dist[1] < n_range_p * 2) {
                    include_oct = true;
                }

                auto dist_c = tar_real_base - cur_real_base;

                if (l != base_level &&
                    dist_c[0] >= -n_range_c && dist_c[0] < n_range_c * 2 &&
                    dist_c[1] >= -n_range_c && dist_c[1] < n_range_c * 2) {
                    include_oct = false;
                }

                if (!include_oct) continue;

                if (l != base_level)
                {
                    int   N = domain_->block_extent()[0] + lBuffer + rBuffer;
                    auto& lin_data_g =
                        it->data_r(idx_w_g_type::tag(), 0).linalg_data();
                    for (int sub_i = 0; sub_i < N; sub_i++)
                    {
                        for (int sub_j = 0; sub_j < N; sub_j++)
                        {
                            auto real_base = it->data_r(idx_w_g_type::tag())
                                                 .real_block()
                                                 .base() -
                                             base;

                            int x_c_tmp = (real_base[0] + sub_i) *
                                          std::pow(2, base_level - l);
                            int y_c_tmp = (real_base[1] + sub_j) *
                                          std::pow(2, base_level - l);

                            int x_c = c_loc[0] - x_c_tmp;
                            if (x_c != (c_loc[0] - x_c_tmp))
                            {
                                std::cout << "Coordinate type not INT"
                                          << std::endl;
                            }

                            int y_c = c_loc[1] - y_c_tmp;

                            float_type lgf_val_00 = lgf_lap_.derived().get(
                                coordinate_type({x_c, y_c}));
                            float_type lgf_val_01 = lgf_lap_.derived().get(
                                coordinate_type({x_c, (y_c + 1)}));
                            float_type lgf_val_10 = lgf_lap_.derived().get(
                                coordinate_type({(x_c + 1), y_c}));

                            float_type u_weight = (lgf_val_01 - lgf_val_00);
                            float_type v_weight = (lgf_val_00 - lgf_val_10);

                            int w_idx = lin_data_g.at(sub_i, sub_j);

                            this->map_add_element(w_idx, -u_weight * dx_base,
                                loc_smat_u);

                            this->map_add_element(w_idx, -v_weight * dx_base,
                                loc_smat_v);
                        }
                    }
                }

                else
                {
                    for (auto& n : it->data())
                    {
                        const auto& coord_loc_level =
                            n.level_coordinate() - base;
                        const auto& coord_loc =
                            coord_loc_level * std::pow(2, base_level - l);

                        int x_c = c_loc[0] - coord_loc[0];
                        if (x_c != (c_loc[0] - coord_loc[0]))
                        {
                            std::cout << "Coordinate type not INT" << std::endl;
                        }

                        int y_c = c_loc[1] - coord_loc[1];

                        float_type lgf_val_00 =
                            lgf_lap_.derived().get(coordinate_type({x_c, y_c}));
                        float_type lgf_val_01 = lgf_lap_.derived().get(
                            coordinate_type({x_c, (y_c + 1)}));
                        float_type lgf_val_10 = lgf_lap_.derived().get(
                            coordinate_type({(x_c + 1), y_c}));

                        float_type u_weight = (lgf_val_01 - lgf_val_00);
                        float_type v_weight = (lgf_val_00 - lgf_val_10);

                        int w_idx = n(idx_w_g_type::tag(), 0);

                        this->map_add_element(w_idx, -u_weight * dx_base,
                            loc_smat_u);

                        this->map_add_element(w_idx, -v_weight * dx_base,
                            loc_smat_v);
                    }
                }
            }
        }
        domain_->client_communicator().barrier();
        std::vector<std::map<int, float_type>> dest_mat_u;
        std::vector<std::map<int, float_type>> dest_mat_v;
        boost::mpi::gather(domain_->client_communicator(), loc_smat_u, dest_mat_u, root);
        boost::mpi::gather(domain_->client_communicator(), loc_smat_v, dest_mat_v, root);

        if (domain_->client_communicator().rank() != root) {
            return;
        }

        for (int i = 0; i < dest_mat_u.size();i++) {
            //accumulate vectors from other processors
            map_add_map(dest_mat_u[i], row_u);
        }

        for (int i = 0; i < dest_mat_v.size();i++) {
            //accumulate vectors from other processors
            map_add_map(dest_mat_v[i], row_v);
        }
    }

    void construction_BCMat_FMM() {
        //Not in use
        boost::mpi::communicator world;
        boost::mpi::communicator client_comm = domain_->client_communicator();
        world.barrier();

        /*if (world.rank() == 0) {
            return;
        }*/

        if (world.rank() == 1) {
            std::cout << "Constructing BC matrix using FMM" << std::endl;
        }
       
        if (max_local_idx == 0 && world.rank() != 0) {
            std::cout << "idx not initialized, please call Assigning_idx()" << std::endl;
        }

        const auto dx_base = domain_->dx_base();
        int base_level = domain_->tree()->base_level();

        boundary_u.clean();

        boundary_u.resizing_row(max_local_idx+1);

        //int counter = 0; //count the number of bc pts in this processor

        //compute the octants needed at each processor
        std::vector<std::set<int>> gid_to_recv;
        gid_to_recv.resize(world.size());
        std::vector<std::set<int>> gid_to_send;
        gid_to_send.resize(world.size());
        for (auto it = domain_->begin(base_level);
             it != domain_->end(base_level); ++it)
        {
            if (world.rank() == 0) continue;
            if (!it->locally_owned() || !it->has_data()) continue;
            if (!it->is_leaf()) continue;
            if (!it->is_correction()) continue;

            octant_t* prt_oct = it.ptr();

            for (int l = base_level - 1; l >= 0; l--)
            {
                //if (world.rank() == 1) std::cout << "rank " << world.rank() << " at level " << l << std::endl;
                octant_t* chd_oct =
                    prt_oct; //chd_oct is the child of the octant at the current level containing the targete octant

                prt_oct =
                    prt_oct
                        ->parent(); //prt_oct is the parent octant on the level l

                if (!prt_oct) break;

                //if (world.rank() == 1) std::cout << "rank " << world.rank() << " at level " << l << " updated parents " << std::endl;

                for (std::size_t i = 0; i < prt_oct->num_neighbors(); ++i)
                {
                    //if (world.rank() == 1) std::cout << "rank " << world.rank() << " at neighbor " << i << std::endl;
                    auto it2 = prt_oct->neighbor(i);
                    if (!it2) continue;
                    for (int j = 0; j < it2->num_children(); ++j)
                    {
                        //if (world.rank() == 1) std::cout << "rank " << world.rank() << " at child " << j << std::endl;
                        //only apply to non-neighboring ones
                        auto child = it2->child(j);
                        if (!child)
                            continue;
                        if (!child->is_leaf() && l == (base_level - 1))
                            continue; //on the base level, the octant needs to be leaf octant
                        bool is_neighbor = false;

                        if (l != (base_level - 1))
                        {
                            //if (world.rank() == 1) std::cout << "rank " << world.rank() << " checking neighbors" << std::endl;
                            //if  child is on the base level, do not exclude neighbors
                            for (std::size_t k = 0;
                                 k < chd_oct->num_neighbors(); ++k)
                            {
                                auto neighbor_tmp = chd_oct->neighbor(k);
                                if (!neighbor_tmp) continue;
                                if (child->global_id() == neighbor_tmp->global_id())
                                {
                                    is_neighbor = true;
                                    break;
                                }
                            }
                        }
                        if (!is_neighbor) {
                            //if (world.rank() == 1) std::cout << "rank " << world.rank() << " registering new octant gids" << std::endl;
                            int gid_loc = child->global_id();
                            int child_rank = child->rank();
                            if (child_rank <= 0) continue;
                            //if (world.rank() == 1) std::cout << "rank " << world.rank() << " child rank " << child_rank << std::endl;
                            gid_to_recv[child_rank].insert(gid_loc);
                        }
                    }
                }
            }
        }

        
        std::cout << "rank " << world.rank() << " tasks to gather" << std::endl;

        int rank = world.rank();

        for (int i = 1; i < world.size();i++) {
            boost::mpi::gather(world, gid_to_recv[i], gid_to_send, i);
        }

        int n_send = 0;

        for (int i =1; i < world.size(); i++) {
            if (i == world.rank()) continue;
            n_send += gid_to_send[i].size();
        }

        //std::cout << "rank " << world.rank() << " tasks gathered" << std::endl;

        if (world.rank() == 0) return;

        std::cout << "rank " << world.rank() << " tasks gathered with size " << n_send << std::endl;

        std::map<int, std::vector<int>> idx_maps_nonloc;
        std::map<int, coordinate_type> base_maps_nonloc;

        
        for (int l = base_level; l >=0 ; l--) {
            //iterate through levels to send idx vectors to influenced processors
            for (auto it = domain_->begin(l); it != domain_->end(l); ++it) {
                if (it->locally_owned() && it->has_data()) {
                    if (!it->is_leaf() && l == base_level) continue;

                    int N = it->data().descriptor().extent()[0];

                    int idx_size = (N+2) * (N+2);

                    std::vector<int> tmp_idx_vec;

                    tmp_idx_vec.resize(idx_size);

                    auto child_idx_vec =
                        it->data_r(idx_w_g_type::tag(), 0).linalg_data();

                    for (int idx_i = 0; idx_i < (N + 2); idx_i++)
                    {
                        for (int idx_j = 0; idx_j < (N + 2); idx_j++)
                        {
                            int loc_idx = idx_i * (N + 2) + idx_j;
                            tmp_idx_vec[loc_idx] =
                                child_idx_vec.at(idx_i, idx_j);
                        }
                    }

                    for (int rk = 1; rk < world.size(); rk++) {
                        if (rk == world.rank()) continue;
                        int gid_cur = it->global_id();
                        auto it2 = gid_to_send[rk].find(gid_cur);
                        if (it2 == gid_to_send[rk].end()) continue;
                        coordinate_type loc_base = it->data_r(idx_w_g_type::tag(), 0).real_block().base();
                        world.isend(rk, 2*gid_cur, tmp_idx_vec);
                        world.isend(rk, 2*gid_cur+1, loc_base);
                    }
                }
            }
        }

        std::cout << "rank " << world.rank() << " tasks sent" << std::endl;

        int counter = 0;

        int N_m = domain_->block_extent()[0]+lBuffer+rBuffer;
        int idx_size = N_m*N_m;

        for (int rk = 1; rk < world.size(); rk++) {
            if (rk == world.rank()) continue;
            std::set<int> oct_to_recv = gid_to_recv[rk];
            for (auto it = oct_to_recv.begin(); it != oct_to_recv.end(); ++it) {
                std::vector<int> idx_vec_tmp;
                idx_vec_tmp.resize(idx_size);
                coordinate_type tmp;
                int cur_gid = *it;
                boost::mpi::request reqs[2];
                reqs[0] = world.irecv(rk, 2*cur_gid, idx_vec_tmp);
                reqs[1] = world.irecv(rk, 2*cur_gid+1, tmp);
                boost::mpi::wait_all(reqs, reqs + 2);
                //counter++;
                idx_maps_nonloc[cur_gid] = idx_vec_tmp;
                base_maps_nonloc[cur_gid] = tmp;
            }
        }
        //boost::mpi::wait_all(reqs_vec.begin(), reqs_vec.end());

        std::cout << "rank " << world.rank() << " recv all idx vecs" << std::endl;

        for (auto it = domain_->begin(base_level);
             it != domain_->end(base_level); ++it)
        {
            

            if (!it->locally_owned() || !it->has_data()) continue;
            if (!it->is_leaf()) continue;
            if (!it->is_correction()) continue;

            int gid_cur = it->global_id();

            std::cout << "Rank " << world.rank() << " computing BC terms gid " << gid_cur << std::endl;

            auto target_ptr = it.ptr();

            for (auto& n : it->data())
            {
                int cur_idx_0 = n(idx_u_type::tag(), 0);
                int cur_idx_1 = n(idx_u_type::tag(), 1);

                if (cur_idx_0 > 0)
                {
                    //int cur_idx_0 = n(idx_u_type::tag(), 0);
                    //int cur_idx_1 = n(idx_u_type::tag(), 1);
                    int glob_idx_0 = n(idx_u_g_type::tag(), 0);
                    int glob_idx_1 = n(idx_u_g_type::tag(), 1);

                    //function that inquires all the idx from other processors
                    const auto& coord_cur = n.level_coordinate();

                    std::map<int, float_type> row_u;
                    std::map<int, float_type> row_v;

                    //const auto& tree_coord = it->tree_coordinate();
                    //int oct_idx = it->idx();
                    //key_type key_tar = it->key();
                    //int gid_cur = it->global_id();

                    
                    get_BC_idx_from_client_FMM(target_ptr, coord_cur, idx_maps_nonloc, base_maps_nonloc,
                        row_u, row_v);

                    for (const auto& [key, val] : row_u)
                    {
                        boundary_u.add_element(cur_idx_0, key, val);
                    }
                    for (const auto& [key, val] : row_v)
                    {
                        boundary_u.add_element(cur_idx_1, key, val);
                    }

                    //add the negative of identity here
                    boundary_u.add_element(cur_idx_0, glob_idx_0, -1.0);
                    boundary_u.add_element(cur_idx_1, glob_idx_1, -1.0);
                }
            }
        }

        domain_->client_communicator().barrier();

        std::cout << "rank " << world.rank() << " constructed matrix" << std::endl;
    }

    void get_BC_idx_from_client_FMM(octant_t* target_octant, coordinate_type c,
        const std::map<int, std::vector<int>>& idx_maps_nonloc,
        const std::map<int, coordinate_type>&  base_maps_nonloc,
        std::map<int, float_type>& row_u, std::map<int, float_type>& row_v)
    {
        //Not in use
        //Send and recv
        //send coordinates to other processes

        //clear the destination matrix rows
        boost::mpi::communicator world;
        row_u.clear();
        row_v.clear();

        coordinate_type base = domain_->bounding_box().base();
        coordinate_type c_loc = c - base;
        //std::cout << root << " " <<domain_->client_communicator().rank() << " c is " << c_loc[0] << " " << c_loc[1] << std::endl;
        //std::map<int, float_type> loc_smat_u; //local resulting matrix to get BC
        //std::map<int, float_type> loc_smat_v; //local resulting matrix to get BC

        int        base_level = domain_->tree()->base_level();
        const auto dx_base = domain_->dx_base();

        /*if (domain_->is_client())
        {
            auto client = domain_->decomposition().client();

            //client->template buffer_exchange<idx_u_type>(base_level);
            client->template buffer_exchange<idx_u_g_type>(base_level);

            //client->template buffer_exchange<idx_p_type>(base_level);
            //client->template buffer_exchange<idx_p_g_type>(base_level);
        }*/

        octant_t* prt_oct =
            target_octant; //store the pointer to the parent octant

        for (int l = base_level - 1; l >= 0; l--)
        {
            //std::cout << "rank " << world.rank() << " at level " << l << std::endl;
            octant_t* chd_oct =
                prt_oct; //chd_oct is the child of the octant at the current level containing the targete octant

            prt_oct =
                prt_oct->parent(); //prt_oct is the parent octant on the level l

            if (!prt_oct) return;

            //std::cout << "rank " << world.rank() << " at level " << l << " updated parents " << std::endl;

            for (std::size_t i = 0; i < prt_oct->num_neighbors(); ++i)
            {
                //std::cout << "rank " << world.rank() << " at neighbor " << i << std::endl;
                auto it2 = prt_oct->neighbor(i);
                if (!it2) continue;
                for (int j = 0; j < it2->num_children(); j++)
                {
                    //std::cout << "rank " << world.rank() << " at child " << j << std::endl;
                    //only apply to non-neighboring ones
                    auto child = it2->child(j);
                    if (!child) continue;
                    if (!child->is_leaf() && l == (base_level - 1))
                        continue; //on the base level, the octant needs to be leaf octant
                    bool is_neighbor = false;

                    if (l != (base_level - 1))
                    {
                        //if  child is on the base level, do not exclude neighbors
                        for (std::size_t k = 0; k < chd_oct->num_neighbors();
                             ++k)
                        {
                            auto neighbor_tmp = chd_oct->neighbor(k);
                            if (!neighbor_tmp) continue;
                            if (child->global_id() == neighbor_tmp->global_id())
                            {
                                is_neighbor = true;
                                break;
                            }
                        }
                    }
                    //std::cout << "rank " << world.rank() << " after at child " << j << std::endl;
                    if (!is_neighbor && child->locally_owned() && child->has_data())
                    {
                        //not neighbor, keep constructing matrix for this octant

                        if (l != (base_level - 1))
                        {
                            int N =
                                domain_->block_extent()[0] + lBuffer + rBuffer;
                            
                            auto& lin_data_g =
                                child->data_r(idx_w_g_type::tag(), 0)
                                    .linalg_data();
                            for (int sub_i = 0; sub_i < N; sub_i++)
                            {
                                for (int sub_j = 0; sub_j < N; sub_j++)
                                {
                                    auto real_base =
                                        child->data_r(idx_w_g_type::tag())
                                            .real_block()
                                            .base() -
                                        base;

                                    int x_c_tmp =
                                        (real_base[0] + sub_i) *
                                        std::pow(2, base_level - (l + 1));
                                    int y_c_tmp =
                                        (real_base[1] + sub_j) *
                                        std::pow(2, base_level - (l + 1));

                                    int x_c = c_loc[0] - x_c_tmp;
                                    if (x_c != (c_loc[0] - x_c_tmp))
                                    {
                                        std::cout << "Coordinate type not INT"
                                                  << std::endl;
                                    }

                                    int y_c = c_loc[1] - y_c_tmp;

                                    float_type lgf_val_00 =
                                        lgf_lap_.derived().get(
                                            coordinate_type({x_c, y_c}));
                                    float_type lgf_val_01 =
                                        lgf_lap_.derived().get(
                                            coordinate_type({x_c, (y_c + 1)}));
                                    float_type lgf_val_10 =
                                        lgf_lap_.derived().get(
                                            coordinate_type({(x_c + 1), y_c}));

                                    float_type u_weight =
                                        (lgf_val_01 - lgf_val_00);
                                    float_type v_weight =
                                        (lgf_val_00 - lgf_val_10);

                                    int w_idx = lin_data_g.at(sub_i, sub_j);

                                    this->map_add_element(w_idx,
                                        -u_weight * dx_base, row_u);

                                    this->map_add_element(w_idx,
                                        -v_weight * dx_base, row_v);
                                }
                            }
                        }

                        else
                        {
                            for (auto& n : child->data())
                            {
                                const auto& coord_loc_level =
                                    n.level_coordinate() - base;
                                const auto& coord_loc =
                                    coord_loc_level *
                                    std::pow(2, base_level - (l + 1));

                                int x_c = c_loc[0] - coord_loc[0];
                                if (x_c != (c_loc[0] - coord_loc[0]))
                                {
                                    std::cout << "Coordinate type not INT"
                                              << std::endl;
                                }

                                int y_c = c_loc[1] - coord_loc[1];

                                float_type lgf_val_00 = lgf_lap_.derived().get(
                                    coordinate_type({x_c, y_c}));
                                float_type lgf_val_01 = lgf_lap_.derived().get(
                                    coordinate_type({x_c, (y_c + 1)}));
                                float_type lgf_val_10 = lgf_lap_.derived().get(
                                    coordinate_type({(x_c + 1), y_c}));

                                float_type u_weight = (lgf_val_01 - lgf_val_00);
                                float_type v_weight = (lgf_val_00 - lgf_val_10);

                                int w_idx = n(idx_w_g_type::tag(), 0);

                                this->map_add_element(w_idx,
                                    -u_weight * dx_base, row_u);

                                this->map_add_element(w_idx,
                                    -v_weight * dx_base, row_v);
                            }
                        }
                    }

                    else if (!is_neighbor)
                    {
                        //std::cout << "rank " << world.rank() << " not child but on other proc " << std::endl;
                        //not neighbor, keep constructing matrix for this octant

                        int  gid_loc = child->global_id();
                        auto it3 = idx_maps_nonloc.find(gid_loc);
                        if (it3 == idx_maps_nonloc.end()) {
                            //std::cout << "Rank " << world.rank() << " cannot find octant with gid " << gid_loc << std::endl;
                            continue;
                        }
                        auto it4 = base_maps_nonloc.find(gid_loc);
                        if (it4 == base_maps_nonloc.end()) {
                            //std::cout << "Rank " << world.rank() << " cannot find octant with gid " << gid_loc << std::endl;
                            continue;
                        }
                        auto base_loc = it4->second;
                        auto idx_loc = it3->second;

                        if (l != (base_level - 1))
                        {
                            int N =
                                domain_->block_extent()[0] + lBuffer + rBuffer;
                            
                            
                            for (int sub_i = 0; sub_i < N; sub_i++)
                            {
                                for (int sub_j = 0; sub_j < N; sub_j++)
                                {
                                    auto real_base = base_loc - base;

                                    int x_c_tmp =
                                        (real_base[0] + sub_i) *
                                        std::pow(2, base_level - (l + 1));
                                    int y_c_tmp =
                                        (real_base[1] + sub_j) *
                                        std::pow(2, base_level - (l + 1));

                                    int x_c = c_loc[0] - x_c_tmp;
                                    if (x_c != (c_loc[0] - x_c_tmp))
                                    {
                                        std::cout << "Coordinate type not INT"
                                                  << std::endl;
                                    }

                                    int y_c = c_loc[1] - y_c_tmp;

                                    float_type lgf_val_00 =
                                        lgf_lap_.derived().get(
                                            coordinate_type({x_c, y_c}));
                                    float_type lgf_val_01 =
                                        lgf_lap_.derived().get(
                                            coordinate_type({x_c, (y_c + 1)}));
                                    float_type lgf_val_10 =
                                        lgf_lap_.derived().get(
                                            coordinate_type({(x_c + 1), y_c}));

                                    float_type u_weight =
                                        (lgf_val_01 - lgf_val_00);
                                    float_type v_weight =
                                        (lgf_val_00 - lgf_val_10);

                                    int loc_idx = sub_i * N + sub_j;
                                    int w_idx = idx_loc[loc_idx];

                                    this->map_add_element(w_idx,
                                        -u_weight * dx_base, row_u);

                                    this->map_add_element(w_idx,
                                        -v_weight * dx_base, row_v);
                                }
                            }
                        }

                        else
                        {
                            int N =
                                domain_->block_extent()[0] + lBuffer + rBuffer;
                            
                            
                            for (int sub_i = 1; sub_i < N-1; sub_i++)
                            {
                                for (int sub_j = 1; sub_j < N-1; sub_j++)
                                {
                                    auto real_base = base_loc - base;

                                    int x_c_tmp =
                                        (real_base[0] + sub_i) *
                                        std::pow(2, base_level - (l + 1));
                                    int y_c_tmp =
                                        (real_base[1] + sub_j) *
                                        std::pow(2, base_level - (l + 1));

                                    int x_c = c_loc[0] - x_c_tmp;
                                    if (x_c != (c_loc[0] - x_c_tmp))
                                    {
                                        std::cout << "Coordinate type not INT"
                                                  << std::endl;
                                    }

                                    int y_c = c_loc[1] - y_c_tmp;

                                    float_type lgf_val_00 =
                                        lgf_lap_.derived().get(
                                            coordinate_type({x_c, y_c}));
                                    float_type lgf_val_01 =
                                        lgf_lap_.derived().get(
                                            coordinate_type({x_c, (y_c + 1)}));
                                    float_type lgf_val_10 =
                                        lgf_lap_.derived().get(
                                            coordinate_type({(x_c + 1), y_c}));

                                    float_type u_weight =
                                        (lgf_val_01 - lgf_val_00);
                                    float_type v_weight =
                                        (lgf_val_00 - lgf_val_10);

                                    int loc_idx = sub_i * N + sub_j;
                                    int w_idx = idx_loc[loc_idx];

                                    this->map_add_element(w_idx,
                                        -u_weight * dx_base, row_u);

                                    this->map_add_element(w_idx,
                                        -v_weight * dx_base, row_v);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /*void get_BC_idx_from_client_FMM(coordinate_type c, int gid, int root, std::map<int, float_type>& row_u, std::map<int, float_type>& row_v) {
        //Send and recv 
        //send coordinates to other processes

        //clear the destination matrix rows
        row_u.clear();
        row_v.clear();

        
        coordinate_type base = domain_->bounding_box().base();
        coordinate_type c_loc = c - base;
        int gid_loc = gid;
        boost::mpi::broadcast(domain_->client_communicator(), c_loc, root);
        boost::mpi::broadcast(domain_->client_communicator(), gid_loc, root);

        //std::cout << root << " " <<domain_->client_communicator().rank() << " c is " << c_loc[0] << " " << c_loc[1] << std::endl;
        std::map<int, float_type> loc_smat_u; //local resulting matrix to get BC
        std::map<int, float_type> loc_smat_v; //local resulting matrix to get BC

        int base_level = domain_->tree()->base_level();
        const auto dx_base = domain_->dx_base();

        
        if (domain_->is_client())
        {
            auto client = domain_->decomposition().client();

            //client->template buffer_exchange<idx_u_type>(base_level);
            client->template buffer_exchange<idx_u_g_type>(base_level);

            //client->template buffer_exchange<idx_p_type>(base_level);
            //client->template buffer_exchange<idx_p_g_type>(base_level);
        }

        auto tree_ptr = domain_->tree()->root();

        auto cur_ptr = domain_->tree()->root();

        octant_t* target_octant = nullptr;

        for (int l = 0; l <= base_level; l++) {
            for (int child_idx = 0; child_idx < cur_ptr->num_children(); child_idx++) {
                auto child = cur_ptr->child(child_idx);
                if ()
            }
        }

        //octant_t* target_octant = tree_ptr->find_octant(key_tar_loc);
        //find the target octant locally

        boost::mpi::communicator world;

        if (!target_octant) {
            std::cout << "rank " << world.rank() << " cannot find octant target" << std::endl;
            throw std::runtime_error("Did not found the target octant");
        }

        octant_t* prt_oct = target_octant; //store the pointer to the parent octant

        for (int l = base_level - 1; l >= 0; l--)
        {
            //std::cout << "rank " << world.rank() << " at level " << l << std::endl;
            octant_t* chd_oct = prt_oct; //chd_oct is the child of the octant at the current level containing the targete octant
            
            prt_oct = prt_oct->parent(); //prt_oct is the parent octant on the level l

            if (!prt_oct) return;

            //std::cout << "rank " << world.rank() << " at level " << l << " updated parents " << std::endl;

            for (std::size_t i = 0; i < prt_oct->num_neighbors(); ++i) {
                //std::cout << "rank " << world.rank() << " at neighbor " << i << std::endl;
                auto it2 = prt_oct->neighbor(i);
                if (!it2) continue;
                for (int j = 0; j < it2->num_children();j++) {
                    //std::cout << "rank " << world.rank() << " at child " << j << std::endl;
                    //only apply to non-neighboring ones
                    auto child = it2->child(j);
                    if (!child || !child->locally_owned() || !child->has_data()) continue;
                    if (!child->is_leaf() && l == (base_level - 1)) continue; //on the base level, the octant needs to be leaf octant
                    bool is_neighbor = false;

                    if (l != (base_level - 1))
                    {
                        //if  child is on the base level, do not exclude neighbors
                        for (std::size_t k = 0; k < chd_oct->num_neighbors();
                             ++k)
                        {
                            auto neighbor_tmp = chd_oct->neighbor(k);
                            if (!neighbor_tmp) continue;
                            if (child->key() == neighbor_tmp->key())
                            {
                                is_neighbor = true;
                                break;
                            }
                        }
                    }
                    //std::cout << "rank " << world.rank() << " after at child " << j << std::endl;
                    if (!is_neighbor) {
                        //not neighbor, keep constructing matrix for this octant

                        if (l != (base_level - 1)) {
                            int N = domain_->block_extent()[0]+lBuffer+rBuffer;
                            auto& lin_data_g = child->data_r(idx_w_g_type::tag(), 0).linalg_data();
                            for (int sub_i = 0; sub_i < N; sub_i++) {
                                for (int sub_j = 0; sub_j < N; sub_j++) {
                                    auto real_base = child->data_r(idx_w_g_type::tag()).real_block().base()  - base;

                                    int x_c_tmp = (real_base[0] + sub_i) * std::pow(2, base_level - (l+1));
                                    int y_c_tmp = (real_base[1] + sub_j) * std::pow(2, base_level - (l+1));

                                    int x_c = c_loc[0] - x_c_tmp;
                                    if (x_c != (c_loc[0] - x_c_tmp))
                                    {
                                        std::cout << "Coordinate type not INT"
                                                  << std::endl;
                                    }

                                    int y_c = c_loc[1] -  y_c_tmp;

                                    float_type lgf_val_00 =
                                        lgf_lap_.derived().get(
                                            coordinate_type({x_c, y_c}));
                                    float_type lgf_val_01 =
                                        lgf_lap_.derived().get(
                                            coordinate_type({x_c, (y_c + 1)}));
                                    float_type lgf_val_10 =
                                        lgf_lap_.derived().get(
                                            coordinate_type({(x_c + 1), y_c}));

                                    float_type u_weight =
                                        (lgf_val_01 - lgf_val_00);
                                    float_type v_weight =
                                        (lgf_val_00 - lgf_val_10);

                                    int w_idx = lin_data_g.at(sub_i, sub_j);

                                    this->map_add_element(w_idx,
                                        -u_weight * dx_base, loc_smat_u);

                                    this->map_add_element(w_idx,
                                        -v_weight * dx_base, loc_smat_v);
                                }
                            }
                        }

                        else
                        {
                            for (auto& n : child->data())
                            {
                                const auto& coord_loc_level =
                                    n.level_coordinate() - base;
                                const auto& coord_loc =
                                    coord_loc_level *
                                    std::pow(2, base_level - (l + 1));

                                int x_c = c_loc[0] - coord_loc[0];
                                if (x_c != (c_loc[0] - coord_loc[0]))
                                {
                                    std::cout << "Coordinate type not INT"
                                              << std::endl;
                                }

                                int y_c = c_loc[1] - coord_loc[1];

                                float_type lgf_val_00 = lgf_lap_.derived().get(
                                    coordinate_type({x_c, y_c}));
                                float_type lgf_val_01 = lgf_lap_.derived().get(
                                    coordinate_type({x_c, (y_c + 1)}));
                                float_type lgf_val_10 = lgf_lap_.derived().get(
                                    coordinate_type({(x_c + 1), y_c}));

                                float_type u_weight = (lgf_val_01 - lgf_val_00);
                                float_type v_weight = (lgf_val_00 - lgf_val_10);

                                int w_idx = n(idx_w_g_type::tag(), 0);

                                this->map_add_element(w_idx,
                                    -u_weight * dx_base, loc_smat_u);

                                this->map_add_element(w_idx,
                                    -v_weight * dx_base, loc_smat_v);
                            }
                        }
                    }
                }
            }
        }

        domain_->client_communicator().barrier();
        std::vector<std::map<int, float_type>> dest_mat_u;
        std::vector<std::map<int, float_type>> dest_mat_v;
        boost::mpi::gather(domain_->client_communicator(), loc_smat_u, dest_mat_u, root);
        boost::mpi::gather(domain_->client_communicator(), loc_smat_v, dest_mat_v, root);

        if (domain_->client_communicator().rank() != root) {
            return;
        }

        for (int i = 0; i < dest_mat_u.size();i++) {
            //accumulate vectors from other processors
            map_add_map(dest_mat_u[i], row_u);
        }

        for (int i = 0; i < dest_mat_v.size();i++) {
            //accumulate vectors from other processors
            map_add_map(dest_mat_v[i], row_v);
        }
    }*/

    void construct_upward_intrp() {
        boost::mpi::communicator world;
        world.barrier();

        if (world.rank() == 0) {
            return;
        }

        if (world.rank() == 1) {
            std::cout << "Constructing upward interpolation matrix" << std::endl;
        }
       
        if (max_local_idx == 0 && world.rank() != 0) {
            std::cout << "idx not initialized, please call Assigning_idx()" << std::endl;
        }

        const auto dx_base = domain_->dx_base();
        int base_level = domain_->tree()->base_level();

        upward_intrp.resizing_row(max_local_idx+1);

        /*if (domain_->is_client())
        {
            auto client = domain_->decomposition().client();

            //client->template buffer_exchange<idx_u_type>(base_level);
            client->template buffer_exchange<idx_w_g_type>(base_level);
            
        }*/

        auto c_cntr_nli = psolver.c_cntr_nli();

        for (int l = base_level - 1; l >= 0; l--)
        {
            /*if (domain_->is_client())
            {
                auto client = domain_->decomposition().client();

                //client->template buffer_exchange<idx_u_type>(base_level);
                client->template buffer_exchange<idx_w_g_type>(l + 1);
            }*/
            for (auto it = domain_->begin(l);
                 it != domain_->end(l); ++it)
            {
                //if (!it->locally_owned() || !it->has_data()) continue;
                //if (!it->is_leaf()) continue;
                //if (!it->is_correction()) continue;
                const int num_child = it->num_children();
                //octant_base_t oct_base_tmp(coordinate_type({0,0}), base_level);
                //std::vector<octant_t> octant_vec(num_child, octant_t(oct_base_tmp));
                std::vector<std::vector<int>> idx_vec;
                idx_vec.resize(num_child);

                int N = it->data().descriptor().extent()[0];

                int idx_size = (N+2) * (N+2);
                /*for (int i = 0; i < num_child;i++) {
                    idx_vec[i].resize(idx_size);
                }*/
                if (it->locally_owned() && it->has_data()) {
                    
                    //octant located in this processor, need to receive data
                    //octant_vec.resize(num_child);
                    for (int i = 0; i < it->num_children(); i++) {
                        auto child = it->child(i);
                        if (child && child->locally_owned() && child->has_data()) {
                            idx_vec[i].resize(idx_size);

                            auto child_idx_vec = child->data_r(idx_w_g_type::tag(), 0).linalg_data();

                            for (int idx_i = 0; idx_i < (N + 2); idx_i++) {
                                for (int idx_j = 0; idx_j < (N + 2); idx_j++) {
                                    int loc_idx = idx_i * (N + 2) + idx_j;
                                    idx_vec[i][loc_idx] = child_idx_vec.at(idx_i, idx_j);
                                }
                            }


                            //octant_vec[i] = (*child);
                            continue;
                        }
                        else if (child) {
                            
                            int source_rank = child->rank();

                            int tag_val = child->global_id();

                            idx_vec[i].resize(idx_size);
                            //std::cout << "Receiving at rank " << world.rank() << " gid is " << tag_val << std::endl;

                            //world.recv(source_rank, tag_val, octant_vec[i]);
                            world.recv(source_rank, tag_val, idx_vec[i]);
                        }
                    }
                }
                else {
                    //search for children is on this processor
                    for (int i = 0; i < it->num_children(); i++)
                    {
                        auto child = it->child(i);
                        int parent_rank = it->rank();
                        if (!child || !child->locally_owned() || !child->has_data()) continue;

                        std::vector<int> tmp_idx_vec;

                        tmp_idx_vec.resize(idx_size);

                        auto child_idx_vec =
                            child->data_r(idx_w_g_type::tag(), 0).linalg_data();

                        for (int idx_i = 0; idx_i < (N + 2); idx_i++)
                        {
                            for (int idx_j = 0; idx_j < (N + 2);
                                 idx_j++)
                            {
                                int loc_idx = idx_i * (N + 2) + idx_j;
                                tmp_idx_vec[loc_idx] =
                                    child_idx_vec.at(idx_i, idx_j);
                            }
                        }

                        int tag_val = child->global_id();
                        //std::cout << "Sending at rank " << world.rank() << " gid is " << tag_val << std::endl;
                        world.send(parent_rank, tag_val, tmp_idx_vec);
                    }
                }

                //domain_->client_communicator().barrier();

                //std::cout << "rank " << world.rank() << " finished communication " << "level " << l << std::endl;

                if (it->locally_owned() && it->has_data()) {
                    int N = it->data().descriptor().extent()[0];
                    auto& lin_data = it->data_r(idx_w_type::tag(), 0).linalg_data();
                    auto& lin_data_g = it->data_r(idx_w_g_type::tag(), 0).linalg_data();
                    //start computing the matrix
                    for (int i = 0; i < N+2; i++) {
                        for (int j = 0; j < N+2; j++) {

                            //if (l <= 4) std::cout << "rank " << world.rank() << " level " << l << " i " << i << " j " << j << std::endl; 
                            //get current idx
                            int cur_idx = lin_data.at(i, j);
                            int cur_g_idx = lin_data_g.at(i, j);
                            if (cur_idx < 0) {
                                std::cout << "cur_idx < 0 in constructing upward interpolation matrix" << std::endl;
                            }
                            upward_intrp.add_element(cur_idx, cur_g_idx, -1.0);

                            //if (l <= 4) std::cout << "rank " << world.rank() << " level " << l << " i " << i << " j " << j << " cur idx " << cur_idx << " g idx " << cur_g_idx << std::endl; 

                            for (int child_idx = 0; child_idx < it->num_children(); child_idx++) {

                                //if (l <= 4) std::cout << "rank " << world.rank() << " level " << l << " child idx " << child_idx << std::endl; 
                                auto child = it->child(child_idx);
                                //int  n = lin_data_g->shape()[0];
                                int  idx_x = (child_idx & (1 << 0)) >> 0;
                                int  idx_y = (child_idx & (1 << 1)) >> 1;

                                //idx_x += 1;
                                //idx_y += 1;
                                //this indexing is only for 2D

                                auto x_intrp_mat = c_cntr_nli_.antrp_mat_sub_[idx_x].data_;
                                auto y_intrp_mat = c_cntr_nli_.antrp_mat_sub_[idx_y].data_;

                                //auto child_cur = octant_vec[child_idx];

                                //if (!child || !child_cur.has_data()) continue;

                                if (idx_vec[child_idx].size() == 0) continue;

                                //auto& child_idx_vec = octant_vec[child_idx].data_r(idx_w_g_type::tag(), 0).linalg_data();
                                //auto child_idx_vec = child_cur.data_r(idx_w_g_type::tag(), 0).linalg_data();
                                if (l != (base_level - 1))
                                {
                                    for (int sub_i = 0; sub_i < (N + 2);
                                         sub_i++)
                                    {
                                        for (int sub_j = 0; sub_j < (N + 2);
                                             sub_j++)
                                        {
                                            /*int              child_idx_g =
                                                child_idx_vec.at(sub_i, sub_j);*/
                                            int loc_idx = sub_i * (N + 2) + sub_j;
                                            int child_idx_g = idx_vec[child_idx][loc_idx];
                                            float_type val_x =
                                                x_intrp_mat.at(i, sub_i);
                                            float_type val_y =
                                                y_intrp_mat.at(j, sub_j);
                                            float_type val_t = val_x * val_y;
                                            //if (std::abs(val_t) < 1e-12)
                                            //    continue;
                                            upward_intrp.add_element(cur_idx,
                                                child_idx_g, val_t);
                                        }
                                    }
                                }

                                else
                                {
                                    for (int sub_i = 1; sub_i < (N + 1);
                                         sub_i++)
                                    {
                                        for (int sub_j = 1; sub_j < (N + 1);
                                             sub_j++)
                                        {
                                            /*int              child_idx_g =
                                                child_idx_vec.at(sub_i, sub_j);*/

                                            int loc_idx = sub_i * (N + 2) + sub_j;
                                            int child_idx_g = idx_vec[child_idx][loc_idx];
                                            float_type val_x =
                                                x_intrp_mat.at(i, sub_i);
                                            float_type val_y =
                                                y_intrp_mat.at(j, sub_j);
                                            float_type val_t = val_x * val_y;
                                            //if (std::abs(val_t) < 1e-12)
                                                //continue;
                                            upward_intrp.add_element(cur_idx,
                                                child_idx_g, val_t);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                //std::cout << "rank " << world.rank() << " finished computing " << "level " << l << std::endl;
            }

            //domain_->client_communicator().barrier();
        }

        upward_intrp.scale_entries(Curl_factor);

        domain_->client_communicator().barrier();
    }

    void map_add_element(int m, float_type val, std::map<int, float_type>& target)
    {
        if (m < 0) return;
        auto it = target.find(m);
        if (it == target.end()) { target[m] = val; }
        else
        {
            it->second += val;
        }
    }

    void map_add_map(const std::map<int, float_type>& source, std::map<int, float_type>& target)
    {
        for (const auto& [key, val] : source)
        {
            map_add_element(key, val, target);
        }
    }

    /*template<class Source_face, class Source_cell, class Target_face, class Target_cell>
    void Solve_Jacobian(force_type& force_source, force_type& force_target) noexcept
    {

        if (domain_->is_server())
            return;
        
        //TODO: solver to solve the inverse of Jacobian with the vectors
        //clean<Target_face>();
        //clean<Target_cell>();
        real_coordinate_type tmp_coord(0.0);
        //std::fill(force_target.begin(), force_target.end(),
        //    tmp_coord);

        real_coordinate_type tmp(0.0);
        //force_type Ax(ib_->size(), tmp);
        //force_type r (ib_->size(), tmp);
        //force_type Ap(ib_->size(), tmp);
        clean<conj_Ap_cell_aux_type>();
        clean<conj_Ap_face_aux_type>();
        clean<conj_Ax_cell_aux_type>();
        clean<conj_Ax_face_aux_type>();
        clean<conj_r_cell_aux_type>();
        clean<conj_r_face_aux_type>();
        clean<conj_p_cell_aux_type>();
        clean<conj_p_face_aux_type>();
        force_type Ax_force(domain_->ib().size(), tmp);
        force_type Ap_force(domain_->ib().size(), tmp);
        force_type r_force(domain_->ib().size(), tmp);

        //Compute the actual source term
        //std::cout << "computing actual forcing term" << std::endl;
        Adjoint_Jacobian<Source_face,Source_cell, conj_r_face_aux_type, conj_r_cell_aux_type>(force_source, r_force);

        

        // Ax
        //this->template ET_H_S_E<Ftmp>(f, Ax, alpha);
        //std::cout << "computing Ax" << std::endl;
        this->template ATA<Target_face, Target_cell, conj_Ax_face_aux_type, conj_Ax_cell_aux_type>(force_target, Ax_force);
        //printvec(Ax, "Ax");

        //  res = uc - Ax
        for (int i=0; i<domain_->ib().size(); ++i)
        {
            if (domain_->ib().rank(i)!=comm_.rank())
                r_force[i]=0;
            else
                r_force[i]=r_force[i]-Ax_force[i];
        }

        //std::cout << "computing rsold" << std::endl;

        add<conj_Ax_face_aux_type, conj_r_face_aux_type>(-1.0);
        add<conj_Ax_cell_aux_type, conj_r_cell_aux_type>(-1.0);

        // p = res
        force_type p_force = r_force;

        copy<conj_r_cell_aux_type, conj_p_cell_aux_type>();
        copy<conj_r_face_aux_type, conj_p_face_aux_type>();

        // rold = r'* r;

        float_type f2 = this->template dotAll<Target_face, Target_cell, Target_face, Target_cell>(force_target, force_target);
        
        float_type rsold = this->template dotAll<conj_r_face_aux_type, conj_r_cell_aux_type, conj_r_face_aux_type, conj_r_cell_aux_type>(r_force, r_force);

        if (comm_.rank()==1)
            std::cout<< "residue square = "<< rsold/f2<<std::endl;;
        if (sqrt(rsold/f2)<cg_threshold_)
            return;

        for (int k=0; k<cg_max_itr_; k++)
        {
            // Ap = A(p)
            this->template ATA<conj_p_face_aux_type, conj_p_cell_aux_type, conj_Ap_face_aux_type, conj_Ap_cell_aux_type>(p_force, Ap_force);
            // alpha = rsold / p'*Ap
            float_type pAp = this->template dotAll<conj_p_face_aux_type, conj_p_cell_aux_type, conj_Ap_face_aux_type, conj_Ap_cell_aux_type>(p_force, Ap_force);
            if (pAp == 0.0)
            {
                return;
            }

            float_type alpha = rsold / pAp;
            // f = f + alpha * p;
            AddAll<conj_p_face_aux_type, conj_p_cell_aux_type, Target_face, Target_cell>(p_force, force_target, alpha);
            //add(f, p, 1.0, alpha);
            // r = r - alpha*Ap
            AddAll<conj_Ap_face_aux_type, conj_Ap_cell_aux_type, conj_r_face_aux_type, conj_r_cell_aux_type>(Ap_force, r_force, -alpha);
            //add(r, Ap, 1.0, -alpha);
            // rsnew = r' * r
            float_type rsnew = this->template dotAll<conj_r_face_aux_type, conj_r_cell_aux_type, conj_r_face_aux_type, conj_r_cell_aux_type>(r_force, r_force);
            float_type f2 = this->template dotAll<Target_face, Target_cell, Target_face, Target_cell>(force_target, force_target);
            //auto ModeError = dot_Mode(r,r);
            if (comm_.rank()==1)
                std::cout<< "residue square = "<< rsnew/f2<<std::endl;;
            if (sqrt(rsnew/f2)<cg_threshold_)
                break;

            // p = r + (rsnew / rsold) * p;
            //add(p, r, rsnew/rsold, 1.0);
            AddAll<conj_r_face_aux_type, conj_r_cell_aux_type, conj_p_face_aux_type, conj_p_cell_aux_type>(r_force, p_force, 1.0, rsnew/rsold);
            rsold = rsnew;
        }
    }*/


    /*template<class Source_face, class Source_cell, class Target_face, class Target_cell>
    void BCG_Stab(force_type& force_source, force_type& force_target) noexcept
    {

        if (domain_->is_server())
            return;
        
        //TODO: solver to solve the inverse of Jacobian with the vectors
        //clean<Target_face>();
        //clean<Target_cell>();
        real_coordinate_type tmp_coord(0.0);
        //std::fill(force_target.begin(), force_target.end(),
        //    tmp_coord);

        real_coordinate_type tmp(0.0);
        //force_type Ax(ib_->size(), tmp);
        //force_type r (ib_->size(), tmp);
        //force_type Ap(ib_->size(), tmp);
        clean<conj_Ap_cell_aux_type>();
        clean<conj_Ap_face_aux_type>();
        clean<conj_Ax_cell_aux_type>();
        clean<conj_Ax_face_aux_type>();
        clean<conj_r_cell_aux_type>();
        clean<conj_r_face_aux_type>();
        clean<conj_p_cell_aux_type>();
        clean<conj_p_face_aux_type>();
        clean<conj_rh_cell_aux_type>();
        clean<conj_rh_face_aux_type>();
        //clean<conj_As_cell_aux_type>();
        //clean<conj_As_face_aux_type>();
        clean<conj_s_cell_aux_type>();
        clean<conj_s_face_aux_type>();
        force_type Ax_force(domain_->ib().size(), tmp);
        force_type Ap_force(domain_->ib().size(), tmp);
        force_type r_force(domain_->ib().size(), tmp);
        force_type p_force(domain_->ib().size(), tmp);
        force_type rh_force(domain_->ib().size(), tmp);
        //force_type As_force(domain_->ib().size(), tmp);
        force_type s_force(domain_->ib().size(), tmp);

        //Compute the actual source term
        //std::cout << "computing actual forcing term" << std::endl;
        //Adjoint_Jacobian<Source_face,Source_cell, conj_r_face_aux_type, conj_r_cell_aux_type>(force_source, r_force);

        

        // Ax
        //this->template ET_H_S_E<Ftmp>(f, Ax, alpha);
        //std::cout << "computing Ax" << std::endl;
        this->template Jacobian<Target_face, Target_cell, conj_Ax_face_aux_type, conj_Ax_cell_aux_type>(force_target, Ax_force);
        //printvec(Ax, "Ax");

        //  res = uc - Ax
        for (int i=0; i<domain_->ib().size(); ++i)
        {
            if (domain_->ib().rank(i)!=comm_.rank())
                r_force[i]=0;
            else
                r_force[i]=force_source[i]-Ax_force[i];
        }

        //std::cout << "computing rsold" << std::endl;

        copy<Target_face, conj_r_face_aux_type>();
        copy<Target_cell, conj_r_cell_aux_type>();

        add<conj_Ax_face_aux_type, conj_r_face_aux_type>(-1.0);
        add<conj_Ax_cell_aux_type, conj_r_cell_aux_type>(-1.0);

        // p = res
        rh_force = r_force;

        copy<conj_r_cell_aux_type, conj_rh_cell_aux_type>();
        copy<conj_r_face_aux_type, conj_rh_face_aux_type>();

        // rold = r'* r;

        float_type f2 = this->template dotAll<Target_face, Target_cell, Target_face, Target_cell>(force_target, force_target);
        
        float_type rsold = this->template dotAll<conj_r_face_aux_type, conj_r_cell_aux_type, conj_r_face_aux_type, conj_r_cell_aux_type>(r_force, r_force);

        if (comm_.rank()==1)
            std::cout<< "residue square = "<< rsold/f2<<std::endl;;
        if (sqrt(rsold/f2)<cg_threshold_)
            return;
        
        float_type rho = 1;
        float_type rho_old = rho;
        float_type w = 1;
        float_type alpha = 1;

        for (int k=0; k<cg_max_itr_; k++)
        {

            rho = this->template dotAll<conj_rh_face_aux_type, conj_rh_cell_aux_type, conj_r_face_aux_type, conj_r_cell_aux_type>(rh_force, r_force);
            float_type beta = (rho/rho_old)*(alpha/w);
            AddAll<conj_Ap_face_aux_type, conj_Ap_cell_aux_type, conj_p_face_aux_type, conj_p_cell_aux_type>(Ap_force, p_force, -w);
            AddAll<conj_r_face_aux_type, conj_r_cell_aux_type, conj_p_face_aux_type, conj_p_cell_aux_type>(r_force, p_force, 1.0, beta);
            
            // Ap = A(p)
            this->template Jacobian<conj_p_face_aux_type, conj_p_cell_aux_type, conj_Ap_face_aux_type, conj_Ap_cell_aux_type>(p_force, Ap_force);
            float_type r_hat_v = this->template dotAll<conj_rh_face_aux_type, conj_rh_cell_aux_type, conj_Ap_face_aux_type, conj_Ap_cell_aux_type>(rh_force, Ap_force);
            alpha = rho/r_hat_v;

            //add(f, p, 1.0, alpha);
            AddAll<conj_p_face_aux_type, conj_p_cell_aux_type, Target_face, Target_cell>(p_force, force_target, alpha);

            //auto s = r;
            copy<conj_r_face_aux_type, conj_s_face_aux_type>();
            copy<conj_r_cell_aux_type, conj_s_cell_aux_type>();
            s_force = r_force;
            //add(s, v, 1.0, -alpha);
            AddAll<conj_Ap_face_aux_type, conj_Ap_cell_aux_type, conj_s_face_aux_type, conj_s_cell_aux_type>(Ap_force, s_force, -alpha);
            this->template Jacobian<conj_s_face_aux_type, conj_s_cell_aux_type, conj_Ax_face_aux_type, conj_Ax_cell_aux_type>(s_force, Ax_force);
            float_type As_s = this->template dotAll<conj_Ax_face_aux_type, conj_Ax_cell_aux_type, conj_s_face_aux_type, conj_s_cell_aux_type>(Ax_force, s_force);
            float_type As_As = this->template dotAll<conj_Ax_face_aux_type, conj_Ax_cell_aux_type, conj_Ax_face_aux_type, conj_Ax_cell_aux_type>(Ax_force, Ax_force);
            w = As_s/As_As;
            //add(f, s, 1.0, w);
            AddAll<conj_s_face_aux_type, conj_s_cell_aux_type, Target_face, Target_cell>(s_force, force_target, w);
            //r = s;
            copy<conj_s_cell_aux_type, conj_r_cell_aux_type>();
            copy<conj_s_face_aux_type, conj_r_face_aux_type>();
            r_force = s_force;
            //add(r, As, 1.0, -w);
            AddAll<conj_Ax_face_aux_type, conj_Ax_cell_aux_type, conj_r_face_aux_type, conj_r_cell_aux_type>(Ax_force, r_force, -w);

            float_type rsnew = this->template dotAll<conj_r_face_aux_type, conj_r_cell_aux_type, conj_r_face_aux_type, conj_r_cell_aux_type>(r_force, r_force);
            float_type f2 = this->template dotAll<Target_face, Target_cell, Target_face, Target_cell>(force_target, force_target);
            //float_type rs_mag = std::abs(rsnew);

            /*this->template ET_H_S_E<Ftmp>(f, Ax, alpha_);
            for (int i = 0; i < ib_->size(); ++i)
            {
                if (ib_->rank(i) != comm_.rank()) Error[i] = 0;
                else
                    Error[i] = uc[i] - Ax[i];
            }
            float_type errorMag = dot(Error, Error);*/
            /*if (comm_.rank()==1)
                std::cout<< "BCGstab residue square = "<< rsnew/f2/*<< " Error is " << errorMag*//* << std::endl;
            if (sqrt(rsnew/f2)<cg_threshold_)
                break;

            // p = r + (rsnew / rsold) * p;
            rho_old = rho;
        }
    }*/

    template<class Source_face, class Source_cell, class Target_face, class Target_cell>
    void ATA(force_type& force_source, force_type& force_target) noexcept
    {
        auto client = domain_->decomposition().client();
        auto forcing_tmp = force_source;
        real_coordinate_type tmp_coord(0.0);
        std::fill(forcing_tmp.begin(), forcing_tmp.end(),
            tmp_coord);
        std::fill(force_target.begin(), force_target.end(),
            tmp_coord);
        clean<r_i_tmp_type>();
        clean<cell_aux_tmp_type>();
        Jacobian<Source_face,Source_cell, r_i_tmp_type, cell_aux_tmp_type>(force_source, forcing_tmp);
        domain_->client_communicator().barrier();
        Adjoint_Jacobian<r_i_tmp_type, cell_aux_tmp_type, Target_face, Target_cell>(forcing_tmp, force_target);
        domain_->client_communicator().barrier();
    }

    template<class Source_face, class Source_cell, class Target_face, class Target_cell>
    void AddAll(force_type& force_source, force_type& force_target, float_type scale1=1.0, float_type scale2=1.0) noexcept
    {
        //Target = Source*scale1+Target*scale2
        addScale<Source_face, Target_face>(scale1, scale2);
        addScale<Source_cell, Target_cell>(scale1, scale2);
        for (int i=0; i<domain_->ib().size(); ++i)
        {
            if (domain_->ib().rank(i)!=comm_.rank())
                force_target[i]=0;
            else
                force_target[i]=force_target[i]*scale2+force_source[i]*scale1;
        }
    }

    template<class Face1, class Cell1, class Face2, class Cell2>
    float_type dotAll(force_type& force1, force_type& force2) noexcept 
    {
        float_type v_cell = dotField<Cell1, Cell2>();
        float_type v_face = dotField<Face1, Face2>();
        float_type v_force = dotVec(force1, force2);
        float_type vel = v_cell + v_face + v_force;
        return vel;
    }

    template<class Source_face, class Source_cell, class Target_face, class Target_cell>
    void Adjoint_Jacobian(force_type& force_source, force_type& force_target) noexcept
    {
        if (domain_->is_server())
            return;
        auto client = domain_->decomposition().client();

        clean<laplacian_face_type>();
        clean<u_i_bc_type>();
        clean<edge_aux_type>();

        //clean<r_i_T_type>(); //use r_i as the result of applying Jcobian in the first block
        //clean<cell_aux_T_type>(); //use cell aux_type to be the second block
        clean<Target_face>();
        clean<Target_cell>();
        real_coordinate_type tmp_coord(0.0);
        std::fill(force_target.begin(), force_target.end(),
            tmp_coord); //use forcing tmp to store the last block,
            //use forcing_old to store the forcing at previous Newton iteration

        curl<Source_face, edge_aux_type>();

        domain_->client_communicator().barrier();
        mDuration_type t_lgf(0);
        /*TIME_CODE(t_lgf,
            SINGLE_ARG(Vel_from_vort<edge_aux_type, u_i_bc_type>();));
        domain_->client_communicator().barrier();
        //pcout << "BCs solved in " << t_lgf.count() << std::endl;

        copy_base_level_BC<u_i_bc_type, Source_face>();

        domain_->client_communicator().barrier();*/

        laplacian<Source_face, laplacian_face_type>();

        nonlinear_jac_adjoint<u_type, Source_face, g_i_type>();

        add<g_i_type, Target_face>(-1);

        add<laplacian_face_type, Target_face>(1.0 / Re_);

        clean<face_aux_tmp_type>();

        //up_and_down<Source_cell>();
        gradient<Source_cell, face_aux_tmp_type>();
        //add<face_aux_tmp_type, Target_face>();

        lsolver.template smearing<face_aux_tmp_type>(force_source, false);

        add<face_aux_tmp_type, Target_face>(1.0);

        divergence<Source_face, Target_cell>(-1.0);

        lsolver.template projection<Source_face>(
            force_target); //need to change this vector in the bracket
    }

    template<class From, class To>
    void Upward_interpolation()
    {
        boost::mpi::communicator world;
        if (world.rank() != 0)
        {
            
            int base_level = domain_->tree()->base_level();
            for (int field_idx = 0; field_idx < From::nFields(); field_idx++)
            {
                for (int l = base_level; l >= 0; l--)
                {
                    for (auto it_s = domain_->begin(l); it_s != domain_->end(l);
                         ++it_s)
                    {

                        if (!it_s->has_data()) continue;
                        if (!it_s->data().is_allocated()) continue;

                        auto& lin_data = it_s->data_r(correction_tmp_type::tag(), 0).linalg_data();
                        std::fill(lin_data.begin(), lin_data.end(), 0.0);
                    }

                }
                for (auto it = domain_->begin(base_level); it != domain_->end(base_level);
                     ++it)
                {
                    if (it->locally_owned() && it->is_leaf() && !it->is_correction())
                    {
                        auto& lin_data_1 =
                            it->data_r(From::tag(), field_idx).linalg_data();
                        auto& lin_data_2 =
                            it->data_r(correction_tmp_type::tag(), field_idx)
                                .linalg_data();

                        xt::noalias(view(lin_data_2, xt::range(1, -1),
                            xt::range(1, -1))) = view(lin_data_1,
                            xt::range(1, -1), xt::range(1, -1));
                    }
                }
                for (int l = base_level - 1; l >= 0; l--)
                {
                    for (auto it_s = domain_->begin(l); it_s != domain_->end(l);
                         ++it_s)
                    {
                        if (!it_s->has_data() || !it_s->data().is_allocated())
                            continue;

                        c_cntr_nli_.template nli_intrp_T_node<correction_tmp_type, correction_tmp_type>(*it_s,
                            From::mesh_type(), field_idx,
                            field_idx, false, false);
                    }
                    domain_->decomposition()
                        .client()
                        ->template communicate_updownward_add<correction_tmp_type, correction_tmp_type>(l, true,
                            false, -1, field_idx, false);

                    for (auto it = domain_->begin(l);
                         it != domain_->end(l); ++it)
                    {
                        if (!it->locally_owned() && it->has_data() &&
                            it->data().is_allocated())
                        {
                            auto& cp2 = it->data_r(correction_tmp_type::tag(), 0).linalg_data();
                            cp2 *= 0.0;
                        }
                    }

                    for (auto it_s = domain_->begin(l); it_s != domain_->end(l);
                         ++it_s)
                    {
                        if (!it_s->locally_owned()) continue;

                        //auto& lin_data_1 = it->data_r(correction_tmp_type::tag(), 0).linalg_data();
                        //auto& lin_data_2 = it->data_r(To::tag(), field_idx).linalg_data();
                        it_s->data_r(To::tag(), field_idx)
                            .linalg()
                            .get()
                            ->cube_noalias_view() =
                            it_s->data_r(correction_tmp_type::tag(), 0).linalg_data();
                    }

                }
            }
        }
    }

    template<class Source_face, class Source_cell, class Target_face, class Target_cell>
    void Jacobian(force_type& force_source, force_type& force_target) noexcept
    {
        if (domain_->is_server())
            return;
        auto client = domain_->decomposition().client();

        clean<laplacian_face_type>();
        clean<u_i_bc_type>();
        clean<edge_aux_type>();

        //clean<r_i_T_type>(); //use r_i as the result of applying Jcobian in the first block
        //clean<cell_aux_T_type>(); //use cell aux_type to be the second block
        clean<Target_face>();
        clean<Target_cell>();
        real_coordinate_type tmp_coord(0.0);
        std::fill(force_target.begin(), force_target.end(),
            tmp_coord); //use forcing tmp to store the last block,
            //use forcing_old to store the forcing at previous Newton iteration


        curl<Source_face, edge_aux_type>();

        domain_->client_communicator().barrier();
        /*mDuration_type t_lgf(0);
        TIME_CODE(t_lgf,
            SINGLE_ARG(Vel_from_vort<edge_aux_type, u_i_bc_type>();));
        domain_->client_communicator().barrier();
        //pcout << "BCs solved in " << t_lgf.count() << std::endl;

        copy_base_level_BC<u_i_bc_type, Source_face>();

        //pcout << "Copied BC "  << std::endl;*/

        domain_->client_communicator().barrier();

        laplacian<Source_face, laplacian_face_type>();

        //pcout << "Computed Laplacian " << std::endl;

        nonlinear_jac<u_type, Source_face, g_i_type>();

        //pcout << "Computed Nonlinear Jac " << std::endl;

        add<g_i_type, Target_face>(-1);

        add<laplacian_face_type, Target_face>(1.0 / Re_);

        clean<face_aux_tmp_type>();
        gradient<Source_cell, face_aux_tmp_type>();

        //pcout << "Computed Gradient" << std::endl;
        //add<face_aux_tmp_type, Target_face>();

        lsolver.template smearing<face_aux_tmp_type>(force_source, false);

        //pcout << "Computed Smearing" << std::endl;

        add<face_aux_tmp_type, Target_face>(1.0);

        divergence<Source_face, Target_cell>(-1.0);

        //pcout << "Computed Divergence" << std::endl;

        lsolver.template projection<Source_face>(
            force_target); //need to change this vector in the bracket
    }

    template<class Source_face, class Source_cell, class Target_face, class Target_cell>
    void NewtonRHS(force_type& force_source, force_type& force_target) noexcept
    {
        if (domain_->is_server())
            return;
        auto client = domain_->decomposition().client();

        clean<laplacian_face_type>();
        clean<u_i_bc_type>();
        clean<edge_aux_type>();

        //clean<r_i_T_type>(); //use r_i as the result of applying Jcobian in the first block
        //clean<cell_aux_T_type>(); //use cell aux_type to be the second block
        clean<Target_face>();
        clean<Target_cell>();
        real_coordinate_type tmp_coord(0.0);
        std::fill(force_target.begin(), force_target.end(),
            tmp_coord); //use forcing tmp to store the last block,
            //use forcing_old to store the forcing at previous Newton iteration


        curl<Source_face, edge_aux_type>();

        domain_->client_communicator().barrier();
        mDuration_type t_lgf(0);
        TIME_CODE(t_lgf,
            SINGLE_ARG(Vel_from_vort<edge_aux_type, u_i_bc_type>();));
        domain_->client_communicator().barrier();
        //pcout << "BCs solved in " << t_lgf.count() << std::endl;

        copy_base_level_BC<u_i_bc_type, Source_face>();

        //pcout << "Copied BC "  << std::endl;

        domain_->client_communicator().barrier();

        laplacian<Source_face, laplacian_face_type>();

        //pcout << "Computed Laplacian " << std::endl;

        nonlinear<Source_face, g_i_type>();

        //pcout << "Computed Nonlinear Jac " << std::endl;

        add<g_i_type, Target_face>(-1);

        add<laplacian_face_type, Target_face>(1.0 / Re_);

        clean<face_aux_tmp_type>();
        gradient<Source_cell, face_aux_tmp_type>();

        //pcout << "Computed Gradient" << std::endl;
        //add<face_aux_tmp_type, Target_face>();

        lsolver.template smearing<face_aux_tmp_type>(force_source, false);

        //pcout << "Computed Smearing" << std::endl;

        add<face_aux_tmp_type, Target_face>(1.0);

        divergence<Source_face, Target_cell>(-1.0);

        //pcout << "Computed Divergence" << std::endl;

        lsolver.template projection<Source_face>(
            force_target); //need to change this vector in the bracket
    }

    template<class Source1, class Source2, class Target>
    void nonlinear_Jac_access() {
        //boost::mpi::communicator world;
        //std::cout << "rank " << (world.rank()) << " starting nonlinear jacobian" << std::endl;
        this->nonlinear_jac<Source1, Source2, Target>();
    }

    template<class Source1, class Source2, class Target>
    void nonlinear_Jac_T_access() {
        //boost::mpi::communicator world;
        //std::cout << "rank " << (world.rank()) << " starting nonlinear jacobian" << std::endl;
        this->nonlinear_jac_adjoint<Source1, Source2, Target>();
    }

    template<class Source, class Target>
    void Curl_access() {


        auto client = domain_->decomposition().client();

        //up_and_down<Velocity_in>();
        clean<Source>(true);
        this->up<Source>(false);
        clean<Target>();
        //clean<stream_f_type>();

        auto dx_base = domain_->dx_base();

        for (int l = domain_->tree()->base_level();
             l < domain_->tree()->depth(); ++l)
        {
            client->template buffer_exchange<Source>(l);

            for (auto it = domain_->begin(l); it != domain_->end(l); ++it)
            {
                if (!it->locally_owned() || !it->has_data()) continue;
                if (it->is_correction()) continue;
                //if(!it->is_leaf()) continue;

                const auto dx_level =
                    dx_base / math::pow2(it->refinement_level());
                //if (it->is_leaf())
                domain::Operator::curl<Source, Target>(it->data(),
                    dx_level);
            }
        }

        //clean<Velocity_out>();
        clean_leaf_correction_boundary<Target>(
            domain_->tree()->base_level(), true, 1);
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


  
    float_type coeff_a(int i, int j) const noexcept
    {
        return a_[i * (i - 1) / 2 + j - 1];
    }

    template<class Source, class Target>
    void Vel_from_vort()
    {
        auto client = domain_->decomposition().client();

        auto dx_base = domain_->dx_base();

        //up_and_down<Velocity_in>();
        clean<Target>();
        up_and_down<Source>();

        //clean<Velocity_out>();
        clean_leaf_correction_boundary<Source>(domain_->tree()->base_level(),
            true, 2);
        //clean_leaf_correction_boundary<edge_aux_type>(l, false,2+stage_idx_);
        psolver.template apply_lgf<Source, stream_f_type>(MASK_TYPE::STREAM);
        //psolver.template apply_lgf<Source, stream_f_type>(MASK_TYPE::Laplacian_BC);

        int l_max = domain_->tree()->depth();
        for (int l = domain_->tree()->base_level(); l < l_max; ++l)
        {
            for (auto it = domain_->begin(l); it != domain_->end(l); ++it)
            {
                if (!it->locally_owned() || !it->has_data()) continue;
                //if(!it->is_correction() && refresh_correction_only) continue;

                const auto dx_level =
                    dx_base / math::pow2(it->refinement_level());
                domain::Operator::curl_transpose<stream_f_type, Target>(
                    it->data(), dx_level, -1.0);
            }
        }

        this->down_to_correction<Target>();
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
                domain::Operator::curl<Velocity_in, edge_aux_type>(it->data(),
                    dx_level);
            }
        }

        //clean<Velocity_out>();
        clean_leaf_correction_boundary<edge_aux_type>(
            domain_->tree()->base_level(), true, 2);
        //clean_leaf_correction_boundary<edge_aux_type>(l, false,2+stage_idx_);
        psolver.template apply_lgf<edge_aux_type, stream_f_type>(
            MASK_TYPE::STREAM);

        int l_max = refresh_correction_only ? domain_->tree()->base_level() + 1
                                            : domain_->tree()->depth();
        for (int l = domain_->tree()->base_level(); l < l_max; ++l)
        {
            client->template buffer_exchange<stream_f_type>(l);
            for (auto it = domain_->begin(l); it != domain_->end(l); ++it)
            {
                if (!it->locally_owned() || !it->has_data()) continue;
                //if(!it->is_correction() && refresh_correction_only) continue;

                const auto dx_level =
                    dx_base / math::pow2(it->refinement_level());
                domain::Operator::curl_transpose<stream_f_type, Velocity_out>(
                    it->data(), dx_level, -1.0);
            }
        }

        this->down_to_correction<Velocity_out>();
    }
    private:

    //TODO maybe to be put directly into operators:
    template<class Source, class Target>
    void nonlinear(float_type _scale = 1.0) noexcept
    {
        clean<edge_aux_type>();
        clean<Target>();
        clean<face_aux_type>();

        auto       client = domain_->decomposition().client();
        const auto dx_base = domain_->dx_base();

        for (int l = domain_->tree()->base_level();
             l < domain_->tree()->depth(); ++l)
        {
            client->template buffer_exchange<Source>(l);

            for (auto it = domain_->begin(l); it != domain_->end(l); ++it)
            {
                if (!it->locally_owned() || !it->has_data()) continue;

                const auto dx_level =
                    dx_base / math::pow2(it->refinement_level());
                domain::Operator::curl<Source, edge_aux_type>(it->data(),
                    dx_level);
            }
        }

        //clean_leaf_correction_boundary<edge_aux_type>(domain_->tree()->base_level(), true, 2);
        // add background velocity
        copy<Source, face_aux_type>();
        domain::Operator::add_field_expression<face_aux_type>(domain_,
            simulation_->frame_vel(), T_stage_, -1.0);

        for (int l = domain_->tree()->base_level();
             l < domain_->tree()->depth(); ++l)
        {
            client->template buffer_exchange<edge_aux_type>(l);
            clean_leaf_correction_boundary<edge_aux_type>(l, false, 2);

            for (auto it = domain_->begin(l); it != domain_->end(l); ++it)
            {
                if (!it->locally_owned() || !it->has_data()) continue;

                domain::Operator::nonlinear<face_aux_type, edge_aux_type,
                    Target>(it->data());

                for (std::size_t field_idx = 0; field_idx < Target::nFields();
                     ++field_idx)
                {
                    auto& lin_data =
                        it->data_r(Target::tag(), field_idx).linalg_data();
                    lin_data *= _scale;
                }
            }

            //client->template buffer_exchange<Target>(l);
            //clean_leaf_correction_boundary<Target>(l, true,3);
        }
    }

    template<class Source_old, class Source_new, class Target>
    void nonlinear_jac(float_type _scale = 1.0) noexcept
    {
        //std::cout << "part begin" << std::endl;
        clean<edge_aux_type>();
        clean<Target>();
        clean<face_aux_tmp_type>();
        clean<nonlinear_tmp_type>();

        //std::cout << "part 0" << std::endl;

        up_and_down<Source_old>();
        up_and_down<Source_new>();

        auto       client = domain_->decomposition().client();
        const auto dx_base = domain_->dx_base();

        for (int l = domain_->tree()->base_level();
             l < domain_->tree()->depth(); ++l)
        {
            client->template buffer_exchange<Source_new>(l);

            for (auto it = domain_->begin(l); it != domain_->end(l); ++it)
            {
                if (!it->locally_owned() || !it->has_data()) continue;

                const auto dx_level =
                    dx_base / math::pow2(it->refinement_level());
                domain::Operator::curl<Source_new, edge_aux_type>(it->data(),
                    dx_level);
            }
        }

        //std::cout << "part 1" << std::endl;

        //clean_leaf_correction_boundary<edge_aux_type>(domain_->tree()->base_level(), true, 2);
        // add background velocity
        copy<Source_old, face_aux_tmp_type>();
        domain::Operator::add_field_expression<face_aux_tmp_type>(domain_,
            simulation_->frame_vel(), T_stage_, -1.0);

        for (int l = domain_->tree()->base_level();
             l < domain_->tree()->depth(); ++l)
        {
            client->template buffer_exchange<edge_aux_type>(l);
            client->template buffer_exchange<face_aux_tmp_type>(l);
            //clean_leaf_correction_boundary<edge_aux_type>(l, false, 2);

            for (auto it = domain_->begin(l); it != domain_->end(l); ++it)
            {
                if (!it->locally_owned() || !it->has_data()) continue;

                domain::Operator::nonlinear<face_aux_tmp_type, edge_aux_type,
                    nonlinear_tmp_type>(it->data());
            }
        }

        //std::cout << "part 2" << std::endl;

        clean<edge_aux_type>();
        clean<face_aux_tmp_type>();

        //auto       client = domain_->decomposition().client();
        //const auto dx_base = domain_->dx_base();

        for (int l = domain_->tree()->base_level();
             l < domain_->tree()->depth(); ++l)
        {
            client->template buffer_exchange<Source_old>(l);

            for (auto it = domain_->begin(l); it != domain_->end(l); ++it)
            {
                if (!it->locally_owned() || !it->has_data()) continue;

                const auto dx_level =
                    dx_base / math::pow2(it->refinement_level());
                domain::Operator::curl<Source_old, edge_aux_type>(it->data(),
                    dx_level);
            }
        }

        //std::cout << "part 3" << std::endl;

        //clean_leaf_correction_boundary<edge_aux_type>(domain_->tree()->base_level(), true, 2);
        // add background velocity
        copy<Source_new, face_aux_tmp_type>();
        //domain::Operator::add_field_expression<face_aux_tmp_type>(domain_, simulation_->frame_vel(), T_stage_, -1.0);

        for (int l = domain_->tree()->base_level();
             l < domain_->tree()->depth(); ++l)
        {
            client->template buffer_exchange<edge_aux_type>(l);
            client->template buffer_exchange<face_aux_tmp_type>(l);
            //clean_leaf_correction_boundary<edge_aux_type>(l, false, 2);

            for (auto it = domain_->begin(l); it != domain_->end(l); ++it)
            {
                if (!it->locally_owned() || !it->has_data()) continue;

                domain::Operator::nonlinear<face_aux_tmp_type, edge_aux_type,
                    Target>(it->data());
            }
        }
        add<nonlinear_tmp_type, Target>();

        //std::cout << "part 4" << std::endl;


        for (int l = domain_->tree()->base_level();
             l < domain_->tree()->depth(); ++l)
        {
            //client->template buffer_exchange<edge_aux_type>(l);
            //clean_leaf_correction_boundary<edge_aux_type>(l, false, 2);

            for (auto it = domain_->begin(l); it != domain_->end(l); ++it)
            {
                if (!it->locally_owned() || !it->has_data()) continue;

                for (std::size_t field_idx = 0; field_idx < Target::nFields();
                     ++field_idx)
                {
                    auto& lin_data =
                        it->data_r(Target::tag(), field_idx).linalg_data();
                    lin_data *= _scale;
                }
            }
        }
    }

    template<class Source_old, class Source_new, class Target>
    void nonlinear_jac_adjoint(float_type _scale = 1.0) noexcept
    {
        clean<edge_aux_type>();
        clean<Target>();
        clean<face_aux_tmp_type>();
        clean<nonlinear_tmp_type>();

        //curl transpose of (vel_old cross vel_new)

        up_and_down<Source_old>();
        up_and_down<Source_new>();

        auto       client = domain_->decomposition().client();
        const auto dx_base = domain_->dx_base();

        copy<Source_old, face_aux_tmp_type>();
        domain::Operator::add_field_expression<face_aux_tmp_type>(domain_,
            simulation_->frame_vel(), T_stage_, -1.0);

        for (int l = domain_->tree()->base_level();
             l < domain_->tree()->depth(); ++l)
        {
            client->template buffer_exchange<face_aux_tmp_type>(l);
            client->template buffer_exchange<Source_new>(l);

            for (auto it = domain_->begin(l); it != domain_->end(l); ++it)
            {
                if (!it->locally_owned() || !it->has_data()) continue;

                domain::Operator::nonlinear_adjoint_p1<face_aux_tmp_type,
                    Source_new, edge_aux_type>(it->data());
            }
        }

        for (int l = domain_->tree()->base_level();
             l < domain_->tree()->depth(); ++l)
        {
            client->template buffer_exchange<edge_aux_type>(l);

            for (auto it = domain_->begin(l); it != domain_->end(l); ++it)
            {
                if (!it->locally_owned() || !it->has_data()) continue;

                const auto dx_level =
                    dx_base / math::pow2(it->refinement_level());
                domain::Operator::curl_transpose<edge_aux_type, Target>(
                    it->data(), dx_level);
            }
        }

        //vort cross vel

        clean<edge_aux_type>();
        clean<face_aux_tmp_type>();

        //auto       client = domain_->decomposition().client();
        //const auto dx_base = domain_->dx_base();

        for (int l = domain_->tree()->base_level();
             l < domain_->tree()->depth(); ++l)
        {
            client->template buffer_exchange<Source_old>(l);

            for (auto it = domain_->begin(l); it != domain_->end(l); ++it)
            {
                if (!it->locally_owned() || !it->has_data()) continue;

                const auto dx_level =
                    dx_base / math::pow2(it->refinement_level());
                domain::Operator::curl<Source_old, edge_aux_type>(it->data(),
                    dx_level);
            }
        }

        //clean_leaf_correction_boundary<edge_aux_type>(domain_->tree()->base_level(), true, 2);
        // add background velocity
        copy<Source_new, face_aux_tmp_type>();
        //domain::Operator::add_field_expression<face_aux_tmp_type>(domain_, simulation_->frame_vel(), T_stage_, -1.0);

        for (int l = domain_->tree()->base_level();
             l < domain_->tree()->depth(); ++l)
        {
            client->template buffer_exchange<edge_aux_type>(l);
            client->template buffer_exchange<face_aux_tmp_type>(l);
            clean_leaf_correction_boundary<edge_aux_type>(l, false, 2);

            for (auto it = domain_->begin(l); it != domain_->end(l); ++it)
            {
                if (!it->locally_owned() || !it->has_data()) continue;

                domain::Operator::nonlinear<face_aux_tmp_type, edge_aux_type,
                    nonlinear_tmp_type>(it->data());
            }
        }
        add<nonlinear_tmp_type, Target>(-1.0);

        for (int l = domain_->tree()->base_level();
             l < domain_->tree()->depth(); ++l)
        {
            client->template buffer_exchange<edge_aux_type>(l);
            clean_leaf_correction_boundary<edge_aux_type>(l, false, 2);

            for (auto it = domain_->begin(l); it != domain_->end(l); ++it)
            {
                if (!it->locally_owned() || !it->has_data()) continue;

                for (std::size_t field_idx = 0; field_idx < Target::nFields();
                     ++field_idx)
                {
                    auto& lin_data =
                        it->data_r(Target::tag(), field_idx).linalg_data();
                    lin_data *= _scale;
                }
            }
        }
    }

    template<class Source, class Target>
    void divergence(float_type _scale = 1.0) noexcept
    {
        auto client = domain_->decomposition().client();

        domain::Operator::domainClean<Target>(domain_);

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
                domain::Operator::divergence<Source, Target>(it->data(),
                    dx_level);

                for (std::size_t field_idx = 0; field_idx < Target::nFields();
                     ++field_idx)
                {
                    auto& lin_data =
                        it->data_r(Target::tag(), field_idx).linalg_data();

                    lin_data *= _scale;
                }
            }

            //client->template buffer_exchange<Target>(l);
            //clean_leaf_correction_boundary<Target>(l, true, 2);
            //clean_leaf_correction_boundary<Target>(l, false,4+stage_idx_);
        }

        clean<Source>(true);
        clean<Target>(true);
    }

    template<class Source, class Target>
    void curl() noexcept
    {
        auto client = domain_->decomposition().client();

        domain::Operator::domainClean<Target>(domain_);

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
                domain::Operator::curl<Source, Target>(it->data(),
                    dx_level);
            }

            //client->template buffer_exchange<Target>(l);
            //clean_leaf_correction_boundary<Target>(l, true, 2);
            //clean_leaf_correction_boundary<Target>(l, false,4+stage_idx_);
        }

        clean<Source>(true);
        clean<Target>(true);
    }

    template<class Source, class Target>
    void laplacian() noexcept
    {
        auto client = domain_->decomposition().client();

        domain::Operator::domainClean<Target>(domain_);

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
                domain::Operator::laplace<Source, Target>(it->data(), dx_level);
            }

            //client->template buffer_exchange<Target>(l);
            //clean_leaf_correction_boundary<Target>(l, true, 2);
            //clean_leaf_correction_boundary<Target>(l, false,4+stage_idx_);
        }

        clean<Source>(true);
        clean<Target>(true);
    }

    template<class Source, class Target>
    void gradient(float_type _scale = 1.0) noexcept
    {
        //up_and_down<Source>();
        domain::Operator::domainClean<Target>(domain_);

        up_and_down<Source>();

        for (int l = domain_->tree()->base_level();
             l < domain_->tree()->depth(); ++l)
        {
            auto client = domain_->decomposition().client();
            client->template buffer_exchange<Source>(l);
            const auto dx_base = domain_->dx_base();
            for (auto it = domain_->begin(l); it != domain_->end(l); ++it)
            {
                if (!it->locally_owned() || !it->has_data()) continue;
                const auto dx_level =
                    dx_base / math::pow2(it->refinement_level());
                domain::Operator::gradient<Source, Target>(it->data(),
                    dx_level);
                for (std::size_t field_idx = 0; field_idx < Target::nFields();
                     ++field_idx)
                {
                    auto& lin_data =
                        it->data_r(Target::tag(), field_idx).linalg_data();

                    lin_data *= _scale;
                }
            }
            client->template buffer_exchange<Target>(l);
            //clean_leaf_correction_boundary<Target>(l, true, 2);
        }

        clean<Source>(true);
        clean<Target>(true);
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
    void addScale(float_type scale1 = 1.0, float_type scale2 = 1.0) noexcept
    {
        static_assert(From::nFields() == To::nFields(),
            "number of fields doesn't match when add");
        for (auto it = domain_->begin(); it != domain_->end(); ++it)
        {
            if (!it->locally_owned() || !it->has_data()) continue;
            for (std::size_t field_idx = 0; field_idx < From::nFields();
                 ++field_idx)
            {
                auto& lin_data =
                    it->data_r(To::tag(), field_idx).linalg_data();
                lin_data *= scale2;
                it->data_r(To::tag(), field_idx)
                    .linalg()
                    .get()
                    ->cube_noalias_view() +=
                    it->data_r(From::tag(), field_idx).linalg_data() * scale1;
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

    template<typename Field1, typename Field2>
    float_type dotField(bool exclude_correction = true) noexcept
    {
        static_assert(Field1::nFields() == Field2::nFields(),
            "number of fields doesn't match when doing dot product");

        float_type m = 0.0;

        for (int l = domain_->tree()->base_level();
             l < domain_->tree()->depth(); ++l)
        {
            auto client = domain_->decomposition().client();
            //client->template buffer_exchange<Source>(l);
            const auto dx_base = domain_->dx_base();
            for (auto it = domain_->begin(l); it != domain_->end(l); ++it)
            {
                if (!it->locally_owned() || !it->has_data()) continue;
                if (!it->is_leaf()) continue;
                if (exclude_correction && it->is_correction()) continue;

                float_type m_tmp =
                    domain::Operator::blockDot<Field1, Field2>(it->data());
                m += m_tmp;
            }
            //client->template buffer_exchange<Target>(l);
        }

        /*static_assert(Field1::nFields() == Field2::nFields(),
            "number of fields doesn't match when doing dot product");

        float_type m = 0.0;

        for (auto it = domain_->begin(); it != domain_->end(); ++it)
        {
            if (!it->locally_owned() || !it->has_data()) continue;
            if (exclude_correction && it->is_correction()) continue;
            float_type m_tmp =
                domain::Operator::blockDot<Field1, Field2>(it->data());
            m += m_tmp;
        }
        //MPI Command to all_reduce and broadcast
        //boost::mpi::communicator world;*/
        float_type m_all=0.0;
        boost::mpi::all_reduce(domain_->client_communicator(), m, m_all,
            std::plus<float_type>());
        return m_all;
    }

    

    template<typename From, typename To>
    void copy_base_level_BC(float_type scale = 1.0) noexcept
    {
        static_assert(From::nFields() == To::nFields(),
            "number of fields doesn't match when copy");

        int base_level = domain_->tree()->base_level();

        for (auto it = domain_->begin(base_level);
             it != domain_->end(base_level); ++it)
        {
            if (!it->locally_owned() || !it->has_data()) continue;
            if (!it->is_correction()) continue;
            for (std::size_t field_idx = 0; field_idx < From::nFields();
                 ++field_idx)
            {
                for (auto& n : it->data().node_field())
                    n(To::tag(), field_idx) = n(From::tag(), field_idx) * scale;
            }
        }
    }

  public:
    class sparse_mat
    {
      public:
        sparse_mat() = default;
        void resizing_row(int n) { mat.resize(n); }
        void clean() { mat.resize(0); }
        void add_element(int n, int m, float_type val) {
            if (m <= 0) return;
            auto it = mat[n].find(m);
            if (it == mat[n].end()) {
                mat[n][m] = val;
            }
            else {
                it->second += val;
            }   
        }

        sparse_mat operator+(const sparse_mat& b) {
            if (this->mat.size() != b.mat.size()) {
                std::cout << "size are " << this->mat.size() << " and " << b.mat.size() << std::endl;
                throw std::runtime_error("sparse matrix does not have the same dim, cannot add");
            }
            sparse_mat res;
            res.mat.resize(this->mat.size());
            for (int i = 1; i < mat.size();i++) {
                for (const auto& [key, val] : this->mat[i])
                {
                    res.mat[i][key] = val;
                }
                for (const auto& [key, val] : b.mat[i])
                {
                    res.add_element(i, key, val);
                }
            }
            return res;
        }

        void add_vec(const sparse_mat& b, float_type factor = 1.0) {
            if (this->mat.size() != b.mat.size()) {
                std::cout << "sparse matrix does not have the same dim, cannot in place add" << std::endl;
                return;
            }
            for (int i = 1; i < mat.size();i++) {
                for (const auto& [key, val] : b.mat[i])
                {
                    this->add_element(i, key, (val*factor));
                }
            }
        }

        void print_row(int n)
        {
            for (const auto& [key, val] : this->mat[n])
            {
                //int idx1 = std::get<0>(key1);
                std::cout << n << " " << key << " " << val << " || ";
            }
            std::cout << std::endl;
        }

        void print_row_full(int n) {
            for (int i = 0; i < mat.size(); i++) {
                auto it = mat[n].find(i);
                if (it == mat[n].end()) {
                    std::cout << 0.00 << " " std::endl;
                }
                else {
                    std::cout << it->second << " ";
                }
            }
            std::cout << std::endl;
        }

        //get the number of element to reserve space for CSR format
        int tot_size(bool include_zero=false) {
            //should not include zero but the result should be the same, this is only for debugging
            int res = 0;
            int begin_num = 1;
            if (include_zero) {
                begin_num = 0;
            }
            for (int i = begin_num; i < mat.size(); i++) {
                res+=mat[i].size();
            }
            return res;
        }
        template<class int_type, class value_type>
        void getCSR(int_type* ia, int_type* ja, value_type* a)
        {
            int_type counter = 0;
            for (int i = 1; i < mat.size(); i++)
            {
                ia[i-1] = counter+1;
                for (const auto& [key, val] : mat[i])
                {
                    //int idx1 = std::get<0>(key1);
                    //std::cout << n << " " << key << " " << val << " || ";
                    if (key < 0) {
                        std::cout << "Negative key" << std::endl;
                    }
                    ja[counter] = key;
                    a[counter] = val;
                    counter++;
                }
            }
            ia[mat.size()-1] = counter+1;
        }

        void clean_entry(float_type th_val = 1e-12) {
            for (int i = 1; i < mat.size(); i++)
            {
                for (const auto& [key, val] : mat[i])
                {
                    if (std::abs(val) < th_val) {
                        mat[i].erase(key);
                    }
                    
                }
            }
        }

        template<class value_type>
        void Apply(value_type* x, value_type* b)
        {
            int counter = 0;
            for (int i = 1; i < mat.size(); i++)
            {
                value_type tmp = 0;
                
                for (const auto& [key, val] : mat[i])
                {
                    
                    //int idx1 = std::get<0>(key1);
                    //std::cout << n << " " << key << " " << val << " || ";
                    if (key < 0) {
                        std::cout << "Negative key" << std::endl;
                    }
                    tmp+=x[key-1]*val;
                }
                b[i-1] = tmp;
            }
            //ia[mat.size()-1] = counter+1;
        }

        void scale_entries(float_type factor) {
            for (int i = 1; i < mat.size(); i++)
            {
                for (const auto& [key, val] : mat[i])
                {
                    //int idx1 = std::get<0>(key1);
                    //std::cout << n << " " << key << " " << val << " || ";
                    if (key < 0) {
                        std::cout << "Negative key" << std::endl;
                    }
                    mat[i][key] *= factor;
                }
            }
        }

        int numRow_loc(){
            return mat.size()-1;
        }

      public:
        std::vector<std::map<int, float_type>> mat;
    };

  public:
    //for testing
    sparse_mat mat;

    //for constructing sparse matrix
    sparse_mat L; //upper left matrix laplacian
    sparse_mat DN; //linearized convective term, i.e. omega_s cross u + omega cross u_s, have not taken negative yet
    sparse_mat Div; //Divergence
    sparse_mat Grad; //Gradient
    sparse_mat Curl; //Curl minus identity
    sparse_mat project; //ib projection
    sparse_mat smearing; //ib smearing
    sparse_mat boundary_u; //the matrix for the boundary of u from LGF
    sparse_mat upward_intrp; //upward interpolation matris for FMM

    //Jacobian Matrix
    sparse_mat Jac;

  private:
    

    simulation_type* simulation_;
    domain_type*     domain_; ///< domain
    poisson_solver_t psolver;
    linsys_solver_t  lsolver;
    lgf_lap_t        lgf_lap_;

    interpolation_type                c_cntr_nli_; //can use this to get the matrix

    bool base_mesh_update_ = false;

    bool add_L = true;
    bool add_DN = true;
    bool add_Div = true;
    bool add_Grad = true;
    bool add_Boundary_u = true;

    int N_sep;
    bool use_FMM = false;
    std::map<int, int> FMM_bin; // the map that structured like N_sep, 1;N_sep*2+N_sep, 2; N_sep*2^2+N_sep*2+N_sep, 2^2;...
    //the first int is the max value for that bin, the second value is the stride for that bin

    float_type              T_, T_stage_, T_max_;
    float_type              dt_base_, dt_, dx_base_;
    float_type              Re_;
    float_type              cfl_max_, cfl_;
    float_type              cg_threshold_;
    float_type              Newton_threshold_;
    float_type              Curl_factor; //the factor to better enforce vorticity criterion
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
    int cg_max_itr_;

    bool use_restart_ = false;
    bool just_restarted_ = false;
    bool write_restart_ = false;
    bool updating_source_max_ = false;
    bool all_time_max_;
    int  restart_base_freq_;
    int  adapt_count_;

    //variables for indexing for constructing matrix
    int max_idx_from_prev_prc = 0;
    int max_local_idx = 0;
    

    std::string                fname_prefix_;
    vector_type<float_type, 6> a_{{1.0 / 3, -1.0, 2.0, 0.0, 0.75, 0.25}};
    vector_type<float_type, 4> c_{{0.0, 1.0 / 3, 1.0, 1.0}};

    force_type forcing_tmp;
    force_type forcing_old;
    force_type forcing_idx;
    force_type forcing_idx_g;
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
