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
class NewtonIteration
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
    using idx_u_g_type = typename Setup::idx_u_g_type;
    using idx_p_g_type = typename Setup::idx_p_g_type;

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
                            domain::Operator::smooth2zero<F>(it->data(), i);
                        }
                    }
                }
            }
    }

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
            if (it->is_leaf() && !it->is_correction())
            {
                for (std::size_t field_idx = 0;
                     field_idx < idx_u_g_type::nFields(); ++field_idx)
                {
                    for (auto& n : it->data())
                    {
                        n(idx_u_g_type::tag(), field_idx) = n(idx_u_type::tag(), field_idx)+max_idx_from_prev_prc;
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
            if (it->is_leaf() && !it->is_correction())
            {
                for (std::size_t field_idx = 0;
                     field_idx < idx_p_g_type::nFields(); ++field_idx)
                {
                    for (auto& n : it->data())
                    {
                        n(idx_p_g_type::tag(), field_idx) = n(idx_p_type::tag(), field_idx)+max_idx_from_prev_prc;
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
        for (std::size_t i=0; i<forcing_idx_g.size(); ++i)
        {
            if (domain_->ib().rank(i)!=comm_.rank()) {
                forcing_idx_g[i]=-1;
                continue;
            }

            for (std::size_t d=0; d<forcing_idx_g[0].size(); ++d) {
                counter++;
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
        Grid2CSR<Face, Cell>(b, this->forcing_tmp);
    }

    template<class Face, class Cell, class val_type>
    void Grid2CSR(val_type* b, force_type& forcing_vec) {
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
            if (it->is_correction()) continue;
            for (std::size_t field_idx = 0; field_idx < idx_u_type::nFields();
                 ++field_idx)
            {
                for (auto& n : it->data()) {
                    int cur_idx = n(idx_u_type::tag(), field_idx);
                    b[cur_idx - 1] = n(Face::tag(), field_idx);
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
            if (it->is_correction()) continue;
            for (std::size_t field_idx = 0; field_idx < idx_p_type::nFields();
                 ++field_idx)
            {
                for (auto& n : it->data()) {
                    int cur_idx = n(idx_p_type::tag(), field_idx);
                    b[cur_idx - 1] = n(Cell::tag(), field_idx);
                }
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

    template<class Face, class Cell, class val_type>
    void CSR2Grid(val_type* b) {
        CSR2Grid<Face, Cell>(b, this->forcing_tmp);
    }

    template<class Face, class Cell, class val_type>
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
            if (it->is_correction()) continue;
            for (std::size_t field_idx = 0; field_idx < idx_u_type::nFields();
                 ++field_idx)
            {
                for (auto& n : it->data()) {
                    int cur_idx = n(idx_u_type::tag(), field_idx);
                    n(Face::tag(), field_idx) = b[cur_idx-1];
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
            if (it->is_correction()) continue;
            for (std::size_t field_idx = 0; field_idx < idx_p_type::nFields();
                 ++field_idx)
            {
                for (auto& n : it->data()) {
                    int cur_idx = n(idx_p_type::tag(), field_idx);
                    n(Cell::tag(), field_idx) = b[cur_idx-1];
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
        for (std::size_t i=0; i<forcing_idx.size(); ++i)
        {
            if (domain_->ib().rank(i)!=comm_.rank()) {
                continue;
            }

            for (std::size_t d=0; d<forcing_idx[0].size(); ++d) {
                int cur_idx = forcing_idx[i][d];
                forcing_vec[i][d] = b[cur_idx - 1];
            }
        }
        domain_->client_communicator().barrier();
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


    void construction_laplacian_u() {
        //construction of laplacian for u during stability, resolvent, and Newton iteration
        boost::mpi::communicator world;
        world.barrier();

        if (world.rank() == 0) {
            return;
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
                    L.add_element(cur_idx, glo_idx, -4.0/dx_base/dx_base);
                    glo_idx = n.at_offset(idx_u_g_type::tag(), 0, 1, field_idx);
                    L.add_element(cur_idx, glo_idx, 1.0/dx_base/dx_base);
                    glo_idx = n.at_offset(idx_u_g_type::tag(), 1, 0, field_idx);
                    L.add_element(cur_idx, glo_idx, 1.0/dx_base/dx_base);
                    glo_idx = n.at_offset(idx_u_g_type::tag(), 0, -1, field_idx);
                    L.add_element(cur_idx, glo_idx, 1.0/dx_base/dx_base);
                    glo_idx = n.at_offset(idx_u_g_type::tag(), -1, 0, field_idx);
                    L.add_element(cur_idx, glo_idx, 1.0/dx_base/dx_base);
                }
            }
        }
        domain_->client_communicator().barrier();
    }

    template<typename U_old>
    void construction_DN_u() {
        boost::mpi::communicator world;
        world.barrier();

        if (world.rank() == 0) {
            return;
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
            client->template buffer_exchange<U_old>(base_level);
            clean<edge_aux_type>();
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

                /*float_type v_s_00 = n(U_old::tag, 1);
                float_type v_s_10 = n.at_offset(U_old::tag, -1, 0, 1);
                float_type v_s_01 = n.at_offset(U_old::tag,  0, 1, 1);
                float_type v_s_11 = n.at_offset(U_old::tag, -1, 1, 1);

                float_type u_s_00 = n(U_old::tag, 0);
                float_type u_s_01 = n.at_offset(U_old::tag, 0, -1, 0);
                float_type u_s_10 = n.at_offset(U_old::tag, 1,  0, 0);
                float_type u_s_11 = n.at_offset(U_old::tag, 1, -1, 0);*/

                float_type v_s_0010 = -(n(U_old::tag, 1)                   + n.at_offset(U_old::tag, -1, 0, 1))*0.25;
                float_type v_s_0111 = -(n.at_offset(U_old::tag,  0, 1, 1)  + n.at_offset(U_old::tag, -1, 1, 1))*0.25;
                float_type u_s_0001 =  (n(U_old::tag, 0)                   + n.at_offset(U_old::tag, 0, -1, 0))*0.25;
                float_type u_s_1011 =  (n.at_offset(U_old::tag,  1, 0, 0)  + n.at_offset(U_old::tag, 1, -1, 0))*0.25;

                //-n.at_offset(edge, 0, 0, 0) *(n.at_offset(face, 0, 0, 1) + n.at_offset(face, -1, 0, 1))
                DN.add_element(cur_idx_0, glob_idx_1,    v_s_0010/dx);
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
                DN.add_element(cur_idx_1, glob_idx_0_4,  u_s_1011/dx);
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


                DN.add_element(cur_idx_0, glob_idx_1_00, -om_00);
                DN.add_element(cur_idx_0, glob_idx_1_10, -om_00);
                DN.add_element(cur_idx_0, glob_idx_1_01, -om_01);
                DN.add_element(cur_idx_0, glob_idx_1_11, -om_01);

                DN.add_element(cur_idx_1, glob_idx_0_00,  om_00);
                DN.add_element(cur_idx_1, glob_idx_0_01,  om_00);
                DN.add_element(cur_idx_1, glob_idx_0_10,  om_10);
                DN.add_element(cur_idx_1, glob_idx_0_11,  om_10);
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
        for (auto it = domain_->begin(base_level);
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
                Grad.add_element(cur_idx_1, glo_idx, 1.0 / dx_base);

                glo_idx = n.at_offset(idx_p_g_type::tag(), -1, 0, 0);
                Grad.add_element(cur_idx_0, glo_idx, -1.0 / dx_base);

                glo_idx = n.at_offset(idx_u_g_type::tag(), 0, -1, 0);
                Grad.add_element(cur_idx_1, glo_idx, -1.0 / dx_base);
            }
        }
        domain_->client_communicator().barrier();
    }

    void construction_Div() {
        //construction of Gradient
        boost::mpi::communicator world;
        world.barrier();

        if (world.rank() == 0) {
            return;
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
        for (auto it = domain_->begin(base_level);
             it != domain_->end(base_level); ++it)
        {
            if (!it->locally_owned() || !it->has_data()) continue;
            if (!it->is_leaf()) continue;
            if (it->is_correction()) continue;

            for (auto& n : it->data())
            {
                int glo_idx_0 = n(idx_u_g_type::tag(), 0);
                int glo_idx_1 = n(idx_u_g_type::tag(), 1);

                int cur_idx = n(idx_p_type::tag(), 0);
                Div.add_element(cur_idx, glo_idx_0, -1.0 / dx_base);
                Div.add_element(cur_idx, glo_idx_1, -1.0 / dx_base);

                glo_idx_0 = n.at_offset(idx_u_g_type::tag(), 1, 0, 0);
                Div.add_element(cur_idx, glo_idx_0, 1.0 / dx_base);

                glo_idx_1 = n.at_offset(idx_u_g_type::tag(), 0, 1, 1);
                Div.add_element(cur_idx, glo_idx_1, 1.0 / dx_base);
            }
        }
        domain_->client_communicator().barrier();
    }


    void construction_Smearing(float_type factor = 1.0) {
        //construction of Gradient
        boost::mpi::communicator world;
        world.barrier();

        if (world.rank() == 0) {
            return;
        }
       
        if (max_local_idx == 0) {
            std::cout << "idx not initialized, please call Assigning_idx()" << std::endl;
        }

        const auto dx_base = domain_->dx_base();

        smearing.resizing_row(max_local_idx+1);

        //int base_level = domain_->tree()->base_level();

        domain_->ib().communicator().compute_indices();
        domain_->ib().communicator().communicate(true, forcing_idx_g);

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

                        float_type val = ddf(dist + off) * factor;
                        int u_loc = node(idx_u_type::tag(),field_idx);
                        int f_glob = forcing_idx_g[i][field_idx];
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


    void construction_Projection() {
        //construction of Gradient
        boost::mpi::communicator world;
        world.barrier();

        if (world.rank() == 0) {
            return;
        }
       
        if (max_local_idx == 0) {
            std::cout << "idx not initialized, please call Assigning_idx()" << std::endl;
        }

        const auto dx_base = domain_->dx_base();

        projection.resizing_row(max_local_idx+1);

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

                    for (std::size_t field_idx = 0; field_idx < idx_u_g_type::nFields();
                         field_idx++)
                    {
                        decltype(ib_coord) off(0.5);
                        off[field_idx] = 0.0; // face data location

                        float_type val = ddf(dist + off);
                        int u_glob = node(idx_u_g_type::tag(),field_idx);
                        int f_loc = forcing_idx[i][field_idx];
                        projection.add_element(f_loc, u_glob, val);
                        /*node(u, field_idx) +=
                            f[field_idx] * ddf(dist + off) * factor;*/
                    }
                }

                /*domain::Operator::ib_projection<U>
                    (ib_coord, f[i], ib_->influence_pts(i, oct_i), ib_->delta_func());*/

                oct_i+=1;
            }
        }
        domain_->client_communicator().barrier();
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


  private:
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
            clean_leaf_correction_boundary<edge_aux_type>(l, false, 2);

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
            clean_leaf_correction_boundary<edge_aux_type>(l, false, 2);

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
            clean_leaf_correction_boundary<Target>(l, true, 2);
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
            clean_leaf_correction_boundary<Target>(l, true, 2);
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
            clean_leaf_correction_boundary<Target>(l, true, 2);
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

  private:
    class sparse_mat
    {
      public:
        sparse_mat() = default;
        void resizing_row(int n) { mat.resize(n); }
        void add_element(int n, int m, float_type val) {
            if (m < 0) return;
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
                std::cout << "sparse matrix does not have the same dim, cannot add" << std::endl;
                return NULL;
            }
            sparse_mat res;
            res.mat(this->mat.size());
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

        void add_vec(const sparse_mat& b) {
            if (this->mat.size() != b.mat.size()) {
                std::cout << "sparse matrix does not have the same dim, cannot in place add" << std::endl;
                return;
            }
            for (int i = 1; i < mat.size();i++) {
                for (const auto& [key, val] : b.mat[i])
                {
                    this->add_element(i, key, val);
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
            int counter = 0;
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
    sparse_mat project; //ib projection
    sparse_mat smearing; //ib smearing

  private:
    

    simulation_type* simulation_;
    domain_type*     domain_; ///< domain
    poisson_solver_t psolver;
    linsys_solver_t  lsolver;

    bool base_mesh_update_ = false;

    float_type              T_, T_stage_, T_max_;
    float_type              dt_base_, dt_, dx_base_;
    float_type              Re_;
    float_type              cfl_max_, cfl_;
    float_type              cg_threshold_;
    float_type              Newton_threshold_;
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
