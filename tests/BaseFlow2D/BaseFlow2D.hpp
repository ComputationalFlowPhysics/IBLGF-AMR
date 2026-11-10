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
//
// 2D steady base-flow solver -- assembled Jacobian + PETSc sparse KSP (no MKL),
// globalised with pseudo-transient continuation (PTC).
//
// Newton solve for the steady state of the IB-LGF AMR incompressible
// Navier-Stokes discretisation. Each step assembles the sparse Jacobian with
// NewtonIteration<SetupNewton> (construct_Jac_p + construct_Jac_from_Jac_p ->
// NewtonIteration::Jac) and solves, with MUMPS LU (any -ksp_/-pc_ at runtime).
// The MUMPS ICNTL/CNTL are set to mirror tests/MKLNewton's PARDISO iparm --
// METIS ordering, MC64 permutation + automatic scaling, null-pivot detection
// with a ~1e-13 absolute threshold, and a few steps of built-in iterative
// refinement -- which is why MKLNewton's bare assembled Newton tolerates the
// near-singular LGF-boundary block. Config knobs mumps_icntl_6/7/8/10/24,
// mumps_cntl_3, ir_ksp_its.
//
//
//   [Jac - (1/dtau) B] dx = R(x) ,   x <- x - dx
//
// -- implicit-Euler pseudo-transient continuation on du/dt = R(u), in
// BaseFlow2D's (x <- x - dx) sign convention. B = construction_B_matrix (1 on
// leaf-velocity DOFs). dtau starts at ptc_cfl0 / max|J_vv| (mass term dominates
// -> near-explicit Euler, guaranteed stable) and is ramped with the SER law
// dtau_{k+1} = dtau_k * ||R_{k-1}|| / ||R_k||; below ptc_switch_res the mass
// term is dropped -> pure Newton (quadratic tail). Optional Reynolds-number
// continuation (config `re_continuation=true`) as a fallback.
//
// Undamped Newton diverges from a cold start for this operator; PTC is what
// makes it converge. Historically the assembled+MUMPS route is the one that
// gave the correct 2D-cylinder eigenvalues (IBLGF-AMR commit 929de3a).
//
// Cold start: u = p = 0 is already the uniform-freestream state -- nonlinear()
// adds +U to the advecting velocity internally, so the stored `u` is the
// disturbance-from-freestream and the lab-frame velocity is u + U.
//
// The converged base flow is written as a restart consumed by
// tests/StabilityEVP2D / tests/StabilityEVP2Dasm.

#ifndef IBLGF_INCLUDED_BASE_FLOW_2D_HPP
#define IBLGF_INCLUDED_BASE_FLOW_2D_HPP

#ifndef DEBUG_IFHERK
#define DEBUG_IFHERK
#endif
#define DEBUG_POISSON

#include "mpi.h"
#include <petscksp.h>
#include <petscsys.h>
#include <array>
#include <cstdio>
#include <iostream>
#include <iomanip>
#include <functional>
#include <tuple>
#include <vector>
#include <limits>
#include <algorithm>
#include <math.h>
#include <chrono>
#include <fftw3.h>
#include <boost/mpi.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <random>

// IBLGF-specific
#include <iblgf/global.hpp>
#include <iblgf/simulation.hpp>
#include <iblgf/domain/domain.hpp>
#include <iblgf/domain/dataFields/dataBlock.hpp>
#include <iblgf/domain/dataFields/datafield.hpp>
#include <iblgf/domain/octree/tree.hpp>
#include <iblgf/IO/parallel_ostream.hpp>
#include <iblgf/lgf/lgf.hpp>
#include <iblgf/fmm/fmm.hpp>

#include <iblgf/utilities/convolution.hpp>
#include <iblgf/interpolation/interpolation.hpp>
#include <iblgf/solver/poisson/poisson.hpp>
#include <iblgf/solver/time_integration/ifherk.hpp>
#include <iblgf/solver/Newton/NewtonMethod.hpp>

#include "../../setups/setup_Newton.hpp"
#include <iblgf/operators/operators.hpp>

namespace iblgf
{
const int Dim = 2;

struct parameters
{
    static constexpr std::size_t Dim = 2;
    // clang-format off
    REGISTER_FIELDS
    (
    Dim,
     (
        //name   type        Dim   lBuffer  hBuffer, storage type
        (test  , float_type, 1,    1,       1,     cell,false ),
        //base flow state
        (u     , float_type, 2,    1,       1,     face,true  ),
        (p     , float_type, 1,    1,       1,     cell,true  ),
        (w     , float_type, 1,    1,       1,     edge,true  )
    ))
    // clang-format on
};

struct NS_AMR_LGF : public SetupNewton<NS_AMR_LGF, parameters>
{
    using super_type = SetupNewton<NS_AMR_LGF, parameters>;
    using simulation_type = typename super_type::simulation_t;
    using domain_type = typename simulation_type::domain_type;
    using ib_t = typename domain_type::ib_t;
    using force_type = typename ib_t::force_type;
    using real_coordinate_type = typename domain_type::real_coordinate_type;

    NS_AMR_LGF(Dictionary* _d)
    : super_type(_d, [this](auto _d, auto _domain)
          { return this->initialize_domain(_d, _domain); })
    {
        if (domain_->is_client()) client_comm_ = client_comm_.split(1);
        else
            client_comm_ = client_comm_.split(0);

        Re_ = simulation_.dictionary()->template get<float_type>("Re");
        R_ = simulation_.dictionary()->template get<float_type>("R");
        dx_base_ = domain_->dx_base();

        // NewtonRHS()/Jacobian()'s nonlinear terms read simulation_.frame_vel()
        // / bc_vel() (background/frame velocity of the moving reference frame);
        // these must be set or those calls throw std::bad_function_call.
        U_.resize(domain_->dimension());
        U_[0] =
            simulation_.dictionary()->template get_or<float_type>("Ux", 1.0);
        U_[1] =
            simulation_.dictionary()->template get_or<float_type>("Uy", 0.0);
        if (domain_->dimension() > 2)
            U_[2] = simulation_.dictionary()->template get_or<float_type>(
                "Uz", 0.0);

        simulation_.frame_vel() =
            [this](std::size_t idx, float_type t, auto coord = {0, 0})
        { return -U_[idx]; };
        simulation_.bc_vel() =
            [this](std::size_t idx, float_type t, auto coord)
        { return -U_[idx]; };

        nLevelRefinement_ =
            simulation_.dictionary_->template get_or<int>("nLevels", 0);
        global_refinement_ = simulation_.dictionary_->template get_or<int>(
            "global_refinement", 0);
        Newton_max_itr_ = simulation_.dictionary()->template get_or<int>(
            "Newton_max_itr", 100);
        Newton_threshold_ =
            simulation_.dictionary()->template get_or<float_type>(
                "Newton_threshold", 1e-6);

        // --- pseudo-transient continuation knobs ----------------------------
        ptc_cfl0_ = simulation_.dictionary()->template get_or<float_type>(
            "ptc_cfl0", 0.05);
        ptc_dtau_max_ = simulation_.dictionary()->template get_or<float_type>(
            "ptc_dtau_max", 1e8);
        ptc_growth_max_ = simulation_.dictionary()->template get_or<float_type>(
            "ptc_growth_max", 8.0);
        ptc_switch_res_ = simulation_.dictionary()->template get_or<float_type>(
            "ptc_switch_res", 1e-3);
        mumps_icntl_14_ = simulation_.dictionary()->template get_or<int>(
            "mumps_icntl_14", 400);
        // MUMPS settings that mirror tests/MKLNewton's PARDISO iparm (which
        // tolerates the near-singular LGF-boundary block far better than a bare
        // KSPPREONLY+PCLU): METIS ordering (PARDISO iparm[1]=2), MC64
        // max-weighted-matching column permutation + scaling (iparm[10]=1,
        // iparm[12]=1), a tiny absolute null-pivot / static-pivot threshold
        // (iparm[9]=13 => perturb by 1e-13), and a few steps of built-in
        // iterative refinement (iparm[7]=5).
        mumps_icntl_7_ = simulation_.dictionary()->template get_or<int>(
            "mumps_icntl_7", 5);   // ordering: 5 = METIS
        mumps_icntl_6_ = simulation_.dictionary()->template get_or<int>(
            "mumps_icntl_6", 7);   // col perm / MC64: 7 = automatic (turns
                                   // itself off if incompatible with icntl_24)
        mumps_icntl_8_ = simulation_.dictionary()->template get_or<int>(
            "mumps_icntl_8", 77);  // scaling: 77 = automatic (MC64-style)
        mumps_icntl_10_ = simulation_.dictionary()->template get_or<int>(
            "mumps_icntl_10", 5);  // iterative-refinement steps (PARDISO iparm[7])
        mumps_icntl_24_ = simulation_.dictionary()->template get_or<int>(
            "mumps_icntl_24", 1);  // 1 = null-pivot detection (bounds the near-
                                   // singular corner). 0 = PARDISO-faithful:
                                   // rely on MUMPS static pivoting + icntl_10.
        mumps_cntl_3_ = simulation_.dictionary()->template get_or<float_type>(
            "mumps_cntl_3", -1e-13); // <0 => |cntl_3| absolute null-pivot thresh
                                     // ~ PARDISO iparm[9]=13 (perturb by 1e-13)
        ir_ksp_its_ = simulation_.dictionary()->template get_or<int>(
            "ir_ksp_its", 0);      // >0 => outer KSPRICHARDSON refinement on top
        ptc_shift_ = simulation_.dictionary()->template get_or<float_type>(
            "ptc_shift", 0.0);  // 0 => auto from the velocity-diagonal sign
        ls_max_ = simulation_.dictionary()->template get_or<int>(
            "ls_max", 8);
        ls_min_alpha_ = simulation_.dictionary()->template get_or<float_type>(
            "ls_min_alpha", 1.0 / 256.0);
        newton_sign_ = simulation_.dictionary()->template get_or<float_type>(
            "newton_sign", -1.0);  // x <- x + newton_sign * dx  (default Newton)
        ptc_creg_ = simulation_.dictionary()->template get_or<float_type>(
            "ptc_creg", 0.0);  // constraint-row diag reg; ANY nonzero value
                               // (even 1e-6) destroys the Newton direction for
                               // this operator -- keep OFF. Left as a knob for
                               // future null-space work.
        ptc_diag_I_ = simulation_.dictionary()->template get_or<bool>(
            "ptc_diag_I", false); // uniform (Levenberg-Marquardt) shift on
                                  // EVERY diagonal instead of the leaf-velocity
                                  // mass matrix B -- needed to bound the direct
                                  // solve, whose near-null space is in the
                                  // constraint / correction-velocity DOFs.
        ptc_pin_reg_ = simulation_.dictionary()->template get_or<float_type>(
            "ptc_pin_reg", 0.05); // fraction of the velocity pseudo-time shift
                                  // applied to the pressure/vorticity/IB-force
                                  // diagonals when ptc_diag_I -- bounds the
                                  // near-null constraint modes; ramps down with
                                  // dtau so the incompressibility penalty -> 0.
        init_mode_ = simulation_.dictionary()->template get_or<std::string>(
            "init_mode", "zero");  // "zero" | "potential"

        // Reynolds-number continuation fallback: `re_continuation=true` runs a
        // geometric ladder {Re/4, Re/2, Re} (e.g. 10 -> 20 -> 40).
        if (simulation_.dictionary()->template get_or<bool>(
                "re_continuation", false))
            re_ladder_ = {0.25 * Re_, 0.5 * Re_, Re_};

        // --- AMR refine box (physical units, D = 2R) ----------------------
        {
            using arr2 = std::array<float_type, 2>;
            arr2 lo = simulation_.dictionary()->template get_or<float_type, 2>(
                "refine_box_lo", arr2{{-8.0, -10.0}});
            arr2 hi = simulation_.dictionary()->template get_or<float_type, 2>(
                "refine_box_hi", arr2{{40.0, 10.0}});
            box_x0_ = lo[0]; box_y0_ = lo[1];
            box_x1_ = hi[0]; box_y1_ = hi[1];
        }

        domain_->register_refinement_condition() =
            [this](auto octant, int diff_level)
        { return this->refinement(octant, diff_level); };

        domain_->ib().init(_d->get_dictionary("simulation_parameters"),
            domain_->dx_base(), nLevelRefinement_, Re_);

        if (!use_restart())
        {
            domain_->init_refine(nLevelRefinement_, global_refinement_, 0);
        }
        else
        {
            domain_->restart_list_construct();
        }

        domain_->distribute<fmm_mask_builder_t, fmm_mask_builder_t>();

        boost::mpi::communicator world;
        if (use_restart())
        {
            if (world.rank() == 0)
                std::cout << "reading initial base flow (u, p) from restart"
                          << std::endl;
            simulation_.template read_h5<u_type>(
                simulation_.restart_field_dir(), "u");
            simulation_.template read_h5<p_type>(
                simulation_.restart_field_dir(), "p");
        }
        else
        {
            if (world.rank() == 0)
                std::cout << "cold start: uniform-freestream guess (stored "
                             "u = p = 0)"
                          << std::endl;
            this->initialize();
        }

        real_coordinate_type tmp_coord(0.0);
        forcing_state_.resize(domain_->ib().size());
        std::fill(forcing_state_.begin(), forcing_state_.end(), tmp_coord);
        forcing_res_.resize(domain_->ib().size());
        std::fill(forcing_res_.begin(), forcing_res_.end(), tmp_coord);

        if (world.rank() == 0)
            std::cout << "on Simulation: \n" << simulation_ << std::endl;
    }

    float_type run(int argc, char* argv[])
    {
        PetscInitialize(&argc, &argv, (char*)0, NULL);

        op_ = std::make_shared<time_integration_t>(&this->simulation_);

        newton_solve();

        PetscFinalize();
        return 0.0;
    }

    /** @brief Orchestrate: one-time assembly of the base-flow-independent
     *  blocks + the mass matrix, then a PTC Newton loop (optionally over a
     *  Reynolds-number ladder), then diagnostics + restart. */
    void newton_solve()
    {
        boost::mpi::communicator world;
        PetscMPIInt rank = world.rank();

        op_->set_stage_time(0.0);
        op_->Assigning_idx();
        if (!domain_->is_server())
            op_->template pad_velocity<u_type, u_type>(true);
        world.barrier();

        // mass matrix B (base-flow independent) -- built once
        op_->construction_B_matrix();
        op_->B.clean_entry(1e-14);

        N_ = op_->total_dim();
        dtau_ = -1.0;  // set from the Jacobian diagonal scale on the first step
        res0_ = -1.0;

        std::vector<float_type> re_seq =
            re_ladder_.empty() ? std::vector<float_type>{Re_} : re_ladder_;

        for (std::size_t r = 0; r < re_seq.size(); ++r)
        {
            op_->set_Re(re_seq[r]);
            if (rank == world.size() - 1 && re_seq.size() > 1)
                std::cout << "=== Reynolds continuation rung " << (r + 1) << "/"
                          << re_seq.size() << "  Re = " << re_seq[r]
                          << " ===" << std::endl;
            // Re_ is baked into the assembled L block at construct_Jac_p() time
            op_->construct_Jac_p();
            rung_converged_ = false;
            reject_streak_  = 0;
            ptc_newton_loop();
            if (!rung_converged_ && rank == world.size() - 1)
                std::cout << "WARNING: PTC Newton did not reach Newton_threshold"
                          << " on this rung" << std::endl;
        }

        // --- diagnostics -----------------------------------------------
        report_wake_length();
        report_drag();

        // --- write the converged base flow as a restart ----------------
        world.barrier();
        bool write_restart = simulation_.dictionary()->template get_or<bool>(
            "write_restart", true);
        if (write_restart)
        {
            if (domain_->is_server()) simulation_.copy_restart();
            world.barrier();
            simulation_.write("", true); // restart_field.hdf5 + tree_info.bin
        }
        simulation_.write("baseflow_final.hdf5");
        world.barrier();
    }

    /** @brief PTC-damped Newton loop for the current Re_. Sets rung_converged_
     *  if the residual reaches Newton_threshold_. Uses/updates the member dtau_.
     *  void return so the PETSc PetscCallVoid macro (which does `return;` on
     *  error) is well-formed. */
    void ptc_newton_loop()
    {
        boost::mpi::communicator world;
        PetscMPIInt rank = world.rank();
        float_type  res_prev = -1.0;

        for (int k = 0; k < Newton_max_itr_; ++k)
        {
            // (re)build the base-flow-dependent Jacobian J = Jac_p - DN
            if (!domain_->is_server())
                op_->template pad_velocity<u_type, u_type>(true);
            op_->template construct_Jac_from_Jac_p<u_type>();
            op_->Jac.clean_entry(1e-10);

            if (k == 0) audit_jacobian();

            // steady residual R(x) -> fu_i / fp_i (+ forcing_res_)
            op_->template NewtonRHS<u_type, p_type, fu_i_type, fp_i_type>(
                forcing_state_, forcing_res_);

            PetscInt nlocal = domain_->is_server() ? 0 : op_->Jac.numRow_loc();
            std::vector<float_type> b_loc(nlocal, 0.0);
            std::vector<float_type> x_old(nlocal, 0.0);
            std::vector<float_type> dx(nlocal, 0.0);

            if (!domain_->is_server())
            {
                op_->template Grid2CSR<u_type, p_type, w_type>(
                    x_old.data(), forcing_state_, false);
                op_->template Grid2CSR<fu_i_type, fp_i_type, fw_i_type>(
                    b_loc.data(), forcing_res_);

                // Pin-row consistency: the assembled Jac row for the pinned
                // pressure cell is [1] on p_pin, so its RHS must be the pin
                // residual -(p_pin - 0) = -x_old[pin], not the -div u|_pin that
                // NewtonRHS wrote there. Without this the pin cell's pressure
                // drifts every iteration and corrupts the near-corner field.
                if (op_->is_p_set_zero())
                {
                    int pin_g = op_->pin_global_idx();      // 1-based global
                    int pin_l = pin_g - (op_->num_start() - 1); // 1-based local
                    if (pin_g > 0 && pin_l >= 1 && pin_l <= nlocal)
                        b_loc[pin_l - 1] = -x_old[pin_l - 1];
                }
            }

            float_type res = std::sqrt(op_->GetStateMag(b_loc.data()));
            if (res0_ < 0.0) res0_ = res;

            // per-DOF-block residual norms (raw L2 of the CSR RHS by block) --
            // shows WHERE a plateau lives (momentum R_u vs continuity R_p vs
            // IB no-slip R_f). Reduced + printed once per iteration.
            {
                double rb[5] = {0, 0, 0, 0, 0};
                if (!domain_->is_server())
                {
                    const int bu  = op_->blk_u_end_, bp = op_->blk_p_end_;
                    const int bwf = op_->blk_wf_end_, bwc = op_->blk_wc_end_;
                    for (int i = 1; i <= (int)b_loc.size(); ++i)
                    {
                        double v = b_loc[i - 1];
                        v *= v;
                        int b = (i <= bu)   ? 0
                                : (i <= bp)  ? 1
                                : (i <= bwf) ? 2
                                : (i <= bwc) ? 3
                                             : 4;
                        rb[b] += v;
                    }
                }
                double g[5];
                boost::mpi::all_reduce(world, rb, 5, g, std::plus<double>());
                if (rank == world.size() - 1)
                    std::cout << "  [res] u=" << std::sqrt(g[0]) << " p="
                              << std::sqrt(g[1]) << " wf=" << std::sqrt(g[2])
                              << " wc=" << std::sqrt(g[3]) << " f="
                              << std::sqrt(g[4]) << std::endl;
            }

            // Initialise dtau from the Jacobian velocity-diagonal scale so the
            // pseudo-time term dominates J on the first steps, then let SER
            // ramp it. Also report the signed velocity diagonal (sets the PTC
            // shift sign so the regularised system is diagonally dominant).
            if (dtau_ < 0.0)
            {
                float_type jvv_absmax = 0.0, jvv_min = 0.0, jvv_max_s = 0.0;
                if (!domain_->is_server())
                    for (int i = 1; i <= op_->B.numRow_loc(); ++i)
                    {
                        if (op_->B.mat[i].empty()) continue;
                        int  gcol = op_->B.mat[i].begin()->first;
                        auto it = op_->Jac.mat[i].find(gcol);
                        if (it == op_->Jac.mat[i].end()) continue;
                        jvv_absmax = std::max(jvv_absmax, std::abs(it->second));
                        jvv_min = std::min(jvv_min, it->second);
                        jvv_max_s = std::max(jvv_max_s, it->second);
                    }
                float_type g_absmax, g_min, g_maxs;
                boost::mpi::all_reduce(world, jvv_absmax, g_absmax,
                    boost::mpi::maximum<float_type>());
                boost::mpi::all_reduce(
                    world, jvv_min, g_min, boost::mpi::minimum<float_type>());
                boost::mpi::all_reduce(
                    world, jvv_max_s, g_maxs, boost::mpi::maximum<float_type>());
                dtau_ = ptc_cfl0_ / std::max(g_absmax, (float_type)1e-30);
                jvv_ref_ = std::max(g_absmax, (float_type)1e-30);
                if (ptc_shift_ == 0.0)
                    ptc_shift_ = (std::abs(g_min) >= std::abs(g_maxs)) ? -1.0
                                                                      : 1.0;
                if (rank == world.size() - 1)
                    std::cout << "[PTC] J_vv diag in [" << g_min << ", "
                              << g_maxs << "]  |max| = " << g_absmax
                              << " ; dtau_0 = " << dtau_
                              << " ; shift sign = " << ptc_shift_ << std::endl;
            }

            if (res < Newton_threshold_)
            {
                if (rank == world.size() - 1)
                    std::cout << "Newton converged: iter " << k
                              << " , ||R|| = " << res << std::endl;
                rung_converged_ = true;
                return;
            }
            if (!std::isfinite(res) || (res0_ > 0.0 && res > 1e4 * res0_))
            {
                if (rank == world.size() - 1)
                    std::cout << "DIVERGED at iter " << k << " , ||R|| = " << res
                              << " (res0 = " << res0_ << ")" << std::endl;
                return;
            }
            if (reject_streak_ >= 15)
            {
                if (rank == world.size() - 1)
                    std::cout << "STALLED at iter " << k << " : " << reject_streak_
                              << " consecutive rejected steps, ||R|| = " << res
                              << " , dtau = " << dtau_ << std::endl;
                return;
            }

            // dtau ramp. SER (dtau *= ||R_prev||/||R||) is too timid when the
            // damped step only chips ~2%/iter off the residual. So: while the
            // full Newton step keeps getting accepted (alpha == 1, i.e. we are
            // in the Newton basin), grow dtau by at least ptc_growth_max each
            // iteration; if a step had to be cut, fall back to SER and never
            // grow.
            if (res_prev > 0.0)
            {
                float_type ser = res_prev / std::max(res, (float_type)1e-30);
                float_type ratio =
                    (last_alpha_ >= 0.999)
                        ? std::max(ser, ptc_growth_max_)
                        : std::min(ser, (float_type)1.0);
                ratio = std::min(ratio, ptc_growth_max_);
                ratio = std::max(ratio, (float_type)0.1);
                dtau_ = std::min(dtau_ * ratio, ptc_dtau_max_);
            }
            res_prev = res;
            float_type inv_dtau =
                (res < ptc_switch_res_ || dtau_ >= ptc_dtau_max_) ? 0.0
                                                                  : 1.0 / dtau_;

            if (!domain_->is_server())
            {
                // (a) Permanent tiny regularisation of the exactly-zero-
                //     diagonal constraint rows (pressure-continuity, IB-force)
                //     so the assembled saddle point is non-singular for the
                //     direct solve. Add -ptc_creg*|J_vv|_max (the stable sign
                //     for a [A B^T; C 0] block) -- O(1e-6), kept for all dtau.
                if (ptc_creg_ != 0.0)
                {
                    const float_type eps = -ptc_creg_ * jvv_ref_;
                    for (int i = 1; i <= op_->Jac.numRow_loc(); ++i)
                    {
                        int gd = i - 1 + op_->num_start();
                        if (op_->Jac.mat[i].find(gd) == op_->Jac.mat[i].end())
                            op_->Jac.add_element(i, gd, eps);
                    }
                }

                // (b) Pseudo-transient / trust-region regularisation.
                //   ptc_diag_I=false: Jac += ptc_shift*(1/dtau) B -- mass term
                //     on the LEAF-velocity block only (classic PTC). The
                //     assembled Jacobian's ~1e-8 near-null space (truncated LGF
                //     boundary closure) lives in the CORRECTION-cell boundary-
                //     velocity DOFs, which B misses, so the direct solve there
                //     is unbounded / MUMPS-nondeterministic.
                //   ptc_diag_I=true : Jac += ptc_shift*(1/dtau) * I over the
                //     WHOLE velocity block [1, blk_u_end_] -- leaf + correction
                //     velocities (all pseudo-time-evolved), but NOT the
                //     pressure / vorticity / IB-force rows (true constraints).
                //     Shifting the pressure diagonal (an earlier attempt)
                //     penalises the Lagrange multiplier and the flow never
                //     becomes divergence-free -> residual plateaus with
                //     ||div u|| ~ O(1). Wiped by next rebuild.
                if (inv_dtau > 0.0)
                {
                    if (ptc_diag_I_)
                    {
                        // full velocity block gets the pseudo-time shift...
                        const float_type s  = ptc_shift_ * inv_dtau;
                        // ...and the pressure / vorticity / IB-force rows get
                        // ptc_pin_reg (~0.05) times that -- a *fraction* of the
                        // pseudo-time shift, ramping down with it. Big enough
                        // early to bound the near-null constraint modes (~1e-3,
                        // so a shift ~1 kills the amplification), small enough
                        // and shrinking so the incompressibility penalty
                        // (||div u|| error ~ sc/|grad|) -> 0 near convergence.
                        const float_type sc =
                            ptc_shift_ * inv_dtau * ptc_pin_reg_;
                        const int nr = op_->Jac.numRow_loc();
                        const int nv = std::min(op_->blk_u_end_, nr);
                        for (int i = 1; i <= nr; ++i)
                            op_->Jac.add_element(i, i - 1 + op_->num_start(),
                                (i <= nv) ? s : sc);
                    }
                    else
                        op_->Jac.add_vec(op_->B, ptc_shift_ * inv_dtau);
                }
            }

            // assemble + solve  [Jac_reg + ptc_shift/dtau B] dx = R
            Mat A;
            Vec x, b;
            KSP ksp;
            PC  pc;
            assemble_jacobian(N_, nlocal, &A);

            PetscCallVoid(MatCreateVecs(A, &x, &b));
            {
                PetscInt rstart, rend;
                PetscCallVoid(VecGetOwnershipRange(b, &rstart, &rend));
                for (PetscInt i = rstart; i < rend; ++i)
                {
                    PetscScalar v = b_loc[i - rstart];
                    PetscCallVoid(VecSetValues(b, 1, &i, &v, INSERT_VALUES));
                }
                PetscCallVoid(VecAssemblyBegin(b));
                PetscCallVoid(VecAssemblyEnd(b));
                PetscCallVoid(VecSet(x, 0.0));
            }

#if defined(PETSC_HAVE_MUMPS)
            // Mirror tests/MKLNewton's PARDISO iparm on the MUMPS side.
            {
                auto seti = [](const char* k, int v) {
                    char buf[24];
                    std::snprintf(buf, sizeof(buf), "%d", v);
                    PetscOptionsSetValue(NULL, k, buf);
                };
                auto setf = [](const char* k, double v) {
                    char buf[32];
                    std::snprintf(buf, sizeof(buf), "%.17g", v);
                    PetscOptionsSetValue(NULL, k, buf);
                };
                seti("-mat_mumps_icntl_14", mumps_icntl_14_); // workspace %
                seti("-mat_mumps_icntl_7", mumps_icntl_7_);   // METIS ordering
                seti("-mat_mumps_icntl_6", mumps_icntl_6_);   // MC64 col perm
                seti("-mat_mumps_icntl_8", mumps_icntl_8_);   // auto scaling
                seti("-mat_mumps_icntl_10", mumps_icntl_10_); // iter refinement
                seti("-mat_mumps_icntl_24", mumps_icntl_24_); // null-pivot detect
                if (mumps_icntl_24_ != 0 && mumps_cntl_3_ != 0.0)
                    setf("-mat_mumps_cntl_3", mumps_cntl_3_);
            }
#endif
            PetscCallVoid(KSPCreate(PETSC_COMM_WORLD, &ksp));
            PetscCallVoid(KSPSetOperators(ksp, A, A));
            PetscCallVoid(KSPGetPC(ksp, &pc));
            PetscCallVoid(PCSetType(pc, PCLU));
#if defined(PETSC_HAVE_MUMPS)
            PetscCallVoid(PCFactorSetMatSolverType(pc, MATSOLVERMUMPS));
#endif
            if (ir_ksp_its_ > 0)
            {
                // Outer iterative refinement on top of MUMPS' internal ICNTL(10)
                // -- Richardson with the LU as preconditioner, a few sweeps.
                PetscCallVoid(KSPSetType(ksp, KSPRICHARDSON));
                PetscCallVoid(KSPSetTolerances(ksp, 1e-12, 1e-30,
                    PETSC_DEFAULT, ir_ksp_its_));
                PetscCallVoid(KSPSetNormType(ksp, KSP_NORM_PRECONDITIONED));
            }
            else
            {
                PetscCallVoid(KSPSetType(ksp, KSPPREONLY));
                PetscCallVoid(KSPSetTolerances(
                    ksp, 1e-10, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT));
            }
            PetscCallVoid(KSPSetFromOptions(ksp));
            PetscCallVoid(KSPSolve(ksp, b, x));

            KSPConvergedReason reason;
            PetscInt           its;
            PetscCallVoid(KSPGetConvergedReason(ksp, &reason));
            PetscCallVoid(KSPGetIterationNumber(ksp, &its));

#if defined(PETSC_HAVE_MUMPS)
            if (k == 0)
            {
                Mat        F;
                PetscInt   ig28 = -1, ig12 = -1;
                PetscReal  rg7 = -1;
                PetscInt   npiv = 0, *piv = nullptr;
                if (PCFactorGetMatrix(pc, &F) == 0 && F)
                {
                    MatMumpsGetInfog(F, 28, &ig28); // # null pivots detected
                    MatMumpsGetInfog(F, 12, &ig12); // # negative pivots
                    MatMumpsGetRinfog(F, 7, &rg7);  // scaled resid after IR
                    MatMumpsGetNullPivots(F, &npiv, &piv); // 0-based glob rows
                }
                if (rank == world.size() - 1)
                    std::cout << "[MUMPS] INFOG(28) null-pivots = " << ig28
                              << " ; INFOG(12) neg-pivots = " << ig12
                              << " ; RINFOG(7) sc.resid = " << rg7
                              << std::endl;
                classify_null_pivots(npiv, piv);
                if (piv) PetscFree(piv);
            }
#endif

            {
                PetscInt rstart, rend;
                PetscCallVoid(VecGetOwnershipRange(x, &rstart, &rend));
                for (PetscInt i = rstart; i < rend; ++i)
                {
                    PetscScalar v;
                    PetscCallVoid(VecGetValues(x, 1, &i, &v));
                    dx[i - rstart] = PetscRealPart(v);
                }
            }

            float_type dx_mag2 = op_->GetStateMag(dx.data());
            float_type x_mag2 = op_->GetStateMag(x_old.data());
            float_type dxrel =
                std::sqrt((x_mag2 > 0.0) ? dx_mag2 / x_mag2 : dx_mag2);

            // --- garbage-step guard --------------------------------------
            // The assembled Jacobian has a handful of near-null pivots at the
            // LGF-boundary corner (see plan file). MUMPS' icntl_24 handling of
            // them is NON-deterministic: most steps are clean but ~1 in 3 comes
            // back with a huge null-space component (||dx||/||x|| ~ 1e12). Do
            // not try to line-search such a step -- reject it, cut dtau hard so
            // the pseudo-time term dominates next iteration, and retry.
            bool garbage = !std::isfinite(dxrel) || dxrel > 1e3;

            // --- backtracking line search on ||R|| (Armijo) ---------------
            // x_trial = x_old - alpha dx ; accept the largest alpha in
            // {1, 1/2, ...} with ||R(x_trial)|| <= (1 - 1e-4 alpha)||R||.
            float_type alpha = garbage ? 0.0 : 1.0, res_new = res;
            if (!garbage)
            {
                std::vector<float_type> x_trial(nlocal, 0.0);
                std::vector<float_type> r_trial(nlocal, 0.0);
                for (int ls = 0; ls < ls_max_; ++ls)
                {
                    if (!domain_->is_server())
                    {
                        for (PetscInt i = 0; i < nlocal; ++i)
                            x_trial[i] = x_old[i] + newton_sign_ * alpha * dx[i];
                        op_->template CSR2Grid<u_type, p_type, w_type>(
                            x_trial.data(), forcing_state_);
                    }
                    world.barrier();
                    op_->template NewtonRHS<u_type, p_type, fu_i_type,
                        fp_i_type>(forcing_state_, forcing_res_);
                    if (!domain_->is_server())
                    {
                        op_->template Grid2CSR<fu_i_type, fp_i_type, fw_i_type>(
                            r_trial.data(), forcing_res_);
                        if (op_->is_p_set_zero())
                        {
                            int pg = op_->pin_global_idx();
                            int pl = pg - (op_->num_start() - 1);
                            if (pg > 0 && pl >= 1 && pl <= nlocal)
                                r_trial[pl - 1] = -x_trial[pl - 1];
                        }
                    }
                    res_new = std::sqrt(op_->GetStateMag(r_trial.data()));
                    if (res_new <= (1.0 - 1e-4 * alpha) * res) break;
                    if (alpha <= ls_min_alpha_) break;
                    alpha *= 0.5;
                }
            }

            // A step that made the residual clearly worse is as useless as a
            // garbage step: don't pollute the state with it. (A tiny increase
            // is tolerated -- pseudo-transient marches need not be monotone.)
            bool rejected = garbage || res_new > 1.02 * res;

            // dtau floor -- driving dtau to ~0 makes 1/dtau blow up, the
            // velocity block gets an enormous shift, dx_vel -> 0 and the step
            // is then pure (unregularised) constraint noise. Keep dtau in a
            // sane band.
            const float_type dtau_floor = ptc_cfl0_ * 1e-3;

            if (rejected)
            {
                // revert grid + forcing to x_old, shrink the trust region
                if (!domain_->is_server())
                    op_->template CSR2Grid<u_type, p_type, w_type>(
                        x_old.data(), forcing_state_);
                world.barrier();
                alpha  = 0.0;
                dtau_  = std::max(dtau_ * 0.35, dtau_floor);
                res_prev = -1.0;               // restart the SER ramp
                ++reject_streak_;
            }
            else
            {
                if (!domain_->is_server())
                    for (PetscInt i = 0; i < nlocal; ++i)
                        x_old[i] += newton_sign_ * alpha * dx[i];
                reject_streak_ = 0;
                // A weak step (line search had to cut alpha a lot) -> shrink
                // the trust region a little so we keep moving.
                if (alpha < 0.25)
                    dtau_ = std::max(dtau_ * 0.5, dtau_floor);
            }
            last_alpha_ = alpha;

            if (rank == world.size() - 1)
                std::cout << "Newton iter " << k << " : ||R|| = " << res
                          << " (rel " << (res0_ > 0 ? res / res0_ : res) << ")"
                          << " -> " << res_new << " , alpha = " << alpha
                          << " , ||dx||/||x|| = " << dxrel
                          << " , dtau = " << dtau_
                          << (rejected ? (garbage ? " [REJECT garbage]"
                                                  : " [REJECT no-decr]")
                                       : (inv_dtau > 0.0 ? " [PTC]" : " [Newton]"))
                          << " , KSP its = " << its << " , reason = " << reason
                          << std::endl;

            simulation_.write("Newton_itr" + std::to_string(k));

            PetscCallVoid(VecDestroy(&x));
            PetscCallVoid(VecDestroy(&b));
            PetscCallVoid(MatDestroy(&A));
            PetscCallVoid(KSPDestroy(&ksp));
        }
    }

    /** @brief Structural audit of the assembled Jacobian, by DOF block.
     *  Reports, per block (velocity / pressure / fine-vorticity /
     *  coarse-vorticity / IB-force): row count, completely-empty rows (a zero
     *  row == a guaranteed null pivot / structural rank deficiency), and rows
     *  with no stored diagonal entry (expected for the saddle constraint rows,
     *  a bug for the velocity block). Also prints the first few offending
     *  global indices so we can trace them back to grid points. Cheap, runs
     *  once (k == 0). See plan file: chasing the MUMPS null pivots. */
    void audit_jacobian()
    {
        boost::mpi::communicator world;
        const int rank      = world.rank();
        const bool is_srv   = domain_->is_server();

        const int nrow  = is_srv ? 0 : op_->Jac.numRow_loc(); // == max_local_idx
        const int start = is_srv ? 0 : op_->num_start();  // 1-based global row 1
        const int bu  = is_srv ? 0 : op_->blk_u_end_;
        const int bp  = is_srv ? 0 : op_->blk_p_end_;
        const int bwf = is_srv ? 0 : op_->blk_wf_end_;
        const int bwc = is_srv ? 0 : op_->blk_wc_end_;

        // 5 blocks: 0 vel, 1 p, 2 w-fine, 3 w-coarse, 4 force
        long tot[5] = {0}, empty[5] = {0}, nodiag[5] = {0};
        long zdiag[5] = {0};                            // |diag| < 1e-30
        std::vector<int> ex_empty, ex_nodiag;           // sample global idx

        auto blk_of = [&](int i) -> int {
            if (i <= bu) return 0;
            if (i <= bp) return 1;
            if (i <= bwf) return 2;
            if (i <= bwc) return 3;
            return 4;
        };

        for (int i = 1; i <= nrow; ++i)
        {
            const int b = blk_of(i);
            const int gdiag = i - 1 + start;            // 1-based global col key
            const auto& row = op_->Jac.mat[i];
            ++tot[b];
            if (row.empty())
            {
                ++empty[b];
                if (ex_empty.size() < 12) ex_empty.push_back(gdiag);
                continue;
            }
            auto it = row.find(gdiag);
            if (it == row.end())
            {
                ++nodiag[b];
                if (ex_nodiag.size() < 12) ex_nodiag.push_back(gdiag);
            }
            else if (std::abs(it->second) < 1e-30)
                ++zdiag[b];
        }

        long g_tot[5], g_empty[5], g_nodiag[5], g_zdiag[5];
        boost::mpi::all_reduce(world, tot, 5, g_tot, std::plus<long>());
        boost::mpi::all_reduce(world, empty, 5, g_empty, std::plus<long>());
        boost::mpi::all_reduce(world, nodiag, 5, g_nodiag, std::plus<long>());
        boost::mpi::all_reduce(world, zdiag, 5, g_zdiag, std::plus<long>());

        // --- velocity-block diagonal-dominance scan -------------------------
        // (the pure-Jac MUMPS null pivots all land in the velocity block, so
        //  look for velocity rows with a tiny |diagonal| relative to their
        //  off-diagonal row sum -- those are the near-singular directions.)
        long vsmall[3] = {0, 0, 0}; // |diag| < 1e-8 , < 1e-4 , < 1e-1
        double vdmin = 1e300, vdmax = 0.0;
        // collect the worst (smallest |diag|) velocity rows on each rank
        std::vector<std::tuple<double, int, int, double, double>> worst; //|d|,g,nnz,d,offsum
        for (int i = 1; i <= bu; ++i)
        {
            const auto& row = op_->Jac.mat[i];
            if (row.empty()) continue;
            const int gdiag = i - 1 + start;
            double d = 0.0, offsum = 0.0;
            for (const auto& [key, val] : row)
            {
                if (key == gdiag) d = std::abs(val);
                else offsum += std::abs(val);
            }
            vdmin = std::min(vdmin, d);
            vdmax = std::max(vdmax, d);
            if (d < 1e-1) ++vsmall[2];
            if (d < 1e-4) ++vsmall[1];
            if (d < 1e-8) ++vsmall[0];
            if (d < 1e-1)
                worst.emplace_back(d, gdiag, (int)row.size(), d, offsum);
        }
        long g_vsmall[3];
        boost::mpi::all_reduce(world, vsmall, 3, g_vsmall, std::plus<long>());
        double g_vdmin, g_vdmax;
        boost::mpi::all_reduce(
            world, vdmin, g_vdmin, boost::mpi::minimum<double>());
        boost::mpi::all_reduce(
            world, vdmax, g_vdmax, boost::mpi::maximum<double>());

        if (rank == world.size() - 1)
        {
            const char* nm[5] = {"velocity      ", "pressure      ",
                "vorticity-fine", "vorticity-crse", "IB-force      "};
            std::cout << "\n[audit] assembled Jacobian structure  (N = " << N_
                      << ")\n"
                      << "[audit]  block           rows      empty   no-diag"
                         "   zero-diag\n";
            for (int b = 0; b < 5; ++b)
                std::cout << "[audit]  " << nm[b] << "  " << std::setw(9)
                          << g_tot[b] << "  " << std::setw(9) << g_empty[b]
                          << "  " << std::setw(9) << g_nodiag[b] << "  "
                          << std::setw(9) << g_zdiag[b] << "\n";
            long te = g_empty[0] + g_empty[1] + g_empty[2] + g_empty[3]
                      + g_empty[4];
            std::cout << "[audit]  => " << te
                      << " completely empty rows (structural null pivots)\n";
            std::cout << "[audit]  velocity |diag|: min " << g_vdmin << "  max "
                      << g_vdmax << "  ; rows with |diag|<1e-1: " << g_vsmall[2]
                      << "  <1e-4: " << g_vsmall[1] << "  <1e-8: " << g_vsmall[0]
                      << "\n"
                      << std::flush;
        }
        // per-rank: the worst velocity rows (smallest |diag|), with full row
        world.barrier();
        for (int rr = 1; rr < world.size(); ++rr)
        {
            if (rank == rr && !worst.empty())
            {
                std::sort(worst.begin(), worst.end());
                std::cout << "[audit] rank " << rr
                          << " weak velocity rows (|diag|, gidx, nnz, diag, "
                             "offdiag-L1):\n";
                int shown = 0;
                for (auto& w : worst)
                {
                    if (shown++ >= 15) break;
                    std::cout << "[audit]   |d|=" << std::get<0>(w)
                              << "  g=" << std::get<1>(w)
                              << "  nnz=" << std::get<2>(w)
                              << "  offL1=" << std::get<4>(w) << "\n";
                }
                std::cout << std::flush;
            }
            world.barrier();
        }
        // (row-content dump of the near-null corner rows removed -- confirmed
        //  they are correction-cell boundary velocities: -1 diagonal + a dense
        //  Biot-Savart LGF stencil on idx_w_g columns; their Schur complement
        //  is near-singular by ~1e-8 from truncation of the discrete K.)

        // per-rank sample indices (each rank prints its own)
        world.barrier();
        for (int rr = 1; rr < world.size(); ++rr)
        {
            if (rank == rr && (!ex_empty.empty() || !ex_nodiag.empty()))
            {
                std::cout << "[audit] rank " << rr << " local blocks: u<=" << bu
                          << " p<=" << bp << " wf<=" << bwf << " wc<=" << bwc
                          << " f<=" << nrow << " (global start " << start
                          << ")\n";
                if (!ex_empty.empty())
                {
                    std::cout << "[audit] rank " << rr << " empty-row gidx:";
                    for (int g : ex_empty) std::cout << ' ' << g;
                    std::cout << "\n";
                }
                if (!ex_nodiag.empty())
                {
                    std::cout << "[audit] rank " << rr << " no-diag-row gidx:";
                    for (int g : ex_nodiag) std::cout << ' ' << g;
                    std::cout << "\n";
                }
                std::cout << std::flush;
            }
            world.barrier();
        }
    }

    /** @brief Take the MUMPS null-pivot row list (0-based global indices) and
     *  report which DOF block each one lives in -- this is the actual null
     *  space of the assembled Jacobian that stalls the Newton solve. Gathers
     *  every rank's block boundaries so a global index can be mapped back to
     *  (rank, block). */
    void classify_null_pivots(PetscInt npiv_loc, const PetscInt* piv_loc)
    {
        boost::mpi::communicator world;
        const int rank    = world.rank();
        const bool is_srv = domain_->is_server();

        // this rank's row ownership + local block boundaries
        std::vector<int> me(6, 0);
        if (!is_srv)
        {
            me[0] = op_->num_start() - 1;       // 0-based first owned row
            me[1] = op_->Jac.numRow_loc();      // # owned rows
            me[2] = op_->blk_u_end_;
            me[3] = op_->blk_p_end_;
            me[4] = op_->blk_wf_end_;
            me[5] = op_->blk_wc_end_;
        }
        std::vector<std::vector<int>> all;
        boost::mpi::all_gather(world, me, all);

        // gather all null-pivot global indices onto the last rank
        std::vector<int> mine(piv_loc, piv_loc + npiv_loc);
        std::vector<std::vector<int>> gathered;
        boost::mpi::gather(world, mine, gathered, world.size() - 1);
        if (rank != world.size() - 1) return;

        std::vector<int> piv;
        for (auto& v : gathered) piv.insert(piv.end(), v.begin(), v.end());
        std::sort(piv.begin(), piv.end());
        piv.erase(std::unique(piv.begin(), piv.end()), piv.end());

        long cnt[6] = {0}; // 0 vel,1 p,2 wf,3 wc,4 force,5 unresolved
        std::vector<int> ex[5];
        for (int g : piv)
        {
            int b = 5;
            for (auto& r : all)
            {
                if (r[1] == 0) continue;
                if (g < r[0] || g >= r[0] + r[1]) continue;
                int lo = g - r[0] + 1; // 1-based local row
                b = (lo <= r[2])   ? 0
                    : (lo <= r[3]) ? 1
                    : (lo <= r[4]) ? 2
                    : (lo <= r[5]) ? 3
                                   : 4;
                break;
            }
            ++cnt[b];
            if (b < 5 && ex[b].size() < 20) ex[b].push_back(g);
        }

        const char* nm[6] = {"velocity", "pressure", "vorticity-fine",
            "vorticity-coarse", "IB-force", "UNRESOLVED"};
        std::cout << "[nullpiv] " << piv.size()
                  << " distinct MUMPS null-pivot rows, by DOF block:\n";
        for (int b = 0; b < 6; ++b)
            if (cnt[b])
            {
                std::cout << "[nullpiv]   " << nm[b] << " : " << cnt[b];
                if (b < 5 && !ex[b].empty())
                {
                    std::cout << "   e.g.";
                    for (int g : ex[b]) std::cout << ' ' << g;
                }
                std::cout << "\n";
            }
        std::cout << std::flush;
    }

    /** @brief Build a PETSc MPIAIJ matrix from op_->Jac (local 1-based rows,
     *  global 1-based columns). Mirrors tests/SLEPcTestLNS/SLEPcTest.hpp. */
    void assemble_jacobian(PetscInt N, PetscInt nlocal, Mat* A)
    {
        Vec sizing;
        PetscCallVoid(VecCreate(PETSC_COMM_WORLD, &sizing));
        PetscCallVoid(VecSetSizes(sizing, nlocal, N));
        PetscCallVoid(VecSetFromOptions(sizing));
        PetscInt rstart, rend;
        PetscCallVoid(VecGetOwnershipRange(sizing, &rstart, &rend));
        PetscCallVoid(VecDestroy(&sizing));

        std::vector<PetscInt> d_nnz(nlocal, 0), o_nnz(nlocal, 0);
        for (PetscInt i = rstart; i < rend; ++i)
        {
            PetscInt              i_loc = i - rstart;
            std::map<int, float_type>& row = op_->Jac.mat[i_loc + 1];
            PetscInt              nd = 0, no = 0;
            bool                  has_diag = false;
            for (const auto& [key, val] : row)
            {
                PetscInt col = key - 1; // 1-based global -> 0-based
                if (col >= rstart && col < rend) ++nd;
                else
                    ++no;
                if (col == i) has_diag = true;
            }
            d_nnz[i_loc] = nd + (has_diag ? 0 : 1);
            o_nnz[i_loc] = no;
        }

        PetscCallVoid(MatCreateAIJ(PETSC_COMM_WORLD, nlocal, nlocal, N, N, 0,
            nlocal ? d_nnz.data() : NULL, 0, nlocal ? o_nnz.data() : NULL, A));
        PetscCallVoid(MatSetFromOptions(*A));
        PetscCallVoid(MatSetUp(*A));

        for (PetscInt i = rstart; i < rend; ++i)
        {
            PetscInt              i_loc = i - rstart;
            std::map<int, float_type>& row = op_->Jac.mat[i_loc + 1];
            std::vector<PetscInt>    cols;
            std::vector<PetscScalar> vals;
            cols.reserve(row.size());
            vals.reserve(row.size());
            bool has_diag = false;
            for (const auto& [key, val] : row)
            {
                cols.push_back(key - 1);
                vals.push_back(val);
                if (key - 1 == i) has_diag = true;
            }
            if (!cols.empty())
                PetscCallVoid(MatSetValues(*A, 1, &i, (PetscInt)cols.size(),
                    cols.data(), vals.data(), INSERT_VALUES));
            if (!has_diag)
            {
                PetscScalar zero = 0.0;
                PetscCallVoid(
                    MatSetValues(*A, 1, &i, 1, &i, &zero, INSERT_VALUES));
            }
        }

        PetscCallVoid(MatAssemblyBegin(*A, MAT_FINAL_ASSEMBLY));
        PetscCallVoid(MatAssemblyEnd(*A, MAT_FINAL_ASSEMBLY));
    }

    /** @brief Cold-start initial guess.
     *  init_mode = "zero": stored u = 0 == uniform freestream (the Jacobian
     *      about this quiescent state is singular -- Newton cannot start here).
     *  init_mode = "potential": inviscid potential flow past the cylinder,
     *      minus the freestream (so `stored u` = disturbance). Smooth, O(1),
     *      no-penetration on the body -- a non-degenerate Newton seed. Freestream
     *      assumed +x (Uy is ignored for the potential part). */
    void initialize()
    {
        if (domain_->is_server()) return;
        const bool       pot = (init_mode_ == "potential");
        const float_type U = U_[0];
        const float_type R2 = R_ * R_;
        for (auto it = domain_->begin(); it != domain_->end(); ++it)
        {
            if (!it->locally_owned()) continue;
            const float_type dxl =
                domain_->dx_base() / math::pow2(it->refinement_level());
            for (auto& node : it->data())
            {
                for (std::size_t fi = 0; fi < w_type::nFields(); ++fi)
                    node(w_type::tag(), fi) = 0.0;
                node(p_type::tag(), 0) = 0.0;
                if (!pot)
                {
                    for (std::size_t fi = 0; fi < u_type::nFields(); ++fi)
                        node(u_type::tag(), fi) = 0.0;
                    continue;
                }
                const auto       c = node.level_coordinate();
                const float_type x = static_cast<float_type>(c[0]) * dxl;
                const float_type y = static_cast<float_type>(c[1]) * dxl;
                const float_type r2 = x * x + y * y;
                if (r2 <= R2)
                {
                    node(u_type::tag(), 0) = -U; // total vel 0 inside body
                    if (u_type::nFields() > 1) node(u_type::tag(), 1) = 0.0;
                }
                else
                {
                    const float_type r4 = r2 * r2;
                    // potential flow: ux_tot = U(1 - R2(x^2-y^2)/r^4),
                    //                 uy_tot = -2 U R2 x y / r^4 ; store minus U
                    node(u_type::tag(), 0) = -U * R2 * (x * x - y * y) / r4;
                    if (u_type::nFields() > 1)
                        node(u_type::tag(), 1) =
                            -2.0 * U * R2 * x * y / r4;
                }
            }
        }
    }

    /** @brief Refine to the finest level inside a physical box around the body
     *  and near wake. Consulted only on the cold-start (init_refine) path. */
    template<class OctantType>
    bool refinement(OctantType* it, int diff_level,
        bool use_all = false) const noexcept
    {
        if (diff_level <= 0) return false;
        auto b = it->data().descriptor();
        b.level() = it->refinement_level();
        const float_type dxl =
            domain_->dx_base() / std::pow(2.0, b.level());
        b.grow(2, 2);
        for (int i = b.base()[0]; i <= b.max()[0]; ++i)
            for (int j = b.base()[1]; j <= b.max()[1]; ++j)
            {
                const float_type x = static_cast<float_type>(i) * dxl;
                const float_type y = static_cast<float_type>(j) * dxl;
                if (x >= box_x0_ && x <= box_x1_ && y >= box_y0_ &&
                    y <= box_y1_)
                    return true;
            }
        return false;
    }

    std::vector<coordinate_t> initialize_domain(Dictionary* _d,
        domain_t*                                           _domain)
    {
        return _domain->construct_basemesh_blocks(_d, _domain->block_extent());
    }

    /** @brief Recirculation-bubble length: largest x on the centreline (y~0,
     *  x>R) where the lab-frame streamwise velocity u_x + U is <= 0. */
    void report_wake_length()
    {
        boost::mpi::communicator world;
        float_type x_reattach_loc = R_;
        if (!domain_->is_server())
        {
            const float_type dx_base = domain_->dx_base();
            for (int l = domain_->tree()->base_level();
                 l < domain_->tree()->depth(); ++l)
            {
                for (auto it = domain_->begin(l); it != domain_->end(l); ++it)
                {
                    if (!it->locally_owned() || !it->has_data()) continue;
                    if (!it->is_leaf() || it->is_correction()) continue;
                    const float_type dxl =
                        dx_base / math::pow2(it->refinement_level());
                    for (auto& n : it->data())
                    {
                        const auto       c = n.level_coordinate();
                        const float_type x = static_cast<float_type>(c[0]) * dxl;
                        const float_type y = static_cast<float_type>(c[1]) * dxl;
                        if (std::abs(y) > dxl) continue;
                        if (x <= R_) continue;
                        const float_type ux = n(u_type::tag(), 0) + U_[0];
                        if (ux <= 0.0 && x > x_reattach_loc) x_reattach_loc = x;
                    }
                }
            }
        }
        float_type x_reattach = R_;
        boost::mpi::all_reduce(world, x_reattach_loc, x_reattach,
            boost::mpi::maximum<float_type>());
        if (world.rank() == world.size() - 1)
            std::cout << "[diag] recirculation length  L_wake/D = "
                      << (x_reattach - R_) / (2.0 * R_)
                      << "   (x_reattach = " << x_reattach
                      << ", expect ~2.2 for Re=40)" << std::endl;
    }

    /** @brief Drag/lift from the converged IB force state. The overall
     *  constant may need one-time calibration against C_D ~ 1.5 (Re=40) via
     *  domain_->ib().force_scale(); secondary diagnostic. */
    void report_drag()
    {
        boost::mpi::communicator world;
        float_type Fx_loc = 0.0, Fy_loc = 0.0;
        for (std::size_t i = 0; i < domain_->ib().size(); ++i)
        {
            if (domain_->ib().rank(i) != world.rank()) continue;
            Fx_loc += forcing_state_[i][0];
            Fy_loc += forcing_state_[i][1];
        }
        float_type Fx = 0.0, Fy = 0.0;
        boost::mpi::all_reduce(world, Fx_loc, Fx, std::plus<float_type>());
        boost::mpi::all_reduce(world, Fy_loc, Fy, std::plus<float_type>());
        const float_type q = U_[0] * U_[0] * (2.0 * R_);
        if (world.rank() == world.size() - 1)
            std::cout << "[diag] sum IB force  Fx = " << Fx << "  Fy = " << Fy
                      << "   C_D ~ " << (q != 0.0 ? 2.0 * Fx / q : 0.0)
                      << "   C_L ~ " << (q != 0.0 ? 2.0 * Fy / q : 0.0)
                      << "  (uncalibrated; expect C_D ~ 1.5 for Re=40)"
                      << std::endl;
    }

  private:
    boost::mpi::communicator           client_comm_;
    std::shared_ptr<time_integration_t> op_;

    std::vector<float_type> U_;
    float_type              Re_ = 0.0;
    float_type              R_ = 0.0;
    float_type              dx_base_ = 0.0;
    int                     nLevelRefinement_ = 0;
    int                     global_refinement_ = 0;
    int                     Newton_max_itr_ = 100;
    float_type              Newton_threshold_ = 1e-6;

    // PTC state
    float_type              ptc_cfl0_ = 0.05;  // dtau_0 = ptc_cfl0 / max|J_vv|
    float_type              ptc_dtau_max_ = 1e8;
    float_type              ptc_growth_max_ = 8.0;
    float_type              ptc_switch_res_ = 1e-3;
    float_type              ptc_shift_ = 0.0;
    float_type              ptc_creg_ = 0.0;
    bool                    ptc_diag_I_ = false;
    float_type              ptc_pin_reg_ = 0.05;
    float_type              jvv_ref_ = 1.0;
    int                     ls_max_ = 8;
    float_type              ls_min_alpha_ = 1.0 / 256.0;
    float_type              newton_sign_ = -1.0;
    std::string             init_mode_ = "zero";
    int                     mumps_icntl_14_ = 400;
    int                     mumps_icntl_7_ = 5;
    int                     mumps_icntl_6_ = 7;
    int                     mumps_icntl_8_ = 77;
    int                     mumps_icntl_10_ = 5;
    int                     mumps_icntl_24_ = 1;
    float_type              mumps_cntl_3_ = -1e-13;
    int                     ir_ksp_its_ = 0;
    std::vector<float_type> re_ladder_;
    float_type              dtau_ = 0.0;
    float_type              last_alpha_ = 1.0;
    float_type              res0_ = -1.0;
    int                     reject_streak_ = 0;
    bool                    rung_converged_ = false;
    PetscInt                N_ = 0;

    float_type box_x0_ = -8.0, box_y0_ = -10.0;
    float_type box_x1_ = 40.0, box_y1_ = 10.0;

    force_type forcing_state_;
    force_type forcing_res_;
};

} // namespace iblgf

#endif
