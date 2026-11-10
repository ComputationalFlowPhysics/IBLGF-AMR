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
// 2D linear stability (EVP) solver -- ASSEMBLED generalized eigenproblem.
//
// Option B counterpart of tests/StabilityEVP2D (matrix-free MatShell + EPS).
// Assembles the Jacobian A (NewtonIteration::construct_Jac_p +
// construct_Jac_from_Jac_p -> ::Jac) and the mass matrix B
// (::construction_B_matrix -> ::B, 1 on leaf-velocity DOFs, 0 elsewhere),
// copies both into PETSc MPIAIJ matrices, and solves the generalized problem
//
//        A q = lambda B q            (EPS_GNHEP, B singular)
//
// with a shift-and-invert spectral transform (STSINVERT, target near 0) whose
// inner solve is a direct MUMPS LU of (A - sigma B). Shift-invert both handles
// the singular B (infinite eigenvalues from the pressure/IB constraints are
// pushed to zero) and converges fast onto the rightmost (least stable) modes.
//
// Base flow u,p is read from a restart, same as StabilityEVP2D; the two tools
// are meant to be run on the identical restart and their spectra compared.

#ifndef IBLGF_INCLUDED_STABILITY_EVP_2D_ASM_HPP
#define IBLGF_INCLUDED_STABILITY_EVP_2D_ASM_HPP

#ifndef DEBUG_IFHERK
#define DEBUG_IFHERK
#endif
#define DEBUG_POISSON

#include "mpi.h"
#include <slepceps.h>
#include <slepcsys.h>
#include <iostream>
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

#include <iblgf/solver/Newton/spmat_petsc.hpp>
#include <iblgf/solver/modal_analysis/field_vec.hpp>

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
        (test  , float_type, 1,    1,       1,     cell,false ),
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
        n_eigs_ = simulation_.dictionary()->template get_or<int>("n_eigs", 5);
        n_init_vecs_ =
            simulation_.dictionary()->template get_or<int>("n_init_vecs", 20);
        sigma_ = simulation_.dictionary()->template get_or<float_type>(
            "eps_shift", 0.0);
        mumps_icntl_14_ = simulation_.dictionary()->template get_or<int>(
            "mumps_icntl_14", 2000);

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
        if (!use_restart())
            throw std::runtime_error(
                "StabilityEVP2Dasm requires use_restart=true (base flow).");
        if (world.rank() == 0)
            std::cout << "reading base flow (u, p) from restart" << std::endl;
        simulation_.template read_h5<u_type>(
            simulation_.restart_field_dir(), "u");
        simulation_.template read_h5<p_type>(
            simulation_.restart_field_dir(), "p");

        if (world.rank() == 0)
            std::cout << "on Simulation: \n" << simulation_ << std::endl;
    }

    float_type run(int argc, char* argv[])
    {
        SlepcInitialize(&argc, &argv, (char*)0, NULL);
        op_ = std::make_shared<time_integration_t>(&this->simulation_);
        eig_solve();
        SlepcFinalize();
        return 0.0;
    }

    void eig_solve()
    {
        boost::mpi::communicator world;
        PetscMPIInt rank = world.rank();
        auto        t0 = std::chrono::high_resolution_clock::now();

        // --- assemble A (Jacobian) and B (mass matrix) ------------------
        op_->Assigning_idx();
        if (!domain_->is_server())
            op_->template pad_velocity<u_type, u_type>(true);
        world.barrier();
        op_->construct_Jac_p();
        op_->template construct_Jac_from_Jac_p<u_type>();
        op_->Jac.clean_entry(1e-10);
        op_->construction_B_matrix();
        op_->B.clean_entry(1e-14);

        PetscInt N = op_->total_dim();
        PetscInt nlocal = domain_->is_server() ? 0 : op_->Jac.numRow_loc();

        Mat A, B;
        PetscInt rstart, rend;
        PetscCallVoid(solver::spmat_petsc::ownership(N, nlocal, &rstart, &rend));
        PetscCallVoid(solver::spmat_petsc::assemble(op_->Jac, N, nlocal, rstart, rend, true, &A));
        PetscCallVoid(solver::spmat_petsc::assemble(op_->B, N, nlocal, rstart, rend, false, &B));

        auto t_asm = std::chrono::high_resolution_clock::now();
        if (rank == world.size() - 1)
            std::cout << "[timing] assemble A,B : "
                      << std::chrono::duration<double>(t_asm - t0).count()
                      << " s" << std::endl;

        // --- generalized EPS with shift-invert / MUMPS -----------------
        EPS eps;
        ST  st;
        KSP ksp;
        PC  pc;
        PetscCallVoid(EPSCreate(PETSC_COMM_WORLD, &eps));
        PetscCallVoid(EPSSetOperators(eps, A, B));
        PetscCallVoid(EPSSetProblemType(eps, EPS_GNHEP));
        PetscCallVoid(EPSSetDimensions(eps, n_eigs_, PETSC_DECIDE, PETSC_DECIDE));
        PetscCallVoid(EPSSetTarget(eps, sigma_));
        PetscCallVoid(EPSSetWhichEigenpairs(eps, EPS_TARGET_REAL));

        PetscCallVoid(EPSGetST(eps, &st));
        PetscCallVoid(STSetType(st, STSINVERT));
        PetscCallVoid(STGetKSP(st, &ksp));
        PetscCallVoid(KSPSetType(ksp, KSPPREONLY));
        PetscCallVoid(KSPGetPC(ksp, &pc));
        PetscCallVoid(PCSetType(pc, PCLU));
#if defined(PETSC_HAVE_MUMPS)
        PetscCallVoid(PCFactorSetMatSolverType(pc, MATSOLVERMUMPS));
        // The assembled (A - sigma B) has heavy fill from the LGF
        // velocity-boundary block; MUMPS' default working-space estimate is
        // far too small (aborts INFOG(1)=-9). Push ICNTL(14) (% workspace
        // increase) and ICNTL(24) (null-pivot detection) into the option DB
        // with the ST's "-st_" prefix, before EPSSetFromOptions/EPSSetUp so
        // they reach the factor matrix. A CLI -st_mat_mumps_icntl_* overrides.
        {
            char buf[32];
            std::snprintf(buf, sizeof(buf), "%d", mumps_icntl_14_);
            PetscCallVoid(PetscOptionsSetValue(
                NULL, "-st_mat_mumps_icntl_14", buf));
            PetscCallVoid(
                PetscOptionsSetValue(NULL, "-st_mat_mumps_icntl_24", "1"));
        }
#endif

        PetscViewer           viewer = NULL;
        PetscViewerAndFormat* vf = NULL;
        if (rank == 0)
        {
            PetscCallVoid(
                PetscViewerASCIIOpen(PETSC_COMM_SELF, "eig_conv.txt", &viewer));
            PetscCallVoid(PetscViewerAndFormatCreate(
                viewer, PETSC_VIEWER_ASCII_DENSE, &vf));
            PetscCallVoid(EPSMonitorSet(eps,
                (PetscErrorCode(*)(EPS, PetscInt, PetscInt, PetscScalar*,
                    PetscScalar*, PetscReal*, PetscInt, void*))EPSMonitorAll,
                (void*)vf, NULL));
        }
        PetscCallVoid(EPSSetFromOptions(eps));
        PetscCallVoid(EPSSolve(eps));

        auto t_solve = std::chrono::high_resolution_clock::now();
        if (rank == world.size() - 1)
            std::cout << "[timing] EPSSolve     : "
                      << std::chrono::duration<double>(t_solve - t_asm).count()
                      << " s" << std::endl;

        PetscInt nconv;
        PetscCallVoid(EPSGetConverged(eps, &nconv));

        // gather each eigenvector to every rank for unpacking by global idx
        VecScatter scat;
        Vec        xr, xseq;
        PetscCallVoid(MatCreateVecs(A, &xr, NULL));
        PetscCallVoid(VecScatterCreateToAll(xr, &scat, &xseq));

        PetscComplex eval;
        for (PetscInt i = 0; i < nconv; i++)
        {
            PetscCallVoid(EPSGetEigenpair(eps, i, &eval, NULL, xr, NULL));
            if (rank == 0)
                std::cout << "Eigenvalue " << i << ": " << PetscRealPart(eval)
                          << (PetscImaginaryPart(eval) < 0 ? " - " : " + ")
                          << std::abs(PetscImaginaryPart(eval)) << "i"
                          << std::endl;

            PetscCallVoid(VecScatterBegin(
                scat, xr, xseq, INSERT_VALUES, SCATTER_FORWARD));
            PetscCallVoid(
                VecScatterEnd(scat, xr, xseq, INSERT_VALUES, SCATTER_FORWARD));
            if (!domain_->is_server())
            {
                op_->template clean<du_i_type>();
                op_->template clean<dp_i_type>();
                PetscCallVoid((solver::field_vec::vec2grid<idx_u_g_type, du_i_type>(domain_.get(), xseq)));
                PetscCallVoid((solver::field_vec::vec2grid<idx_p_g_type, dp_i_type>(domain_.get(), xseq)));
            }
            world.barrier();
            simulation_.write("eigenvector_" + std::to_string(i));
        }

        PetscCallVoid(VecScatterDestroy(&scat));
        PetscCallVoid(VecDestroy(&xr));
        PetscCallVoid(VecDestroy(&xseq));
        PetscCallVoid(EPSDestroy(&eps));
        PetscCallVoid(MatDestroy(&A));
        PetscCallVoid(MatDestroy(&B));
    }


    template<class OctantType>
    bool refinement(OctantType* it, int diff_level,
        bool use_all = false) const noexcept
    {
        return false;
    }

    std::vector<coordinate_t> initialize_domain(Dictionary* _d,
        domain_t*                                           _domain)
    {
        return _domain->construct_basemesh_blocks(_d, _domain->block_extent());
    }

  private:
    boost::mpi::communicator           client_comm_;
    std::shared_ptr<time_integration_t> op_;

    std::vector<float_type> U_;
    float_type              Re_ = 0.0;
    float_type              R_ = 0.0;
    int                     nLevelRefinement_ = 0;
    int                     global_refinement_ = 0;
    int                     n_eigs_ = 5;
    int                     n_init_vecs_ = 20;
    float_type              sigma_ = 0.0;
    int                     mumps_icntl_14_ = 2000;
};

} // namespace iblgf

#endif
