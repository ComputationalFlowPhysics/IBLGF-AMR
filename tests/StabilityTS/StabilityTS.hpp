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
// Matrix-free linear stability via time-stepper Arnoldi, 2D and 3D: one
// driver, the dimension set by TS_DIM (StabilityTS2D.cpp / StabilityTS3D.cpp
// define it before including this header). Without a restart it builds the
// mesh itself with u_base = 0.
//
// Instead of the eigenvalues of the linearized operator J (StabilityEVP2D,
// which applies J directly and has to treat the pressure / IB-force
// constraint rows as if they had a time derivative), this finds eigenvalues
// mu of the propagator P = exp(J T): P q = u'(T), the linearized IF-HERK
// (solver::Ifherk_linear, the same scheme as the nonlinear DNS, linearized
// about u_base) advanced n_steps = T/dt from u'(0) = q. Then
//   lambda = log(mu) / T.
// The time stepper enforces div u' = 0 and u' = 0 on the immersed boundary
// itself, so the state is velocity only and there are no spurious
// constraint eigenvalues; the stiff damped spectrum maps to |mu| ~ 0 and the
// physical modes to the largest |mu| (EPS_LARGEST_MAGNITUDE).
//
// Base flow: u of a restart (read into u_base). Perturbation state: u on
// locally owned leaf, non-correction blocks, packed into contiguous local
// PETSc slots (idx_u). Perturbation boundary conditions: bc_vel = 0
// (homogeneous no-slip for u') while frame_vel = -U, so the linearized
// advection uses the full base velocity u_base + U.
//
// Frequency aliasing: log(mu) is only unique up to 2*pi*i/T, so pick
// T < pi / |omega_max| for the modes of interest.

#ifndef IBLGF_INCLUDED_STABILITY_TS_HPP
#define IBLGF_INCLUDED_STABILITY_TS_HPP

#ifndef DEBUG_IFHERK
#define DEBUG_IFHERK
#endif
#define DEBUG_POISSON

#include "mpi.h"
#include <slepceps.h>
#include <slepcsys.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <complex>
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
#include <iblgf/solver/time_integration/ifherk_linear.hpp>

#include <iblgf/solver/modal_analysis/field_vec.hpp>

#include "../../setups/setup_linear.hpp"
#include <iblgf/operators/operators.hpp>

namespace iblgf
{
#ifndef TS_DIM
#error "define TS_DIM (2 or 3) before including StabilityTS.hpp"
#endif
const int Dim = TS_DIM;

struct parameters
{
    static constexpr std::size_t Nf = 2; // Ifherk_linear forcing bins (unused)
    static constexpr std::size_t Dim = TS_DIM;
    // clang-format off
    REGISTER_FIELDS
    (
    Dim,
     (
        //name            type        Dim     lBuffer hBuffer storage type
        (test           , float_type, 1,      1,      1,      cell,false ),
        // perturbation velocity / pressure (the IF-HERK state)
        (u              , float_type, Dim,    1,      1,      face,true  ),
        (p              , float_type, 1,      1,      1,      cell,true  ),
        // base flow the operator is linearized about
        (u_base         , float_type, Dim,    1,      1,      face,true  ),
        (p_base         , float_type, 1,      1,      1,      cell,true  ),
        // Ifherk_linear work fields
        (nonlinear_tmp  , float_type, Dim,    1,      1,      face,true  ),
        (face_aux_tmp   , float_type, Dim,    1,      1,      face,true  ),
        (edge_aux2      , float_type, (Dim*2-3), 1,   1,      edge,true  ),
        (f_hat_re       , float_type, Dim*Nf, 1,      1,      face,false ),
        (f_hat_im       , float_type, Dim*Nf, 1,      1,      face,false ),
        (u_hat_re       , float_type, Dim*Nf, 1,      1,      face,false ),
        (u_hat_im       , float_type, Dim*Nf, 1,      1,      face,false ),
        // PETSc local slot of each velocity DOF (1-based, 0 = not a DOF)
        (idx_u          , float_type, Dim,    1,      1,      face,false ),
        // output: real part of the eigenvector (same names as the
        // StabilityEVP2D/2Dasm outputs so the plotting scripts carry over)
        (du_i           , float_type, Dim,    1,      1,      face,true  )
    ))
    // clang-format on
};

struct NS_AMR_LGF : public SetupLinear<NS_AMR_LGF, parameters>
{
    using super_type = SetupLinear<NS_AMR_LGF, parameters>;
    using simulation_type = typename super_type::simulation_t;
    using domain_type = typename simulation_type::domain_type;

    NS_AMR_LGF(Dictionary* _d)
    : super_type(_d, [this](auto _d, auto _domain)
          { return this->initialize_domain(_d, _domain); })
    {
        if (domain_->is_client()) client_comm_ = client_comm_.split(1);
        else
            client_comm_ = client_comm_.split(0);

        auto dict = simulation_.dictionary();
        Re_ = dict->template get<float_type>("Re");

        U_.resize(domain_->dimension());
        U_[0] = dict->template get_or<float_type>("Ux", 1.0);
        U_[1] = dict->template get_or<float_type>("Uy", 0.0);
        if (domain_->dimension() > 2)
            U_[2] = dict->template get_or<float_type>("Uz", 0.0);

        // Linearized advection uses (u_base - frame_vel) = u_base + U.
        simulation_.frame_vel() =
            [this](std::size_t idx, float_type t, auto coord = {0, 0, 0})
        { return -U_[idx]; };
        // The IB solve drives (u - bc_vel) to zero on the body. The
        // perturbation obeys homogeneous no-slip, so bc_vel must be 0 --
        // anything else makes the propagator affine, not linear.
        simulation_.bc_vel() =
            [](std::size_t idx, float_type t, auto coord) { return 0.0; };

        nLevelRefinement_ = dict->template get_or<int>("nLevels", 0);
        n_eigs_ = dict->template get_or<int>("n_eigs", 6);
        T_arnoldi_ = dict->template get_or<float_type>("T_arnoldi", 2.0);
        // Optional Krylov seeds (e.g. modes from a coarser mesh interpolated
        // onto this one): seed_dir/seed_<i>.hdf5, restart-format files on
        // this run's tree whose "u" holds the seed velocity.
        n_seeds_ = dict->template get_or<int>("n_seeds", 0);
        seed_dir_ = dict->template get_or<std::string>("seed_dir", "seeds");

        domain_->ib().init(_d->get_dictionary("simulation_parameters"),
            domain_->dx_base(), nLevelRefinement_, Re_);

        boost::mpi::communicator world;
        if (use_restart())
        {
            domain_->restart_list_construct();
            domain_->distribute<fmm_mask_builder_t, fmm_mask_builder_t>();
            if (world.rank() == 0)
                std::cout << "reading base flow u -> u_base from restart"
                          << std::endl;
            simulation_.template read_h5<u_base_type>(
                simulation_.restart_field_dir(), "u");
        }
        else
        {
            // no restart: build the mesh here (IB blocks refined to nLevels)
            // with u_base = 0 -- the linearized operator is then the Oseen
            // operator about the uniform stream U (frame_vel) with the IB
            domain_->init_refine(nLevelRefinement_, 0, 0);
            domain_->distribute<fmm_mask_builder_t, fmm_mask_builder_t>();
            if (world.rank() == 0)
                std::cout << "no restart: u_base = 0" << std::endl;
        }

        // time step: same rule as Ifherk_linear (dt_base / 2^levels)
        float_type dt = dict->template get_or<float_type>("dt", -1.0);
        if (dt < 0)
            dt = domain_->dx_base() *
                 dict->template get_or<float_type>("cfl", 0.2);
        int levels =
            domain_->tree()->depth() - domain_->tree()->base_level() - 1;
        dt_ = dt / std::pow(2.0, levels);
        n_steps_ = std::max(1, int(std::lround(T_arnoldi_ / dt_)));
        T_arnoldi_ = n_steps_ * dt_; // the T actually integrated

        if (world.rank() == 0)
            std::cout << "on Simulation: \n" << simulation_ << std::endl;
    }

    float_type run(int argc, char* argv[])
    {
        SlepcInitialize(&argc, &argv, (char*)0, NULL);
        ifherk_ = std::make_shared<time_integration_t>(&this->simulation_);
        ifherk_->init_single_step();
        init_idx();
        eig_solve();
        SlepcFinalize();
        return 0.0;
    }

    /** @brief Number the velocity DOFs on locally owned leaf,
     *         non-correction blocks with contiguous 1-based local slots. */
    void init_idx()
    {
        n_local_ = solver::field_vec::assign_leaf_idx<idx_u_type>(domain_.get(), false);
    }

    template<class It>
    bool is_dof_block(const It& it) const
    {
        return it->locally_owned() && it->has_data() && it->is_leaf() &&
               !it->is_correction();
    }

    void eig_solve()
    {
        boost::mpi::communicator world;
        PetscMPIInt              rank = world.rank();

        PetscInt m_local = n_local_;
        PetscInt M;
        boost::mpi::all_reduce(world, m_local, M, std::plus<PetscInt>());
        if (rank == 0)
            std::cout << "[StabilityTS2D] velocity DOFs = " << M
                      << ", dt = " << dt_ << ", steps per matvec = "
                      << n_steps_ << ", T = " << T_arnoldi_ << std::endl;

        Mat A;
        PetscCallVoid(
            MatCreateShell(PETSC_COMM_WORLD, m_local, m_local, M, M, this, &A));
        PetscCallVoid(MatShellSetOperation(
            A, MATOP_MULT, (void (*)(void))MatMult_propagator_access));

        Vec x, y;
        PetscCallVoid(MatCreateVecs(A, &x, &y));

        // Initial space: the seed files if given; otherwise random u pushed
        // through one propagation, so the Krylov space starts solenoidal and
        // satisfying the IB constraint.
        int  n_init = std::max(1, n_seeds_);
        Vec* init;
        PetscCallVoid(VecDuplicateVecs(x, n_init, &init));
        if (n_seeds_ > 0)
        {
            for (int i = 0; i < n_seeds_; ++i)
            {
                std::string f =
                    seed_dir_ + "/seed_" + std::to_string(i) + ".hdf5";
                if (rank == 0)
                    std::cout << "[StabilityTS2D] seed " << i << " <- " << f
                              << std::endl;
                clean_all<u_type>();
                simulation_.template read_h5<u_type>(f, "u");
                world.barrier();
                pack<u_type>(init[i], false);
                PetscCallVoid(VecAssemblyBegin(init[i]));
                PetscCallVoid(VecAssemblyEnd(init[i]));
                PetscReal nrm;
                PetscCallVoid(VecNorm(init[i], NORM_2, &nrm));
                PetscCallVoid(VecScale(init[i], 1.0 / nrm));
            }
        }
        else
        {
            clean_all<u_type>();
            perturb<u_type>();
            world.barrier();
            pack<u_type>(x, false);
            PetscCallVoid(VecAssemblyBegin(x));
            PetscCallVoid(VecAssemblyEnd(x));
            PetscCallVoid(MatMult(A, x, init[0]));
            PetscReal ynorm;
            PetscCallVoid(VecNorm(init[0], NORM_2, &ynorm));
            PetscCallVoid(VecScale(init[0], 1.0 / ynorm));
        }

        EPS eps;
        PetscCallVoid(EPSCreate(PETSC_COMM_WORLD, &eps));
        PetscCallVoid(EPSSetOperators(eps, A, NULL));
        PetscCallVoid(EPSSetProblemType(eps, EPS_NHEP));
        PetscCallVoid(EPSSetWhichEigenpairs(eps, EPS_LARGEST_MAGNITUDE));
        PetscCallVoid(
            EPSSetDimensions(eps, n_eigs_, PETSC_DECIDE, PETSC_DECIDE));
        PetscCallVoid(EPSSetInitialSpace(eps, n_init, init));

        PetscViewer           viewer = NULL;
        PetscViewerAndFormat* viewer_format = NULL;
        if (rank == 0)
        {
            PetscCallVoid(
                PetscViewerASCIIOpen(PETSC_COMM_SELF, "eig_conv.txt", &viewer));
            PetscCallVoid(PetscViewerAndFormatCreate(viewer,
                PETSC_VIEWER_ASCII_DENSE, &viewer_format));
            PetscCallVoid(EPSMonitorSet(eps,
                (PetscErrorCode(*)(EPS, PetscInt, PetscInt, PetscScalar*,
                    PetscScalar*, PetscReal*, PetscInt,
                    void*))EPSMonitorAll,
                (void*)viewer_format, NULL));
        }
        PetscCallVoid(EPSSetFromOptions(eps));

        auto t0 = std::chrono::high_resolution_clock::now();
        PetscCallVoid(EPSSolve(eps));
        auto t1 = std::chrono::high_resolution_clock::now();

        PetscInt nconv;
        PetscCallVoid(EPSGetConverged(eps, &nconv));
        if (rank == 0)
            std::cout << "[timing] EPSSolve     : "
                      << std::chrono::duration<double>(t1 - t0).count()
                      << " s, matvecs = " << n_matvec_ << std::endl;

        std::ofstream evals;
        if (rank == 0)
        {
            evals.open("eigenvalues.txt");
            evals << "# i  Re(mu)  Im(mu)  |mu|  Re(lambda)  Im(lambda)  "
                     "rel_error   (lambda = log(mu)/T, T = "
                  << T_arnoldi_ << ")\n";
        }
        PetscScalar mu;
        for (PetscInt i = 0; i < nconv; i++)
        {
            PetscCallVoid(EPSGetEigenpair(eps, i, &mu, NULL, x, NULL));
            PetscReal err;
            PetscCallVoid(EPSComputeError(eps, i, EPS_ERROR_RELATIVE, &err));
            std::complex<double> m(PetscRealPart(mu), PetscImaginaryPart(mu));
            std::complex<double> lam = std::log(m) / double(T_arnoldi_);
            if (rank == 0)
            {
                // same format as StabilityEVP2Dasm so scripts parse both
                std::cout << "Eigenvalue " << i << ": " << lam.real()
                          << (lam.imag() < 0 ? " - " : " + ")
                          << std::abs(lam.imag()) << "i" << std::endl;
                evals << std::setprecision(10) << i << " " << m.real() << " "
                      << m.imag() << " " << std::abs(m) << " " << lam.real()
                      << " " << lam.imag() << " " << err << "\n";
            }

            clean_all<du_i_type>();
            unpack<du_i_type>(x, false);
            world.barrier();
            simulation_.write("eigenvector_" + std::to_string(i));
        }

        PetscCallVoid(EPSDestroy(&eps));
        PetscCallVoid(VecDestroyVecs(n_init, &init));
        PetscCallVoid(VecDestroy(&x));
        PetscCallVoid(VecDestroy(&y));
        PetscCallVoid(MatDestroy(&A));
    }

    static PetscErrorCode MatMult_propagator_access(Mat A, Vec x, Vec y)
    {
        NS_AMR_LGF* ctx;
        PetscCall(MatShellGetContext(A, (void**)&ctx));
        return ctx->MatMult_propagator(x, y);
    }

    /** @brief y = P x = u'(T) from u'(0) = x. P is real, so the real and
     *         imaginary parts of the complex x are propagated separately. */
    PetscErrorCode MatMult_propagator(Vec x, Vec y)
    {
        boost::mpi::communicator world;
        // Krylov vectors stay real until a restart brings in complex Ritz
        // vectors -- skip the (expensive) imaginary pass when it is zero.
        const PetscScalar* xa;
        PetscCall(VecGetArrayRead(x, &xa));
        float_type imag_max_local = 0.0, imag_max = 0.0;
        for (PetscInt i = 0; i < n_local_; ++i)
            imag_max_local =
                std::max(imag_max_local, float_type(std::abs(PetscImaginaryPart(xa[i]))));
        PetscCall(VecRestoreArrayRead(x, &xa));
        boost::mpi::all_reduce(world, imag_max_local, imag_max,
            boost::mpi::maximum<float_type>());
        const int n_parts = imag_max > 0.0 ? 2 : 1;

        for (int part = 0; part < n_parts; ++part)
        {
            bool imag_part = (part == 1);
            clean_all<u_type>();
            clean_all<p_type>();
            unpack<u_type>(x, imag_part);
            world.barrier();
            ifherk_->time_step_once(n_steps_, true);
            world.barrier();
            pack<u_type>(y, imag_part);
        }
        PetscCall(VecAssemblyBegin(y));
        PetscCall(VecAssemblyEnd(y));
        ++n_matvec_;
        if (world.rank() == 0)
            std::cout << "[StabilityTS2D] matvec " << n_matvec_ << std::endl;
        return 0;
    }

    /** @brief Real (imag_part=false) or imaginary part of x -> Field. */
    template<class Field>
    void unpack(Vec x, bool imag_part)
    {
        PetscCallVoid((solver::field_vec::vec2grid<idx_u_type, Field>(domain_.get(), x,
            imag_part ? solver::field_vec::Part::imag : solver::field_vec::Part::real)));
    }

    /** @brief Field -> real part of y (overwrite) or imaginary part
     *         (accumulate). */
    template<class Field>
    void pack(Vec y, bool imag_part)
    {
        // real pass overwrites the DOF entries, imaginary pass adds i * value
        PetscCallVoid((solver::field_vec::grid2vec<idx_u_type, Field>(domain_.get(), y,
            imag_part ? solver::field_vec::Part::imag : solver::field_vec::Part::real,
            false)));
    }

    template<class Field>
    void clean_all()
    {
        for (auto it = domain_->begin(); it != domain_->end(); ++it)
        {
            if (!it->has_data() || !it->data().is_allocated()) continue;
            for (std::size_t f = 0; f < Field::nFields(); ++f)
            {
                auto& lin = it->data_r(Field::tag(), f).linalg_data();
                std::fill(lin.begin(), lin.end(), 0.0);
            }
        }
    }

    template<class Field>
    void perturb()
    {
        if (domain_->is_server()) return;
        boost::mpi::communicator             world;
        std::mt19937                         gen(100 + world.rank());
        std::normal_distribution<float_type> dist(0.0, 1.0);
        for (auto it = domain_->begin(); it != domain_->end(); ++it)
        {
            if (!is_dof_block(it)) continue;
            for (std::size_t f = 0; f < Field::nFields(); ++f)
                for (auto& n : it->data()) n(Field::tag(), f) = dist(gen);
        }
    }

    /** @brief Refinement condition -- unused, the tree comes from the
     *         restart. */
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
    boost::mpi::communicator            client_comm_;
    std::shared_ptr<time_integration_t> ifherk_;

    std::vector<float_type> U_;
    float_type              Re_;
    int                     nLevelRefinement_ = 0;
    int                     n_eigs_ = 6;
    float_type              T_arnoldi_ = 2.0;
    float_type              dt_ = 0.0;
    int                     n_steps_ = 1;
    int                     n_seeds_ = 0;
    std::string             seed_dir_ = "seeds";
    PetscInt                n_local_ = 0;
    int                     n_matvec_ = 0;
};

} // namespace iblgf

#endif
