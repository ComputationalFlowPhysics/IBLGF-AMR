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
// 2D matrix-free linear stability (EVP) solver.
//
// Reuses NewtonIteration<SetupNewton>::Jacobian(), the already-implemented
// matrix-free, IB-force-coupled linearized Navier-Stokes operator (the same
// operator the Newton fixed-point solver applies via its hand-rolled BiCGStab
// to find base flows). Here it is instead wrapped in a PETSc MatShell and fed
// to SLEPc's EPS to find the leading eigenvalues of the operator linearized
// about a base flow loaded from a converged NewtonIteration restart.
//
// State layout in the (complex) PETSc Vec, packed per-rank as contiguous
// local slots: [ velocity perturbation (idx_u) | pressure perturbation
// (idx_p) | immersed-boundary force perturbation (force_local_idx_) ].
// Only local slot numbers matter here (PETSc's own contiguous per-rank
// partitioning handles global numbering) -- unlike the explicit-CSR Jacobian
// in LinearNS.hpp, MatShell never needs true global row/col indices.

#ifndef IBLGF_INCLUDED_STABILITY_EVP_2D_HPP
#define IBLGF_INCLUDED_STABILITY_EVP_2D_HPP

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
        //base flow, loaded from a converged NewtonIteration restart
        (u     , float_type, 2,    1,       1,     face,true  ),
        (p     , float_type, 1,    1,       1,     cell,true  )
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

        // Jacobian()'s nonlinear_jac term reads simulation_.frame_vel()/
        // bc_vel() (background/frame velocity of the moving reference
        // frame) -- these must be set even though this tool never time
        // marches, or calling Jacobian() throws std::bad_function_call.
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
        {
            throw std::runtime_error(
                "StabilityEVP2D requires use_restart=true, loading a base "
                "flow converged by NewtonIteration.");
        }
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
        boost::mpi::communicator world;
        SlepcInitialize(&argc, &argv, (char*)0, NULL);

        solver_ = std::make_shared<time_integration_t>(&this->simulation_);

        init_idx_all();
        eig_solve();

        PetscFinalize();
        return 0.0;
    }

    /** @brief Assign contiguous 1-based local PETSc slots across the
     *         (velocity, pressure, IB-force) perturbation state, in that
     *         order. Only local uniqueness matters -- see file header. */
    void init_idx_all()
    {
        max_local_idx = -1;
        force_local_idx_.assign(domain_->ib().size() * Dim, -1);
        if (domain_->is_server()) return;

        solver_->template clean<idx_u_type>();
        solver_->template clean<idx_p_type>();

        int local_count = 0;
        int base_level = domain_->tree()->base_level();
        for (int l = base_level; l < domain_->tree()->depth(); l++)
        {
            for (auto it = domain_->begin(l); it != domain_->end(l); ++it)
            {
                if (!it->locally_owned() || !it->has_data()) continue;
                if (!it->is_leaf()) continue;
                for (std::size_t field_idx = 0;
                     field_idx < idx_u_type::nFields(); ++field_idx)
                {
                    for (auto& n : it->data())
                    {
                        local_count++;
                        n(idx_u_type::tag(), field_idx) = local_count;
                    }
                }
            }
        }
        for (int l = base_level; l < domain_->tree()->depth(); l++)
        {
            for (auto it = domain_->begin(l); it != domain_->end(l); ++it)
            {
                if (!it->locally_owned() || !it->has_data()) continue;
                if (!it->is_leaf()) continue;
                for (std::size_t field_idx = 0;
                     field_idx < idx_p_type::nFields(); ++field_idx)
                {
                    for (auto& n : it->data())
                    {
                        local_count++;
                        n(idx_p_type::tag(), field_idx) = local_count;
                    }
                }
            }
        }

        boost::mpi::communicator world;
        for (std::size_t i = 0; i < domain_->ib().size(); ++i)
        {
            if (domain_->ib().rank(i) != world.rank()) continue;
            for (std::size_t d = 0; d < Dim; ++d)
            {
                local_count++;
                force_local_idx_[i * Dim + d] = local_count;
            }
        }

        max_local_idx = local_count;
    }

    void eig_solve()
    {
        boost::mpi::communicator world;
        world.barrier();
        PetscMPIInt rank = world.rank();

        PetscInt m_local = domain_->is_server() ? 0 : max_local_idx;
        PetscInt M;
        boost::mpi::all_reduce(world, m_local, M, std::plus<int>());

        Vec  x;
        Mat  A;
        Vec* vic;
        PetscCallVoid(
            MatCreateShell(PETSC_COMM_WORLD, m_local, m_local, M, M, NULL, &A));
        NS_AMR_LGF* ctx = this;
        PetscCallVoid(MatShellSetContext(A, ctx));
        PetscCallVoid(
            MatShellSetOperation(A, MATOP_MULT, (void (*)(void))MatMult_Jacobian_access));

        Vec b;
        PetscCallVoid(MatCreateVecs(A, &x, &b));

        PetscCallVoid(VecDuplicateVecs(b, n_init_vecs_, &vic));
        for (int i = 0; i < n_init_vecs_; i++)
        {
            if (!domain_->is_server())
            {
                solver_->template clean<du_i_type>();
                this->template perturbIC<du_i_type>(1000 + i);
            }
            world.barrier();
            pack_real<idx_u_type, du_i_type>(vic[i], false);
            PetscCallVoid(VecAssemblyBegin(vic[i]));
            PetscCallVoid(VecAssemblyEnd(vic[i]));
        }
        simulation_.write("checkIC");

        EPS eps;
        PetscCallVoid(EPSCreate(PETSC_COMM_WORLD, &eps));
        PetscCallVoid(EPSSetOperators(eps, A, NULL));
        PetscCallVoid(EPSSetProblemType(eps, EPS_NHEP));
        PetscCallVoid(EPSSetWhichEigenpairs(eps, EPS_LARGEST_REAL));
        PetscCallVoid(EPSSetDimensions(eps, n_eigs_, PETSC_DECIDE, PETSC_DECIDE));
        PetscCallVoid(EPSSetInitialSpace(eps, n_init_vecs_, vic));

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
        PetscCallVoid(EPSSolve(eps));

        PetscInt nconv;
        PetscCallVoid(EPSGetConverged(eps, &nconv));
        world.barrier();

        PetscComplex eval;
        for (int i = 0; i < nconv; i++)
        {
            PetscCallVoid(EPSGetEigenpair(eps, i, &eval, NULL, x, NULL));
            if (rank == 0)
                std::cout << "Eigenvalue " << i << ": " << eval << std::endl;

            if (!domain_->is_server())
            {
                solver_->template clean<du_i_type>();
                solver_->template clean<dp_i_type>();
            }
            unpack_real<idx_u_type, du_i_type>(x, false);
            unpack_real<idx_p_type, dp_i_type>(x, false);
            world.barrier();
            simulation_.write("eigenvector_" + std::to_string(i));
        }

        PetscCallVoid(EPSDestroy(&eps));
        PetscCallVoid(VecDestroyVecs(n_init_vecs_, &vic));
        PetscCallVoid(VecDestroy(&x));
        PetscCallVoid(VecDestroy(&b));
        PetscCallVoid(MatDestroy(&A));
    }

    static PetscErrorCode MatMult_Jacobian_access(Mat A, Vec x, Vec y)
    {
        NS_AMR_LGF* ctx;
        PetscCall(MatShellGetContext(A, (void**)&ctx));
        ctx->MatMult_Jacobian_mf(A, x, y);
        return 0;
    }

    /** @brief y = J*x, J the linearized-NS operator about the loaded base
     *         flow (u_type/p_type), applied twice matrix-free -- once for
     *         the real part of x, once for the imaginary part -- since the
     *         operator itself is real-linear. */
    PetscErrorCode MatMult_Jacobian_mf(Mat A, Vec x, Vec y)
    {
        boost::mpi::communicator world;
        real_coordinate_type tmp_coord(0.0);

        for (int part = 0; part < 2; ++part)
        {
            bool imag_part = (part == 1);

            force_type force_in(domain_->ib().size(), tmp_coord);
            force_type force_out(domain_->ib().size(), tmp_coord);

            if (!domain_->is_server())
            {
                solver_->template clean<du_i_type>();
                solver_->template clean<dp_i_type>();
                solver_->template clean<fu_i_type>();
                solver_->template clean<fp_i_type>();
            }
            unpack_real<idx_u_type, du_i_type>(x, imag_part);
            unpack_real<idx_p_type, dp_i_type>(x, imag_part);
            unpack_force_real(x, force_in, imag_part);
            world.barrier();

            if (!domain_->is_server())
            {
                solver_->template Jacobian<du_i_type, dp_i_type, fu_i_type,
                    fp_i_type>(force_in, force_out);
            }
            world.barrier();

            pack_real<idx_u_type, fu_i_type>(y, imag_part);
            pack_real<idx_p_type, fp_i_type>(y, imag_part);
            pack_force_real(y, force_out, imag_part);
        }

        PetscCall(VecAssemblyBegin(y));
        PetscCall(VecAssemblyEnd(y));
        return 0;
    }

    /** @brief Read the real (imag_part=false) or imaginary (imag_part=true)
     *         part of complex Vec x's grid segment into real-valued Field. */
    template<class Field_idx, class Field>
    void unpack_real(Vec x, bool imag_part)
    {
        if (domain_->is_server()) return;
        const PetscComplex* x_array;
        PetscCallVoid(VecGetArrayRead(x, &x_array));
        int base_level = domain_->tree()->base_level();
        for (int l = base_level; l < domain_->tree()->depth(); l++)
        {
            for (auto it = domain_->begin(l); it != domain_->end(l); ++it)
            {
                if (!it->locally_owned() || !it->has_data()) continue;
                if (!it->is_leaf()) continue;
                for (std::size_t field_idx = 0; field_idx < Field::nFields();
                     ++field_idx)
                {
                    for (auto& n : it->data())
                    {
                        int i_local =
                            n(Field_idx::tag(), field_idx) - 1; //1-based
                        if (i_local < 0) continue;
                        PetscComplex value = x_array[i_local];
                        n(Field::tag(), field_idx) = imag_part
                            ? PetscImaginaryPart(value)
                            : PetscRealPart(value);
                    }
                }
            }
        }
        PetscCallVoid(VecRestoreArrayRead(x, &x_array));
    }

    /** @brief Write real-valued Field into the real or imaginary part of
     *         complex Vec y's grid segment (overwrite on the real pass,
     *         accumulate on the imaginary pass). */
    template<class Field_idx, class Field>
    void pack_real(Vec y, bool imag_part)
    {
        if (domain_->is_server()) return;
        PetscComplex* y_array;
        PetscCallVoid(VecGetArray(y, &y_array));
        int base_level = domain_->tree()->base_level();
        for (int l = base_level; l < domain_->tree()->depth(); l++)
        {
            for (auto it = domain_->begin(l); it != domain_->end(l); ++it)
            {
                if (!it->locally_owned() || !it->has_data()) continue;
                if (!it->is_leaf()) continue;
                for (std::size_t field_idx = 0; field_idx < Field::nFields();
                     ++field_idx)
                {
                    for (auto& n : it->data())
                    {
                        int i_local =
                            n(Field_idx::tag(), field_idx) - 1; //1-based
                        if (i_local < 0) continue;
                        float_type value = n(Field::tag(), field_idx);
                        if (imag_part) y_array[i_local] += value * PETSC_i;
                        else
                            y_array[i_local] = value;
                    }
                }
            }
        }
        PetscCallVoid(VecRestoreArray(y, &y_array));
    }

    void unpack_force_real(Vec x, force_type& f, bool imag_part)
    {
        boost::mpi::communicator world;
        const PetscComplex* x_array;
        PetscCallVoid(VecGetArrayRead(x, &x_array));
        for (std::size_t i = 0; i < domain_->ib().size(); ++i)
        {
            if (domain_->ib().rank(i) != world.rank()) continue;
            for (std::size_t d = 0; d < Dim; ++d)
            {
                int i_local = force_local_idx_[i * Dim + d] - 1; //1-based
                if (i_local < 0) continue;
                PetscComplex value = x_array[i_local];
                f[i][d] =
                    imag_part ? PetscImaginaryPart(value) : PetscRealPart(value);
            }
        }
        PetscCallVoid(VecRestoreArrayRead(x, &x_array));
    }

    void pack_force_real(Vec y, force_type& f, bool imag_part)
    {
        boost::mpi::communicator world;
        PetscComplex* y_array;
        PetscCallVoid(VecGetArray(y, &y_array));
        for (std::size_t i = 0; i < domain_->ib().size(); ++i)
        {
            if (domain_->ib().rank(i) != world.rank()) continue;
            for (std::size_t d = 0; d < Dim; ++d)
            {
                int i_local = force_local_idx_[i * Dim + d] - 1; //1-based
                if (i_local < 0) continue;
                float_type value = f[i][d];
                if (imag_part) y_array[i_local] += value * PETSC_i;
                else
                    y_array[i_local] = value;
            }
        }
        PetscCallVoid(VecRestoreArray(y, &y_array));
    }

    template<typename Source>
    void perturbIC(float_type offset = 0.0) noexcept
    {
        boost::mpi::communicator world;
        if (domain_->is_server()) return;
        int seed = 100;
        std::mt19937 gen(seed + world.rank() + offset);
        std::normal_distribution<float_type> dist(0.0, 1.0);

        for (auto it = domain_->begin(); it != domain_->end(); ++it)
        {
            if (!it->locally_owned()) continue;
            if (!it->is_leaf()) continue;

            for (auto& node : it->data())
            {
                for (std::size_t field_idx = 0; field_idx < Source::nFields();
                     ++field_idx)
                {
                    node(Source::tag(), field_idx) = dist(gen);
                }
            }
        }
    }

    /** @brief  Refinement condition for octants -- unused here, the base
     *          flow's tree is restored from the Newton restart. */
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
    boost::mpi::communicator client_comm_;
    std::shared_ptr<time_integration_t> solver_;

    std::vector<float_type> U_;
    float_type Re_;
    float_type R_;
    int        nLevelRefinement_ = 0;
    int        global_refinement_ = 0;
    int        n_eigs_ = 5;
    int        n_init_vecs_ = 20;

    int              max_local_idx = -1;
    std::vector<int> force_local_idx_;
};

} // namespace iblgf

#endif
