// 3D linearized-NS assembly verification harness.
//
// Milestone-by-milestone port of NewtonMethod.hpp's assembled Jacobian to
// Dim=3 (see docs/3d_stability_scope.md). This test builds a small uniform 3D
// mesh and, for each ported block, checks the assembled sparse matrix against
// the corresponding applied operator from include/iblgf/operators/operators.hpp
// (which already has verified 3D branches).
//
// M1: Assigning_idx() DOF-block counts.
// M2: construction_laplacian_u() -- assembled L @ u vs Operator::laplace(u),
//     on a Gaussian test field, checked on interior (non-correction) leaf
//     nodes only (the correction-cell / AMR BC terms are M9-M11).

#ifndef IBLGF_INCLUDED_LNS3D_VERIFY_HPP
#define IBLGF_INCLUDED_LNS3D_VERIFY_HPP

#include <iostream>
#include <vector>
#include <cmath>
#include <petscksp.h>
#include <boost/mpi.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>

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
using namespace domain;
using namespace types;
using namespace dictionary;

const int Dim = 3;

struct parameters
{
    static constexpr std::size_t Dim = 3;
    // clang-format off
    REGISTER_FIELDS
    (
    Dim,
     (
        (test    , float_type, 1,           1, 1, cell,false),
        (u        , float_type, Dim,         1, 1, face,true ),
        (p        , float_type, 1,           1, 1, cell,true ),
        (w        , float_type, (Dim*2 - 3), 1, 1, edge,true ),
        (u_exact  , float_type, Dim,         1, 1, face,true ),
        (u_err    , float_type, Dim,         1, 1, face,true ),
        (fu_i     , float_type, Dim,         1, 1, face,true ),
        (fp_i     , float_type, 1,           1, 1, cell,true ),
        (div_exact, float_type, 1,           1, 1, cell,true ),
        (grad_exact,float_type, Dim,         1, 1, face,true ),
        (curl_exact,float_type, (Dim*2 - 3), 1, 1, edge,true ),
        // scratch fields for CSR2Grid write-back -- NEVER used as test input,
        // so writing an assembled-matrix result here can't clobber u/p/w.
        (scr_u    , float_type, Dim,         1, 1, face,true ),
        (scr_p    , float_type, 1,           1, 1, cell,true ),
        (scr_w    , float_type, (Dim*2 - 3), 1, 1, edge,true ),
        // M6 finite-difference check scratch
        (du_test  , float_type, Dim,         1, 1, face,true ),
        (u_pm     , float_type, Dim,         1, 1, face,true ),
        (edge_pm  , float_type, (Dim*2 - 3), 1, 1, edge,true ),
        (n_plus   , float_type, Dim,         1, 1, face,true ),
        (n_minus  , float_type, Dim,         1, 1, face,true ),
        (dn_fd    , float_type, Dim,         1, 1, face,true )
    ))
    // clang-format on
};

struct NS_AMR_LGF : public SetupNewton<NS_AMR_LGF, parameters>
{
    using super_type      = SetupNewton<NS_AMR_LGF, parameters>;
    using simulation_type = typename super_type::simulation_t;
    using domain_type     = typename simulation_type::domain_type;
    using time_integration_t = typename super_type::time_integration_t;
    using ib_t              = typename domain_type::ib_t;
    using force_type        = typename ib_t::force_type;
    using real_coordinate_type = typename domain_type::real_coordinate_type;

    NS_AMR_LGF(Dictionary* _d)
        : super_type(_d, [this](auto _d, auto _domain) {
            return this->initialize_domain(_d, _domain);
        })
    {
        boost::mpi::communicator world;
        if (domain_->is_client()) client_comm_ = client_comm_.split(1);
        else client_comm_ = client_comm_.split(0);

        Re_ = simulation_.dictionary()->template get_or<float_type>("Re", 40.0);
        R_  = simulation_.dictionary()->template get_or<float_type>("R", 0.5);
        nLevelRefinement_ = simulation_.dictionary_->template get_or<int>(
            "nLevels", 0);
        global_refinement_ = simulation_.dictionary_->template get_or<int>(
            "global_refinement", 0);

        simulation_.frame_vel() =
            [](std::size_t idx, float_type, auto = {0, 0, 0}) { return 0.0; };
        simulation_.bc_vel() =
            [](std::size_t idx, float_type, auto = {0, 0, 0}) { return 0.0; };

        domain_->register_refinement_condition() =
            [this](auto oct, int dl) { return this->refinement(oct, dl); };
        domain_->ib().init(_d->get_dictionary("simulation_parameters"),
            domain_->dx_base(), nLevelRefinement_, Re_);
        domain_->init_refine(nLevelRefinement_, global_refinement_, 0);
        domain_->distribute<fmm_mask_builder_t, fmm_mask_builder_t>();

        this->initialize();

        if (world.rank() == 0)
            std::cout << "on Simulation:\n" << simulation_ << std::endl;
    }

    float_type run(int argc, char* argv[])
    {
        PetscInitialize(&argc, &argv, (char*)0, NULL);
        boost::mpi::communicator world;

        auto op = std::make_shared<time_integration_t>(&this->simulation_);
        op->set_stage_time(0.0);

        // ---- M1: Assigning_idx -------------------------------------------
        if (world.rank() == 0)
            std::cout << "\n=== M1: Assigning_idx (Dim=3) ===" << std::endl;
        op->Assigning_idx();

        const bool srv = domain_->is_server();
        const long bu  = srv ? 0 : op->blk_u_end_;
        const long bp  = srv ? 0 : op->blk_p_end_;
        const long bwf = srv ? 0 : op->blk_wf_end_;
        const long bwc = srv ? 0 : op->blk_wc_end_;
        const long bf  = srv ? 0 : op->blk_f_end_;
        long g[5] = {0};
        long loc[5] = {bu, bp - bu, bwf - bp, bwc - bwf, bf - bwc};
        boost::mpi::all_reduce(world, loc, 5, g, std::plus<long>());
        if (world.rank() == world.size() - 1)
        {
            std::cout << "[dof] velocity        " << g[0] << "\n"
                      << "[dof] pressure        " << g[1] << "\n"
                      << "[dof] vorticity-fine  " << g[2] << "\n"
                      << "[dof] vorticity-coarse" << g[3] << "\n"
                      << "[dof] IB-force        " << g[4] << "\n"
                      << "[dof] TOTAL           " << (g[0]+g[1]+g[2]+g[3]+g[4])
                      << std::endl;
        }

        // ---- M2: construction_laplacian_u vs Operator::laplace -----------
        if (world.rank() == 0)
            std::cout << "\n=== M2: construction_laplacian_u (Dim=3) ==="
                       << std::endl;

        set_gaussian_test_field();
        // Operator::laplace reference (interior leaf, non-correction only)
        if (domain_->is_client())
        {
            const float_type dx_base = domain_->dx_base();
            auto client = domain_->decomposition().client();
            const int base_level = domain_->tree()->base_level();
            client->template buffer_exchange<u_type>(base_level);
            for (auto it = domain_->begin_leaves(); it != domain_->end_leaves();
                 ++it)
            {
                if (!it->locally_owned() || !it->has_data()) continue;
                if (it->is_correction()) continue;
                auto dx_level = dx_base / std::pow(2, it->refinement_level());
                domain::Operator::laplace<u_type, u_exact_type>(
                    it->data(), dx_level);
            }
        }

        op->construction_laplacian_u();

        PetscInt N = op->total_dim();
        PetscInt nlocal = srv ? 0 : op->L.numRow_loc();
        Mat A;
        assemble_from_spmat(op->L, N, nlocal, A);

        Vec x, y;
        MatCreateVecs(A, &x, &y);
        {
            std::vector<float_type> xloc(nlocal, 0.0);
            real_coordinate_type tmp_coord(0.0);
            force_type dummy_f(domain_->ib().size(), tmp_coord);
            if (!srv)
                op->template Grid2CSR<u_type, p_type, w_type>(
                    xloc.data(), dummy_f, false);
            PetscInt rstart, rend;
            VecGetOwnershipRange(x, &rstart, &rend);
            for (PetscInt i = rstart; i < rend; ++i)
            {
                PetscScalar v = xloc[i - rstart];
                VecSetValues(x, 1, &i, &v, INSERT_VALUES);
            }
            VecAssemblyBegin(x);
            VecAssemblyEnd(x);
        }
        MatMult(A, x, y);

        std::vector<float_type> yloc(nlocal, 0.0);
        {
            PetscInt rstart, rend;
            VecGetOwnershipRange(y, &rstart, &rend);
            for (PetscInt i = rstart; i < rend; ++i)
            {
                PetscScalar v;
                VecGetValues(y, 1, &i, &v);
                yloc[i - rstart] = PetscRealPart(v);
            }
        }
        if (!srv)
        {
            real_coordinate_type tmp_coord(0.0);
            force_type dummy_f(domain_->ib().size(), tmp_coord);
            // Write into the scratch fields, NOT u/p/w -- those still hold
            // the Gaussian test field and must survive for M3.
            op->template CSR2Grid<scr_u_type, scr_p_type, scr_w_type>(
                yloc.data(), dummy_f);
        }
        // compare: L@u_test (in scr_u) vs Re*u_exact (L includes the 1/Re
        // scale, Operator::laplace does not), interior leaf only.
        float_type l2 = 0.0, l2ex = 0.0, linf = 0.0;
        long count = 0;
        if (domain_->is_client())
        {
            for (auto it = domain_->begin_leaves(); it != domain_->end_leaves();
                 ++it)
            {
                if (!it->locally_owned() || !it->has_data()) continue;
                if (it->is_correction()) continue;
                for (auto& n : it->data())
                {
                    for (std::size_t d = 0; d < Dim; ++d)
                    {
                        float_type got = n(scr_u_type::tag(), d);
                        float_type exp = n(u_exact_type::tag(), d) / Re_;
                        float_type e = got - exp;
                        l2 += e * e;
                        l2ex += exp * exp;
                        linf = std::max(linf, std::abs(e));
                        ++count;
                    }
                }
            }
        }
        float_type gl2, gl2ex, glinf; long gcount;
        boost::mpi::all_reduce(world, l2, gl2, std::plus<float_type>());
        boost::mpi::all_reduce(world, l2ex, gl2ex, std::plus<float_type>());
        boost::mpi::all_reduce(world, linf, glinf, boost::mpi::maximum<float_type>());
        boost::mpi::all_reduce(world, count, gcount, std::plus<long>());
        if (world.rank() == world.size() - 1)
        {
            std::cout << "[M2] interior nodes checked: " << gcount << "\n"
                       << "[M2] rel L2 err  = " << std::sqrt(gl2 / std::max(gl2ex, 1e-300))
                       << "\n[M2] max abs err = " << glinf << std::endl;
        }

        // ---- M3: construction_Div / construction_Grad --------------------
        if (world.rank() == 0)
            std::cout << "\n=== M3: construction_Div / Grad (Dim=3) ==="
                       << std::endl;
        if (domain_->is_client())
        {
            const float_type dx_base = domain_->dx_base();
            auto client = domain_->decomposition().client();
            const int base_level = domain_->tree()->base_level();
            client->template buffer_exchange<u_type>(base_level);
            client->template buffer_exchange<p_type>(base_level);
            for (auto it = domain_->begin_leaves(); it != domain_->end_leaves();
                 ++it)
            {
                if (!it->locally_owned() || !it->has_data()) continue;
                if (it->is_correction()) continue;
                auto dx_level = dx_base / std::pow(2, it->refinement_level());
                domain::Operator::divergence<u_type, div_exact_type>(
                    it->data(), dx_level);
                domain::Operator::gradient<p_type, grad_exact_type>(
                    it->data(), dx_level);
            }
        }

        const long nrow = srv ? 0 : op->L.numRow_loc();
        std::vector<float_type> p_exact(nrow, 0.0), u_exact_g(nrow, 0.0);
        std::vector<char> p_mask(nrow, 0), u_mask(nrow, 0);
        if (domain_->is_client())
        {
            for (auto it = domain_->begin_leaves(); it != domain_->end_leaves();
                 ++it)
            {
                if (!it->locally_owned() || !it->has_data()) continue;
                if (it->is_correction()) continue;
                for (auto& n : it->data())
                {
                    int pi = (int)n(idx_p_type::tag(), 0);
                    if (pi >= 1 && pi <= nrow)
                    {
                        p_exact[pi - 1] = n(div_exact_type::tag(), 0);
                        p_mask[pi - 1] = 1;
                    }
                    for (std::size_t d = 0; d < Dim; ++d)
                    {
                        int ui = (int)n(idx_u_type::tag(), d);
                        if (ui >= 1 && ui <= nrow)
                        {
                            u_exact_g[ui - 1] = n(grad_exact_type::tag(), d);
                            u_mask[ui - 1] = 1;
                        }
                    }
                }
            }
        }
        op->construction_Div();
        op->construction_Grad();
        verify_block("M3-Div ", op->Div, *op, op->blk_u_end_ + 1,
            op->blk_p_end_, p_exact, p_mask);
        verify_block("M3-Grad", op->Grad, *op, 1, op->blk_u_end_, u_exact_g,
            u_mask);

        // ---- M4: construction_Curl ----------------------------------------
        if (world.rank() == 0)
            std::cout << "\n=== M4: construction_Curl (Dim=3) ===" << std::endl;
        if (domain_->is_client())
        {
            const float_type dx_base = domain_->dx_base();
            auto client = domain_->decomposition().client();
            const int base_level = domain_->tree()->base_level();
            client->template buffer_exchange<u_type>(base_level);
            for (auto it = domain_->begin_leaves(); it != domain_->end_leaves();
                 ++it)
            {
                if (!it->locally_owned() || !it->has_data()) continue;
                if (it->is_correction()) continue;
                auto dx_level = dx_base / std::pow(2, it->refinement_level());
                domain::Operator::curl<u_type, curl_exact_type>(
                    it->data(), dx_level);
            }
        }
        // w test field == 0 (never set), so Curl@x == Curl_factor * curl(u_test)
        std::vector<float_type> w_exact(nrow, 0.0);
        std::vector<char> w_mask(nrow, 0);
        if (domain_->is_client())
        {
            for (auto it = domain_->begin_leaves(); it != domain_->end_leaves();
                 ++it)
            {
                if (!it->locally_owned() || !it->has_data()) continue;
                if (it->is_correction()) continue;
                for (auto& n : it->data())
                    for (int c = 0; c < 3; ++c)
                    {
                        int wi = (int)n(idx_w_type::tag(), c);
                        if (wi >= 1 && wi <= nrow)
                        {
                            w_exact[wi - 1] = n(curl_exact_type::tag(), c);
                            w_mask[wi - 1] = 1;
                        }
                    }
            }
        }
        op->construction_Curl();
        verify_block("M4-Curl", op->Curl, *op, op->blk_p_end_ + 1,
            op->blk_wf_end_, w_exact, w_mask, op->get_curl_factor());

        // ---- M5: construction_B_matrix (already Dim-generic; sanity check) --
        if (world.rank() == 0)
            std::cout << "\n=== M5: construction_B_matrix (Dim=3) ==="
                       << std::endl;
        op->construction_B_matrix();
        // B == I on leaf-velocity rows -> B@u_test == u_test there.
        std::vector<float_type> u_id(nrow, 0.0);
        std::vector<char> u_id_mask(nrow, 0);
        if (domain_->is_client())
        {
            for (auto it = domain_->begin_leaves(); it != domain_->end_leaves();
                 ++it)
            {
                if (!it->locally_owned() || !it->has_data()) continue;
                if (it->is_correction()) continue;
                for (auto& n : it->data())
                    for (std::size_t d = 0; d < Dim; ++d)
                    {
                        int ui = (int)n(idx_u_type::tag(), d);
                        if (ui >= 1 && ui <= nrow)
                        {
                            u_id[ui - 1] = n(u_type::tag(), d);
                            u_id_mask[ui - 1] = 1;
                        }
                    }
            }
        }
        verify_block("M5-B   ", op->B, *op, 1, op->blk_u_end_, u_id,
            u_id_mask);

        // ---- M6: construction_DN_u (finite-difference check) -------------
        if (world.rank() == 0)
            std::cout << "\n=== M6: construction_DN_u (Dim=3, FD) ==="
                       << std::endl;
        run_M6_fd_check(*op);

        // ---- M7: construction_Smearing / Projection (adjoint check) ------
        // Both were already Dim-generic (no hardcoded 2D indices) -- verify
        // the IB delta-function weights are consistent: v^T S g == g^T P v
        // for random v (velocity), g (force), i.e. Smearing == Projection^T.
        if (world.rank() == 0)
            std::cout << "\n=== M7: Smearing/Projection adjoint (Dim=3) ==="
                       << std::endl;
        op->construction_Smearing();
        op->construction_Projection();
        run_M7_adjoint_check(*op);

        // ---- M8: pad_velocity (already Dim-generic; sanity check) --------
        if (world.rank() == 0)
            std::cout << "\n=== M8: pad_velocity (Dim=3) ===" << std::endl;
        run_M8_pad_velocity_check(*op);

        // ---- M9: construct_upward_intrp (partition-of-unity check) -------
        if (world.rank() == 0)
            std::cout << "\n=== M9: construct_upward_intrp (Dim=3) ==="
                       << std::endl;
        run_M9_upward_intrp_check(*op);

        // ---- M10: construction_BCMat (3D LGF Biot-Savart boundary block) --
        if (world.rank() == 0)
            std::cout << "\n=== M10: construction_BCMat (Dim=3) ==="
                       << std::endl;
        run_M10_bcmat_check(*op);

        MatDestroy(&A);
        VecDestroy(&x);
        VecDestroy(&y);
        PetscFinalize();
        return 0.0;
    }

    /** @brief Build a PETSc MPIAIJ matrix from a sparse_mat (local 1-based
     *  rows, global 1-based columns). Mirrors tests/StabilityEVP2Dasm. */
    void assemble_from_spmat(solver::sparse_mat& S, PetscInt N, PetscInt nlocal,
        Mat& M)
    {
        Vec sizing;
        VecCreate(PETSC_COMM_WORLD, &sizing);
        VecSetSizes(sizing, nlocal, N);
        VecSetFromOptions(sizing);
        PetscInt rstart, rend;
        VecGetOwnershipRange(sizing, &rstart, &rend);
        VecDestroy(&sizing);

        std::vector<PetscInt> d_nnz(nlocal, 0), o_nnz(nlocal, 0);
        for (PetscInt i = rstart; i < rend; ++i)
        {
            PetscInt i_loc = i - rstart;
            std::map<int, float_type>& row = S.mat[i_loc + 1];
            PetscInt nd = 0, no = 0;
            bool has_diag = false;
            for (const auto& [key, val] : row)
            {
                PetscInt col = key - 1;
                if (col >= rstart && col < rend) ++nd; else ++no;
                if (col == i) has_diag = true;
            }
            d_nnz[i_loc] = nd + (has_diag ? 0 : 1);
            o_nnz[i_loc] = no;
        }
        MatCreateAIJ(PETSC_COMM_WORLD, nlocal, nlocal, N, N, 0,
            nlocal ? d_nnz.data() : NULL, 0, nlocal ? o_nnz.data() : NULL, &M);
        MatSetFromOptions(M);
        MatSetUp(M);
        for (PetscInt i = rstart; i < rend; ++i)
        {
            PetscInt i_loc = i - rstart;
            std::map<int, float_type>& row = S.mat[i_loc + 1];
            std::vector<PetscInt> cols;
            std::vector<PetscScalar> vals;
            bool has_diag = false;
            for (const auto& [key, val] : row)
            {
                cols.push_back(key - 1);
                vals.push_back(val);
                if (key - 1 == i) has_diag = true;
            }
            if (!cols.empty())
                MatSetValues(M, 1, &i, (PetscInt)cols.size(), cols.data(),
                    vals.data(), INSERT_VALUES);
            if (!has_diag)
            {
                PetscScalar zero = 0.0;
                MatSetValues(M, 1, &i, 1, &i, &zero, INSERT_VALUES);
            }
        }
        MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY);
    }

    template<class OctantType>
    bool refinement(OctantType*, int, bool = false) const noexcept
    {
        return false;
    }

    /** @brief FD check: DN @ du_test  ~=  [N(u+eps*du) - N(u-eps*du)] / 2eps,
     *  N(v) = Operator::nonlinear(v, curl(v)), on interior leaf nodes. */
    void run_M6_fd_check(time_integration_t& op)
    {
        boost::mpi::communicator world;
        const bool srv = domain_->is_server();
        const float_type dx_base = domain_->dx_base();
        const int base_level = domain_->tree()->base_level();
        const float_type eps = 1e-4;

        // du_test: a second, differently-shaped smooth field.
        if (domain_->is_client())
        {
            auto center = (domain_->bounding_box().max() -
                              domain_->bounding_box().min() + 1) / 2.0 +
                domain_->bounding_box().min();
            for (auto it = domain_->begin_leaves(); it != domain_->end_leaves();
                 ++it)
            {
                if (!it->locally_owned() || !it->has_data()) continue;
                auto dx_level = dx_base / std::pow(2, it->refinement_level());
                auto scaling = std::pow(2, it->refinement_level());
                for (auto& node : it->data())
                {
                    const auto& coord = node.level_coordinate();
                    float_type xc = (coord[0]-center[0]*scaling+0.5)*dx_level;
                    float_type yc = (coord[1]-center[1]*scaling+0.5)*dx_level;
                    float_type zc = (coord[2]-center[2]*scaling+0.5)*dx_level;
                    float_type xf0=(coord[0]-center[0]*scaling)*dx_level, yf0=yc, zf0=zc;
                    float_type xf1=xc, yf1=(coord[1]-center[1]*scaling)*dx_level, zf1=zc;
                    float_type xf2=xc, yf2=yc, zf2=(coord[2]-center[2]*scaling)*dx_level;
                    const float_type b_ = 0.7; // different width -> independent of u_test
                    node(du_test_type::tag(), 0) = std::exp(-b_*((xf0-1)*(xf0-1)+yf0*yf0+zf0*zf0));
                    node(du_test_type::tag(), 1) = std::exp(-b_*(xf1*xf1+(yf1-1)*(yf1-1)+zf1*zf1));
                    node(du_test_type::tag(), 2) = std::exp(-b_*(xf2*xf2+yf2*yf2+(zf2-1)*(zf2-1)));
                }
            }
        }

        auto compute_N = [&](auto FaceVal, auto EdgeTmp, auto Dest) {
            using Face = decltype(FaceVal);
            using Edge = decltype(EdgeTmp);
            using Dst = decltype(Dest);
            if (!domain_->is_client()) return;
            auto client = domain_->decomposition().client();
            client->template buffer_exchange<typename Face::type>(base_level);
            for (auto it = domain_->begin_leaves(); it != domain_->end_leaves();
                 ++it)
            {
                if (!it->locally_owned() || !it->has_data()) continue;
                auto dx_level = dx_base / std::pow(2, it->refinement_level());
                domain::Operator::curl<typename Face::type,
                    typename Edge::type>(it->data(), dx_level);
            }
            client->template buffer_exchange<typename Edge::type>(base_level);
            for (auto it = domain_->begin_leaves(); it != domain_->end_leaves();
                 ++it)
            {
                if (!it->locally_owned() || !it->has_data()) continue;
                if (it->is_correction()) continue;
                domain::Operator::nonlinear<typename Face::type,
                    typename Edge::type, typename Dst::type>(it->data());
            }
        };
        // u_pm = u_type +/- eps*du_test ; curl into edge_pm ; nonlinear -> n_plus/n_minus
        if (domain_->is_client())
            for (auto it = domain_->begin_leaves(); it != domain_->end_leaves();
                 ++it)
            {
                if (!it->locally_owned() || !it->has_data()) continue;
                for (auto& n : it->data())
                    for (std::size_t d = 0; d < Dim; ++d)
                        n(u_pm_type::tag(), d) = n(u_type::tag(), d) +
                            eps * n(du_test_type::tag(), d);
            }
        compute_N(tmpl_tag<u_pm_type>{}, tmpl_tag<edge_pm_type>{},
            tmpl_tag<n_plus_type>{});
        if (domain_->is_client())
            for (auto it = domain_->begin_leaves(); it != domain_->end_leaves();
                 ++it)
            {
                if (!it->locally_owned() || !it->has_data()) continue;
                for (auto& n : it->data())
                    for (std::size_t d = 0; d < Dim; ++d)
                        n(u_pm_type::tag(), d) = n(u_type::tag(), d) -
                            eps * n(du_test_type::tag(), d);
            }
        compute_N(tmpl_tag<u_pm_type>{}, tmpl_tag<edge_pm_type>{},
            tmpl_tag<n_minus_type>{});

        long nrow = srv ? 0 : op.L.numRow_loc();
        std::vector<float_type> dn_exact(nrow, 0.0);
        std::vector<char> dn_mask(nrow, 0);
        if (domain_->is_client())
        {
            for (auto it = domain_->begin_leaves(); it != domain_->end_leaves();
                 ++it)
            {
                if (!it->locally_owned() || !it->has_data()) continue;
                if (it->is_correction()) continue;
                for (auto& n : it->data())
                    for (std::size_t d = 0; d < Dim; ++d)
                    {
                        int ui = (int)n(idx_u_type::tag(), d);
                        if (ui >= 1 && ui <= nrow)
                        {
                            dn_exact[ui - 1] = (n(n_plus_type::tag(), d) -
                                n(n_minus_type::tag(), d)) / (2.0 * eps);
                            dn_mask[ui - 1] = 1;
                        }
                    }
            }
        }

        op.template construction_DN_u<u_type>();

        PetscInt N = op.total_dim();
        Mat A;
        assemble_from_spmat(op.DN, N, nrow, A);
        Vec x, y;
        MatCreateVecs(A, &x, &y);
        {
            std::vector<float_type> xloc(nrow, 0.0);
            real_coordinate_type tmp_coord(0.0);
            force_type dummy_f(domain_->ib().size(), tmp_coord);
            if (!srv)
                op.template Grid2CSR<du_test_type, scr_p_type, scr_w_type>(
                    xloc.data(), dummy_f, false);
            PetscInt rstart, rend;
            VecGetOwnershipRange(x, &rstart, &rend);
            for (PetscInt i = rstart; i < rend; ++i)
            {
                PetscScalar v = xloc[i - rstart];
                VecSetValues(x, 1, &i, &v, INSERT_VALUES);
            }
            VecAssemblyBegin(x); VecAssemblyEnd(x);
        }
        MatMult(A, x, y);
        float_type l2 = 0.0, l2ex = 0.0, linf = 0.0; long count = 0;
        {
            PetscInt rstart, rend;
            VecGetOwnershipRange(y, &rstart, &rend);
            for (PetscInt i = rstart; i < rend; ++i)
            {
                PetscInt iloc1 = i - rstart + 1;
                if (!dn_mask[iloc1 - 1]) continue;
                PetscScalar v; VecGetValues(y, 1, &i, &v);
                float_type got = PetscRealPart(v);
                float_type exp = dn_exact[iloc1 - 1];
                float_type e = got - exp;
                l2 += e*e; l2ex += exp*exp;
                linf = std::max(linf, std::abs(e));
                ++count;
            }
        }
        float_type gl2, gl2ex, glinf; long gcount;
        boost::mpi::all_reduce(world, l2, gl2, std::plus<float_type>());
        boost::mpi::all_reduce(world, l2ex, gl2ex, std::plus<float_type>());
        boost::mpi::all_reduce(world, linf, glinf, boost::mpi::maximum<float_type>());
        boost::mpi::all_reduce(world, count, gcount, std::plus<long>());
        if (world.rank() == world.size() - 1)
            std::cout << "[M6-DN ] nodes=" << gcount
                      << "  rel L2=" << std::sqrt(gl2 / std::max(gl2ex, (float_type)1e-300))
                      << "  max abs=" << glinf
                      << "  (eps=" << eps << ", FD truncation ~ eps^2)"
                      << std::endl;
        MatDestroy(&A); VecDestroy(&x); VecDestroy(&y);
    }

    template<class T> struct tmpl_tag { using type = T; };

    /** @brief v^T (Smearing g) == g^T (Projection v) for random v (velocity),
     *  g (IB force) -- Smearing == Projection^T iff the delta-function
     *  weights are consistent (both are already Dim-generic, no 3D port
     *  needed; this just confirms it). */
    void run_M7_adjoint_check(time_integration_t& op)
    {
        boost::mpi::communicator world;
        const bool srv = domain_->is_server();
        PetscInt N = op.total_dim();
        PetscInt nrow = srv ? 0 : op.L.numRow_loc();
        Mat S, P;
        assemble_from_spmat(op.smearing, N, nrow, S);
        assemble_from_spmat(op.project, N, nrow, P);

        real_coordinate_type tmp_coord(0.0);
        force_type g_force(domain_->ib().size(), tmp_coord);
        for (std::size_t i = 0; i < g_force.size(); ++i)
            for (std::size_t d = 0; d < g_force[0].size(); ++d)
                g_force[i][d] = std::sin(0.7 * i + d + 1.0);
        force_type zero_force(domain_->ib().size(), tmp_coord);

        Vec x_v, x_g, y_S, y_P;
        MatCreateVecs(S, &x_g, &y_S);
        MatCreateVecs(P, &x_v, &y_P);
        {
            std::vector<float_type> xg(nrow, 0.0), xv(nrow, 0.0);
            if (!srv)
            {
                op.template Grid2CSR<scr_u_type, scr_p_type, scr_w_type>(
                    xg.data(), g_force, false);
                op.template Grid2CSR<du_test_type, scr_p_type, scr_w_type>(
                    xv.data(), zero_force, false);
            }
            PetscInt rstart, rend;
            VecGetOwnershipRange(x_g, &rstart, &rend);
            for (PetscInt i = rstart; i < rend; ++i)
            {
                PetscScalar vg = xg[i - rstart], vv = xv[i - rstart];
                VecSetValues(x_g, 1, &i, &vg, INSERT_VALUES);
                VecSetValues(x_v, 1, &i, &vv, INSERT_VALUES);
            }
            VecAssemblyBegin(x_g); VecAssemblyEnd(x_g);
            VecAssemblyBegin(x_v); VecAssemblyEnd(x_v);
        }
        MatMult(S, x_g, y_S); // Smearing @ g  (nonzero on velocity rows)
        MatMult(P, x_v, y_P); // Projection @ v (nonzero on force rows)

        PetscScalar lhs, rhs;
        VecDot(y_S, x_v, &lhs); // v . (S g)
        VecDot(y_P, x_g, &rhs); // g . (P v)
        float_type l = PetscRealPart(lhs), r = PetscRealPart(rhs);
        if (world.rank() == world.size() - 1)
            std::cout << "[M7-adj] v.(S g) = " << l << "  g.(P v) = " << r
                      << "  rel diff = "
                      << std::abs(l - r) / std::max(std::abs(l), (float_type)1e-300)
                      << std::endl;
        MatDestroy(&S); MatDestroy(&P);
        VecDestroy(&x_v); VecDestroy(&x_g); VecDestroy(&y_S); VecDestroy(&y_P);
    }

    /** @brief pad_velocity is already Dim-generic (Operator::curl /
     *  curl_transpose / apply_lgf all verified 3D). Sanity check: run it on
     *  the Gaussian u_test field and confirm the correction-cell velocities
     *  it fills in are finite and nonzero (no numeric change vs 2D needed;
     *  this just confirms nothing crashes / silently zeroes out in 3D). */
    void run_M8_pad_velocity_check(time_integration_t& op)
    {
        boost::mpi::communicator world;
        double sumsq = 0.0; long n = 0, nbad = 0;
        if (!domain_->is_server())
        {
            // pad_velocity is client-only, like all other callers (e.g.
            // BaseFlow2D) guard it -- but the all_reduce below is collective
            // over `world`, so the server must still reach it.
            op.template pad_velocity<u_type, u_type>(true);
            for (auto it = domain_->begin_leaves(); it != domain_->end_leaves(); ++it)
            {
                if (!it->locally_owned() || !it->has_data()) continue;
                if (!it->is_correction()) continue;
                for (auto& nd : it->data())
                    for (std::size_t d = 0; d < Dim; ++d)
                    {
                        float_type v = nd(u_type::tag(), d);
                        if (!std::isfinite(v)) ++nbad;
                        sumsq += v * v;
                        ++n;
                    }
            }
        }
        double gsumsq; long gn, gnbad;
        boost::mpi::all_reduce(world, sumsq, gsumsq, std::plus<double>());
        boost::mpi::all_reduce(world, n, gn, std::plus<long>());
        boost::mpi::all_reduce(world, nbad, gnbad, std::plus<long>());
        if (world.rank() == world.size() - 1)
            std::cout << "[M8-pad] correction-cell velocity nodes=" << gn
                      << "  ||u_corr||=" << std::sqrt(gsumsq)
                      << "  non-finite=" << gnbad << std::endl;
    }

    /** @brief Partition-of-unity check for the 3D FMM anterpolation: set
     *  w == 1 (constant) at every octree level, both fine and coarse. A
     *  correct upward_intrp row is Curl_factor*(-w_coarse + sum(weights *
     *  w_fine_children)); if the anterpolation weights sum to 1 (as any
     *  reasonable restriction operator's must), this is exactly 0 for a
     *  constant field -- a real, dimension-agnostic correctness check that
     *  doesn't require knowing the anterpolation weights themselves. */
    void run_M9_upward_intrp_check(time_integration_t& op)
    {
        boost::mpi::communicator world;
        const bool srv = domain_->is_server();
        if (!srv)
        {
            for (int l = 0; l < domain_->tree()->depth(); ++l)
                for (auto it = domain_->begin(l); it != domain_->end(l); ++it)
                {
                    if (!it->locally_owned() || !it->has_data()) continue;
                    for (auto& n : it->data())
                        for (std::size_t c = 0; c < (2 * Dim - 3); ++c)
                            n(w_type::tag(), c) = 1.0;
                }
        }
        op.construct_upward_intrp();

        PetscInt N = op.total_dim();
        PetscInt nrow = srv ? 0 : op.L.numRow_loc();
        Mat A;
        assemble_from_spmat(op.upward_intrp, N, nrow, A);
        Vec x, y;
        MatCreateVecs(A, &x, &y);
        {
            // Grid2CSR only walks the FINE levels [base_level, depth) -- it
            // does not know about the coarse FMM levels construct_upward_intrp
            // operates on. Build x directly from the local idx_w index
            // instead, at every level (fine + coarse), so it truly is the
            // constant-1 field this check needs.
            std::vector<float_type> xloc(nrow, 0.0);
            if (!srv)
                for (int l = 0; l < domain_->tree()->depth(); ++l)
                    for (auto it = domain_->begin(l); it != domain_->end(l); ++it)
                    {
                        if (!it->locally_owned() || !it->has_data()) continue;
                        for (auto& n : it->data())
                            for (std::size_t c = 0; c < (2 * Dim - 3); ++c)
                            {
                                int wi = (int)n(idx_w_type::tag(), c);
                                if (wi >= 1 && wi <= nrow) xloc[wi - 1] = 1.0;
                            }
                    }
            PetscInt rstart, rend;
            VecGetOwnershipRange(x, &rstart, &rend);
            for (PetscInt i = rstart; i < rend; ++i)
            {
                PetscScalar v = xloc[i - rstart];
                VecSetValues(x, 1, &i, &v, INSERT_VALUES);
            }
            VecAssemblyBegin(x); VecAssemblyEnd(x);
        }
        MatMult(A, x, y);
        // check only the coarse-vorticity rows [blk_wf_end_+1, blk_wc_end_],
        // broken down by which octree level each row's node lives at (to
        // localize a mismatch to the base_level-1 boundary case vs deeper
        // coarse levels).
        const int lo = (srv ? 0 : op.blk_wf_end_) + 1;
        const int hi = srv ? 0 : op.blk_wc_end_;
        std::map<int, int> row_to_level;
        if (!srv)
        {
            for (int l = 0; l < domain_->tree()->base_level(); ++l)
                for (auto it = domain_->begin(l); it != domain_->end(l); ++it)
                {
                    if (!it->locally_owned() || !it->has_data()) continue;
                    for (auto& n : it->data())
                        for (std::size_t c = 0; c < (2 * Dim - 3); ++c)
                        {
                            int wi = (int)n(idx_w_type::tag(), c);
                            if (wi >= 1 && wi <= nrow) row_to_level[wi] = l;
                        }
                }
        }
        // Fixed, rank-independent set of level keys: [0, base_level) plus a
        // sentinel -1 for any row that didn't map to a known correction
        // level. Every rank must call all_reduce the SAME number of times in
        // the SAME order (a collective), so we cannot iterate over each
        // rank's local (possibly differently-populated) map keys directly.
        const int base_level_local = srv ? 0 : domain_->tree()->base_level();
        int base_level_g = 0;
        boost::mpi::all_reduce(world, base_level_local, base_level_g,
            boost::mpi::maximum<int>());
        std::vector<double> sumsq(base_level_g + 1, 0.0);
        std::vector<long> count(base_level_g + 1, 0);
        {
            PetscInt rstart, rend;
            VecGetOwnershipRange(y, &rstart, &rend);
            for (PetscInt i = rstart; i < rend; ++i)
            {
                PetscInt iloc1 = i - rstart + 1;
                if (iloc1 < lo || iloc1 > hi) continue;
                PetscScalar v; VecGetValues(y, 1, &i, &v);
                double e = PetscRealPart(v);
                auto itl = row_to_level.find((int)iloc1);
                int lvl = (itl != row_to_level.end()) ? itl->second : base_level_g;
                sumsq[lvl] += e * e;
                count[lvl] += 1;
            }
        }
        std::vector<double> gsumsq(base_level_g + 1, 0.0);
        std::vector<long> gcount(base_level_g + 1, 0);
        boost::mpi::all_reduce(world, sumsq.data(), (int)sumsq.size(),
            gsumsq.data(), std::plus<double>());
        boost::mpi::all_reduce(world, count.data(), (int)count.size(),
            gcount.data(), std::plus<long>());
        if (world.rank() == world.size() - 1)
        {
            std::cout << "[M9-upw] per-level ||upward_intrp @ 1|| "
                         "(base_level="
                      << base_level_g << "):" << std::endl;
            for (int lvl = 0; lvl <= base_level_g; ++lvl)
            {
                std::string label = (lvl == base_level_g) ? "unmatched"
                                                            : std::to_string(lvl);
                std::cout << "[M9-upw]   level " << label
                          << "  rows=" << gcount[lvl]
                          << "  rms=" << std::sqrt(gsumsq[lvl] / std::max(gcount[lvl], 1L))
                          << std::endl;
            }
        }
        MatDestroy(&A); VecDestroy(&x); VecDestroy(&y);
    }

    /** @brief construction_BCMat 3D vs pad_velocity -- an independent
     *  second implementation of the same physical operation (LGF/Biot-Savart
     *  boundary velocity due to vorticity), so unlike M9 this has a genuine
     *  cross-check oracle:
     *
     *    boundary_u @ [u;w]  ==  LGF_response(w) - u   (by construction: the
     *    matrix is "LGF weights on w" + "-1 * identity on u")
     *
     *  pad_velocity computes exactly LGF_response(curl(Velocity_in)) and
     *  writes it into Velocity_in's own correction-cell slots. So setting
     *  w := curl(u_test) (Operator::curl, unfiltered over all base_level
     *  nodes -- matching both get_BC_idx_from_client_3d's and pad_velocity's
     *  own internal edge_aux loop, neither of which skips correction cells),
     *  then comparing
     *    boundary_u @ [u_before; w]   vs   (u_after - u_before)
     *  at correction-cell velocity rows, tests the two independently-coded
     *  paths (hand-rolled per-point LGF matrix assembly vs production
     *  apply_lgf/curl_transpose) against each other. */
    void run_M10_bcmat_check(time_integration_t& op)
    {
        boost::mpi::communicator world;
        const bool srv = domain_->is_server();

        // Fresh test field -- M8's pad_velocity already overwrote the
        // correction-cell u from the earlier Gaussian setup.
        set_gaussian_test_field();

        // w_test = curl(u_test), unfiltered at base_level.
        if (!srv)
        {
            const float_type dx_base = domain_->dx_base();
            auto client = domain_->decomposition().client();
            const int base_level = domain_->tree()->base_level();
            client->template buffer_exchange<u_type>(base_level);
            for (auto it = domain_->begin(base_level);
                 it != domain_->end(base_level); ++it)
            {
                if (!it->locally_owned() || !it->has_data()) continue;
                domain::Operator::curl<u_type, curl_exact_type>(
                    it->data(), dx_base);
            }
            for (auto it = domain_->begin(base_level);
                 it != domain_->end(base_level); ++it)
            {
                if (!it->locally_owned() || !it->has_data()) continue;
                for (auto& n : it->data())
                    for (std::size_t c = 0; c < (2 * Dim - 3); ++c)
                        n(w_type::tag(), c) = n(curl_exact_type::tag(), c);
            }
        }

        op.construction_BCMat();

        PetscInt N    = op.total_dim();
        PetscInt nrow = srv ? 0 : op.L.numRow_loc();
        Mat A;
        assemble_from_spmat(op.boundary_u, N, nrow, A);
        Vec x, y;
        MatCreateVecs(A, &x, &y);
        {
            std::vector<float_type> xloc(nrow, 0.0);
            real_coordinate_type tmp_coord(0.0);
            force_type dummy_f(domain_->ib().size(), tmp_coord);
            if (!srv)
                op.template Grid2CSR<u_type, p_type, w_type>(
                    xloc.data(), dummy_f, false);
            PetscInt rstart, rend;
            VecGetOwnershipRange(x, &rstart, &rend);
            for (PetscInt i = rstart; i < rend; ++i)
            {
                PetscScalar v = xloc[i - rstart];
                VecSetValues(x, 1, &i, &v, INSERT_VALUES);
            }
            VecAssemblyBegin(x); VecAssemblyEnd(x);
        }
        MatMult(A, x, y);

        std::vector<float_type> y_loc(nrow, 0.0);
        {
            PetscInt rstart, rend;
            VecGetOwnershipRange(y, &rstart, &rend);
            for (PetscInt i = rstart; i < rend; ++i)
            {
                PetscScalar v; VecGetValues(y, 1, &i, &v);
                y_loc[i - rstart] = PetscRealPart(v);
            }
        }

        // u_before at correction-cell velocity DOFs, keyed by local row.
        std::vector<float_type> u_before(nrow, 0.0);
        std::vector<char> mask(nrow, 0);
        if (!srv)
        {
            for (auto it = domain_->begin_leaves(); it != domain_->end_leaves();
                 ++it)
            {
                if (!it->locally_owned() || !it->has_data()) continue;
                if (!it->is_correction()) continue;
                for (auto& n : it->data())
                    for (std::size_t d = 0; d < Dim; ++d)
                    {
                        int ui = (int)n(idx_u_type::tag(), d);
                        if (ui >= 1 && ui <= nrow)
                        {
                            u_before[ui - 1] = n(u_type::tag(), d);
                            mask[ui - 1]     = 1;
                        }
                    }
            }
            op.template pad_velocity<u_type, u_type>(true);
        }

        double sumsq_err = 0.0, sumsq_ref = 0.0, maxabs = 0.0;
        long   count     = 0;
        if (!srv)
        {
            for (auto it = domain_->begin_leaves(); it != domain_->end_leaves();
                 ++it)
            {
                if (!it->locally_owned() || !it->has_data()) continue;
                if (!it->is_correction()) continue;
                for (auto& n : it->data())
                    for (std::size_t d = 0; d < Dim; ++d)
                    {
                        int ui = (int)n(idx_u_type::tag(), d);
                        if (ui >= 1 && ui <= nrow && mask[ui - 1])
                        {
                            float_type delta =
                                n(u_type::tag(), d) - u_before[ui - 1];
                            float_type err = y_loc[ui - 1] - delta;
                            sumsq_err += err * err;
                            sumsq_ref += delta * delta;
                            maxabs = std::max(maxabs, std::abs(err));
                            ++count;
                        }
                    }
            }
        }
        double g_err, g_ref, g_max;
        long   g_count;
        boost::mpi::all_reduce(world, sumsq_err, g_err, std::plus<double>());
        boost::mpi::all_reduce(world, sumsq_ref, g_ref, std::plus<double>());
        boost::mpi::all_reduce(
            world, maxabs, g_max, boost::mpi::maximum<double>());
        boost::mpi::all_reduce(world, count, g_count, std::plus<long>());
        if (world.rank() == world.size() - 1)
            std::cout << "[M10-BCMat] rows=" << g_count
                       << "  rel L2=" << std::sqrt(g_err / std::max(g_ref, 1e-300))
                       << "  max abs=" << g_max << std::endl;

        MatDestroy(&A); VecDestroy(&x); VecDestroy(&y);
    }

    void initialize()
    {
        if (domain_->is_server()) return;
        for (auto it = domain_->begin_leaves(); it != domain_->end_leaves();
             ++it)
        {
            if (!it->locally_owned() || !it->has_data()) continue;
            for (auto& n : it->data())
            {
                for (std::size_t d = 0; d < Dim; ++d) n(u_type::tag(), d) = 0.0;
                n(p_type::tag(), 0) = 0.0;
            }
        }
    }

    /** @brief u_i = exp(-a r_i^2) at the i-th face location; Operator::laplace
     *  reference computed separately via the applied operator. */
    void set_gaussian_test_field()
    {
        if (domain_->is_server()) return;
        const float_type dx_base = domain_->dx_base();
        auto center = (domain_->bounding_box().max() -
                          domain_->bounding_box().min() + 1) / 2.0 +
            domain_->bounding_box().min();
        for (auto it = domain_->begin_leaves(); it != domain_->end_leaves();
             ++it)
        {
            if (!it->locally_owned() || !it->has_data()) continue;
            auto dx_level = dx_base / std::pow(2, it->refinement_level());
            auto scaling = std::pow(2, it->refinement_level());
            for (auto& node : it->data())
            {
                const auto& coord = node.level_coordinate();
                float_type xc = (coord[0] - center[0] * scaling + 0.5) * dx_level;
                float_type yc = (coord[1] - center[1] * scaling + 0.5) * dx_level;
                float_type zc = (coord[2] - center[2] * scaling + 0.5) * dx_level;

                float_type xf0 = (coord[0] - center[0] * scaling) * dx_level;
                float_type yf0 = yc, zf0 = zc;
                float_type xf1 = xc;
                float_type yf1 = (coord[1] - center[1] * scaling) * dx_level;
                float_type zf1 = zc;
                float_type xf2 = xc, yf2 = yc;
                float_type zf2 = (coord[2] - center[2] * scaling) * dx_level;

                const float_type a = a_;
                float_type rf0_2 = xf0*xf0 + yf0*yf0 + zf0*zf0;
                float_type rf1_2 = xf1*xf1 + yf1*yf1 + zf1*zf1;
                float_type rf2_2 = xf2*xf2 + yf2*yf2 + zf2*zf2;

                node(u_type::tag(), 0) = std::exp(-a * rf0_2);
                node(u_type::tag(), 1) = std::exp(-a * rf1_2);
                node(u_type::tag(), 2) = std::exp(-a * rf2_2);

                float_type rc2 = xc*xc + yc*yc + zc*zc;
                node(p_type::tag(), 0) = std::exp(-a * rc2);
            }
        }
    }

    /** @brief Compare an assembled sparse_mat block (velocity-out rows for
     *  Grad, pressure-out rows for Div) against the applied Operator::
     *  reference. row_lo/row_hi select the local row range [1-based) to
     *  check (the block that operator writes into). */
    void verify_block(const char* name, solver::sparse_mat& S,
        time_integration_t& op, PetscInt row_lo, PetscInt row_hi,
        const std::vector<float_type>& exact_loc,
        const std::vector<char>& mask, float_type scale = 1.0)
    {
        boost::mpi::communicator world;
        const bool srv = domain_->is_server();
        PetscInt N = op.total_dim();
        PetscInt nlocal = srv ? 0 : op.Jac.numRow_loc(); // sizing only
        // reuse Jac's row partition for sizing (same max_local_idx)
        Mat A;
        assemble_from_spmat(S, N, srv ? 0 : op.L.numRow_loc(), A);
        Vec x, y;
        MatCreateVecs(A, &x, &y);
        {
            std::vector<float_type> xloc(srv ? 0 : op.L.numRow_loc(), 0.0);
            real_coordinate_type tmp_coord(0.0);
            force_type dummy_f(domain_->ib().size(), tmp_coord);
            if (!srv)
                op.template Grid2CSR<u_type, p_type, w_type>(
                    xloc.data(), dummy_f, false);
            PetscInt rstart, rend;
            VecGetOwnershipRange(x, &rstart, &rend);
            for (PetscInt i = rstart; i < rend; ++i)
            {
                PetscScalar v = xloc[i - rstart];
                VecSetValues(x, 1, &i, &v, INSERT_VALUES);
            }
            VecAssemblyBegin(x); VecAssemblyEnd(x);
        }
        MatMult(A, x, y);
        float_type l2 = 0.0, l2ex = 0.0, linf = 0.0; long count = 0;
        {
            PetscInt rstart, rend;
            VecGetOwnershipRange(y, &rstart, &rend);
            for (PetscInt i = rstart; i < rend; ++i)
            {
                PetscInt iloc1 = i - rstart + 1; // 1-based local row
                if (iloc1 < row_lo || iloc1 > row_hi) continue;
                if (!mask[iloc1 - 1]) continue;
                PetscScalar v;
                VecGetValues(y, 1, &i, &v);
                float_type got = PetscRealPart(v);
                float_type exp = exact_loc[iloc1 - 1] * scale;
                float_type e = got - exp;
                l2 += e*e; l2ex += exp*exp;
                linf = std::max(linf, std::abs(e));
                ++count;
            }
        }
        float_type gl2, gl2ex, glinf; long gcount;
        boost::mpi::all_reduce(world, l2, gl2, std::plus<float_type>());
        boost::mpi::all_reduce(world, l2ex, gl2ex, std::plus<float_type>());
        boost::mpi::all_reduce(world, linf, glinf, boost::mpi::maximum<float_type>());
        boost::mpi::all_reduce(world, count, gcount, std::plus<long>());
        if (world.rank() == world.size() - 1)
            std::cout << "[" << name << "] nodes=" << gcount
                      << "  rel L2=" << std::sqrt(gl2 / std::max(gl2ex, (float_type)1e-300))
                      << "  max abs=" << glinf << std::endl;
        MatDestroy(&A); VecDestroy(&x); VecDestroy(&y);
    }

    std::vector<coordinate_t> initialize_domain(Dictionary* _d, domain_t* _domain)
    {
        return _domain->construct_basemesh_blocks(_d, _domain->block_extent());
    }

  private:
    boost::mpi::communicator client_comm_;
    float_type               Re_ = 40.0;
    float_type               R_  = 0.5;
    float_type               a_  = 2.0; // Gaussian width for the M2 test field
    int                      nLevelRefinement_ = 0;
    int                      global_refinement_ = 0;
};

} // namespace iblgf

#endif
