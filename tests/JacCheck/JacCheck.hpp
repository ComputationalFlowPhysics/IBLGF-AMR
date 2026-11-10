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
// One driver for 2D and 3D (JC_DIM, set by JacCheck2D.cpp / JacCheck3D.cpp):
// A = the StabilityEVP2Dasm / StabilityEVP3D assembled Jacobian, compared
// against the central difference of NewtonRHS and the matrix-free
// Jacobian(); optional analytic Gaussian test direction (jac_check_gauss).
//
// Diagnostic: is the assembled Jacobian (NewtonIteration::Jac, the A of
// StabilityEVP2Dasm) the derivative of the discrete steady residual the DNS
// actually uses (NewtonRHS, which recomputes the correction-cell velocities
// with pad_velocity)?
//
//   v   = random (u' on leaf faces, p' on leaf cells, f' on IB points),
//         completed as the DNS does: u'_corr = pad_velocity(u'), and
//         w' chosen so A's own vorticity rows vanish (A's own w convention)
//   y   = A v
//   ref = [R(x0 + eps v) - R(x0 - eps v)] / (2 eps)   (R = NewtonRHS, packed
//         with Grid2CSR; exact for R quadratic in u, linear in p, f)
//
// Rows are compared by group: leaf-velocity rows split into IB blocks,
// outer-ring blocks (touching correction / missing blocks) and interior
// blocks; pressure rows; IB-force rows. The constraint rows of A, which the
// residual does not have (u_corr rows: -u_corr + K w; w rows), are reported
// as |A v| on the consistent v: the u_corr rows vanish iff K reproduces
// pad_velocity. B (1 on leaf velocities, 0 elsewhere) is not involved.

#ifndef IBLGF_INCLUDED_JAC_CHECK_HPP
#define IBLGF_INCLUDED_JAC_CHECK_HPP

#ifndef DEBUG_IFHERK
#define DEBUG_IFHERK
#endif
#define DEBUG_POISSON

#include "mpi.h"
#include <petscmat.h>
#include <petscsys.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <array>
#include <string>
#include <vector>
#include <map>
#include <cmath>
#include <random>
#include <fftw3.h>
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

#include <iblgf/solver/Newton/spmat_petsc.hpp>

#include "../../setups/setup_Newton.hpp"
#include <iblgf/operators/operators.hpp>

namespace iblgf
{
#ifndef JC_DIM
#error "define JC_DIM (2 or 3) before including JacCheck.hpp"
#endif
const int Dim = JC_DIM;

struct parameters
{
    static constexpr std::size_t Dim = JC_DIM;
    // clang-format off
    REGISTER_FIELDS
    (
    Dim,
     (
        (test  , float_type, 1,    1,       1,     cell,false ),
        // base flow (u, p, w as in StabilityEVP2Dasm)
        (u     , float_type, Dim,  1,       1,     face,true  ),
        (p     , float_type, 1,    1,       1,     cell,true  ),
        (w     , float_type, (Dim*2-3), 1,  1,     edge,true  ),
        // saved base state while the residual is evaluated at x0 +- eps v
        (u0    , float_type, Dim,  1,       1,     face,false ),
        (p0    , float_type, 1,    1,       1,     cell,false ),
        // zero edge field: w' is filled in the packed vector, not the grid
        (wzero , float_type, (Dim*2-3), 1,  1,     edge,false )
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
        U_.resize(domain_->dimension());
        U_[0] = simulation_.dictionary()->template get_or<float_type>("Ux", 1.0);
        U_[1] = simulation_.dictionary()->template get_or<float_type>("Uy", 0.0);
        if (domain_->dimension() > 2)
            U_[2] = simulation_.dictionary()->template get_or<float_type>("Uz", 0.0);
        simulation_.frame_vel() =
            [this](std::size_t idx, float_type t, auto coord = {0, 0, 0})
        { return -U_[idx]; };
        simulation_.bc_vel() =
            [this](std::size_t idx, float_type t, auto coord)
        { return -U_[idx]; };

        nLevelRefinement_ =
            simulation_.dictionary_->template get_or<int>("nLevels", 0);
        eps_ = simulation_.dictionary()->template get_or<float_type>(
            "jac_check_eps", 1e-4);
        // optional smooth test direction: u' = du_i of this file (e.g. a
        // stability eigenvector on the same tree), p' = f' = 0
        du_file_ = simulation_.dictionary()->template get_or<std::string>(
            "jac_check_du_file", "none");
        terms_ = simulation_.dictionary()->template get_or<bool>(
            "jac_check_terms", false);
        // analytic smooth direction: u' = curl(0,0,g), p' = g, f' = 0,
        // g = exp(-|x - x0|^2 / sigma^2) (physical coordinates, x = i dx)
        gauss_ = simulation_.dictionary()->template get_or<bool>(
            "jac_check_gauss", false);
        gx_[0] = simulation_.dictionary()->template get_or<float_type>("gauss_x0", 0.0);
        gx_[1] = simulation_.dictionary()->template get_or<float_type>("gauss_y0", 0.0);
        gx_[2] = simulation_.dictionary()->template get_or<float_type>("gauss_z0", 0.0);
        gsig_ = simulation_.dictionary()->template get_or<float_type>("gauss_sigma", 1.0);

        domain_->ib().init(_d->get_dictionary("simulation_parameters"),
            domain_->dx_base(), nLevelRefinement_, Re_);
        // base flow: a restart's u, p if use_restart; otherwise build the mesh
        // here (IB blocks refined to nLevels) with u = p = 0 -- no 3D base
        // flow exists yet, and with U != 0 the linearized advection by the
        // freestream is still part of A
        if (use_restart())
        {
            domain_->restart_list_construct();
            domain_->distribute<fmm_mask_builder_t, fmm_mask_builder_t>();
            simulation_.template read_h5<u_type>(simulation_.restart_field_dir(), "u");
            simulation_.template read_h5<p_type>(simulation_.restart_field_dir(), "p");
        }
        else
        {
            domain_->init_refine(nLevelRefinement_, 0, 0);
            domain_->distribute<fmm_mask_builder_t, fmm_mask_builder_t>();
        }
    }

    float_type run(int argc, char* argv[])
    {
        PetscInitialize(&argc, &argv, (char*)0, NULL);
        op_ = std::make_shared<time_integration_t>(&this->simulation_);
        check();
        PetscFinalize();
        return 0.0;
    }

    // ---------------------------------------------------------------------
    void check()
    {
        boost::mpi::communicator world;
        const bool server = domain_->is_server();

        // --- A about the base flow (as StabilityEVP2Dasm) ------------------
        op_->Assigning_idx();
        if (!server) op_->template pad_velocity<u_type, u_type>(true);
        world.barrier();
        op_->construct_Jac_p();
        op_->template construct_Jac_from_Jac_p<u_type>();

        const PetscInt N = op_->total_dim();
        const PetscInt nl = server ? 0 : op_->Jac.numRow_loc();
        PetscInt rstart, rend;
        PetscCallVoid(solver::spmat_petsc::ownership(N, nl, &rstart, &rend));
        Mat A;
        PetscCallVoid(solver::spmat_petsc::assemble(op_->Jac, N, nl, rstart, rend, false, &A));

        // save the (padded) base state
        copy_leaf<u_type, u0_type>();
        copy_leaf<p_type, p0_type>();

        // --- random v, completed consistently ------------------------------
        const std::size_t nib = domain_->ib().size();
        real_coordinate_type zc(0.0);
        force_type df(nib, zc), f0(nib, zc), fres(nib, zc);
        if (du_file_ != "none")
            simulation_.template read_h5<du_i_type>(du_file_, "du_i");
        if (!server)
        {
            std::mt19937 gen(1234 + world.rank());
            std::normal_distribution<float_type> nd(0.0, 1.0);
            if (du_file_ == "none" || gauss_) op_->template clean<du_i_type>();
            op_->template clean<dp_i_type>();
            if (gauss_)
            {
                const float_type dxb = domain_->dx_base();
                for (auto it = domain_->begin(); it != domain_->end(); ++it)
                {
                    if (!it->locally_owned() || !it->has_data()) continue;
                    if (!it->is_leaf() || it->is_correction()) continue;
                    const float_type dl = dxb / std::pow(2.0, it->refinement_level());
                    for (auto& n : it->data())
                    {
                        const auto& c = n.level_coordinate();
                        // value of g, dg/dx, dg/dy at (cell index + offsets) * dl
                        auto gd = [&](float_type ox, float_type oy, float_type oz) {
                            float_type x = (c[0] + ox) * dl - gx_[0];
                            float_type y = (c[1] + oy) * dl - gx_[1];
                            float_type z = 0.0;
                            if constexpr (Dim == 3) z = (c[2] + oz) * dl - gx_[2];
                            float_type g = std::exp(-(x * x + y * y + z * z) / (gsig_ * gsig_));
                            return std::array<float_type, 3>{g,
                                -2.0 * x / (gsig_ * gsig_) * g,
                                -2.0 * y / (gsig_ * gsig_) * g};
                        };
                        n(du_i_type::tag(), 0) = gd(0.0, 0.5, 0.5)[2];   //  dg/dy
                        n(du_i_type::tag(), 1) = -gd(0.5, 0.0, 0.5)[1];  // -dg/dx
                        if constexpr (Dim == 3) n(du_i_type::tag(), 2) = 0.0;
                        n(dp_i_type::tag(), 0) = gd(0.5, 0.5, 0.5)[0];
                    }
                }
            }
            else if (du_file_ == "none")
            {
                for_leaf([&](auto& n) {
                    for (std::size_t d = 0; d < Dim; ++d) n(du_i_type::tag(), d) = nd(gen);
                    n(dp_i_type::tag(), 0) = nd(gen);
                });
                for (std::size_t i = 0; i < nib; ++i)
                    if (domain_->ib().rank(i) == world.rank())
                        for (std::size_t d = 0; d < Dim; ++d) df[i][d] = nd(gen);
            }
            // correction-cell velocity as the DNS computes it
            op_->template pad_velocity<du_i_type, du_i_type>(true);
        }
        world.barrier();

        std::vector<float_type> v(nl, 0.0), y(nl, 0.0);
        if (!server)
            op_->template Grid2CSR<du_i_type, dp_i_type, wzero_type>(
                v.data(), df, false);

        // w' from A's own vorticity rows (block (blk_p_end_, blk_wc_end_]):
        // Jacobi sweeps w_i -= r_i / a_ii until those rows vanish
        const int bu = server ? 0 : op_->blk_u_end_;
        const int bp = server ? 0 : op_->blk_p_end_;
        const int bwf = server ? 0 : op_->blk_wf_end_;
        const int bwc = server ? 0 : op_->blk_wc_end_;
        std::vector<float_type> diag(nl, 0.0);
        for (PetscInt i = 0; i < nl; ++i)
        {
            auto& row = op_->Jac.mat[i + 1];
            auto it = row.find(int(rstart + i + 1));
            if (it != row.end()) diag[i] = it->second;
        }
        double wres = 0.0;
        for (int sweep = 0; sweep < 50; ++sweep)
        {
            PetscCallVoid(solver::spmat_petsc::matvec(A, v, y, nl));
            double loc = 0.0;
            for (int i = bp; i < bwc; ++i)
            {
                loc += y[i] * y[i];
                if (diag[i] != 0.0) v[i] -= y[i] / diag[i];
            }
            boost::mpi::all_reduce(world, loc, wres, std::plus<double>());
            wres = std::sqrt(wres);
            if (world.rank() == 1)
                std::cout << "[JacCheck] w-row sweep " << sweep
                          << " |A v|_w = " << wres << std::endl;
            if (wres < 1e-12) break;
        }
        PetscCallVoid(solver::spmat_petsc::matvec(A, v, y, nl));

        // --- matrix-free Jacobian() on the same (padded) v, about the base u
        //     (before residual_at changes u): the PETscJacobian comparison
        std::vector<float_type> jmf(nl, 0.0);
        {
            force_type fmf(nib, zc);
            if (!server)
                op_->template Jacobian<du_i_type, dp_i_type, fu_i_type, fp_i_type>(
                    df, fmf);
            world.barrier();
            if (!server)
                op_->template Grid2CSR<fu_i_type, fp_i_type, fw_i_type>(
                    jmf.data(), fmf);
        }

        // --- reference: central difference of NewtonRHS ---------------------
        std::vector<float_type> rp(nl, 0.0), rm(nl, 0.0);
        residual_at(+eps_, df, f0, fres, rp);
        residual_at(-eps_, df, f0, fres, rm);
        std::vector<float_type> ref(nl, 0.0);
        for (PetscInt i = 0; i < nl; ++i) ref[i] = (rp[i] - rm[i]) / (2 * eps_);

        // --- row groups ------------------------------------------------------
        // 0 leaf-u interior, 1 leaf-u outer ring, 2 leaf-u IB blocks,
        // 3 u_corr (constraint), 4 p, 5 w (constraint), 6 f
        std::vector<int> grp(nl, -1);
        if (!server)
        {
            for (int i = 0; i < nl; ++i)
            {
                if (i < bu) grp[i] = 3;      // default: correction, refined below
                else if (i < bp) grp[i] = 4;
                else if (i < bwc) grp[i] = 5;
                else grp[i] = 6;
            }
            for (auto it = domain_->begin(); it != domain_->end(); ++it)
            {
                if (!it->locally_owned() || !it->has_data()) continue;
                if (!it->is_leaf() || it->is_correction()) continue;
                int g = 0;
                if (it->is_ib() || it->is_extended_ib()) g = 2;
                else
                    for (std::size_t k = 0; k < it->num_neighbors(); ++k)
                    {
                        auto nb = it->neighbor(k);
                        if (!nb || !nb->has_data() || nb->is_correction()) g = 1;
                    }
                for (std::size_t d = 0; d < Dim; ++d)
                    for (auto& n : it->data())
                    {
                        int r = int(n(idx_u_type::tag(), d)) - 1;
                        if (r >= 0 && r < bu) grp[r] = g;
                    }
            }
        }
        const char* names[7] = {"leaf u, interior blocks", "leaf u, outer-ring blocks",
            "leaf u, IB blocks", "u_corr rows (constraint)", "pressure rows",
            "vorticity rows (constraint)", "IB-force rows"};
        double loc[7][5] = {{0}}; // |y-ref|^2, |ref|^2, |y|^2, |y-jmf|^2, |jmf|^2
        double vcorr_loc = 0.0;
        for (int i = 0; i < nl; ++i)
        {
            int g = grp[i];
            if (g < 0) continue;
            double e = y[i] - ref[i];
            loc[g][0] += e * e;
            loc[g][1] += ref[i] * ref[i];
            loc[g][2] += y[i] * y[i];
            loc[g][3] += (y[i] - jmf[i]) * (y[i] - jmf[i]);
            loc[g][4] += jmf[i] * jmf[i];
            if (g == 3) vcorr_loc += v[i] * v[i];
        }
        double glob[7][5], vcorr = 0.0;
        boost::mpi::all_reduce(world, &loc[0][0], 35, &glob[0][0], std::plus<double>());
        boost::mpi::all_reduce(world, vcorr_loc, vcorr, std::plus<double>());

        if (world.rank() == 1)
        {
            std::ofstream out("jac_check.txt");
            auto emit = [&](std::ostream& os) {
                os << "# JacCheck: A v vs central-difference NewtonRHS (ref) and vs "
                      "matrix-free Jacobian() (Jmf), eps = " << eps_ << ", u' = "
                   << (gauss_ ? "Gaussian at (" + std::to_string(gx_[0]) + "," +
                                    std::to_string(gx_[1]) + "," + std::to_string(gx_[2]) + ")"
                              : (du_file_ == "none" ? std::string("random") : du_file_))
                   << "  (w-row residual after completion " << wres << ")\n";
                os << std::left << std::setw(30) << "# row group" << std::right
                   << std::setw(14) << "|Av - ref|" << std::setw(14) << "|ref|"
                   << std::setw(14) << "|Av|" << std::setw(12) << "rel(ref)"
                   << std::setw(14) << "|Av - Jmf|" << std::setw(12) << "rel(Jmf)\n";
                for (int g = 0; g < 7; ++g)
                {
                    double d = std::sqrt(glob[g][0]), r = std::sqrt(glob[g][1]),
                           a = std::sqrt(glob[g][2]);
                    os << std::left << std::setw(30) << names[g] << std::right
                       << std::scientific << std::setprecision(3)
                       << std::setw(14) << d << std::setw(14) << r
                       << std::setw(14) << a << std::setw(12)
                       << ((g == 3 || g == 5) ? a / std::max(1e-300, std::sqrt(vcorr))
                                              : d / std::max(1e-300, r))
                       << std::setw(14) << std::sqrt(glob[g][3]) << std::setw(12);
                    // Jacobian() has no constraint rows (u_corr, w): no ratio
                    if (g == 3 || g == 5) os << "-";
                    else
                        os << std::sqrt(glob[g][3]) /
                                  std::max(1e-300, std::sqrt(glob[g][4]));
                    os << "\n";
                }
                os << "# constraint rows (u_corr, w): rel err column = |Av| / |u'_corr|;"
                      " the residual has no such rows, so |ref| there is not comparable\n";
            };
            emit(std::cout);
            emit(out);
        }
        PetscCallVoid(MatDestroy(&A));

        if (terms_) check_terms(v, grp, df, f0, fres, N, nl, rstart, rend);
    }

    /** @brief Term-by-term: A built from one assembled piece vs the central
     *  difference of the matching NewtonRHS term, on the same v. */
    void check_terms(std::vector<float_type>& v, std::vector<int>& grp,
        force_type& df, force_type& f0, force_type& fres, PetscInt N,
        PetscInt nl, PetscInt rstart, PetscInt rend)
    {
        boost::mpi::communicator world;
        struct term { const char* name; unsigned rhs_mask, jac_mask; };
        const term terms[6] = {{"laplacian (L)", 1u, 2u},
            {"advection (DN)", 2u, 4u}, {"gradient (Grad)", 4u, 16u},
            {"smearing (f->u)", 8u, 128u}, {"divergence (Div)", 16u, 8u},
            {"IB projection (project)", 32u, 64u}};
        const int  show[5] = {0, 1, 2, 4, 6};
        const char* gname[7] = {"interior", "outer ring", "IB blocks", "u_corr",
            "pressure", "w", "IB force"};
        std::ofstream out;
        if (world.rank() == 1)
        {
            out.open("jac_check.txt", std::ios::app);
            out << "\n# per term: rel err = |A_t v - ref_t| / |ref_t| "
                   "(|A_t v - ref_t|, |ref_t| in brackets)\n";
        }
        std::vector<float_type> y(nl, 0.0), rp(nl, 0.0), rm(nl, 0.0);
        for (const auto& t : terms)
        {
            // A_t about the base flow: restore u0/p0, pad, rebuild
            if (!domain_->is_server())
            {
                copy_leaf<u0_type, u_type>();
                copy_leaf<p0_type, p_type>();
                op_->template pad_velocity<u_type, u_type>(true);
            }
            world.barrier();
            op_->set_jac_terms(t.jac_mask);
            op_->construct_Jac_p();
            op_->template construct_Jac_from_Jac_p<u_type>();
            Mat At;
            PetscCallVoid(solver::spmat_petsc::assemble(op_->Jac, N, nl, rstart, rend, false, &At));
            PetscCallVoid(solver::spmat_petsc::matvec(At, v, y, nl));
            PetscCallVoid(MatDestroy(&At));

            residual_at(+eps_, df, f0, fres, rp, t.rhs_mask);
            residual_at(-eps_, df, f0, fres, rm, t.rhs_mask);

            double loc[7][2] = {{0}}, glob[7][2];
            for (PetscInt i = 0; i < nl; ++i)
            {
                int g = grp[i];
                if (g < 0) continue;
                double ref = (rp[i] - rm[i]) / (2 * eps_);
                loc[g][0] += (y[i] - ref) * (y[i] - ref);
                loc[g][1] += ref * ref;
            }
            boost::mpi::all_reduce(world, &loc[0][0], 14, &glob[0][0],
                std::plus<double>());
            if (world.rank() == 1)
            {
                std::ostringstream os;
                os << std::left << std::setw(26) << t.name << std::right;
                for (int g : show)
                {
                    double d = std::sqrt(glob[g][0]), r = std::sqrt(glob[g][1]);
                    os << "  " << gname[g] << " " << std::scientific
                       << std::setprecision(2)
                       << (r > 0 ? d / r : d) << " [" << d << ", " << r << "]";
                }
                std::cout << os.str() << std::endl;
                out << os.str() << "\n";
            }
        }
    }

    /** @brief NewtonRHS at (u0 + s du, p0 + s dp, f0 + s df), packed. */
    void residual_at(float_type s, force_type& df, force_type& f0,
        force_type& fres, std::vector<float_type>& r, unsigned mask = 63u)
    {
        boost::mpi::communicator world;
        force_type f = f0;
        if (!domain_->is_server())
        {
            for_leaf([&](auto& n) {
                for (std::size_t d = 0; d < Dim; ++d)
                    n(u_type::tag(), d) = n(u0_type::tag(), d) + s * n(du_i_type::tag(), d);
                n(p_type::tag(), 0) = n(p0_type::tag(), 0) + s * n(dp_i_type::tag(), 0);
            });
            for (std::size_t i = 0; i < f.size(); ++i)
                for (std::size_t d = 0; d < Dim; ++d) f[i][d] += s * df[i][d];
        }
        world.barrier();
        op_->template NewtonRHS_terms<u_type, p_type, fu_i_type, fp_i_type>(
            f, fres, mask);
        world.barrier();
        if (!domain_->is_server())
            op_->template Grid2CSR<fu_i_type, fp_i_type, fw_i_type>(r.data(), fres);
    }

    template<class F>
    void for_leaf(F&& fn)
    {
        for (auto it = domain_->begin(); it != domain_->end(); ++it)
        {
            if (!it->locally_owned() || !it->has_data()) continue;
            if (!it->is_leaf() || it->is_correction()) continue;
            for (auto& n : it->data()) fn(n);
        }
    }

    template<class From, class To>
    void copy_leaf()
    {
        if (domain_->is_server()) return;
        for_leaf([](auto& n) {
            for (std::size_t d = 0; d < From::nFields(); ++d)
                n(To::tag(), d) = n(From::tag(), d);
        });
    }



    template<class OctantType>
    bool refinement(OctantType* it, int diff_level, bool use_all = false) const noexcept
    {
        return false;
    }

    std::vector<coordinate_t> initialize_domain(Dictionary* _d, domain_t* _domain)
    {
        return _domain->construct_basemesh_blocks(_d, _domain->block_extent());
    }

  private:
    boost::mpi::communicator            client_comm_;
    std::shared_ptr<time_integration_t> op_;
    std::vector<float_type>             U_;
    float_type                          Re_ = 0.0;
    float_type                          eps_ = 1e-4;
    std::string                         du_file_ = "none";
    bool                                terms_ = false;
    bool                                gauss_ = false;
    float_type                          gx_[3] = {0.0, 0.0, 0.0};
    float_type                          gsig_ = 1.0;
    int                                 nLevelRefinement_ = 0;
};

} // namespace iblgf

#endif
