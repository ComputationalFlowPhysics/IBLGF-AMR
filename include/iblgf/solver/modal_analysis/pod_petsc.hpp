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

#ifndef IBLGF_SOLVER_MODAL_ANALYSIS_POD_PETSC_HPP
#define IBLGF_SOLVER_MODAL_ANALYSIS_POD_PETSC_HPP

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
#include <slepceps.h>
#include <slepcsys.h>
#include <petscvec.h>
#include <petscksp.h>

#include <slepcsvd.h>

#include <random>

namespace iblgf
{
namespace solver
{
using namespace domain;
template<class Setup>
class POD
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
    using time_integration_t = typename Setup::time_integration_t;

    using ib_t = typename domain_type::ib_t;
    using force_type = typename ib_t::force_type;

    //FMM
    using Fmm_t = typename Setup::Fmm_t;

    using test_type = typename Setup::test_type;
    using idx_u_type = typename Setup::idx_u_type;
    using edge_aux_type = typename Setup::edge_aux_type;
    using u_type = typename Setup::u_type;
    using stream_f_type = typename Setup::stream_f_type;
    using cell_aux_type = typename Setup::cell_aux_type;
    // using cell_aux_tmp_type = typename Setup::cell_aux_tmp_type;
    // using face_aux_tmp_type = typename Setup::face_aux_tmp_type;
    using face_aux_type = typename Setup::face_aux_type;
    static constexpr int lBuffer = 1; ///< Lower left buffer for interpolation
    static constexpr int rBuffer = 1; ///< Lower left buffer for interpolation

    static constexpr int Dim = Setup::Dim; ///< Number of dimensions

    POD(simulation_type* _simulation)
    : simulation_(_simulation)
    , domain_(_simulation->domain_.get())
    , psolver(_simulation)
    {
        boost::mpi::communicator world;
        this->init_idx<idx_u_type>();
        world.barrier();
    }
    float_type run_POD()
    {
        boost::mpi::communicator world;
        std::cout << "RSVD_DT::run()" << std::endl;
        world.barrier();
        PetscMPIInt rank;
        rank = world.rank();
        idxStart = simulation_->dictionary()->template get_or<int>("nStart", 100);
        nTotal = simulation_->dictionary()->template get_or<int>("nTotal", 100);
        nskip = simulation_->dictionary()->template get_or<int>("nskip", 100);

        PetscInt m_local, M;
        m_local = max_local_idx; // since 1 based
        Vec x, b;
        if (rank == 0) m_local = 0;

        boost::mpi::all_reduce(world, m_local, M, std::plus<int>());
        Mat                    A;
        ISLocalToGlobalMapping ltog_row, ltog_col;
        PetscCall(MatCreateDense(PETSC_COMM_WORLD, m_local, PETSC_DECIDE, M, nTotal, NULL, &A));
        // PetscCall(MatSetSizes(A, m_local, PETSC_DECIDE, M, nTotal));
        PetscCall(MatSetUp(A));
        PetscCall(MatCreateVecs(A, &x, &b));
        PetscInt rstartx, rendx, rstartb, rendb;
        PetscCall(VecGetOwnershipRange(x, &rstartx, &rendx));
        PetscCall(VecGetOwnershipRange(b, &rstartb, &rendb));

        //get index set
        int* global_rows;

        global_rows = new int[m_local];
        for (int i = 0; i < m_local; i++) { global_rows[i] = i + static_cast<int>(rstartb); }

        int* global_cols = new int[nTotal];
        for (int j = 0; j < nTotal; j++) { global_cols[j] = j; }

        PetscInt* rows = global_rows;
        PetscInt* cols = global_cols;
        IS        isrow, iscol;

        PetscCall(ISCreateGeneral(PETSC_COMM_WORLD, m_local, rows, PETSC_COPY_VALUES, &isrow));
        PetscCall(ISCreateGeneral(PETSC_COMM_WORLD, nTotal, cols, PETSC_COPY_VALUES, &iscol));

        PetscCall(ISLocalToGlobalMappingCreateIS(isrow, &ltog_row));
        PetscCall(ISLocalToGlobalMappingCreateIS(iscol, &ltog_col));

        PetscCall(MatSetLocalToGlobalMapping(A, ltog_row, ltog_col));

        PetscInt size_x, size_b;

        PetscCall(VecGetSize(x, &size_x));
        PetscCall(VecGetSize(b, &size_b));

        delete[] global_rows;
        delete[] global_cols;
        // Vec x, b;
        this->load_snapshots<idx_u_type, u_type>(A);
        world.barrier();
        PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
        PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

        //do SVD
        SVD       svd;
        PetscReal sigma;
        PetscInt  nConv;
        Vec       v;
        std::cout << rank << " here" << std::endl;
        PetscCall(SVDCreate(PETSC_COMM_WORLD, &svd));
        PetscCall(SVDSetType(svd, SVDLANCZOS)); // or SVDTRLANCZOS, SVDCYCLIC, etc.
        // SVDSetDimensions(svd, 3, PETSC_DEFAULT, PETSC_DEFAULT);
        PetscCall(SVDSetOperators(svd, A, NULL));
        PetscCall(SVDSetFromOptions(svd));
        std::cout << rank << " here2" << std::endl;
        PetscCall(SVDSolve(svd));
        std::cout << rank << " here3" << std::endl;
        PetscCall(SVDGetConverged(svd, &nConv));
        PetscCall(MatCreateVecs(A, NULL, &v));

        simulation_->write("podmodetest");

        for (PetscInt i = 0; i < nConv; i++)
        {
            world.barrier();

            PetscCall(SVDGetSingularTriplet(svd, i, &sigma, b, NULL));
            if (rank == 1) std::cout << "Singular value " << i << " : " << sigma << std::endl;
            vec2grid<idx_u_type, u_type>(b, 1);
            world.barrier();
            simulation_->write("podmode_" + std::to_string(i));
            world.barrier();
        }

        return 0.0;
    }
    float_type run_MOS()
    {
        boost::mpi::communicator world;
        std::cout << "MOS::run()" << std::endl;
        world.barrier();
        PetscMPIInt rank;
        rank = world.rank();
        idxStart = simulation_->dictionary()->template get_or<int>("nStart", 100);
        nTotal = simulation_->dictionary()->template get_or<int>("nTotal", 100);
        nskip = simulation_->dictionary()->template get_or<int>("nskip", 100);

        PetscInt m_local, M;
        m_local = max_local_idx; // since 1 based
        Vec x, b;
        if (rank == 0) m_local = 0;

        boost::mpi::all_reduce(world, m_local, M, std::plus<int>());
        Mat                    A;
        ISLocalToGlobalMapping ltog_row, ltog_col;
        PetscCall(MatCreateDense(PETSC_COMM_WORLD, m_local, PETSC_DECIDE, M, nTotal, NULL, &A));
        // PetscCall(MatSetSizes(A, m_local, PETSC_DECIDE, M, nTotal));
        PetscCall(MatSetUp(A));
        PetscCall(MatCreateVecs(A, &x, &b));
        PetscInt rstartx, rendx, rstartb, rendb;
        PetscCall(VecGetOwnershipRange(x, &rstartx, &rendx));
        PetscCall(VecGetOwnershipRange(b, &rstartb, &rendb));

        //get index set
        int* global_rows;

        global_rows = new int[m_local];
        for (int i = 0; i < m_local; i++) { global_rows[i] = i + static_cast<int>(rstartb); }

        int* global_cols = new int[nTotal];
        for (int j = 0; j < nTotal; j++) { global_cols[j] = j; }

        PetscInt* rows = global_rows;
        PetscInt* cols = global_cols;
        IS        isrow, iscol;

        PetscCall(ISCreateGeneral(PETSC_COMM_WORLD, m_local, rows, PETSC_COPY_VALUES, &isrow));
        PetscCall(ISCreateGeneral(PETSC_COMM_WORLD, nTotal, cols, PETSC_COPY_VALUES, &iscol));

        PetscCall(ISLocalToGlobalMappingCreateIS(isrow, &ltog_row));
        PetscCall(ISLocalToGlobalMappingCreateIS(iscol, &ltog_col));

        PetscCall(MatSetLocalToGlobalMapping(A, ltog_row, ltog_col));

        PetscInt size_x, size_b;

        PetscCall(VecGetSize(x, &size_x));
        PetscCall(VecGetSize(b, &size_b));

        // Vec x, b;
        this->load_snapshots<idx_u_type, u_type>(A);
        world.barrier();
        PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
        PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
        world.barrier();
        Mat At;
        PetscCall(MatDuplicate(A, MAT_DO_NOT_COPY_VALUES, &At));
        this->subtractMatmean<idx_u_type>(A,At); // subtract mean from A
        world.barrier();
        PetscCall(MatAssemblyBegin(At, MAT_FINAL_ASSEMBLY));
        PetscCall(MatAssemblyEnd(At, MAT_FINAL_ASSEMBLY));
        world.barrier();

        delete[] global_rows;
        delete[] global_cols;

        

        // this->subtractMatmean<idx_u_type>(A); // subtract mean from A
        //method of snapshots
        Mat C;
        PetscCall(MatTransposeMatMult(At, At, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &C));
        EPS eps;
        PetscCall(EPSCreate(PETSC_COMM_WORLD, &eps));
        PetscCall(EPSSetOperators(eps, C, NULL));
        PetscCall(EPSSetProblemType(eps, EPS_HEP));
        PetscCall(EPSSetFromOptions(eps));
        PetscCall(EPSSolve(eps));
        Vec vi, phi_i;
        PetscCall(MatCreateVecs(A, &vi,NULL));     // size m

        PetscScalar lambda_i;
        for (int i = 0; i < 4; ++i) {
            PetscCall(EPSGetEigenpair(eps, i, &lambda_i, NULL, vi, NULL));
            PetscCall(MatCreateVecs(At, NULL, &phi_i));     // size m
            PetscCall(MatMult(At, vi, phi_i));              // phi_i = A * v_i
            if (rank == 1) std::cout << "Singular value " << i << " : " << lambda_i << std::endl;
            // PetscCall(VecScale(phi_i));
            // Store or output phi_i
            vec2grid<idx_u_type, u_type>(phi_i, 1);
            world.barrier();
            this->curl<u_type>();
            world.barrier();
            simulation_->write("podmode_" + std::to_string(i));
            world.barrier();
        }

        return 0.0;
    }
    float_type run_vec_test()
    {
        boost::mpi::communicator world;
        std::cout << "RSVD_DT::run()" << std::endl;
        world.barrier();
        PetscMPIInt rank;
        rank = world.rank();
        PetscInt m_local, M;
        m_local = max_local_idx; // since 1 based
        if (rank == 0) m_local = 0;

        boost::mpi::all_reduce(world, m_local, M, std::plus<int>());

        Vec x, b;
        // Mat A;

        PetscCall(VecCreate(PETSC_COMM_WORLD, &x));
        PetscCall(VecSetSizes(x, m_local, M));
        PetscCall(VecSetFromOptions(x));
        PetscCall(VecSet(x, 2.0));
        PetscCall(VecAssemblyBegin(x));
        PetscCall(VecAssemblyEnd(x));

        grid2vec<idx_u_type, idx_u_type>(x);
        clean<u_type>();
        vec2grid<idx_u_type, u_type>(x); //vec_check is x
        copy<idx_u_type, face_aux_type>();
        add<u_type, face_aux_type>(-1.0);
        grid2vec<idx_u_type, face_aux_type>(x);
        PetscReal norm;
        PetscCall(VecNorm(x, NORM_2, &norm));
        std::cout << "Norm of x: " << norm << std::endl;

        return 0.0;
    }
    template<class Field_idx, class Field>
    float_type load_snapshots(Mat A)
    {
        boost::mpi::communicator world;
        std::string dir = simulation_->dictionary()->get_dictionary("output")->template get<std::string>("directory");
        for (int i = 0; i < nTotal; i++)
        {
            world.barrier();
            int         timeIdx = idxStart + i * nskip;
            std::string flow_file_i = "./" + dir + "/flow_adapted_to_ref_" + std::to_string(timeIdx) + ".hdf5";
            simulation_->template read_h5<Field>(flow_file_i, "u");
            world.barrier();
            if (!domain_->is_client()) continue;
            int base_level = domain_->tree()->base_level();
            for (int l = base_level; l < domain_->tree()->depth(); l++)
            {
                const auto dx_level = domain_->dx_base() / math::pow2(l);
                const auto w_1_2 =1;// std::pow(dx_level, domain_->dimension() / 2.0); //W^(1/2) so i can do standard SVD
                for (auto it = domain_->begin(l); it != domain_->end(l); ++it)
                {
                    if (!it->locally_owned() || !it->has_data()) continue;
                    // if (!it->data().is_allocated()) continue;
                    if (it->is_leaf() && !it->is_correction())
                    {
                        for (std::size_t field_idx = 0; field_idx < Dim; ++field_idx)
                        {
                            for (auto& n : it->data())
                            {
                                int i_local = n(Field_idx::tag(), field_idx) - 1;
                                if (i_local < 0) continue;
                                // int i_global =
                                //     (rank - 1) * m_local + i_local - 1;
                                PetscScalar value_r = n(Field::tag(), field_idx) * w_1_2;

                                PetscComplex value = value_r;
                                // PetscScalar value = 1.0;
                                PetscCall(MatSetValuesLocal(A, 1, &i_local, 1, &i, &value, INSERT_VALUES));
                            }
                        }
                    }
                }
            }
        }

        return 0.0;
    }

    template<class Field_idx>
    float_type subtractMatmean(Mat A,Mat At)
    {
        // std::cout<<"here"<<std::endl;
        if (!domain_->is_client()) return 0.0;
            int base_level = domain_->tree()->base_level();
            for (int l = base_level; l < domain_->tree()->depth(); l++)
            {
                const auto dx_level = 1.0 / math::pow2(l);
                const auto w_1_2 =std::pow(dx_level, domain_->dimension() / 2.0); //W^(1/2) so i can do standard SVD

                for (auto it = domain_->begin(l); it != domain_->end(l); ++it)
                {
                    if (!it->locally_owned() || !it->has_data()) continue;
                    // if (!it->data().is_allocated()) continue;
                    if (it->is_leaf() && !it->is_correction())
                    {
                        for (std::size_t field_idx = 0; field_idx < Dim; ++field_idx)
                        {
                            for (auto& n : it->data())
                            {
                                int i_local = n(Field_idx::tag(), field_idx) - 1;
                                if (i_local < 0) continue;
                                // int i_global =
                                //     (rank - 1) * m_local + i_local - 1;
                                PetscScalar row_mean= 0.0;
                                PetscScalar value = 0.0;
                                for(int k=0; k<nTotal; ++k)
                                {
                                    PetscCall(MatGetValuesLocal(A, 1, &i_local, 1, &k, &value));
                                    row_mean += value;
                                   
                                }
                                row_mean /= nTotal; // mean of the row
                                // std::cout<<"Row mean for i_local " << i_local << " : " << row_mean << std::endl;
                                for(int k=0; k<nTotal; ++k)
                                {
                                    PetscCall(MatGetValuesLocal(A, 1, &i_local, 1, &k, &value));
                                    value -= row_mean; // subtract mean from each element
                                    value *= w_1_2; // scale by w_1_2
                                    PetscCall(MatSetValuesLocal(At, 1, &i_local, 1, &k, &value, INSERT_VALUES));
                                }
                                // PetscScalar value = 1.0;
                                // PetscCall(MatSetValuesLocal(A, 1, &i_local, 1, &i, &value, INSERT_VALUES));
                            }
                        }
                    }
                }
            }
        // if (!domain_->is_client())
        // {   
        //     PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
        //     PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
        //     return 0.0;

        // }
        // // subtract mean of each row from each element in that row
        // PetscInt m_local, M;
        // PetscCall(MatGetLocalSize(A, &m_local, NULL));
        // PetscCall(MatGetSize(A, &M, NULL));
        // PetscScalar* A_array;
        // PetscCall(MatDenseGetArray(A, &A_array));
        // for (PetscInt i = 0; i < m_local; ++i)
        // {
        //     PetscScalar mean = 0.0;
        //     for (PetscInt j = 0; j < M; ++j)
        //     {
        //         mean += A_array[i * M + j];
        //     }
        //     mean /= M;
        //     for (PetscInt j = 0; j < M; ++j)
        //     {
        //         A_array[i * M + j] -= mean;
        //     }
        // }
        
        // PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
        // PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
        // PetscCall(MatDenseRestoreArray(A, &A_array));
        return 0.0;


    }

    template<class Field_idx, class Field_re>
    float_type grid2vec(Vec x)
    {
        PetscComplex* x_array;
        PetscCall(VecGetArray(x, &x_array));
        for (int i = 0; i < max_local_idx; ++i) { x_array[i] = 0.0; }
        if (domain_->is_client())
        {
            int base_level = domain_->tree()->base_level();
            for (int l = base_level; l < domain_->tree()->depth(); l++)
            {
                for (auto it = domain_->begin(l); it != domain_->end(l); ++it)
                {
                    if (!it->locally_owned() || !it->has_data()) continue;
                    // if (!it->data().is_allocated()) continue;
                    if (it->is_leaf())
                    {
                        for (std::size_t field_idx = 0; field_idx < Dim; ++field_idx)
                        {
                            for (auto& n : it->data())
                            {
                                int i_local = n(Field_idx::tag(), field_idx) - 1; //since used to be 1 based
                                if (i_local < 0) continue;
                                PetscScalar  value_r = n(Field_re::tag(), field_idx);
                                PetscComplex value = value_r;
                                x_array[i_local] = value;
                            }
                        }
                    }
                }
            }
        }

        PetscCall(VecAssemblyBegin(x));
        PetscCall(VecAssemblyEnd(x));
        PetscCall(VecRestoreArray(x, &x_array));
        return 0.0;
    }
    template<class Field_idx, class Field_re>
    float_type vec2grid(Vec x, int w_type = 0)
    {
        //loop through points
        const PetscComplex* x_array;
        PetscCall(VecGetArrayRead(x, &x_array));
        int base_level = domain_->tree()->base_level();
        for (int l = base_level; l < domain_->tree()->depth(); l++)
        {
            const auto dx_level = 1.0 / math::pow2(l);
            float_type w_1_2;
            if (w_type == 1)
            {
                w_1_2 = std::pow(dx_level, domain_->dimension() / 2.0); //W^(1/2) so i can do standard SVD
            }
            else
            {
                w_1_2 = 1.0; //no scaling
            }
            for (auto it = domain_->begin(l); it != domain_->end(l); ++it)
            {
                if (!it->locally_owned() || !it->has_data()) continue;
                // if (!it->data().is_allocated()) continue;
                if (it->is_leaf())
                {
                    for (std::size_t field_idx = 0; field_idx < Dim; ++field_idx)
                    {
                        for (auto& n : it->data())
                        {
                            int i_local = n(Field_idx::tag(), field_idx) - 1; //since used to be 1 based
                            if (i_local < 0) continue;
                            // int i_global =
                            //     (rank - 1) * m_local + i_local - 1;
                            PetscComplex value;
                            // PetscCall(MatGetValuesLocal(B, 1, &i_local, 1, &j, &value));
                            value = x_array[i_local];
                            n(Field_re::tag(), field_idx) = PetscRealPart(value) / w_1_2;

                            // n(Field_re::tag(), field_idx * N_tv + idx) = PetscRealPart(value) / w_1_2;
                            // n(Field_im::tag(), field_idx * N_tv + idx) = PetscImaginaryPart(value) / w_1_2;
                        }
                    }
                }
            }
        }
        VecRestoreArrayRead(x, &x_array);
        return 0.0;
    }
    template<typename F>
    void clean(bool non_leaf_only = false, int clean_width = 1) noexcept
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
                    if (domain_->dimension() == 3)
                    {
                        view(lin_data, xt::all(), xt::all(), xt::range(0, clean_width)) *= 0.0;
                        view(lin_data, xt::all(), xt::range(0, clean_width), xt::all()) *= 0.0;
                        view(lin_data, xt::range(0, clean_width), xt::all(), xt::all()) *= 0.0;
                        view(lin_data, xt::range(N + 2 - clean_width, N + 3), xt::all(), xt::all()) *= 0.0;
                        view(lin_data, xt::all(), xt::range(N + 2 - clean_width, N + 3), xt::all()) *= 0.0;
                        view(lin_data, xt::all(), xt::all(), xt::range(N + 2 - clean_width, N + 3)) *= 0.0;
                    }
                    else
                    {
                        view(lin_data, xt::all(), xt::range(0, clean_width)) *= 0.0;
                        view(lin_data, xt::range(0, clean_width), xt::all()) *= 0.0;
                        view(lin_data, xt::range(N + 2 - clean_width, N + 3), xt::all()) *= 0.0;
                        view(lin_data, xt::all(), xt::range(N + 2 - clean_width, N + 3)) *= 0.0;
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

    template<typename From, typename To>
    void add(float_type scale = 1.0) noexcept
    {
        static_assert(From::nFields() == To::nFields(), "number of fields doesn't match when add");
        for (auto it = domain_->begin(); it != domain_->end(); ++it)
        {
            if (!it->locally_owned() || !it->has_data()) continue;
            for (std::size_t field_idx = 0; field_idx < From::nFields(); ++field_idx)
            {
                it->data_r(To::tag(), field_idx).linalg().get()->cube_noalias_view() +=
                    it->data_r(From::tag(), field_idx).linalg_data() * scale;
            }
        }
    }
    template<typename From, typename To>
    void copy(float_type scale = 1.0) noexcept
    {
        static_assert(From::nFields() == To::nFields(), "number of fields doesn't match when copy");

        for (auto it = domain_->begin(); it != domain_->end(); ++it)
        {
            if (!it->locally_owned() || !it->has_data()) continue;
            for (std::size_t field_idx = 0; field_idx < From::nFields(); ++field_idx)
            {
                for (auto& n : it->data().node_field()) n(To::tag(), field_idx) = n(From::tag(), field_idx) * scale;
            }
        }
    }
    template<typename F>
    void init_idx(bool assign_base_correction = true)
    {
        boost::mpi::communicator world;
        this->clean<F>();
        // ifherk_.template clean<F_g>();
        int local_count = 0; //local count of the number of indices
        max_local_idx = -1;
        int max_idx_from_prev_prc = -1;
        if (domain_->is_server()) return; // server has no data

        int base_level = domain_->tree()->base_level();
        for (int l = base_level; l < domain_->tree()->depth(); l++) //other levels found from up down
        {
            for (auto it = domain_->begin(l); it != domain_->end(l); ++it)
            {
                if (!it->locally_owned() || !it->has_data()) continue;
                if (!it->is_leaf()) continue;
                if (it->is_correction() && !assign_base_correction) continue;
                for (std::size_t field_idx = 0; field_idx < F::nFields(); ++field_idx)
                {
                    for (auto& n : it->data())
                    {
                        local_count++;
                        n(F::tag(), field_idx) = local_count; //0  means not part of matrix so make 1 based
                    }
                }
            }
        }
        max_local_idx = local_count;
        domain_->client_communicator().barrier();
        boost::mpi::scan(domain_->client_communicator(), max_local_idx, max_idx_from_prev_prc, std::plus<float_type>());
        max_idx_from_prev_prc -=
            max_local_idx; //max idx from previous processor is the sum of all local counts from previous processors
        for (int i = 1; i < world.size(); i++)
        {
            if (world.rank() == i)
                std::cout << "rank " << world.rank() << " counter is " << local_count << " counter + max idx is "
                          << (local_count + max_idx_from_prev_prc) << " max idx from prev prc " << max_idx_from_prev_prc
                          << std::endl;
            domain_->client_communicator().barrier();
        }

        int offset_to_global =
            max_idx_from_prev_prc; //if max from last index is 1 (1 based, so 1 data point) then offset is 1
        int min_local_g_idx = max_idx_from_prev_prc +
                              1; //min global index on current processor is max idx from previous + 1 since next point
        int max_local_g_idx =
            offset_to_global + max_local_idx; //max global index on current processor is offset + local count
        // for (int l = base_level; l < domain_->tree()->depth(); l++)
        // {
        //     for (auto it = domain_->begin(l); it != domain_->end(l); ++it)
        //     {
        //         if (!it->locally_owned() || !it->has_data()) continue;
        //         if (!it->is_leaf()) continue;
        //         if (it->is_correction() && !assign_base_correction) continue;
        //         for (std::size_t field_idx = 0; field_idx < F::nFields(); ++field_idx)
        //         {
        //             for (auto& n : it->data())
        //             {
        //                 if(n(F::tag(), field_idx) > 0)
        //                 {
        //                     n(F_g::tag(), field_idx) =n(F::tag(), field_idx)+ offset_to_global;
        //                 }
        //             }
        //         }
        //     }
        // }
        if (local_count != max_local_g_idx - min_local_g_idx + 1)
        {
            std::cout << "local count is " << local_count << " max local g idx is " << max_local_g_idx
                      << " min local g idx is " << min_local_g_idx << std::endl;
            std::cout << "local count is not equal to max local g idx - min local g idx + 1" << std::endl;
        }

        return;
    }
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
  private:
    simulation_type* simulation_;
    domain_type*     domain_; ///< domain
    poisson_solver_t psolver;

    int              max_ref_level_ = 0;
    int              max_local_idx = -1;
    int              min_local_idx = -1;
    int              idxStart, nTotal, nskip;

    // poisson_solver_t   psolver_;
};

} // namespace solver
} // namespace iblgf

#endif