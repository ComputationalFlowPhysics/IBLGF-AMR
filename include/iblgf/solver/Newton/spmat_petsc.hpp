#ifndef IBLGF_INCLUDED_SOLVER_SPMAT_PETSC_HPP
#define IBLGF_INCLUDED_SOLVER_SPMAT_PETSC_HPP

// NewtonIteration's assembled sparse_mat (Jac, B: local 1-based rows, global
// 1-based columns) -> PETSc MPIAIJ, plus the small helpers the assembled
// stability / Jacobian-check drivers share (row ownership, y = A x on
// std::vector local slices).

#include <vector>

#include <petscmat.h>

#include <iblgf/global.hpp>
#include <iblgf/solver/Newton/SparseMat.hpp>

namespace iblgf
{
namespace solver
{
namespace spmat_petsc
{
/** @brief PETSc's default contiguous row ownership for nlocal local rows of
 *  an N-row distributed object. */
inline PetscErrorCode ownership(
    PetscInt N, PetscInt nlocal, PetscInt* rstart, PetscInt* rend)
{
    Vec s;
    PetscCall(VecCreate(PETSC_COMM_WORLD, &s));
    PetscCall(VecSetSizes(s, nlocal, N));
    PetscCall(VecSetFromOptions(s));
    PetscCall(VecGetOwnershipRange(s, rstart, rend));
    PetscCall(VecDestroy(&s));
    return 0;
}

/** @brief PETSc MPIAIJ from a sparse_mat (local 1-based rows, global 1-based
 *  cols). force_diag: insert an explicit 0 on rows without a diagonal entry
 *  (keeps a shift-invert factorization structurally sound). */
inline PetscErrorCode assemble(sparse_mat& S, PetscInt N, PetscInt nlocal,
    PetscInt rstart, PetscInt rend, bool force_diag, Mat* M)
{
    std::vector<PetscInt> d_nnz(nlocal, 0), o_nnz(nlocal, 0);
    for (PetscInt i = rstart; i < rend; ++i)
    {
        PetscInt i_loc = i - rstart;
        bool     has_diag = false;
        for (const auto& [key, val] : S.mat[i_loc + 1])
        {
            PetscInt col = key - 1;
            if (col >= rstart && col < rend) ++d_nnz[i_loc];
            else
                ++o_nnz[i_loc];
            if (col == i) has_diag = true;
        }
        if (force_diag && !has_diag) ++d_nnz[i_loc];
    }

    PetscCall(MatCreateAIJ(PETSC_COMM_WORLD, nlocal, nlocal, N, N, 0,
        nlocal ? d_nnz.data() : NULL, 0, nlocal ? o_nnz.data() : NULL, M));
    PetscCall(MatSetFromOptions(*M));
    PetscCall(MatSetUp(*M));

    for (PetscInt i = rstart; i < rend; ++i)
    {
        std::vector<PetscInt>    cols;
        std::vector<PetscScalar> vals;
        bool                     has_diag = false;
        for (const auto& [key, val] : S.mat[i - rstart + 1])
        {
            cols.push_back(key - 1);
            vals.push_back(val);
            if (key - 1 == i) has_diag = true;
        }
        if (!cols.empty())
            PetscCall(MatSetValues(*M, 1, &i, (PetscInt)cols.size(),
                cols.data(), vals.data(), INSERT_VALUES));
        if (force_diag && !has_diag)
        {
            PetscScalar zero = 0.0;
            PetscCall(MatSetValues(*M, 1, &i, 1, &i, &zero, INSERT_VALUES));
        }
    }
    PetscCall(MatAssemblyBegin(*M, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(*M, MAT_FINAL_ASSEMBLY));
    return 0;
}

/** @brief y = A x for local slices held in std::vectors (real parts). */
inline PetscErrorCode matvec(Mat A, const std::vector<float_type>& x,
    std::vector<float_type>& y, PetscInt nlocal)
{
    Vec X, Y;
    PetscCall(MatCreateVecs(A, &X, &Y));
    PetscScalar* a;
    PetscCall(VecGetArray(X, &a));
    for (PetscInt i = 0; i < nlocal; ++i) a[i] = x[i];
    PetscCall(VecRestoreArray(X, &a));
    PetscCall(MatMult(A, X, Y));
    const PetscScalar* b;
    PetscCall(VecGetArrayRead(Y, &b));
    for (PetscInt i = 0; i < nlocal; ++i) y[i] = PetscRealPart(b[i]);
    PetscCall(VecRestoreArrayRead(Y, &b));
    PetscCall(VecDestroy(&X));
    PetscCall(VecDestroy(&Y));
    return 0;
}

} // namespace spmat_petsc
} // namespace solver
} // namespace iblgf

#endif
