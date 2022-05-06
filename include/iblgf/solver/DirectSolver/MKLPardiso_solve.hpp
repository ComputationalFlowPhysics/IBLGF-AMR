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

#ifndef IBLGF_INCLUDED_MKLPARDISO_HPP
#define IBLGF_INCLUDED_MKLPARDISO_HPP

#ifndef DEBUG_IFHERK
#define DEBUG_IFHERK
#endif



//need c2f
#include "mpi.h"
#include "/home/root/intel-oneAPI/oneAPI/mkl/latest/include/mkl.h"
#include "/home/root/intel-oneAPI/oneAPI/mkl/latest/include/mkl_cluster_sparse_solver.h"

//need those defined so that xTensor does not load its own CBLAS and resulting in conflicts
//#define CXXBLAS_DRIVERS_MKLBLAS_H
//#define CXXBLAS_DRIVERS_CBLAS_H
#define CXXLAPACK_CXXLAPACK_CXX
#define WITH_MKLBLAS 1
#define CXXBLAS_DRIVERS_MKLBLAS_H 1
#define UNDEF_XT_CBLAS
//Need to add line to undef HAVE_CBLAS at driver.h in xtensor-blas

#include <iostream>
#include <vector>
#include <limits>
#include <algorithm>
#include <vector>
#include <math.h>
#include <boost/mpi.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>

// IBLGF-specific
#include <iblgf/global.hpp>
#include <iblgf/simulation.hpp>
#include <iblgf/domain/domain.hpp>
#include <iblgf/IO/parallel_ostream.hpp>
#include <iblgf/solver/poisson/poisson.hpp>
#include <iblgf/solver/Newton/Interpolation_mat.hpp>
#include <iblgf/solver/linsys/linsys.hpp>
#include <iblgf/operators/operators.hpp>
#include <iblgf/utilities/misc_math_functions.hpp>

#include <boost/serialization/map.hpp>
#include <iblgf/solver/Newton/NewtonMethod.hpp> //definition of sparse_mat here

#ifdef MKL_ILP64
#define MPI_DT MPI_LONG
#else
#define MPI_DT MPI_INT
#endif

#define MPI_REDUCE_AND_BCAST \
        MPI_Reduce(&err_mem, &error, 1, MPI_DT, MPI_SUM, 0, MPI_COMM_WORLD); \
        MPI_Bcast(&error, 1, MPI_DT, 0, MPI_COMM_WORLD);


namespace iblgf
{
namespace solver
{
using namespace domain;

template<class float_type>
class IntelPardisoSolve {
    public:
    using parm_type = std::vector<MKL_INT>;

    IntelPardisoSolve() {
        set_parms=false;
    }

    IntelPardisoSolve(parm_type _iparm, int ndim) {
        set_parms=true;
        n = ndim;
        if (_iparm.size() > 64) {
            std::cout << "custom iparm too many arguments, will use default" << std::endl;
        }

		for (int i = 0; i < 64;i++) {
			iparm[i] = 0;
		}

        if (_iparm.size() > 64) {
            //default parameters
            iparm[0] = 1; /* Solver default parameters overriden with provided by iparm */
            iparm[1] = 2;  /* Use METIS for fill-in reordering */
            iparm[5] = 0;  /* Write solution into x */
            iparm[7] = 10; /* Max number of iterative refinement steps */
            iparm[9] = 13; /* Perturb the pivot elements with 1E-13 */
            iparm[10] = 1; /* Use nonsymmetric permutation and scaling MPS */
            iparm[12] = 1; /* Switch on Maximum Weighted Matching algorithm (default for non-symmetric) */
            iparm[17] = -1; /* Output: Number of nonzeros in the factor LU */
            iparm[18] = -1; /* Output: Mflops for LU factorization */
            iparm[26] = 1; /* Check input data for correctness */
            iparm[39] = 2; /* Input: matrix/rhs/solution are distributed between MPI processes  */
        }
        else {
            for (int i = 0; i < _iparm.size(); i++) { iparm[i] = _iparm[i]; }
        }

        mpi_stat = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        mpi_stat = MPI_Comm_size(MPI_COMM_WORLD, &size);
        comm = MPI_Comm_c2f(MPI_COMM_WORLD);

        maxfct = 1;
        mnum = 1;
        msglvl = 1;
        error = 0;
        err_mem = 0;
    }

    void set_parameters(parm_type _iparm, int ndim) {
        set_parms=true;
        n = ndim;
        if (_iparm.size() > 64) {
            std::cout << "custom iparm too many arguments, will use default" << std::endl;
        }

		for (int i = 0; i < 64;i++) {
			iparm[i] = 0;
		}

        if (_iparm.size() > 64) {
            //default parameters
            iparm[0] = 1; /* Solver default parameters overriden with provided by iparm */
            iparm[1] = 2;  /* Use METIS for fill-in reordering */
            iparm[5] = 0;  /* Write solution into x */
            iparm[7] = 10; /* Max number of iterative refinement steps */
            iparm[9] = 13; /* Perturb the pivot elements with 1E-13 */
            iparm[10] = 1; /* Use nonsymmetric permutation and scaling MPS */
            iparm[12] = 1; /* Switch on Maximum Weighted Matching algorithm (default for non-symmetric) */
            iparm[17] = -1; /* Output: Number of nonzeros in the factor LU */
            iparm[18] = -1; /* Output: Mflops for LU factorization */
            iparm[26] = 1; /* Check input data for correctness */
            iparm[39] = 2; /* Input: matrix/rhs/solution are distributed between MPI processes  */
        }
        else {
            for (int i = 0; i < _iparm.size(); i++) { iparm[i] = _iparm[i]; }
        }

        mpi_stat = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        mpi_stat = MPI_Comm_size(MPI_COMM_WORLD, &size);
        comm = MPI_Comm_c2f(MPI_COMM_WORLD);

        maxfct = 1;
        mnum = 1;
        msglvl = 1;
        error = 0;
        err_mem = 0;
    }
    template<class NewtonIteration>
    int load_matrix(sparse_mat& mat, NewtonIteration& ifherk) {
        if (!set_parms) {
            std::cout << "parameters are not set in Pardiso, call set_parameters()" << std::endl;
            return 1;
        }
        if (world.rank() != 0) {
            int begin_row = ifherk.num_start();
            int end_row = ifherk.num_end();
            iparm[40] = begin_row;
            iparm[41] = end_row;

            int loc_size = end_row - begin_row+1;
            int check_size = mat.numRow_loc();

            if (loc_size != check_size) {
                std::cout << "Rank " << world.rank() << " local matrix size does not match " << loc_size << " vs " << check_size << std::endl;
                return 1;
            }

            int tot_size = mat.tot_size();
            ia = (MKL_INT*)MKL_malloc(sizeof(MKL_INT) * (loc_size + 1), 64);
            ja = (MKL_INT*)MKL_malloc(sizeof(MKL_INT) * tot_size, 64);
            a = (float_type*)MKL_malloc(sizeof(float_type) * tot_size, 64);
            for (int k =0 ; k < tot_size; k++) {
                a[k] = 0;
            }

            mat.getCSR(ia, ja, a);
        }
        else {
            iparm[40] = 2;
			iparm[41] = 1;
        }
        return 0;
    }

    template<class NewtonIteration>
    int load_RHS(sparse_mat& mat, NewtonIteration& ifherk, float_type* b_res) {
        if (!set_parms) {
            std::cout << "parameters are not set in Pardiso, call set_parameters()" << std::endl;
            return 1;
        }
        if (world.rank() != 0) {
            int begin_row = ifherk.num_start();
            int end_row = ifherk.num_end();
            int loc_size = end_row - begin_row+1;
            int check_size = mat.numRow_loc();

            if (loc_size != check_size) {
                std::cout << "Rank " << world.rank() << " local matrix size does not match " << loc_size << " vs " << check_size << std::endl;
                return 1;
            }

            x = (float_type*)MKL_malloc(sizeof(float_type) * loc_size, 64);
            b = (float_type*)MKL_malloc(sizeof(float_type) * loc_size, 64);
            for (int k =0 ; k < loc_size; k++) {
                x[k] = 0;
                b[k] = b_res[k];
            }
        }
        else {

        }
        return 0;
    }

    int reordering() {
        if (!set_parms) {
            std::cout << "parameters are not set in Pardiso, call set_parameters()" << std::endl;
            return 1;
        }
        MPI_Barrier(MPI_COMM_WORLD);
        phase = 11;
        cluster_sparse_solver(pt, &maxfct, &mnum, &mtype, &phase, &n, a, ia, ja,
            &idum, &nrhs, iparm, &msglvl, &ddum, &ddum, &comm, &error);
        if (error != 0)
        {
            if (rank == 0)
                std::cout << "\nERROR during symbolic factorization: " << 
                    error << std::endl;
            return 1;
        }

        if (rank == 0) std::cout << "\nReordering completed ... " << std::endl;
        return 0;
    }

    int factorization() {
        if (!set_parms) {
            std::cout << "parameters are not set in Pardiso, call set_parameters()" << std::endl;
            return 1;
        }
        phase = 22;
        cluster_sparse_solver(pt, &maxfct, &mnum, &mtype, &phase, &n, a, ia, ja,
            &idum, &nrhs, iparm, &msglvl, &ddum, &ddum, &comm, &error);
        if (error != 0)
        {
            if (rank == 0)
                printf("\nERROR during numerical factorization: %lli",
                    (long long int)error);
            return 1;
        }
        if (rank == 0) printf("\nFactorization completed ... ");
        return 0;
    }

    int back_substitution() {
        if (!set_parms) {
            std::cout << "parameters are not set in Pardiso, call set_parameters()" << std::endl;
            return 1;
        }
        phase = 33;

        if (rank == 0) printf("\nSolving system...");
        cluster_sparse_solver(pt, &maxfct, &mnum, &mtype, &phase, &n, a, ia, ja,
            &idum, &nrhs, iparm, &msglvl, b, x, &comm, &error);
        if (error != 0)
        {
            if (world.rank() == 0)
                std::cout << "\nERROR during solution: " << error << std::endl;
            return 1;
        }
        return 0;
    }

    int release_internal_mem () {
        if (!set_parms) {
            std::cout << "parameters are not set in Pardiso, call set_parameters()" << std::endl;
            return 1;
        }
        phase = -1; /* Release internal memory. */
        cluster_sparse_solver(pt, &maxfct, &mnum, &mtype, &phase, &n, &ddum, ia,
            ja, &idum, &nrhs, iparm, &msglvl, &ddum, &ddum, &comm, &error);
        if (error != 0)
        {
            if (rank == 0)
                std::cout << "\nERROR during release memory: " << 
                    error << std::endl;
            return 1;
        }

        MPI_Barrier(MPI_COMM_WORLD);
        return 0;
    }
    template<class NewtonIteration>
    int getSolution(sparse_mat& mat, NewtonIteration& ifherk, float_type* output) {
        if (!set_parms) {
            std::cout << "parameters are not set in Pardiso, call set_parameters()" << std::endl;
            return 1;
        }
        if (world.rank() != 0) {
            int begin_row = ifherk.num_start();
            int end_row = ifherk.num_end();
            int loc_size = end_row - begin_row+1;
            int check_size = mat.numRow_loc();

            if (loc_size != check_size) {
                std::cout << "Rank " << world.rank() << " local matrix size does not match " << loc_size << " vs " << check_size << std::endl;
                return 1;
            }

            //x = (float_type*)MKL_malloc(sizeof(float_type) * loc_size, 64);
            //output = (float_type*)MKL_malloc(sizeof(float_type) * loc_size, 64);
            for (int k =0 ; k < loc_size; k++) {
                output[k] = x[k];
            }
        }
        return 0;
    }

    void FreeSolver() {
        if (!set_parms) {
            std::cout << "parameters are not set in Pardiso, call set_parameters()" << std::endl;
            return;
        }
        if (rank  != 0)
        {
            MKL_free(ia);
            MKL_free(ja);
            MKL_free(a);
            MKL_free(x);
            MKL_free(b);
        }
    }

  private:
    MKL_INT iparm[64];
    MKL_INT n;
    MKL_INT mtype = 11;

    MKL_INT nrhs = 1;
    void*   pt[64] = {0};

    MKL_INT maxfct, mnum, phase, msglvl, error, err_mem;

    //int comm;

    float_type ddum; /* Double dummy   */
    MKL_INT    idum; /* Integer dummy. */
    MKL_INT    j;

    

    MKL_INT*    ia = NULL;
    MKL_INT*    ja = NULL;
    float_type* a = NULL;
    /* RHS and solution vectors. */
    float_type* b = NULL;
    float_type* x = NULL;

    bool set_parms = false;

    int    mpi_stat = 0;
    int    argc = 0;
    int    comm, rank, size;
    char** argv;

    boost::mpi::communicator world;
};

} // namespace solver
} // namespace iblgf

#endif
