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

#ifndef IBLGF_INCLUDED_SCALAPACK_IB_HPP
#define IBLGF_INCLUDED_SCALAPACK_IB_HPP

#ifndef DEBUG_IFHERK
#define DEBUG_IFHERK
#endif



//need c2f
#include "mpi.h"
#include <petscksp.h>

//need those defined so that xTensor does not load its own CBLAS and resulting in conflicts
//#define CXXBLAS_DRIVERS_MKLBLAS_H
//#define CXXBLAS_DRIVERS_CBLAS_H
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


namespace iblgf
{
namespace solver
{
using namespace domain;

template<class Setup>
class DirectIB {
    //This routine depend on PETSC, thus need to run PetscInitialize at the very beginning
    public:

    static constexpr std::size_t N_modes = Setup::N_modes; 

    using simulation_type = typename Setup::simulation_t;
	using domain_type = typename simulation_type::domain_type;
	using ib_t = typename domain_type::ib_t;
    using force_type = typename ib_t::force_type;
    using point_force_type = types::vector_type<float_type, 3 * 2 * N_modes>;
	using real_coordinate_type = typename domain_type::real_coordinate_type;

    using u_type = typename Setup::u_type;    

    DirectIB(simulation_type* _simulation)
    : domain_(_simulation->domain_.get()) {
        boost::mpi::communicator world;

        PetscMPIInt    rank, size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        Color = -1;

        int force_dim = domain_->ib().size()*(u_type::nFields())/N_modes;

        n = force_dim;

        int ProcDist[N_modes]; //array holding the Processes distribution on each mode

        

        for (int i = 0; i < N_modes; i++) {
            ProcDist[i] = 0;
        }

        if ((size - 1) > N_modes) {
            localModes.resize(1);
            int nProc = (size - 1) / N_modes;
            int res_proc = (size - 1) % N_modes;
            for (int i = 0; i < N_modes; i++) { 
                if (i < res_proc) ProcDist[i] = nProc + 1;
                else  ProcDist[i] = nProc;
            }
            int cumNum = 0;
            for (int i = 0; i < N_modes; i++) {
                if ((rank - 1) >= cumNum && (rank - 1) < (cumNum + ProcDist[i])) {
                    localModes[0] = i;
                    break;
                }
                cumNum += ProcDist[i];
            }
        }
        else {
            //processes <= N_modes
            
            int numLocModes = N_modes / (size - 1);
            int startMode = 0;
            if (N_modes % (size - 1) > (rank - 1)) {
                numLocModes += 1;
            }
            if (N_modes % (size - 1) > (rank - 1)) {
                startMode = (rank - 1) * numLocModes;
            }
            else {
                startMode = (N_modes % (size - 1)) * (numLocModes + 1) + (rank - 1 - (N_modes % (size - 1))) * numLocModes;
            }
            localModes.resize(numLocModes);
            for (int i = 0; i < numLocModes; i++) {
                localModes[i] = startMode + i;
            }
        }
        if (world.rank() != 0) {
            if ((size - 1) > N_modes) Color = localModes[0];
            else Color = rank;
        }

        MPI_Comm_split(MPI_COMM_WORLD, Color, 0, &PETSC_COMM_WORLD);


        int ModeSize = localModes.size();
        
        x.resize(ModeSize);
        b.resize(ModeSize);
        u.resize(ModeSize);
        
        A.resize(ModeSize);
        
        ksp.resize(ModeSize);
        
        pc.resize(ModeSize);
        
        norm.resize(ModeSize);
        tol.resize(ModeSize);
        for (int i = 0; i < ModeSize; i++) {
            tol[i] = 1000. * PETSC_MACHINE_EPSILON;
        }
        rstartx.resize(ModeSize);
        rstartb.resize(ModeSize);
        rendx.resize(ModeSize);
        rendb.resize(ModeSize);

    }

    DirectIB(simulation_type* _simulation, std::vector<int> localModes_, bool splitting_comm)
    : domain_(_simulation->domain_.get()) {
        boost::mpi::communicator world;

        PetscMPIInt    rank, size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        Color = -1;

        int force_dim = domain_->ib().size()*(u_type::nFields())/N_modes;

        n = force_dim;

        localModes = localModes_;
        if (world.rank() != 0) {
            if ((size - 1) > N_modes) Color = localModes[0];
            else Color = rank;
        }

        if (splitting_comm) MPI_Comm_split(MPI_COMM_WORLD, Color, 0, &PETSC_COMM_WORLD);


        int ModeSize = localModes.size();
        
        x.resize(ModeSize);
        b.resize(ModeSize);
        u.resize(ModeSize);
        
        A.resize(ModeSize);
        
        ksp.resize(ModeSize);
        
        pc.resize(ModeSize);
        
        norm.resize(ModeSize);
        tol.resize(ModeSize);
        for (int i = 0; i < ModeSize; i++) {
            tol[i] = 1000. * PETSC_MACHINE_EPSILON;
        }
        rstartx.resize(ModeSize);
        rstartb.resize(ModeSize);
        rendx.resize(ModeSize);
        rendb.resize(ModeSize);

    }

    int load_matrix(std::vector<force_type>& mat_, bool summed = true) {
        boost::mpi::communicator world;
        mat = mat_;
        std::vector<force_type> mat_loc = mat_;
        if (world.rank() != 0)
        {
            if (!summed)
            {
                for (int k = 0; k < mat_loc.size(); k++)
                {
                    for (int i = 0; i < mat_loc[i].size(); i++)
                    {
                        if (domain_->ib().rank(i) != world.rank())
                        {
                            for (int j = 0; j < mat_loc[i][0].size(); j++)
                            {
                                mat_loc[k][i][j] = 0.0;
                            }
                        }
                    }
                    domain_->client_communicator().barrier();
                    boost::mpi::all_reduce(domain_->client_communicator(),
                        &mat_loc[k][0], domain_->ib().size(), &mat[k][0],
                        std::plus<point_force_type>());
                }
            }
        }

        if (Color == -1) return 0;


        for (int ModeNum = 0; ModeNum < localModes.size(); ModeNum++) {

            int ModeIdx = localModes[ModeNum];

            PetscMPIInt rank, size;
            MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
            MPI_Comm_size(PETSC_COMM_WORLD, &size);


            PetscScalar one = 1.0, value[3], zero = 0.0;        
            
            //PetscCall(VecGetLocalSize(x, &nlocal));

            //PetscCall(MatCreateDense(PETSC_COMM_WORLD,PETSC_DECIDE, PETSC_DECIDE, n, n, NULL, &A));
            //MatSetType(A,MATSCALAPACK);
            PetscCall(MatCreateScaLAPACK(PETSC_COMM_WORLD,PETSC_DECIDE , PETSC_DECIDE , n, n, 0,0, &A[ModeNum]));
            //PetscCall(MatSetSizes(A, nlocal, nlocal, n, n));
            PetscCall(MatSetFromOptions(A[ModeNum]));
            PetscCall(MatSetUp(A[ModeNum]));

            PetscCall(MatCreateVecs(A[ModeNum],&x[ModeNum],&b[ModeNum]));

            //PetscCall(VecSetFromOptions(x[ModeNum]));
            PetscCall(VecDuplicate(x[ModeNum], &u[ModeNum]));

            PetscCall(VecGetOwnershipRange(x[ModeNum], &rstartx[ModeNum], &rendx[ModeNum]));

            PetscCall(VecGetOwnershipRange(b[ModeNum], &rstartb[ModeNum], &rendb[ModeNum]));

            //std::cout << "rank " << rank << " start and end " << rstartx[ModeNum] << " "
            //          << rendx[ModeNum] << " " << rstartb[ModeNum] << " " << rendb[ModeNum] << std::endl;

            for (int i = rstartb[ModeNum]; i < rendb[ModeNum]; i++)
            {
                
                for (int j = 0; j < n; j++)
                {
                    int ib_idx_i = i / ((u_type::nFields()) / N_modes);
                    int field_idx_i = i % ((u_type::nFields()) / N_modes);
                    int idx_complex_i =
                        field_idx_i /
                        2; //the number of components (zero for u, one for v, two for w)
                    int realcomp_i =
                        field_idx_i %
                        2; //zero if real part but one if complex part

                    int field_idx_now_i =
                        idx_complex_i * N_modes * 2 + realcomp_i;
                    int field_idx_short_i = idx_complex_i*N_modes*2 + realcomp_i;

                    int ib_idx_j = j / ((u_type::nFields()) / N_modes);
                    int field_idx_j = j % ((u_type::nFields()) / N_modes);
                    int idx_complex_j =
                        field_idx_j /
                        2; //the number of components (zero for u, one for v, two for w)
                    int realcomp_j =
                        field_idx_j %
                        2; //zero if real part but one if complex part

                    int field_idx_now_j =
                        idx_complex_j * N_modes * 2 + realcomp_j  + ModeIdx*2;

                    float_type val_ij =
                        mat[i][ib_idx_j][field_idx_now_j];

                    PetscScalar valA = val_ij;
                    PetscCall(MatSetValues(A[ModeNum], 1, &i, 1, &j, &valA,
                        INSERT_VALUES));
                }
            }

            MatAssemblyBegin(A[ModeNum],MAT_FINAL_ASSEMBLY);
            MatAssemblyEnd(A[ModeNum],MAT_FINAL_ASSEMBLY);

            PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp[ModeNum]));
            PetscCall(KSPSetOperators(ksp[ModeNum], A[ModeNum], A[ModeNum]));

            PetscCall(KSPGetPC(ksp[ModeNum], &pc[ModeNum]));
            PetscCall(PCSetType(pc[ModeNum], PCLU));

            PetscCall(KSPSetTolerances(ksp[ModeNum], 1.e-7, PETSC_DEFAULT, PETSC_DEFAULT,
                PETSC_DEFAULT));

            PetscCall(KSPSetFromOptions(ksp[ModeNum]));
        }
        return 0;
    }

    int load_RHS(force_type& forcing, bool summed = true) {

        boost::mpi::communicator world;
        if (world.rank() == 0) return 0;

        force_type forcing_loc = forcing;
        force_type forcing_glob = forcing;
        if (world.rank() != 0)
        {
            if (!summed)
            {
                
                    for (int i = 0; i < forcing_loc.size(); i++)
                    {
                        if (domain_->ib().rank(i) != world.rank())
                        {
                            for (int j = 0; j < forcing_loc[0].size(); j++)
                            {
                                forcing_loc[i][j] = 0.0;
                            }
                        }
                    }
                    boost::mpi::all_reduce(domain_->client_communicator(),
                        &forcing_loc[0], domain_->ib().size(), &forcing_glob[0],
                        std::plus<point_force_type>());
            }
        }

        for (int ModeNum = 0; ModeNum < localModes.size(); ModeNum++) {

            int ModeIdx = localModes[ModeNum];

            PetscMPIInt rank, size;
            MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
            MPI_Comm_size(PETSC_COMM_WORLD, &size);


            for (int i = rstartb[ModeNum]; i < rendb[ModeNum]; i++)
            {
                int ib_idx_i = i / ((u_type::nFields()) / N_modes);
                int field_idx_i = i % ((u_type::nFields()) / N_modes);
                int idx_complex_i =
                    field_idx_i /
                    2; //the number of components (zero for u, one for v, two for w)
                int realcomp_i =
                    field_idx_i % 2; //zero if real part but one if complex part

                int field_idx_now_i = idx_complex_i * N_modes * 2 + realcomp_i + ModeIdx*2;

                PetscScalar v = forcing_glob[ib_idx_i][field_idx_now_i];
                PetscCall(VecSetValues(b[ModeNum], 1, &i, &v, INSERT_VALUES));
            }

            VecAssemblyBegin(b[ModeNum]);
            VecAssemblyEnd(b[ModeNum]);
        }
        return 0;
    }

    
    int getSolution(force_type& res) {
        boost::mpi::communicator world;

        point_force_type tmp(0.0);
        force_type res_loc(domain_->ib().size(), tmp);

        if (world.rank() == 0) return 0;
        for (int ModeNum = 0; ModeNum < localModes.size(); ModeNum++) {

            PetscCall(KSPSolve(ksp[ModeNum], b[ModeNum], x[ModeNum]));

            int ModeIdx = localModes[ModeNum];

            PetscMPIInt rank, size;
            MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
            MPI_Comm_size(PETSC_COMM_WORLD, &size);


            for (int i = rstartx[ModeNum]; i < rendx[ModeNum]; i++)
            {
                int ib_idx_i = i / ((u_type::nFields()) / N_modes);
                int field_idx_i = i % ((u_type::nFields()) / N_modes);
                int idx_complex_i =
                    field_idx_i /
                    2; //the number of components (zero for u, one for v, two for w)
                int realcomp_i =
                    field_idx_i % 2; //zero if real part but one if complex part

                int field_idx_now_i = idx_complex_i * N_modes * 2 + realcomp_i + ModeIdx*2;

                PetscScalar v;
                PetscCall(VecGetValues(x[ModeNum], 1, &i, &v));
                res_loc[ib_idx_i][field_idx_now_i] = PetscRealPart(v);
            }
        }

        boost::mpi::all_reduce(domain_->client_communicator(),
                        &res_loc[0], domain_->ib().size(), &res[0],
                        std::plus<point_force_type>());

        return 0;

    }

    std::vector<int> getlocal_modes() {
        std::vector<int> _localModes = localModes;
        return _localModes;
    }

  private:
    std::vector<force_type> mat;

    PetscMPIInt Color = -1; //this is the Color for splitting communicators
    std::vector<int> localModes;
    domain_type*     domain_;
    PetscInt         n; //dimension of the matrix 
    //storing the modes need to be computed in this processor
    //if number of modes is larger than number of processors, the size can be bigger than one
    //otherwise, equal to one

    std::vector<Vec>       x, b, u; /* approx solution, RHS, exact solution */
    std::vector<Mat>       A;       /* linear system matrix */
    std::vector<KSP>       ksp;     /* linear solver context */
    std::vector<PC>        pc;      /* preconditioner context */
    std::vector<PetscReal> norm, tol; /* norm of solution error */
    std::vector<PetscInt> rstartx, its, rendx, rstartb, rendb;
};

} // namespace solver
} // namespace iblgf

#endif
