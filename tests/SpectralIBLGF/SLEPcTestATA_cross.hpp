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

#ifndef IBLGF_INCLUDED_NEWTON_OPERATPOR_HPP
#define IBLGF_INCLUDED_NEWTON_OPERATPOR_HPP

#ifndef DEBUG_IFHERK
#define DEBUG_IFHERK
#endif
#define DEBUG_POISSON

static char help[] = "Solves a tridiagonal linear system.\n\n";

//need c2f
#include "mpi.h"
#include <iostream>
#include <slepceps.h>
#include <slepcsvd.h>
#include <petscksp.h>
#include <petscsys.h>
//#include "/home/root/intel-oneAPI/oneAPI/mkl/latest/include/mkl.h"
//#include "/home/root/intel-oneAPI/oneAPI/mkl/latest/include/mkl_cluster_sparse_solver.h"

//need those defined so that xTensor does not load its own CBLAS and resulting in conflicts
//#define CXXBLAS_DRIVERS_MKLBLAS_H
//#define CXXBLAS_DRIVERS_CBLAS_H
//#define CXXLAPACK_CXXLAPACK_CXX

#include <iostream>
#include <vector>
#include <limits>
#include <algorithm>
#include <vector>
#include <math.h>
#include <chrono>
#include <fftw3.h>
#include <boost/mpi.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>




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

#include "../../setups/setup_Helm_stab.hpp"
#include <iblgf/operators/operators.hpp>

KSP kspA, kspAT;
PC pcA, pcAT;
Vec y_tmp, Cy, By;
Mat C, Q_hi; //Q_hi is Q^(-1/2)

PetscErrorCode Id(Mat A_, Vec x_, Vec y_) {
        //PetscCall(KSPSolve(kspA, x, y_tmp));
        //PetscCall(KSPSolve(kspAT, y_tmp, y));
        PetscFunctionReturn(0);
    }

/*PetscErrorCode MatMul_ATA_inv(Mat A_, Vec x_, Vec y_) {
    PetscCall(MatMult(C, x_, By));
    PetscCall(KSPSolve(kspAT, By, y_tmp));
    PetscCall(MatMult(C, y_tmp, By));
    PetscCall(KSPSolve(kspA, By, Cy));
    PetscCall(MatMult(C, Cy, y_));
    //PetscCall(KSPSolve(kspAT, By, y_));
    PetscFunctionReturn(0);
}*/


PetscErrorCode MatMul_ATA_inv(Mat A_, Vec x_, Vec y_) {
    PetscCall(MatMult(Q_hi, x_, By));
    PetscCall(KSPSolve(kspA, By, y_tmp));
    PetscCall(MatMult(C, y_tmp, By));
    PetscCall(KSPSolve(kspAT, By, Cy));
    PetscCall(MatMult(Q_hi, Cy, y_));
    //PetscCall(KSPSolve(kspAT, By, y_));
    PetscFunctionReturn(0);
}

PetscErrorCode MatMulA_inv(Mat A_, Vec x_, Vec y_) {
    //PetscCall(MatMul(C, x_, Cy));
    PetscCall(KSPSolve(kspA, x_, By));
    PetscCall(MatMult(C, By, y_));
    //PetscCall(KSPSolve(kspAT, y_tmp, y_));
    PetscFunctionReturn(0);
}


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
        //name               type        Dim   lBuffer  hBuffer, storage type
        
         (test             , float_type, 1,    1,       1,     cell,false ),
        //IF-HERK
		 (u                , float_type, 2,    1,       1,     face,true  ),
         (u_n              , float_type, 2,    1,       1,     face,true  ), //temp var to store velocity for time averaging
         (uz               , float_type, 1,    1,       1,     cell,true  ),
         (p                , float_type, 1,    1,       1,     cell,true  ),
         (u_ref            , float_type, 2,    1,       1,     face,true  ),
         (uz_ref           , float_type, 1,    1,       1,     cell,true  ),
		 (p_ref            , float_type, 1,    1,       1,     cell,true  ),
         (Nz_ref           , float_type, 1,    1,       1,     cell,true  ),
         (w_ref            , float_type, 3,    1,       1,     edge,true  ),
         (N_ref            , float_type, 2,    1,       1,     face,true  ),
         (cs_ref           , float_type, 1,    1,       1,     cell,true  ),
         (uz_n             , float_type, 1,    1,       1,     cell,true  )  //temp var to store u_z for time averaging
		 
    ))
    // clang-format on
};

struct NS_AMR_LGF : public SetupHelmStab<NS_AMR_LGF, parameters>
{
    using super_type = SetupHelmStab<NS_AMR_LGF,parameters>;
    using vr_fct_t = std::function<float_type(float_type x, float_type y, int field_idx, bool perturbation)>;

    //Timings
    using clock_type = std::chrono::high_resolution_clock;
    using milliseconds = std::chrono::milliseconds;
    using duration_type = typename clock_type::duration;
    using time_point_type = typename clock_type::time_point;

	using simulation_type = typename super_type::simulation_t;
	using domain_type = typename simulation_type::domain_type;
	using ib_t = typename domain_type::ib_t;
    using force_type = typename ib_t::force_type;

	using real_coordinate_type = typename domain_type::real_coordinate_type;

	NS_AMR_LGF(Dictionary* _d)
		: super_type(_d, [this](auto _d, auto _domain) {
		return this->initialize_domain(_d, _domain);
			})
	{
		if (domain_->is_client()) client_comm_ = client_comm_.split(1);
		else
			client_comm_ = client_comm_.split(0);

		//smooth_start_ = simulation_.dictionary()->template get_or<bool>("smooth_start", false);
		//U_.resize(domain_->dimension());
        U_.resize(3);
		U_[0] = simulation_.dictionary()->template get_or<float_type>("Ux", 1.0);
		U_[1] = simulation_.dictionary()->template get_or<float_type>("Uy", 0.0);
        //U_[1] = simulation_.dictionary()->template get_or<float_type>("Uy", 0.0);
		//if (domain_->dimension()>2)
		U_[2] = simulation_.dictionary()->template get_or<float_type>("Uz", 0.0);

		smooth_start_ = simulation_.dictionary()->template get_or<bool>("smooth_start", false);

		vortexType = simulation_.dictionary()->template get_or<int>("Vort_type", 0);

		pert_mag = simulation_.dictionary()->template get_or<float_type>("pert_mag", 0.1);

		simulation_.frame_vel() =
			[this](std::size_t idx, float_type t, auto coord = {0, 0})
			{
                return -U_[idx];
				/*float_type T0 = 0.5;
				if (t<=0.0 && smooth_start_)
					return 0.0;
				else if (t<T0-1e-10 && smooth_start_)
				{
					float_type h1 = exp(-1/(t/T0));
					float_type h2 = exp(-1/(1 - t/T0));

					return -U_[idx] * (h1/(h1+h2));
				}
				else
				{
					return -U_[idx];
				}*/
			};

		/*simulation_.frame_vel() =
			[this](std::size_t idx, float_type t, auto coord = {0, 0, 0})
			{return 0.0;};*/


		dx_ = domain_->dx_base();
		cfl_ =
			simulation_.dictionary()->template get_or<float_type>("cfl", 0.2);
		dt_ = simulation_.dictionary()->template get_or<float_type>("dt", -1.0);

		tot_steps_ = simulation_.dictionary()->template get<int>("nBaseLevelTimeSteps");
		Re_ = simulation_.dictionary()->template get<float_type>("Re");
		R_ = simulation_.dictionary()->template get<float_type>("R");
		d2v_ = simulation_.dictionary()->template get_or<float_type>("DistanceOfVortexRings", R_);
		v_delta_ = simulation_.dictionary()->template get_or<float_type>("vDelta", 0.2 * R_);
		single_ring_ = simulation_.dictionary()->template get_or<bool>("single_ring", true);
		perturbation_ = simulation_.dictionary()->template get_or<bool>("perturbation", false);
		vort_sep = simulation_.dictionary()->template get_or<float_type>("vortex_separation", 1.0 * R_);
		hard_max_refinement_ = simulation_.dictionary()->template get_or<bool>("hard_max_refinement", false);

        clean_p_tar = simulation_.dictionary()->template get_or<bool>("clean_p_tar", false);
        target_real = simulation_.dictionary()->template get_or<float_type>("target_real", 0.0);
        target_imag = simulation_.dictionary()->template get_or<float_type>("target_imag", 0.0);
        testing_smearing = simulation_.dictionary()->template get_or<bool>("testing_smearing", false);
        num_input = simulation_.dictionary()->template get_or<bool>("num_input", false);
        check_mat_res = simulation_.dictionary()->template get_or<bool>("check_mat_res", false);
        addImagBC = simulation_.dictionary()->template get_or<bool>("addImagBC", false);
        Omega_w = simulation_.dictionary()->template get_or<float_type>("Omega_w", 0.0);


        Omega_l = simulation_.dictionary()->template get_or<float_type>("Omega_l", 0.0);
        Omega_r = simulation_.dictionary()->template get_or<float_type>("Omega_r", 1.0);
        Omega_int = simulation_.dictionary()->template get_or<float_type>("Omega_int", 0.1);

		auto domain_range = domain_->bounding_box().max() - domain_->bounding_box().min();
		Lx = domain_range[0] * dx_;



		ctr_dis_x = 0.0*dx_; //this is setup as the center of the vortex in the unit of grid spacing
		ctr_dis_y = 0.0*dx_;


		



		bool use_fat_ring = simulation_.dictionary()->template get_or<bool>("fat_ring", false);

		ic_filename_ = simulation_.dictionary_->template get_or<std::string>(
			"hdf5_ic_name", "null");

		ref_filename_ = simulation_.dictionary_->template get_or<std::string>(
			"hdf5_ref_name", "null");

		source_max_ = simulation_.dictionary_->template get_or<float_type>(
			"source_max", 1.0);

		refinement_factor_ = simulation_.dictionary_->template get<float_type>(
			"refinement_factor");

		base_threshold_ = simulation_.dictionary()->
			template get_or<float_type>("base_level_threshold", 1e-4);

		nLevelRefinement_ = simulation_.dictionary_->
			template get_or<int>("nLevels", 0);

		hard_max_level_ = simulation_.dictionary()->template get_or<int>("hard_max_level", nLevelRefinement_);

		global_refinement_ = simulation_.dictionary_->template get_or<int>(
			"global_refinement", 0);

        row_to_print = simulation_.dictionary_->template get_or<int>(
			"row_to_print", 100);

        N_pts = simulation_.dictionary()->template get_or<int>("N_pts", 27);

        print_mat = simulation_.dictionary_->template get_or<bool>(
			"print_mat",  false);

        set_deflation = simulation_.dictionary_->template get_or<bool>(
			"set_deflation",  false);

        n_times = simulation_.dictionary_->template get<int>(
			"n_times");

        n_start = simulation_.dictionary_->template get<int>(
			"n_start");

        flow_interval = simulation_.dictionary_->template get<int>(
			"flow_interval");

		//bool subtract_non_leaf_ = simulation_.dictionaty()->template get_or<bool>("subtract_non_leaf", true);

		if (dt_ < 0) dt_ = dx_ * cfl_;

		dt_ /= pow(2.0, nLevelRefinement_);
		tot_steps_ *= pow(2, nLevelRefinement_);

		pcout << "\n Setup:  Test - Simple IC \n" << std::endl;
		pcout << "Number of refinement levels: " << nLevelRefinement_
			<< std::endl;

		//domain_->decomposition().subtract_non_leaf() = true;

		domain_->register_refinement_condition() = [this](auto octant,
                                                       int     diff_level) {
            return this->refinement(octant, diff_level);
        };

		domain_->ib().init(_d->get_dictionary("simulation_parameters"), domain_->dx_base(), nLevelRefinement_, Re_);

		if (!use_restart())
		{
			domain_->init_refine(_d->get_dictionary("simulation_parameters")
				->template get_or<int>("nLevels", 0),
				global_refinement_, 0);
		}
		else
		{
			domain_->restart_list_construct();
		}

		domain_->distribute<fmm_mask_builder_t, fmm_mask_builder_t>();
		boost::mpi::communicator world;
		if (!use_restart()) { this->initialize(); }
		else
		{
			if (world.rank() == 0)
				std::cout << "reading u profile from data" << std::endl;
			simulation_.template read_h5<u_type>(simulation_.restart_field_dir(), "u");
			simulation_.template read_h5<p_type>(simulation_.restart_field_dir(), "p");

            //simulation_.template read_h5<u_ref_type>(simulation_.restart_field_dir(), "u");
			//simulation_.template read_h5<p_ref_type>(simulation_.restart_field_dir(), "p");
			//this->initialize(); 
		}

		real_coordinate_type tmp_coord(0.0);
        forcing_tar.resize(domain_->ib().size());
        std::fill(forcing_tar.begin(), forcing_tar.end(), tmp_coord);
        forcing_num.resize(domain_->ib().size());
        std::fill(forcing_num.begin(), forcing_num.end(), tmp_coord);
		forcing_ref.resize(domain_->ib().size());
        std::fill(forcing_ref.begin(), forcing_ref.end(), tmp_coord);

		
		if (world.rank() == 0)
			std::cout << "on Simulation: \n" << simulation_ << std::endl;
	}

	float_type run(int argc, char *argv[])
	{
		boost::mpi::communicator world;

		helm_stab_t ifherk(&this->simulation_);
        ifherk.Assigning_idx();
        std::vector<float_type> fz(domain_->ib().size(), 0.0);
        std::vector<float_type> fz_tar(domain_->ib().size(), 0.0);
        std::vector<float_type> fz_num(domain_->ib().size(), 0.0);
        std::vector<float_type> error_fz(domain_->ib().size(), 0.0);
		for (int i = 0; i < forcing_ref.size();i++) {
            if (domain_->ib().rank(i)!=world.rank()) {
                forcing_ref[i]=0;
                continue;
            }

			float_type t = static_cast<float_type>(i)/static_cast<float_type>(forcing_ref.size());
			float_type val_u = std::sin(2*M_PI*t); 
			float_type val_v = std::cos(2*M_PI*t); 

            /*float_type val_u = 1.0; 
			float_type val_v = 1.0;*/ 

			forcing_ref[i][0] = val_u;
			forcing_ref[i][1] = val_v;

            fz[i] = val_u*val_v;
		}

        world.barrier();

        //if (world.rank() != 0) ifherk.clean_up_initial_velocity<u_type>();
        //if (world.rank() != 0) ifherk.clean_up_initial_velocity<u_ref_type>();

        if (world.rank() != 0)
        {
            ifherk.clean<u_type>();
        }


        for (int i = 0; i < n_times; i++)
        {
            float_type  ratio = 1.0 / static_cast<float_type>(n_times);
            int         step_num = n_start + i * flow_interval;
            std::string flow_name = "./uData/flowTime_" + std::to_string(step_num)+".hdf5";
            simulation_.template read_h5<u_n_type>(flow_name, "u");
            if (world.rank() != 0) ifherk.add<u_n_type, u_type>(ratio);
            if (world.rank() != 0) ifherk.clean<u_n_type>();
        }
        

        world.barrier();
		



        ifherk.construct_linear_mat<u_type, uz_type>();
        ifherk.construction_imaginary();
        if (addImagBC) ifherk.construction_BCMat_u_imag();
        ifherk.construction_B_matrix();
        ifherk.construction_Q_matrix();
        ifherk.Jac.clean_entry(1e-13);
        ifherk.Imag.clean_entry(1e-13);

        simulation_.write("init.hdf5");

        int ndim = ifherk.total_dim();
        int loc_size = ifherk.Jac.numRow_loc();

        world.barrier();



        //need to store eigenvectors
        float_type*  x0_real= NULL;
        float_type*  x0_imag= NULL;
        float_type*  x1_real= NULL;
        float_type*  x1_imag= NULL;
        float_type*  x2_real= NULL;
        float_type*  x2_imag= NULL;
        float_type*  x3_real= NULL;
        float_type*  x3_imag= NULL;
        float_type*  x4_real= NULL;
        float_type*  x4_imag= NULL;

        PetscMPIInt    rank, size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        PetscMPIInt Color = 0;
        if (world.rank() != 0) {
            Color = 1;
        }

        MPI_Comm_split(MPI_COMM_WORLD, Color, 0, &PETSC_COMM_WORLD);
        if (Color != 0) {

        Vec            x, b, b1, u, v, y;          /* approx solution, RHS, exact solution */
        //Vec*           Cv;               /*deflation of 
        Mat            A, B, A_shell, AT;             /* linear system matrix */

        Mat            K, K1;
        SVD            svd;              /* eigenproblem solver context */
        EPS            eps;
        SVDType        type;
        PetscReal      error,tol,sigma,mu=PETSC_SQRT_MACHINE_EPSILON;
        PetscBool      flg,terse;
        PetscInt       nev, maxit, its;
        char           filename[PETSC_MAX_PATH_LEN];
        PetscViewer    viewer;
        KSP            ksp;              /* linear solver context */
        PC             pc;               /* preconditioner context */
        ST             st;
        PetscReal      norm;  /* norm of solution error */
        PetscInt       i,j,n = ndim,col[3],rstart,rend,nlocal, nconv1, nconv2, ctx;

        
        PetscScalar    one = 1.0,value[3], zero = 0.0;

        PetscCall(SlepcInitialize(&argc,&argv,(char*)0,help));

        PetscCall(VecCreate(PETSC_COMM_WORLD, &x));

        PetscInt* ia = NULL;
        PetscInt* ja = NULL;
        PetscScalar*  a = NULL;
        double*  b_val= NULL;

        float_type* P_1 = NULL;
        

        loc_size = 0;

        
        std::vector<int> iparm;

        
        iparm.resize(64);

        for (int i = 0; i < 64; i++) {
            iparm[i] = 0;
        }        

		iparm[0] = 1; /* Solver default parameters overriden with provided by iparm */
        //iparm[1] = 0;  /* Use METIS for fill-in reordering */
        iparm[1] = 3;  /* Use PARALLEL METIS for fill-in reordering */
        iparm[5] = 0;  /* Write solution into x */
        iparm[7] = 10;  /* Max number of iterative refinement steps */
        iparm[9] = 13; /* Perturb the pivot elements with 1E-13 */
        iparm[10] = 1; /* Use nonsymmetric permutation and scaling MPS */
        iparm[12] = 1; /* Switch on Maximum Weighted Matching algorithm (default for non-symmetric) */
        iparm[17] = -1; /* Output: Number of nonzeros in the factor LU */
        iparm[18] = -1; /* Output: Mflops for LU factorization */
        //iparm[23] = 10;
        iparm[26] = 1;  /* Check input data for correctness */ 
        iparm[39] = 2; /* Input: matrix/rhs/solution are distributed between MPI processes  */

        


        if (world.rank() != 0) {
            int begin_row = ifherk.num_start();
            int end_row = ifherk.num_end();

            loc_size = end_row - begin_row+1;
            int check_size = ifherk.Jac.numRow_loc();

            if (loc_size != check_size) {
                std::cout << "Rank " << world.rank() << " local matrix size does not match " << loc_size << " vs " << check_size << std::endl;
                return 1;
            }

            int tot_size = ifherk.Jac.tot_size();
            //PetscMalloc(sizeof(PetscInt) * tot_size, &ia);
            //PetscMalloc(sizeof(PetscInt) * tot_size, &ja);
            //PetscMalloc(sizeof(PetscScalar) * tot_size, &a);

            /*if (world.rank() == 1) 
            {
                PetscMalloc(sizeof(float_type) * (loc_size-1), &b_val);
                PetscMalloc(sizeof(float_type) * loc_size, &x_val);
            }*/

            PetscMalloc(sizeof(float_type) * loc_size, &P_1);

            for (int i = 0; i < loc_size; i++) {
                P_1[i] = 0.0;
            }
            ifherk.Pressure_nullspace(P_1);
            //x = (float_type*)PetscMalloc(sizeof(float_type) * loc_size);
            //b = (float_type*)PetscMalloc(sizeof(float_type) * loc_size);

            //ifherk.Jac.getIdx(ia, ja, a, rstart);
            //ifherk.Grid2CSR<u_type, p_type>(b_val);
        }
        else {
            //PetscMalloc(sizeof(PetscInt) * 0, &ia);
            //PetscMalloc(sizeof(PetscInt) * 0, &ja);
            //PetscMalloc(sizeof(float_type) * 0, &a);

            //PetscMalloc(sizeof(float_type) * 0, &b_val);
            //PetscMalloc(sizeof(float_type) * 0, &x_val);
        }

        //if (world.rank() == 1) {loc_size = loc_size - 10;}
        //if (world.rank() == 0) {loc_size = 10;}

        PetscCall(VecSetSizes(x, loc_size, n));
        PetscCall(VecSetFromOptions(x));
        PetscCall(VecDuplicate(x, &b));
        PetscCall(VecDuplicate(x, &y));
        PetscCall(VecDuplicate(x, &y_tmp));
        PetscCall(VecDuplicate(x, &Cy));
        PetscCall(VecDuplicate(x, &By));
        //PetscCall(VecDuplicate(x, &b1));
        PetscCall(VecDuplicate(x, &u));
        PetscCall(VecDuplicate(x, &v));

        PetscInt* idx_ = NULL;

        float_type sum = 0;

        for (int i = 0; i < loc_size; i++) {
            //idx_[i] = i;
            sum += P_1[i];
            PetscScalar tmp = P_1[i];
            VecSetValues(b, 1, &i, &tmp, INSERT_VALUES);
            //PetscScalar tmp = P_1[i]*PETSC_i;
            //VecSetValues(b1, 1, &i, &tmp, INSERT_VALUES);
        }

        if (world.rank() == 1) {
            std::cout << "Sum of all P_1 is " << sum << std::endl;
        }

        PetscCall(VecAssemblyBegin(b));
        PetscCall(VecAssemblyEnd(b));

        //PetscCall(VecAssemblyBegin(b1));
        //PetscCall(VecAssemblyEnd(b1));

        

        /* Identify the starting and ending mesh points on each
        processor for the interior part of the mesh. We let PETSc decide
        above. */

        PetscCall(VecGetOwnershipRange(x, &rstart, &rend));

        PetscMalloc(sizeof(float_type) * loc_size, &x0_real);
        PetscMalloc(sizeof(float_type) * loc_size, &x0_imag);
        PetscMalloc(sizeof(float_type) * loc_size, &x1_real);
        PetscMalloc(sizeof(float_type) * loc_size, &x1_imag);
        PetscMalloc(sizeof(float_type) * loc_size, &x2_real);
        PetscMalloc(sizeof(float_type) * loc_size, &x2_imag);
        PetscMalloc(sizeof(float_type) * loc_size, &x3_real);
        PetscMalloc(sizeof(float_type) * loc_size, &x3_imag);
        PetscMalloc(sizeof(float_type) * loc_size, &x4_real);
        PetscMalloc(sizeof(float_type) * loc_size, &x4_imag);
        

        std::cout << "rank " << rank << " start and end " << rstart << " " << rend <<std::endl;
        PetscCall(VecGetLocalSize(x, &nlocal));

        if (nlocal != loc_size) {
            std::cout << "rank " << rank << " nlocal and loc_size does not match " << std::endl;
        }


        //compute the d_nnz and o_nnz vectors
        PetscInt* d_nnz = NULL;
        PetscInt* o_nnz = NULL;
        if (world.rank() == 1) {
            std::cout << "Initialized d_nnz and o_nnz" << std::endl;
        }
        PetscMalloc(sizeof(PetscInt) * loc_size, &d_nnz);
        PetscMalloc(sizeof(PetscInt) * loc_size, &o_nnz);

        if (world.rank() == 1) {
            std::cout << "Allocated d_nnz and o_nnz" << std::endl;
        }


        for (i = rstart; i < rend; i++)
        {
            int i_loc = i - rstart;
            std::map<int, float_type> row = ifherk.Jac.mat[i_loc+1];
            PetscInt n_loc = 0;
            PetscInt n_out = 0;
            
            for (const auto& [key, val] : row) {
                PetscInt key1 = key;
                if (key1 <= rend && key1 > rstart) {
                    //key is 1-based while the rstart rend indices are zero based
                    n_loc += 1;
                }
                else {
                    n_out += 1;
                }
            }

            std::map<int, float_type> row_c = ifherk.Imag.mat[i_loc+1];

            for (const auto& [key, val] : row_c) {
                PetscInt key1 = key;
                if (key1 <= rend && key1 > rstart) {
                    //key is 1-based while the rstart rend indices are zero based
                    n_loc += 1;
                }
                else {
                    n_out += 1;
                }
            }

            std::map<int, float_type> row1 = ifherk.B.mat[i_loc+1];
            for (const auto& [key, val] : row1) {
                PetscInt key1 = key;
                if (key1 <= rend && key1 > rstart) {
                    //key is 1-based while the rstart rend indices are zero based
                    n_loc += 1;
                }
                else {
                    n_out += 1;
                }
            }

            d_nnz[i_loc] = n_loc;
            o_nnz[i_loc] = n_out;
        }

        if (world.rank() == 1) {
            std::cout << "Computed d_nnz and o_nnz" << std::endl;
        }
        
        PetscCall(MatCreateAIJ(PETSC_COMM_WORLD, nlocal, nlocal, n, n, NULL, d_nnz, NULL, o_nnz, &A));
        MatSetType(A,MATMPIAIJ);
        //PetscCall(MatCreateMPIAIJWithArrays(PETSC_COMM_WORLD, nlocal, nlocal, n, n, ia, ja, a, &A));
        //PetscCall(MatSetSizes(A, nlocal, nlocal, n, n));
        PetscCall(MatSetFromOptions(A));
        PetscCall(MatSetUp(A));

        PetscCall(MatCreate(PETSC_COMM_WORLD, &B));
        MatSetType(B,MATMPIAIJ);
        //PetscCall(MatCreateMPIAIJWithArrays(PETSC_COMM_WORLD, nlocal, nlocal, n, n, ia, ja, a, &A));
        PetscCall(MatSetSizes(B, nlocal, nlocal, n, n));
        PetscCall(MatSetFromOptions(B));
        PetscCall(MatSetUp(B));



        PetscCall(MatCreate(PETSC_COMM_WORLD, &C));
        MatSetType(C,MATMPIAIJ);
        //PetscCall(MatCreateMPIAIJWithArrays(PETSC_COMM_WORLD, nlocal, nlocal, n, n, ia, ja, a, &A));
        PetscCall(MatSetSizes(C, nlocal, nlocal, n, n));
        PetscCall(MatSetFromOptions(C));
        PetscCall(MatSetUp(C));


        PetscCall(MatCreate(PETSC_COMM_WORLD, &Q_hi));
        MatSetType(Q_hi,MATMPIAIJ);
        //PetscCall(MatCreateMPIAIJWithArrays(PETSC_COMM_WORLD, nlocal, nlocal, n, n, ia, ja, a, &A));
        PetscCall(MatSetSizes(Q_hi, nlocal, nlocal, n, n));
        PetscCall(MatSetFromOptions(Q_hi));
        PetscCall(MatSetUp(Q_hi));

        int counter_zero_diag = 0;
        std::vector<int> zero_diag_idx;


        for (i = rstart; i < rend; i++)
        {
            PetscScalar valComplex = i + 10000;
            PetscCall(MatSetValue(B, i, i, valComplex, INSERT_VALUES));
        }

        for (i = rstart; i < rend; i++)
        {
            //int range = rend - rstart;
            //int range_nonzero = range/2;
            int i_loc = i - rstart;


            
            std::map<int, float_type> row = ifherk.Jac.mat[i_loc+1];
            std::map<int, float_type> row_c = ifherk.Imag.mat[i_loc+1];
            std::map<int, float_type> row1 = ifherk.B.mat[i_loc+1];

            std::map<int, float_type> rowQ = ifherk.Q.mat[i_loc+1];
            std::map<int, float_type> rowQ_hi = ifherk.Q_halfi.mat[i_loc+1];
            
            std::vector<PetscScalar> row_val;
            std::vector<PetscInt> row_idx;
            row_val.resize(row.size() + row_c.size()+row1.size());
            row_idx.resize(row.size() + row_c.size()+row1.size());
            int counter = 0;

            for (const auto& [key, val] : row_c)
            {
                if (key <= 0) continue;
                auto it = row.find(key);
                if (it == row.end())
                {
                    row_val[counter] = val * PETSC_i;
                    row_idx[counter] = key - 1;
                    counter++;
                }
                else { 
                    row_val[counter] = it->second + val * PETSC_i;
                    row_idx[counter] = key - 1;
                    counter++;
                    row.erase(it); 
                }
            }


            //std::map<int, float_type> row1 = ifherk.B.mat[i_loc+1];
            for (const auto& [key, val] : row1)
            {
                if (key <= 0) continue;
                auto it = row.find(key);
                if (it == row.end())
                {
                    row_val[counter] = -val * PETSC_i * Omega_l;
                    row_idx[counter] = key - 1;
                    counter++;
                }
                else { 
                    row_val[counter] = it->second - val * PETSC_i * Omega_l;
                    row_idx[counter] = key - 1;
                    counter++;
                    row.erase(it); 
                }
            }

            for (const auto& [key, val] : row) {
                row_val[counter] = val;
                row_idx[counter] = key - 1;
                counter++;
            }

            PetscCall(MatSetValues(A, 1, &i, counter, row_idx.data(), row_val.data(), INSERT_VALUES));





            /*if (rowQ.size() == 0) {
                float_type val = 0.0;
                PetscScalar valComplex = val;
                //PetscCall(MatSetValues(B, 1, &i, 1, &i, &val, INSERT_VALUES));
                PetscCall(MatSetValue(C, i, i, valComplex, INSERT_VALUES));
                //PetscCall(MatSetValue(A, i, i, valA, INSERT_VALUES));
            }*/
            for (const auto& [key, val] : rowQ)
            {
                PetscInt loc_col = key-1;
                if (!std::isfinite(val)) {
                    std::cout << "mat C rank " << world.rank() << " row " << i << " loc_col " << loc_col << std::endl;
                }
                PetscScalar valComplex = val;
                PetscCall(MatSetValue(C, i, loc_col, valComplex, INSERT_VALUES));
                //for debugging, setting a diagonal matrix with some zeros
                //PetscCall(MatSetValue(A, i, loc_col, valComplex, INSERT_VALUES));
            }

            for (const auto& [key, val] : rowQ_hi)
            {
                PetscInt loc_col = key-1;
                if (!std::isfinite(val)) {
                    std::cout << "mat C rank " << world.rank() << " row " << i << " loc_col " << loc_col << std::endl;
                }
                PetscScalar valComplex = val;
                PetscCall(MatSetValue(Q_hi, i, loc_col, valComplex, INSERT_VALUES));
                //for debugging, setting a diagonal matrix with some zeros
                //PetscCall(MatSetValue(A, i, loc_col, valComplex, INSERT_VALUES));
            }

        }

        if (world.rank() == 1) {
            std::cout << "finished setting values" << std::endl;
            std::cout << "number of zero diagnonals " << counter_zero_diag << std::endl;
            int n = zero_diag_idx.size();
            if (n != 0) std::cout << "min number zero diags " << zero_diag_idx[0] << " max number zero diags " << zero_diag_idx[n-1] << std::endl;
        }

        MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);

        MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);

        MatAssemblyBegin(Q_hi,MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(Q_hi,MAT_FINAL_ASSEMBLY);


        PetscCall(MatHermitianTranspose(A, MAT_INITIAL_MATRIX, &AT));

        //SolverCtx<KSP, Vec, PC> ATA_inv;

        //ATA_inv.initialize(PETSC_COMM_WORLD, A, AT, y_tmp);



        PetscCall(KSPCreate(PETSC_COMM_WORLD, &kspA));
        KSPSetType(kspA, KSPPREONLY);
        PetscCall(KSPCreate(PETSC_COMM_WORLD, &kspAT));
        KSPSetType(kspAT, KSPPREONLY);

            
        PetscCall(KSPSetOperators(kspA, A, A));

        PetscCall(KSPSetOperators(kspAT, AT, AT));

        PetscCall(KSPGetPC(kspA, &pcA));
        PetscCall(PCSetType(pcA, PCLU));

        PCFactorSetMatSolverType(pcA, MATSOLVERMUMPS);
        PCFactorSetUpMatSolverType(pcA);

        PetscCall(KSPSetTolerances(kspA, 1.e-12, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT));


        PCFactorGetMatrix(pcA,&K);
        MatMumpsSetIcntl(K,14,50);
        MatMumpsSetCntl(K,3,1e-12);

        PetscCall(KSPSetFromOptions(kspA));

        PetscCall(KSPGetPC(kspAT, &pcAT));
        PetscCall(PCSetType(pcAT, PCLU));

        PCFactorSetMatSolverType(pcAT, MATSOLVERMUMPS);
        PCFactorSetUpMatSolverType(pcAT);


        PetscCall(KSPSetTolerances(kspAT, 1.e-12, PETSC_DEFAULT,PETSC_DEFAULT, PETSC_DEFAULT));

        PCFactorGetMatrix(pcAT,&K1);
        MatMumpsSetIcntl(K1,14,50);
        MatMumpsSetCntl(K1,3,1e-12);

        PetscCall(KSPSetFromOptions(kspAT));



        PetscCall(MatCreateShell(PETSC_COMM_WORLD,nlocal, nlocal, n, n,&ctx,&A_shell));
        PetscCall(MatShellSetOperation(A_shell,MATOP_MULT,(void(*)(void))MatMul_ATA_inv));
        //PetscCall(MatShellSetOperation(A_shell,MATOP_MULT,(void(*)(void))MatMulA_inv));
        //PetscCall(MatShellSetOperation(A_shell,MATOP_MULT,(void(*)(void))Id));
        //PetscCall(MatShellSetOperation(A_shell,MATOP_MULT_TRANSPOSE,(void(*)(void))MatMul));
        //PetscCall(MatShellSetOperation(A,MATOP_MULT_TRANSPOSE,(void(*)(void))MatMult_Laplacian2D));
        //PetscCall(MatShellSetOperation(A,MATOP_GET_DIAGONAL,(void(*)(void))MatGetDiagonal_Laplacian2D));

        MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);


        if (world.rank() == 1) {
            std::cout << "finished Assemblying" << std::endl;
        }

        if (world.rank() == 1) {
            std::cout << "finished Assemblying" << std::endl;
        }

        PetscCall(EPSCreate(PETSC_COMM_WORLD, &eps));

        /*
     Set operators. In this case, it is a Generalized eigenvalue problem
  */
        PetscCall(EPSSetOperators(eps, A_shell, NULL));

        //PetscCall(EPSSetOperators(eps, A_shell, C));
        //PetscCall(EPSSetOperators(eps, A, NULL));
        EPSSetProblemType(eps,EPS_HEP);
        //EPSSetProblemType(eps,EPS_NHEP);

        //EPSSetProblemType(eps,EPS_PGNHEP);
        
        EPSSetDimensions(eps,1,PETSC_DEFAULT,PETSC_DEFAULT);

        
        
        EPSSetWhichEigenpairs(eps,	EPS_LARGEST_MAGNITUDE);
        

        if (world.rank() == 2) {
            std::cout << "set up of eps finished" << std::endl;
        }

        /*
     Set solver parameters at runtime
  */
        PetscCall(EPSSetFromOptions(eps));

        if (world.rank() == 2) {
            std::cout << "set up of eps finished" << std::endl;
        }

        MPI_Barrier(PETSC_COMM_WORLD);



        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Solve the eigensystem
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        PetscCall(EPSSolve(eps));

        if (world.rank() == 2) {
            std::cout << "EPS solved" << std::endl;
        }

        PetscCall(EPSGetIterationNumber(eps, &its));

        PetscCall(PetscPrintf(PETSC_COMM_WORLD,
            " Number of iterations of the method: %" PetscInt_FMT "\n", its));

        /*
     Optional: Get some information from the solver and display it
  */
        PetscCall(EPSGetType(eps, &type));
        PetscCall(
            PetscPrintf(PETSC_COMM_WORLD, " Solution method: %s\n\n", type));
        PetscCall(EPSGetDimensions(eps, &nev, NULL, NULL));
        PetscCall(PetscPrintf(PETSC_COMM_WORLD,
            " Number of requested eigenvalues: %" PetscInt_FMT "\n", nev));
        PetscCall(EPSGetTolerances(eps, &tol, &maxit));
        PetscCall(PetscPrintf(PETSC_COMM_WORLD,
            " Stopping condition: tol=%.4g, maxit=%" PetscInt_FMT "\n",
            (double)tol, maxit));

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                    Display solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        /* show detailed info unless -terse option is given by user */
        PetscCall(PetscOptionsHasName(NULL, NULL, "-terse", &terse));
        if (terse) PetscCall(EPSErrorView(eps, EPS_ERROR_RELATIVE, NULL));
        else
        {
            PetscCall(PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,
                PETSC_VIEWER_ASCII_INFO_DETAIL));
            PetscCall(EPSConvergedReasonView(eps, PETSC_VIEWER_STDOUT_WORLD));
            PetscCall(EPSErrorView(eps, EPS_ERROR_RELATIVE,
                PETSC_VIEWER_STDOUT_WORLD));
            PetscCall(PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD));
        }

        for (float_type Cur_o = Omega_int+Omega_l; Cur_o <= Omega_r; Cur_o+= Omega_int) 
        {
            MPI_Barrier(PETSC_COMM_WORLD);
            auto t0 = clock_type::now();
            for (i = rstart; i < rend; i++)
            {
                int i_loc = i - rstart;
  
                std::map<int, float_type> row1 = ifherk.B.mat[i_loc+1];

                if (row1.size() == 0) continue;

                PetscScalar valComplex = -PETSC_i*Omega_int;
                PetscScalar valComplex1 = PETSC_i*Omega_int;

                PetscCall(MatSetValue(A, i, i, valComplex, ADD_VALUES));
                PetscCall(MatSetValue(AT, i, i, valComplex1, ADD_VALUES));

            }


            MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);
            MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);

            MatAssemblyBegin(AT,MAT_FINAL_ASSEMBLY);
            MatAssemblyEnd(AT,MAT_FINAL_ASSEMBLY);

            PetscCall(EPSDestroy(&eps));
        
            PetscCall(KSPSetOperators(kspA, A, A));
            PetscCall(KSPSetOperators(kspAT, AT, AT));

            PetscCall(KSPSetFromOptions(kspA));
            PetscCall(KSPSetFromOptions(kspAT));
            PetscCall(EPSCreate(PETSC_COMM_WORLD, &eps));
            PetscCall(EPSSetOperators(eps, A_shell, NULL));
            EPSSetProblemType(eps,EPS_HEP);        
            EPSSetDimensions(eps,1,PETSC_DEFAULT,PETSC_DEFAULT);    
            EPSSetWhichEigenpairs(eps,	EPS_LARGEST_MAGNITUDE);
            
            PetscCall(EPSSetFromOptions(eps));

            if (world.rank() == 2) {
                std::cout << "set up of eps finished" << std::endl;
            }

            MPI_Barrier(PETSC_COMM_WORLD);

            PetscCall(EPSSolve(eps));

            PetscCall(EPSGetIterationNumber(eps, &its));

            
            PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                " Number of iterations of the method: %" PetscInt_FMT "\n", its));

            PetscCall(EPSGetType(eps, &type));
            PetscCall(
                PetscPrintf(PETSC_COMM_WORLD, " Solution method: %s\n\n", type));
            PetscCall(EPSGetDimensions(eps, &nev, NULL, NULL));
            PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                " Number of requested eigenvalues: %" PetscInt_FMT "\n", nev));
            PetscCall(EPSGetTolerances(eps, &tol, &maxit));
            PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                " Stopping condition: tol=%.4g, maxit=%" PetscInt_FMT "\n",
                (double)tol, maxit));

            PetscCall(PetscOptionsHasName(NULL, NULL, "-terse", &terse));
            if (terse) PetscCall(EPSErrorView(eps, EPS_ERROR_RELATIVE, NULL));
            else
            {
                PetscCall(PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,
                    PETSC_VIEWER_ASCII_INFO_DETAIL));
                PetscCall(EPSConvergedReasonView(eps, PETSC_VIEWER_STDOUT_WORLD));
                PetscCall(EPSErrorView(eps, EPS_ERROR_RELATIVE,
                    PETSC_VIEWER_STDOUT_WORLD));
                PetscCall(PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD));
            }

            MPI_Barrier(PETSC_COMM_WORLD);
            auto t1 = clock_type::now();
            mDuration_type ms_int = t1 - t0;
            if (world.rank() == 1) {
                std::cout << "Solving Resolvent Norm at Omega = " << Cur_o 
                          << " in " << ms_int.count() << " ms " << std::endl;
            }

            if (world.rank() == 1) {
            PetscScalar EV;
            PetscScalar EVi;
            PetscCall(EPSGetEigenvalue(eps, 0, &EV, &EVi));
            float_type evr = PetscRealPart(EV);
            

            std::ofstream outfile;
            int width = 20;

            outfile.open("EVs.txt", std::ios_base::app);
            outfile << std::setw(width) << Cur_o << std::setw(width)
                    << std::scientific << std::setprecision(9);
            outfile << std::setw(width) << evr << std::setw(width)
                    << std::scientific << std::setprecision(9);
            outfile << std::endl;
            outfile.close();
            }
        }

        EPSGetEigenvector(eps,0, x, NULL);
        PetscCall(MatMult(Q_hi, x, By));

        for (i = rstart; i < rend;i++) {
            int i_loc = i - rstart;
            PetscScalar tmp;
            PetscCall(VecGetValues(By, 1, &i, &tmp));
            x0_real[i_loc] = PetscRealPart(tmp);
            x0_imag[i_loc] = PetscImaginaryPart(tmp);
        }

        
        PetscCall(KSPSolve(kspA, By, y));

        for (i = rstart; i < rend;i++) {
            int i_loc = i - rstart;
            PetscScalar tmp;
            PetscCall(VecGetValues(y, 1, &i, &tmp));
            x1_real[i_loc] = PetscRealPart(tmp);
            x1_imag[i_loc] = PetscImaginaryPart(tmp);
        }

        for (i = rstart; i < rend;i++) {
            int i_loc = i - rstart;
            PetscScalar tmp;
            PetscCall(VecGetValues(x, 1, &i, &tmp));
            x2_real[i_loc] = PetscRealPart(tmp);
            x2_imag[i_loc] = PetscImaginaryPart(tmp);
        }

        PetscCall(MatMult(A, y, x));

        for (i = rstart; i < rend;i++) {
            int i_loc = i - rstart;
            PetscScalar tmp;
            PetscCall(VecGetValues(x, 1, &i, &tmp));
            x3_real[i_loc] = PetscRealPart(tmp);
            x3_imag[i_loc] = PetscImaginaryPart(tmp);
        }

        MPI_Barrier(PETSC_COMM_WORLD);
        //PetscCall(EPSDestroy(&eps));
        PetscCall(MatDestroy(&A));
        PetscCall(MatDestroy(&B));
        

        

        PetscCall(VecDestroy(&x));
        PetscCall(VecDestroy(&u));
        PetscCall(VecDestroy(&b));
        
        //PetscCall(KSPDestroy(&ksp));

        PetscCall(SlepcFinalize());
        }

        world.barrier();
        if (world.rank() == 0) {
            std::cout << "finished computing singular values" << std::endl;
        }
        world.barrier();


        ifherk.CSR2Grid<u_ref_type, p_ref_type, w_ref_type, N_ref_type, cs_ref_type, uz_ref_type, Nz_ref_type>(x0_real, forcing_ref, fz);
        //if (world.rank() != 0) ifherk.pad_velocity<u_num_inv_type, u_num_inv_type>(true);
        //if (world.rank() != 0) ifherk.Curl_access<u_ref_type, uz_ref_type, w_ref_type>();
        simulation_.write("x0_real.hdf5");
        ifherk.CSR2Grid<u_ref_type, p_ref_type, w_ref_type, N_ref_type, cs_ref_type, uz_ref_type, Nz_ref_type>(x0_imag, forcing_ref, fz);
        //if (world.rank() != 0) ifherk.pad_velocity<u_num_inv_type, u_num_inv_type>(true);
        //if (world.rank() != 0) ifherk.Curl_access<u_ref_type, uz_ref_type, w_ref_type>();
        simulation_.write("x0_imag.hdf5");
        ifherk.CSR2Grid<u_ref_type, p_ref_type, w_ref_type, N_ref_type, cs_ref_type, uz_ref_type, Nz_ref_type>(x1_real, forcing_ref, fz);
        //if (world.rank() != 0) ifherk.pad_velocity<u_num_inv_type, u_num_inv_type>(true);
        //if (world.rank() != 0) ifherk.Curl_access<u_ref_type, uz_ref_type, w_ref_type>();
        simulation_.write("x1_real.hdf5");
        ifherk.CSR2Grid<u_ref_type, p_ref_type, w_ref_type, N_ref_type, cs_ref_type, uz_ref_type, Nz_ref_type>(x1_imag, forcing_ref, fz);
        //if (world.rank() != 0) ifherk.pad_velocity<u_num_inv_type, u_num_inv_type>(true);
        //if (world.rank() != 0) ifherk.Curl_access<u_ref_type, uz_ref_type, w_ref_type>();
        simulation_.write("x1_imag.hdf5");
        ifherk.CSR2Grid<u_ref_type, p_ref_type, w_ref_type, N_ref_type, cs_ref_type, uz_ref_type, Nz_ref_type>(x2_real, forcing_ref, fz);
        //if (world.rank() != 0) ifherk.pad_velocity<u_num_inv_type, u_num_inv_type>(true);
        //if (world.rank() != 0) ifherk.Curl_access<u_ref_type, uz_ref_type, w_ref_type>();
        simulation_.write("x2_real.hdf5");
        ifherk.CSR2Grid<u_ref_type, p_ref_type, w_ref_type, N_ref_type, cs_ref_type, uz_ref_type, Nz_ref_type>(x2_imag, forcing_ref, fz);
        //if (world.rank() != 0) ifherk.pad_velocity<u_num_inv_type, u_num_inv_type>(true);
        //if (world.rank() != 0) ifherk.Curl_access<u_ref_type, uz_ref_type, w_ref_type>();
        simulation_.write("x2_imag.hdf5");
        ifherk.CSR2Grid<u_ref_type, p_ref_type, w_ref_type, N_ref_type, cs_ref_type, uz_ref_type, Nz_ref_type>(x3_real, forcing_ref, fz);
        //if (world.rank() != 0) ifherk.pad_velocity<u_num_inv_type, u_num_inv_type>(true);
        //if (world.rank() != 0) ifherk.Curl_access<u_ref_type, uz_ref_type, w_ref_type>();
        simulation_.write("x3_real.hdf5");
        ifherk.CSR2Grid<u_ref_type, p_ref_type, w_ref_type, N_ref_type, cs_ref_type, uz_ref_type, Nz_ref_type>(x3_imag, forcing_ref, fz);
        //if (world.rank() != 0) ifherk.pad_velocity<u_num_inv_type, u_num_inv_type>(true);
        //if (world.rank() != 0) ifherk.Curl_access<u_ref_type, uz_ref_type, w_ref_type>();
        simulation_.write("x3_imag.hdf5");

        /*ifherk.CSR2Grid<u_ref_type, p_ref_type, w_ref_type, N_ref_type, cs_ref_type, uz_ref_type, Nz_ref_type>(x4_real, forcing_ref, fz);
        if (world.rank() != 0) ifherk.Curl_access<u_ref_type, uz_ref_type, w_ref_type>();
        simulation_.write("x4_real.hdf5");
        ifherk.CSR2Grid<u_ref_type, p_ref_type, w_ref_type, N_ref_type, cs_ref_type, uz_ref_type, Nz_ref_type>(x4_imag, forcing_ref, fz);
        //if (world.rank() != 0) ifherk.pad_velocity<u_num_inv_type, u_num_inv_type>(true);
        if (world.rank() != 0) ifherk.Curl_access<u_ref_type, uz_ref_type, w_ref_type>();
        simulation_.write("x4_imag.hdf5");*/

        world.barrier();

        
        
        return 0;

	}

    

    /** @brief  Refienment conditon for octants.  */
    template<class OctantType>
    bool refinement(
        OctantType* it, int diff_level, bool use_all = false) const noexcept
    {
        auto b = it->data().descriptor();
        auto center =
            (domain_->bounding_box().max() - domain_->bounding_box().min()) /
                2.0 +
            domain_->bounding_box().min();
        
        const auto dx_base = domain_->dx_base();

        auto scaling = std::pow(2, b.level());
        center *= scaling;

        
        auto dx_level = dx_base / std::pow(2, b.level());

        float_type max_c = N_pts*dx_level;

        b.grow(2, 2);
        auto corners = b.get_corners();
        for (int i = b.base()[0]; i <= b.max()[0]; ++i)
        {
            for (int j = b.base()[1]; j <= b.max()[1]; ++j)
            {
                const float_type x =
                    static_cast<float_type>(i - center[0] + 0.5) * dx_level;
                const float_type y =
                    static_cast<float_type>(j - center[1] + 0.5) * dx_level;

                if (std::fabs(x) < max_c && std::fabs(y) < max_c) {
                    return true;
                }

                /*const auto vort = std::exp(-x*x-y*y)*(4*x*x - 2.0 + 4*y*y - 2.0);
                if (std::fabs(vort) >
                    source_max_ * pow(refinement_factor_, diff_level))
                {
                    return true;
                }*/
            }
        }
        return false;
    }

    /** @brief  Initialization of the domain blocks. This is registered in the
     *          domain through the base setup class, passing it to the domain ctor.
     */
    std::vector<coordinate_t> initialize_domain(
        Dictionary* _d, domain_t* _domain)
    {
        return _domain->construct_basemesh_blocks(_d, _domain->block_extent());
    }


    /** @brief Initialization of poisson problem.
     *  @detail Testing poisson with manufactured solutions: exp
     */
	void initialize()
	{
		poisson_solver_t psolver(&this->simulation_);

		boost::mpi::communicator world;
		if (domain_->is_server()) return;
		auto center = (domain_->bounding_box().max() -
			domain_->bounding_box().min() + 1) / 2.0 +
			domain_->bounding_box().min();

		// Adapt center to always have peak value in a cell-center
		//center+=0.5/std::pow(2,nRef);
		const float_type dx_base = domain_->dx_base();

		//if (ic_filename_ != "null") return;

		// Voriticity IC
		for (auto it = domain_->begin(); it != domain_->end(); ++it)
		{
			if (!it->locally_owned()) continue;

			auto dx_level = dx_base / std::pow(2, it->refinement_level());
			auto scaling = std::pow(2, it->refinement_level());

			for (auto& node : it->data())
			{

				const auto& coord = node.level_coordinate();

				float_type xc = static_cast<float_type>(
                                    coord[0] - center[0] * scaling + 0.5) *
                                dx_level;
                float_type yc = static_cast<float_type>(
                                    coord[1] - center[1] * scaling + 0.5) *
                                dx_level;
                /*float_type zc = static_cast<float_type>(
                                    coord[2] - center[2] * scaling + 0.5) *
                                dx_level;*/

                //Face centered coordinates
                float_type xf0 =
                    static_cast<float_type>(coord[0] - center[0] * scaling) *
                    dx_level;
                float_type yf0 = yc;
                //float_type zf0 = zc;

                float_type xf1 = xc;
                float_type yf1 =
                    static_cast<float_type>(coord[1] - center[1] * scaling) *
                    dx_level;
                //float_type zf1 = zc;

                float_type xf2 = xc;
                float_type yf2 = yc;
                /*float_type zf2 =
                    static_cast<float_type>(coord[2] - center[2] * scaling) *
                    dx_level;*/

                //Edge centered coordinates
                float_type xe0 = static_cast<float_type>(
                                     coord[0] - center[0] * scaling + 0.5) *
                                 dx_level;
                float_type ye0 =
                    static_cast<float_type>(coord[1] - center[1] * scaling) *
                    dx_level;
                /*float_type ze0 =
                    static_cast<float_type>(coord[2] - center[2] * scaling) *
                    dx_level;*/
                float_type xe1 =
                    static_cast<float_type>(coord[0] - center[0] * scaling) *
                    dx_level;
                float_type ye1 = static_cast<float_type>(
                                     coord[1] - center[1] * scaling + 0.5) *
                                 dx_level;
                /*float_type ze1 =
                    static_cast<float_type>(coord[2] - center[2] * scaling) *
                    dx_level;*/
                float_type xe2 =
                    static_cast<float_type>(coord[0] - center[0] * scaling) *
                    dx_level;
                float_type ye2 =
                    static_cast<float_type>(coord[1] - center[1] * scaling) *
                    dx_level;
                /*float_type ze2 = static_cast<float_type>(
                                     coord[2] - center[2] * scaling + 0.5) *
                                 dx_level;*/

                const float_type r = std::sqrt(xc * xc + yc * yc);
                const float_type rf0 =
                    std::sqrt(xf0 * xf0 + yf0 * yf0);
                const float_type rf1 =
                    std::sqrt(xf1 * xf1 + yf1 * yf1);
                const float_type rf2 =
                    std::sqrt(xf2 * xf2 + yf2 * yf2);
                const float_type re0 =
                    std::sqrt(xe0 * xe0 + ye0 * ye0);
                const float_type re1 =
                    std::sqrt(xe1 * xe1 + ye1 * ye1);
                const float_type re2 =
                    std::sqrt(xe2 * xe2 + ye2 * ye2);
                const float_type a2 = a_ * a_;
                const float_type xc2 = xc * xc;
                const float_type yc2 = yc * yc;
				float_type r_2 = r * r;

				const auto fct = std::exp(-r_2);
                const auto tmpc = std::exp(-r_2);

                const auto tmpf0 = std::exp(-rf0 * rf0);
                const auto tmpf1 = std::exp(-rf1 * rf1);
                const auto tmpf2 = std::exp(-rf2 * rf2);

                const auto tmpe0 = std::exp(-re0 * re0);
                const auto tmpe1 = std::exp(-re1 * re1);
                const auto tmpe2 = std::exp(-re2 * re2);

				//node(u_ref, 0) = -tmpf0*yf0;
				//node(u_ref, 1) = tmpf1*xf1;

				int rand_val = rand();

				float_type v_1 = static_cast<float_type>(rand_val)/static_cast<float_type>(RAND_MAX) * pert_mag*2 - pert_mag;
				rand_val = rand();
				float_type v_2 = static_cast<float_type>(rand_val)/static_cast<float_type>(RAND_MAX) * pert_mag*2 - pert_mag;
				rand_val = rand();
				float_type v_3 = static_cast<float_type>(rand_val)/static_cast<float_type>(RAND_MAX) * pert_mag*2 - pert_mag;

				node(u,0) = tmpf0*(4*xf0*xf0 - 2.0 + 4*yf0*yf0 - 2.0);
				node(u,1) = tmpf1*(4*xf1*xf1 - 2.0 + 4*yf1*yf1 - 2.0);
				node(p,0) = tmpc*(4*xc*xc - 2.0 + 4*yc*yc - 2.0);

                node(u_ref,0) = tmpf0;
                node(u_ref,1) = tmpf1;
                node(p_ref,0) = tmpc;

				//node(p_ref, 0) = tmpc;

				//int rand_val = rand();

				/*float_type v_1 = static_cast<float_type>(rand_val)/static_cast<float_type>(RAND_MAX) * pert_mag*2 - pert_mag;

				rand_val = rand();
				float_type v_2 = static_cast<float_type>(rand_val)/static_cast<float_type>(RAND_MAX) * pert_mag*2 - pert_mag;

				rand_val = rand();
				float_type v_3 = static_cast<float_type>(rand_val)/static_cast<float_type>(RAND_MAX) * pert_mag*2 - pert_mag;

				node(u_num, 0) = node(u_ref,0) + v_1*tmpf0;
				node(u_num, 1) = node(u_ref,1) + v_2*tmpf1;
				node(p_num, 0) = tmpc + v_3*tmpc;*/
			}

		}
	}

    private:

    boost::mpi::communicator client_comm_;

    bool single_ring_=true;
    bool perturbation_=false;
    bool hard_max_refinement_=false;
    bool smooth_start_;
    bool print_mat;
    bool num_input;
    bool check_mat_res; //true if we need to check of the matrix constructed is true by applying and checking with discrete operator
                        //false if we only want the eigenvalues
    bool addImagBC = true;
    float_type Omega_w; // Omega in resolvent analysis
    int vortexType = 0;

    float_type Omega_l, Omega_r, Omega_int;

    int row_to_print = 0;
    int N_pts;

    float_type target_real = 0.0;
    float_type target_imag = 0.0;

    bool clean_p_tar;
    bool testing_smearing = false;
    bool set_deflation=false;

    std::vector<float_type> U_;
    //bool subtract_non_leaf_  = true;
    float_type R_;
    float_type v_delta_;
    float_type d2v_;
    float_type source_max_;
    float_type vort_sep;


	force_type forcing_tar;
    force_type forcing_num;
	force_type forcing_ref;
    force_type forcing_num_inv;

	float_type ctr_dis_x = 0.0;
	float_type ctr_dis_y = 0.0;

    float_type rmin_ref_;
    float_type rmax_ref_;
    float_type rz_ref_;
    float_type c1=0;
    float_type c2=0;
    float_type eps_grad_=1.0e6;;

	float_type pert_mag;
    int nLevelRefinement_=0;
    int hard_max_level_ = 0;
    int global_refinement_=0;
    fcoord_t offset_;

    float_type a_ = 10.0;

    float_type dt_,dx_;
    float_type cfl_;
    float_type Re_;
    int tot_steps_;
    float_type refinement_factor_=1./8;
    float_type base_threshold_=1e-4;

    vr_fct_t vr_fct_;

    std::string ic_filename_, ref_filename_;

    float_type Lx;

    //Resolvent analysis specific
    int n_times, n_start, flow_interval; //number of time shots to average
};

double vortex_run(std::string input, int argc = 0, char** argv = nullptr);

} // namespace iblgf

#endif
