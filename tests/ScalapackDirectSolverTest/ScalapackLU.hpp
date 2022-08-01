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

#ifndef IBLGF_INCLUDED_SCALAPACKLU_HPP
#define IBLGF_INCLUDED_SCALAPACKLU_HPP

#ifndef DEBUG_IFHERK
#define DEBUG_IFHERK
#endif
#define DEBUG_POISSON

static char help[] = "Solves a IB system.\n\n";

#include "mpi.h"
#include <petscksp.h>

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
#include <iblgf/solver/time_integration/ifherk_helm.hpp>

#include "../../setups/setup_helmholtz.hpp"
#include <iblgf/operators/operators.hpp>


namespace iblgf
{
const int Dim = 2;

struct parameters
{
    static constexpr std::size_t Dim = 2;
	static constexpr std::size_t N_modes = 16;
	static constexpr std::size_t PREFAC  = 2; //2 for complex values 
    // clang-format off
    REGISTER_FIELDS
    (
    Dim,
     (
        //name               type        Dim   lBuffer  hBuffer, storage type
         (error_u          , float_type, 3*2*N_modes,    1,       1,     face,true ),
         (error_p          , float_type, 1*2*N_modes,    1,       1,     cell,true ),
         (test             , float_type, 1*2*N_modes,    1,       1,     cell,false ),
        //IF-HERK
         (u                , float_type, 3*2*N_modes,    1,       1,     face,true  ),
         (u_ref            , float_type, 3*2*N_modes,    1,       1,     face,true  ),
         (p_ref            , float_type, 1*2*N_modes,    1,       1,     cell,true  ),
         (p                , float_type, 1*2*N_modes,    1,       1,     cell,true  ),
         (w_num            , float_type, 1*2*N_modes,    1,       1,     edge,false ),
         (w_exact          , float_type, 1*2*N_modes,    1,       1,     edge,false ),
         (error_w          , float_type, 1*2*N_modes,    1,       1,     edge,false ),
         //for radial velocity
         (exact_u_theta    , float_type, 3*2*N_modes,    1,       1,     edge,false ),
         (num_u_theta      , float_type, 3*2*N_modes,    1,       1,     edge,false ),
         (error_u_theta    , float_type, 3*2*N_modes,    1,       1,     edge,false )
    ))
    // clang-format on
};

struct NS_AMR_LGF : public Setup_helmholtz<NS_AMR_LGF, parameters>
{
    using super_type = Setup_helmholtz<NS_AMR_LGF,parameters>;
    using vr_fct_t = std::function<float_type(float_type x, float_type y, int field_idx, bool perturbation)>;

    //Timings
    using clock_type = std::chrono::high_resolution_clock;
    using milliseconds = std::chrono::milliseconds;
    using duration_type = typename clock_type::duration;
    using time_point_type = typename clock_type::time_point;
    using linsys_solve_t = typename super_type::linsys_solver_t;

	using simulation_type = typename super_type::simulation_t;
	using domain_type = typename simulation_type::domain_type;
	using ib_t = typename domain_type::ib_t;
    using force_type = typename ib_t::force_type;
    using point_force_type = types::vector_type<float_type, 3 * 2 * N_modes>;

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
		U_.resize(domain_->dimension());
		U_[0] = simulation_.dictionary()->template get_or<float_type>("Ux", 1.0);
		U_[1] = simulation_.dictionary()->template get_or<float_type>("Uy", 0.0);
		if (domain_->dimension()>2)
			U_[2] = simulation_.dictionary()->template get_or<float_type>("Uz", 0.0);

		smooth_start_ = simulation_.dictionary()->template get_or<bool>("smooth_start", false);

		vortexType = simulation_.dictionary()->template get_or<int>("Vort_type", 0);

		pert_mag = simulation_.dictionary()->template get_or<float_type>("pert_mag", 0.1);

		simulation_.frame_vel() =
			[this](std::size_t idx, float_type t, auto coord = {0, 0})
			{
				float_type T0 = 0.5;
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
				}
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
			//this->initialize(); 
		}

		
		if (world.rank() == 0)
			std::cout << "on Simulation: \n" << simulation_ << std::endl;
	}

	float_type run(int argc, char *argv[])
	{
		boost::mpi::communicator world;

        point_force_type tmp(0.0);

        force_type tmp_f(domain_->ib().size(), tmp);

        for (int i = 0; i < tmp_f.size(); i++) {
            for (int j = 0; j < tmp_f[0].size(); j++) {
                tmp_f[i][j] = 0.0;
            }
        }

        linsys_solve_t ib_solve(&this->simulation_);

        int force_dim = tmp_f.size()*(u_type::nFields())/N_modes;

        float_type alpha = 0.01;

        std::vector<force_type> matrix_force;

        if (world.rank() == 1) {
            std::cout << "finished initializing" << std::endl;
        }

        if (world.rank() != 0) {
        for (int num = 0; num < force_dim;num++) {
            if (world.rank() == 1)
            {
                if (num % ((u_type::nFields())/N_modes) == 0)
                {
                    std::cout << "number solved " << num << " over "
                              << force_dim << std::endl;
                }
            }
            for (int i = 0; i < tmp_f.size(); i++)
            {
                for (int j = 0; j < tmp_f[0].size(); j++) { tmp_f[i][j] = 0.0; }
            }
            int ib_idx = num / ((u_type::nFields())/N_modes);
            int field_idx = num % ((u_type::nFields())/N_modes);
            int idx_complex = field_idx/2; //the number of components (zero for u, one for v, two for w)
            int realcomp = field_idx % 2;  //zero if real part but one if complex part

            if (domain_->ib().rank(ib_idx)==world.rank()) {
                for (int ModeN = 0; ModeN < N_modes; ModeN++) {
                    int field_idx_now = idx_complex*N_modes*2 + ModeN*2 + realcomp;
                    tmp_f[ib_idx][field_idx_now] = 1.0;
                }
                //tmp_f[ib_idx][field_idx] = 1.0;
            }
            /*for (int i = 0; i < tmp_f.size(); i++)
            {
                if (domain_->ib().rank(i)!=world.rank()) continue;
                tmp_f[i][num] = 1.0;
            }*/
            force_type Ap(domain_->ib().size(), tmp);
            for (int i = 0; i < Ap.size(); i++)
            {
                for (int j = 0; j < Ap[0].size(); j++) { Ap[i][j] = 0.0; }
            }
            ib_solve.ET_H_S_E<face_aux2_type>(tmp_f, Ap, alpha);
            for (int i = 0; i < Ap.size(); i++)
            {
                if (domain_->ib().rank(i)!=world.rank()) {
                    for (int j = 0; j < Ap[0].size(); j++) { Ap[i][j] = 0.0; }
                }
            }
            matrix_force.emplace_back(Ap);
        }

        if (world.rank() == 1) {
            std::cout << "finished constructing" << std::endl;
            std::cout << "number of rows " << matrix_force.size() << std::endl;
        }
        }

        int zeros = 0;
        int nonzeros = 0;

        if (world.rank() != 0) {

        for (int i = 0; i < matrix_force.size(); i++) {
            //if (domain_->ib().rank(i)!=world.rank()) continue;
            force_type force_tmp = matrix_force[i];
            for (int j = 0; j < force_tmp.size(); j++) {
                if (domain_->ib().rank(j)!=world.rank()) continue;
                for (int k = 0; k < force_tmp[0].size(); k++) {
                    float_type v_tmp = force_tmp[j][k];
                    if (std::abs(v_tmp) > 1e-12) {
                        nonzeros += 1;
                    }
                    else {
                        zeros += 1;
                    }
                }
            }
        }
        }

        
        

        if (world.rank() == 0) {
            zeros = 0;
            nonzeros = 0;
        }

        int zeros_glob;
        int non_zeros_glob;
		
        boost::mpi::all_reduce(world, zeros, zeros_glob, std::plus<int>());
        boost::mpi::all_reduce(world, nonzeros, non_zeros_glob, std::plus<int>());

        if (world.rank() == 1) {
            std::cout << "nonzeros are " << non_zeros_glob << std::endl;
            std::cout << "zeros are " << zeros_glob << std::endl;
        }

        std::vector<force_type> matrix_force_glob = matrix_force;

        for (int i = 0; i < matrix_force_glob.size(); i++) {
            boost::mpi::all_reduce(domain_->client_communicator(), &matrix_force[i][0], domain_->ib().size(), &matrix_force_glob[i][0], std::plus<point_force_type>());
        }

        float_type sum = 0;
        if (world.rank() == 1) {
            for (int nModes = 0; nModes < N_modes; nModes++) {
                for (int i = 0; i < force_dim; i++) {
                    for (int j = i; j < force_dim; j++) {
                        int ib_idx_i = i / ((u_type::nFields())/N_modes);
                        int field_idx_i = i % ((u_type::nFields())/N_modes);
                        int idx_complex_i = field_idx_i/2; //the number of components (zero for u, one for v, two for w)
                        int realcomp_i = field_idx_i % 2;  //zero if real part but one if complex part

                        int field_idx_now_i = idx_complex_i*N_modes*2 + nModes*2 + realcomp_i;
                        int field_idx_short_i = idx_complex_i*N_modes*2 + realcomp_i;

                        int ib_idx_j = j / ((u_type::nFields())/N_modes);
                        int field_idx_j = j % ((u_type::nFields())/N_modes);
                        int idx_complex_j = field_idx_j/2; //the number of components (zero for u, one for v, two for w)
                        int realcomp_j = field_idx_j % 2;  //zero if real part but one if complex part

                        int field_idx_now_j = idx_complex_j*N_modes*2 + nModes*2 + realcomp_j;
                        int field_idx_short_j = idx_complex_j*N_modes*2 + realcomp_j;

                        float_type val_ij =
                            matrix_force_glob[i][ib_idx_j][field_idx_now_j];
                        float_type val_ji =
                            matrix_force_glob[j][ib_idx_i][field_idx_now_i];

                        sum += (val_ij - val_ji) * (val_ij - val_ji);
                    }
                }
            }
            std::cout << "asymetry is " << sum << std::endl;

            //for (int nModes = 0; nModes < N_modes; nModes++) {
            std::ofstream outfile;
            outfile.open("Matrix_first.txt", std::ios_base::app);
            for (int i = 0; i < force_dim; i++)
            {
                for (int j = 0; j < force_dim; j++)
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
                        idx_complex_j * N_modes * 2 + realcomp_j;
                    int field_idx_short_j = idx_complex_j*N_modes*2 + realcomp_j;

                    float_type val_ij =
                        matrix_force_glob[i][ib_idx_j][field_idx_now_j];
                    float_type val_ji =
                        matrix_force_glob[j][ib_idx_i][field_idx_now_i];

                    outfile << val_ij << " ";
                }
                outfile << std::endl;
            }
            outfile.close();
            //}
        }

        //the construction of IB matrix is finished, now testing if Scalapack works with simple examples
        //find a manufactured solution
        force_type Ap_glob(domain_->ib().size(), tmp);
        force_type p(domain_->ib().size(), tmp);
        force_type Ap(domain_->ib().size(), tmp);
        if (world.rank() != 0) {
        for (int i = 0; i < p.size(); i++) {
            for (int j = 0; j < p[0].size(); j++) {
                p[i][j] = 1.0;
            }
        }
        for (int i = 0; i < p.size(); i++)
        {
            if (domain_->ib().rank(i) != world.rank())
            {
                for (int j = 0; j < p[0].size(); j++) { p[i][j] = 0.0; }
            }
        }
        
        for (int i = 0; i < Ap.size(); i++)
        {
            for (int j = 0; j < Ap[0].size(); j++) { Ap[i][j] = 0.0; }
        }
        ib_solve.ET_H_S_E<face_aux2_type>(p, Ap, alpha);
        for (int i = 0; i < Ap.size(); i++)
        {
            if (domain_->ib().rank(i) != world.rank())
            {
                for (int j = 0; j < Ap[0].size(); j++) { Ap[i][j] = 0.0; }
            }
        }

        boost::mpi::all_reduce(domain_->client_communicator(), &Ap[0], domain_->ib().size(), &Ap_glob[0], std::plus<point_force_type>());
        }

        PetscMPIInt    rank, size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        PetscMPIInt Color = 0;
        if (world.rank() != 0) {
            Color = 1;
        }

        MPI_Comm_split(MPI_COMM_WORLD, Color, 0, &PETSC_COMM_WORLD);
        if (Color != 0)
        {
            Vec       x, b, u; /* approx solution, RHS, exact solution */
            Mat       A;       /* linear system matrix */
            KSP       ksp;     /* linear solver context */
            PC        pc;      /* preconditioner context */
            PetscReal norm,
                tol =
                    1000. * PETSC_MACHINE_EPSILON; /* norm of solution error */
            PetscInt i, n = force_dim, col[3], its, rstart, rend, nlocal;

            

            PetscMPIInt rank, size;
            MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
            MPI_Comm_size(PETSC_COMM_WORLD, &size);

            if (rank != (size-1)) nlocal = n / (size - 1);
            else nlocal = n % (size - 1);


            PetscScalar one = 1.0, value[3], zero = 0.0;

            PetscCall(PetscInitialize(&argc, &argv, (char*)0, help));

            PetscCall(VecCreate(PETSC_COMM_WORLD, &x));

            PetscCall(VecSetSizes(x, nlocal, n));
            PetscCall(VecSetFromOptions(x));
            PetscCall(VecDuplicate(x, &b));
            PetscCall(VecDuplicate(x, &u));

            PetscCall(VecGetOwnershipRange(x, &rstart, &rend));

            std::cout << "rank " << rank << " start and end " << rstart << " "
                      << rend << std::endl;
            PetscCall(VecGetLocalSize(x, &nlocal));

            PetscCall(MatCreateDense(PETSC_COMM_WORLD,nlocal, nlocal, n, n, NULL, &A));
            MatSetType(A,MATSCALAPACK);
            //PetscCall(MatCreateScaLAPACK(PETSC_COMM_WORLD,nlocal, n, n, n, 0,0, &A));
            //PetscCall(MatSetSizes(A, nlocal, nlocal, n, n));
            PetscCall(MatSetFromOptions(A));
            PetscCall(MatSetUp(A));

            for (i = rstart; i < rend; i++)
            {
                
                for (int j = 0; j < force_dim; j++)
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
                        idx_complex_j * N_modes * 2 + realcomp_j;
                    int field_idx_short_j = idx_complex_j*N_modes*2 + realcomp_j;

                    float_type val_ij =
                        matrix_force_glob[i][ib_idx_j][field_idx_now_j];
                    float_type val_ji =
                        matrix_force_glob[j][ib_idx_i][field_idx_now_i];

                    PetscScalar valA = val_ij;
                    PetscCall(MatSetValues(A, 1, &i, 1, &j, &valA,
                        INSERT_VALUES));
                }
            }

            MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);
            MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);

            for (i = rstart; i < rend; i++)
            {
                int ib_idx_i = i / ((u_type::nFields()) / N_modes);
                int field_idx_i = i % ((u_type::nFields()) / N_modes);
                int idx_complex_i =
                    field_idx_i /
                    2; //the number of components (zero for u, one for v, two for w)
                int realcomp_i =
                    field_idx_i % 2; //zero if real part but one if complex part

                int field_idx_now_i = idx_complex_i * N_modes * 2 + realcomp_i;
                int field_idx_short_i =
                    idx_complex_i * N_modes * 2 + realcomp_i;

                PetscScalar v = Ap_glob[ib_idx_i][field_idx_now_i];
                PetscCall(VecSetValues(b, 1, &i, &v, INSERT_VALUES));

                PetscScalar vx = 1.0;
                PetscCall(VecSetValues(u, 1, &i, &vx, INSERT_VALUES));
            }

            VecAssemblyBegin(b);
            VecAssemblyEnd(b);

            VecAssemblyBegin(u);
            VecAssemblyEnd(u);

            PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
            PetscCall(KSPSetOperators(ksp, A, A));
            auto t0 = clock_type::now();
            PetscCall(KSPGetPC(ksp, &pc));
            PetscCall(PCSetType(pc, PCLU));
            auto t1 = clock_type::now();
            mDuration_type ms_PC = t1 - t0;
            if (rank == 0) {
            std::cout << "Set LU in " << ms_PC.count() << std::endl;
            }


            PetscCall(KSPSetTolerances(ksp, 1.e-7, PETSC_DEFAULT, PETSC_DEFAULT,
                PETSC_DEFAULT));

            PetscCall(KSPSetFromOptions(ksp));
            auto t2 = clock_type::now();

            PetscCall(KSPSolve(ksp, b, x));
            auto t3 = clock_type::now();
            mDuration_type ms_solve = t3 - t2;

            PetscCall(VecAXPY(x, -1.0, u));
            PetscCall(VecNorm(x, NORM_2, &norm));
            if (rank == 0)
            {
                std::cout << "solution mag is " << norm << std::endl;
            }
            if (rank == 0) {
            std::cout << "First solve in " << ms_solve.count() << std::endl;
            }
            auto t4 = clock_type::now();

            PetscCall(KSPSolve(ksp, b, x));
            auto t5 = clock_type::now();
            mDuration_type ms_solve2 = t5 - t4;
            if (rank == 0) {
            std::cout << "Second solve in " << ms_solve2.count() << std::endl;
            }

            PetscCall(KSPView(ksp, PETSC_VIEWER_STDOUT_WORLD));

            PetscCall(VecAXPY(x, -1.0, u));
            PetscCall(VecNorm(x, NORM_2, &norm));
            if (rank == 0)
            {
                std::cout << "solution mag is " << norm << std::endl;
            }
            PetscCall(KSPGetIterationNumber(ksp, &its));
        }

        return 0;

	}

    /** @brief  Refienment conditon for octants.  */
    template<class OctantType>
    bool refinement(
        OctantType* it, int diff_level, bool use_all = false) const noexcept
    {
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

				node(u,0) = v_1;
				node(u,1) = v_2;
				node(p,0) = v_3;

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
    int vortexType = 0;

    int row_to_print = 0;

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
};

double vortex_run(std::string input, int argc = 0, char** argv = nullptr);

} // namespace iblgf

#endif
