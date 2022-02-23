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
#include <iblgf/solver/time_integration/ifherk.hpp>
#include <iblgf/solver/Newton/NewtonMethod.hpp>

#include "../../setups/setup_Newton.hpp"
#include <iblgf/operators/operators.hpp>

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
         (error_u          , float_type, 2,    1,       1,     face,true  ),
		 (error_p          , float_type, 1,    1,       1,     cell,true  ),
         (test             , float_type, 1,    1,       1,     cell,false ),
        //IF-HERK
         (u_tar            , float_type, 2,    1,       1,     face,true  ),
		 (u                , float_type, 2,    1,       1,     face,true  ),
		 (p                , float_type, 1,    1,       1,     cell,true  ),
         (u_num_inv        , float_type, 2,    1,       1,     face,true  ),
		 (p_num_inv        , float_type, 1,    1,       1,     cell,true  ),
		 (u_ref            , float_type, 2,    1,       1,     face,true  ),
		 (p_ref            , float_type, 1,    1,       1,     cell,true  ),
		 (p_tar            , float_type, 1,    1,       1,     cell,true  ),
		 (u_num            , float_type, 2,    1,       1,     face,true  ),
		 (p_num            , float_type, 1,    1,       1,     cell,true  )
    ))
    // clang-format on
};

struct NS_AMR_LGF : public SetupNewton<NS_AMR_LGF, parameters>
{
    using super_type =SetupNewton<NS_AMR_LGF,parameters>;
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

        clean_p_tar = simulation_.dictionary()->template get_or<bool>("clean_p_tar", false);
        testing_smearing = simulation_.dictionary()->template get_or<bool>("testing_smearing", false);

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

        print_mat = simulation_.dictionary_->template get_or<bool>(
			"print_mat",  false);

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

            simulation_.template read_h5<u_ref_type>(simulation_.restart_field_dir(), "u");
			simulation_.template read_h5<p_ref_type>(simulation_.restart_field_dir(), "p");

			//this->initialize();

			//simulation_.template read_h5<du_i_type>(simulation_.restart_field_dir(), "u");
			//simulation_.template read_h5<dp_i_type>(simulation_.restart_field_dir(), "p");
			//this->initialize(); 
		}

		real_coordinate_type tmp_coord(0.0);
        forcing_tar.resize(domain_->ib().size());
        std::fill(forcing_tar.begin(), forcing_tar.end(), tmp_coord);
        forcing_num.resize(domain_->ib().size());
        std::fill(forcing_num.begin(), forcing_num.end(), tmp_coord);
		forcing_ref.resize(domain_->ib().size());
        std::fill(forcing_ref.begin(), forcing_ref.end(), tmp_coord);
        forcing_num_inv.resize(domain_->ib().size());
        std::fill(forcing_num_inv.begin(), forcing_num_inv.end(), tmp_coord);

		
		if (world.rank() == 0)
			std::cout << "on Simulation: \n" << simulation_ << std::endl;
	}

	float_type run()
	{
		boost::mpi::communicator world;

		time_integration_t ifherk(&this->simulation_);
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
		}
        if (world.rank() != 0) ifherk.pad_velocity<u_type, u_ref_type>(false);
        if (world.rank() != 0) ifherk.pad_velocity<u_type, u_type>(false);
		ifherk.Jacobian<u_ref_type, p_ref_type, u_tar_type, p_tar_type>(forcing_ref, forcing_tar);
        if (clean_p_tar) {
            ifherk.clean<p_tar_type>();
        }

        for (int i = 0; i < forcing_ref.size();i++) {
            if (domain_->ib().rank(i)!=world.rank()) {
                /*if (forcing_ref[i][0]!=0 || forcing_ref[i][1]!=0) {
                    std::cout << "Now local forcing got reassigned" << std::endl;
                }*/
                forcing_ref[i] = 0;
                continue;
            }

			float_type t = static_cast<float_type>(i)/static_cast<float_type>(forcing_ref.size());
			float_type val_u = std::sin(2*M_PI*t); 
			float_type val_v = std::cos(2*M_PI*t); 

            /*float_type val_u = 1.0; 
			float_type val_v = 1.0;*/ 

			forcing_ref[i][0] = val_u;
			forcing_ref[i][1] = val_v;
		}
        ifherk.Assigning_idx();
		world.barrier();
		simulation_.write("init.hdf5");

        ifherk.construct_linear_mat<u_type>();

		if (print_mat)
        {
            /*for (int k = 1; k < world.size(); k++)
            {
                if (world.rank() == k)
                {
                    for (int i = 1; i < ifherk.Jac.numRow_loc(); i++)
                    {
                        std::cout << i << " ";
                        ifherk.Jac.print_row(i);
                    }
                }
                world.barrier();
            }*/

            if (world.rank() == 0) {
                std::cout << "L" << std::endl;                
            }
            world.barrier();

            for (int k = 1; k < world.size(); k++)
            {
                if (world.rank() == k)
                {
                    for (int i = 1; i < ifherk.Jac.numRow_loc(); i++)
                    {
                        std::cout << i << " ";
                        ifherk.L.print_row(i);
                    }
                }
                world.barrier();
            }
            world.barrier();

            if (world.rank() == 0) {
                std::cout << "DN" << std::endl;          
            }
            world.barrier();

            for (int k = 1; k < world.size(); k++)
            {
                if (world.rank() == k)
                {
                    for (int i = 1; i < ifherk.Jac.numRow_loc(); i++)
                    {
                        std::cout << i << " ";
                        ifherk.DN.print_row(i);
                    }
                }
                world.barrier();
            }
            world.barrier();

            if (world.rank() == 0) {
                std::cout << "Div" << std::endl;          
            }
            world.barrier();

            for (int k = 1; k < world.size(); k++)
            {
                if (world.rank() == k)
                {
                    for (int i = 1; i < ifherk.Jac.numRow_loc(); i++)
                    {
                        std::cout << i << " ";
                        ifherk.Div.print_row(i);
                    }
                }
                world.barrier();
            }
            world.barrier();

            if (world.rank() == 0) {
                std::cout << "Grad" << std::endl;          
            }
            world.barrier();

            for (int k = 1; k < world.size(); k++)
            {
                if (world.rank() == k)
                {
                    for (int i = 1; i < ifherk.Jac.numRow_loc(); i++)
                    {
                        std::cout << i << " ";
                        ifherk.Grad.print_row(i);
                    }
                }
                world.barrier();
            }
            world.barrier();

            if (world.rank() == 0) {
                std::cout << "project" << std::endl;          
            }
            world.barrier();

            for (int k = 1; k < world.size(); k++)
            {
                if (world.rank() == k)
                {
                    for (int i = 1; i < ifherk.Jac.numRow_loc(); i++)
                    {
                        std::cout << i << " ";
                        ifherk.project.print_row(i);
                    }
                }
                world.barrier();
            }
            world.barrier();

            if (world.rank() == 0) {
                std::cout << "smearing" << std::endl;          
            }
            world.barrier();

            for (int k = 1; k < world.size(); k++)
            {
                if (world.rank() == k)
                {
                    for (int i = 1; i < ifherk.Jac.numRow_loc(); i++)
                    {
                        std::cout << i << " ";
                        ifherk.smearing.print_row(i);
                    }
                }
                world.barrier();
            }
            world.barrier();

            if (world.rank() == 0) {
                std::cout << "boundary_u" << std::endl;          
            }
            world.barrier();

            for (int k = 1; k < world.size(); k++)
            {
                if (world.rank() == k)
                {
                    for (int i = 1; i < ifherk.Jac.numRow_loc(); i++)
                    {
                        std::cout << i << " ";
                        ifherk.boundary_u.print_row(i);
                    }
                }
                world.barrier();
            }
            world.barrier();
        }

        world.barrier();
        if (world.rank() == 1) {
            std::cout << "finishing constructing matrix" << std::endl;
        }

		if (world.rank() == 1) {
            std::cout << "including zero size is " << ifherk.Jac.tot_size(true) << std::endl;
            std::cout << "not including zero size is " << ifherk.Jac.tot_size(false) << std::endl;
        }

		int ndim = ifherk.total_dim();

        float_type* allVec;
        float_type* loc_vec;
        

        std::vector<int> allSize;

        int loc_size = ifherk.Jac.numRow_loc();

        if (world.rank() == 0) {

            allVec = (float_type*)MKL_malloc(sizeof(float_type) * ndim, 64);
            //loc_vec = (float_type*)MKL_malloc(sizeof(float_type) * ndim, 64);
        }
        else {
            allVec = (float_type*)MKL_malloc(sizeof(float_type) * ndim, 64);
            loc_vec = (float_type*)MKL_malloc(sizeof(float_type) * loc_size, 64);
            ifherk.Grid2CSR<u_ref_type, p_ref_type>(loc_vec, forcing_ref, false);
        }

        boost::mpi::gather(world, loc_size, allSize, 0);

        if (world.rank() == 0)
        {
            int begin_idx = 0;

            for (int i = 1; i < world.size(); i++)
            {
                //std::cout << "receiving data from " << i << std::endl;
                world.recv(i, i, &allVec[begin_idx], allSize[i]);
                begin_idx += allSize[i];
            }
        }
        else {
            //std::cout << "rank " << world.rank() << " sending data" << std::endl;
            world.send(0, world.rank(), &loc_vec[0], loc_size);
        }

        boost::mpi::broadcast(world, &allVec[0], ndim, 0);

        float_type sum = 0.0;

        for (int i = 0; i < ndim;i++) {
            sum += allVec[i]*allVec[i];
        }

        /*for (int i = 0; i < world.size();i++) {
            if (world.rank() == i) {
                std::cout << "Rank " << i << " sum " << sum << std::endl;
            }
            world.barrier();
        }*/

        world.barrier();

        if (testing_smearing) {
            if (world.rank() == 0) {
                std::cout << "smearing" << std::endl;          
            }
            world.barrier();

            for (int k = 1; k < world.size(); k++)
            {
                if (world.rank() == k)
                {
                    for (int i = 1; i <= ifherk.Jac.numRow_loc(); i++)
                    {
                        std::cout << i << " ";
                        ifherk.smearing.print_row(i);
                    }
                }
                world.barrier();
            }
            world.barrier();

            if (world.rank() == 0) {
                std::cout << "Value" << std::endl;          
            }
            for (int k = 1; k < world.size(); k++)
            {
                if (world.rank() == k)
                {
                    std::cout << "Rank " << k << std::endl;
                    for (int i = 0; i < ndim; i++)
                    {
                        std::cout << i << " " << allVec[i] << std::endl;
                    }
                }
                world.barrier();
            }
        }

        float_type* res_tmp;
        float_type* res;
        res_tmp = (float_type*)MKL_malloc(sizeof(float_type) * loc_size, 64);
        res = (float_type*)MKL_malloc(sizeof(float_type) * loc_size, 64);

        ifherk.Jac.Apply(allVec, res);
        //ifherk.smearing.Apply(allVec, res_tmp);
        //apply once to get the BC
        /*ifherk.CSR2CSR_correction(res_tmp, loc_vec, 1.0); 

        //clean to receive values again
        for (int i = 0; i < ndim;i++) {
            allVec[i] = 0.0;
        }


        if (world.rank() == 0)
        {
            int begin_idx = 0;

            for (int i = 1; i < world.size(); i++)
            {
                std::cout << "receiving data from " << i << std::endl;
                world.recv(i, i, &allVec[begin_idx], allSize[i]);
                begin_idx += allSize[i];
            }
        }
        else {
            std::cout << "rank " << world.rank() << " sending data" << std::endl;
            world.send(0, world.rank(), &loc_vec[0], loc_size);
        }

        boost::mpi::broadcast(world, &allVec[0], ndim, 0);

        ifherk.Jac.Apply(allVec, res);*/

        float_type* BC_diff;

        BC_diff = (float_type*)MKL_malloc(sizeof(float_type) * loc_size, 64);

        for (int i = 0; i < loc_size;i++) {
            BC_diff[i] = 0.0;
        }

        ifherk.CSR2CSR_correction(res, BC_diff, 1.0); 

        float_type err_BC = 0.0;
        if (world.rank() != 0)
        {
            for (int i = 0; i < loc_size; i++)
            {
                err_BC += BC_diff[i] * BC_diff[i];
            }
        }

        float_type BC_diff_sum = 0.0;

        boost::mpi::all_reduce(world, err_BC, BC_diff_sum, std::plus<float_type>());

        if (world.rank() == 1) {
            std::cout << "BC error is " << BC_diff_sum << std::endl;
        }

        ifherk.CSR2Grid<u_num_type, p_num_type>(res, forcing_num);

        world.barrier();
        for (int i = 1; i < world.size(); i++)
        {
            if (print_mat && (i == world.rank()))
            {
                for (int i = 0; i < loc_size; i++)
                {
                    std::cout << i << " " << res[i] << std::endl;
                }
            }
            world.barrier();
        }

        float_type u1_inf = this->compute_errors<u_num_type, u_tar_type, error_u_type>(
			std::string("u_0_"), 0);
		float_type u2_inf = this->compute_errors<u_num_type, u_tar_type, error_u_type>(
			std::string("u_1_"), 1);

		float_type p_inf = this->compute_errors<p_num_type, p_tar_type, error_p_type>(
			std::string("p_0_"), 0);

		force_type errVec;

		real_coordinate_type tmp_coord(0.0);
        errVec.resize(domain_->ib().size());
        std::fill(errVec.begin(), errVec.end(), tmp_coord);

		for (int i=0; i<domain_->ib().size(); ++i)
        {
            if (domain_->ib().rank(i)!=world.rank())
                errVec[i]=0;
            else
                errVec[i]=forcing_tar[i]-forcing_num[i];
        }
		
		float_type err_forcing = ifherk.dotVec(errVec, errVec);

		
		if (world.rank() == 1)
			std::cout << "L2 Error of forcing is " << std::sqrt(err_forcing) << std::endl;

		simulation_.write("final_apply.hdf5");

        //ifherk.clean<u_num_type>();
        ifherk.clean<p_num_type>();

        ifherk.clean<error_u_type>();
        ifherk.clean<error_p_type>();

        //std::fill(forcing_num.begin(), forcing_num.end(), tmp_coord);

        MKL_INT n = ndim;
		MKL_INT mtype = 11;

		MKL_INT nrhs = 1;
		void *pt[64] = { 0 };
		MKL_INT iparm[64];

		for (int i = 0; i < 64;i++) {
			iparm[i] = 0;
		}		

		MKL_INT maxfct, mnum, phase, msglvl, error, err_mem;

		//int comm;

		float_type  ddum; /* Double dummy   */
    	MKL_INT idum; /* Integer dummy. */
    	MKL_INT j;


		maxfct = 1;
		mnum = 1;
		msglvl = 1;
		error = 0;
		err_mem = 0;

		/*MKL_INT ia[21];
		MKL_INT ja[20];
		float_type b[20];
		float_type a[20];
		float_type x[20];*/

        MKL_INT* ia = NULL;
        MKL_INT* ja = NULL;
        float_type*  a = NULL;
        /* RHS and solution vectors. */
        float_type* b = NULL;
        float_type* x = NULL;

        int    mpi_stat = 0;
        int    argc = 0;
        int    comm, rank, size;
        char** argv;

        mpi_stat = MPI_Init(&argc, &argv);
        mpi_stat = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        mpi_stat = MPI_Comm_size(MPI_COMM_WORLD, &size);
        comm = MPI_Comm_c2f(MPI_COMM_WORLD);

		iparm[0] = 1; /* Solver default parameters overriden with provided by iparm */
        iparm[1] = 2;  /* Use METIS for fill-in reordering */
        iparm[5] = 0;  /* Write solution into x */
        iparm[7] = 2;  /* Max number of iterative refinement steps */
        iparm[9] = 13; /* Perturb the pivot elements with 1E-13 */
        iparm[10] = 1; /* Use nonsymmetric permutation and scaling MPS */
        iparm[12] = 1; /* Switch on Maximum Weighted Matching algorithm (default for non-symmetric) */
        iparm[17] = -1; /* Output: Number of nonzeros in the factor LU */
        iparm[18] = -1; /* Output: Mflops for LU factorization */
        iparm[26] = 1;  /* Check input data for correctness */
        iparm[39] = 2; /* Input: matrix/rhs/solution are distributed between MPI processes  */

        if (world.rank() != 0) {
            int begin_row = ifherk.num_start();
            int end_row = ifherk.num_end();
            iparm[40] = begin_row;
            iparm[41] = end_row;

            int loc_size = end_row - begin_row+1;
            int check_size = ifherk.Jac.numRow_loc();

            if (loc_size != check_size) {
                std::cout << "Rank " << world.rank() << " local matrix size does not match " << loc_size << " vs " << check_size << std::endl;
                return 1;
            }

            int tot_size = ifherk.Jac.tot_size();
            ia = (MKL_INT*)MKL_malloc(sizeof(MKL_INT) * (loc_size + 1), 64);
            ja = (MKL_INT*)MKL_malloc(sizeof(MKL_INT) * tot_size, 64);
            a = (float_type*)MKL_malloc(sizeof(float_type) * tot_size, 64);
            for (int k =0 ; k < tot_size; k++) {
                a[k] = 0;
            }
            x = (float_type*)MKL_malloc(sizeof(float_type) * loc_size, 64);
            b = (float_type*)MKL_malloc(sizeof(float_type) * loc_size, 64);
            for (int k =0 ; k < loc_size; k++) {
                x[k] = 0;
                b[k] = 0;
            }

            ifherk.Jac.getCSR(ia, ja, a);
            //ifherk.Grid2CSR<u_num_type, p_num_type>(b, forcing_num);
            ifherk.Grid2CSR<u_tar_type, p_tar_type>(b, forcing_tar);

            if (print_mat) {
                for (int i = 0; i < loc_size; i++) {
                    std::cout << i << " " << b[i] << std::endl;
                }
            }
        }
        else {
            iparm[40] = 2;
			iparm[41] = 1;
        }

        MPI_Barrier(MPI_COMM_WORLD);
        if (world.rank() == 1) std::cout << "Start computing" << std::endl;

        /*if (world.rank() == 0) {
			iparm[40] = 1;
			iparm[41] = 20;
			for (int i = 0; i < 20; i++) {
				ia[i] = i+1;
				ja[i] = i+1;
				a[i] = static_cast<float_type>(i+1);
				b[i] = 1.0/static_cast<float_type>(i+1);
			}
			ia[20]=21;
		}
		else if (world.rank() == 1) {
			iparm[40] = 21;
			iparm[41] = 40;
			for (int i = 0; i < 20; i++) {
				ia[i] = i+1;
				ja[i] = i+21;
				a[i] = static_cast<float_type>(i+21);
				b[i] = 1.0/static_cast<float_type>(i+21);
			}
			ia[20]=21;
		}
		else {
			iparm[40] = 2;
			iparm[41] = 1;
		}*/

        /* -------------------------------------------------------------------- */
        /* .. Reordering and Symbolic Factorization. This step also allocates   */
        /* all memory that is necessary for the factorization.                  */
        /* -------------------------------------------------------------------- */
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

        /* -------------------------------------------------------------------- */
        /* .. Numerical factorization.                                          */
        /* -------------------------------------------------------------------- */
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

        /* -------------------------------------------------------------------- */
        /* .. Back substitution and iterative refinement.                       */
        /* -------------------------------------------------------------------- */
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
        /* The solution of the system is distributed between MPI processes like as input matrix
       so MPI processes with rank 0 and 1 keep only part of solution */
        if (rank == 2)
        {
            printf("\nThe solution of the system is: ");
            for (j = 0; j < 3; j++)
            {
                std::cout << "\n on zero process x " << j << "=" <<
                    x[j];
            }
            
        }
        MPI_Barrier(MPI_COMM_WORLD);

        if (rank == 1)
        {
            printf("\nThe solution of the system is: ");
            for (j = 0; j < 3; j++)
            {
                std::cout << "\n on first process x " << j << "=" <<
                    x[j];
            }
            std::cout << std::endl;
            
        }
        MPI_Barrier(MPI_COMM_WORLD);

        /* -------------------------------------------------------------------- */
        /* .. Termination and release of memory. */
        /* -------------------------------------------------------------------- */
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
        
        //Do not finalize MPI again since boost mpi calls it already
        //mpi_stat = MPI_Finalize();
        //std::cout << "  MPI_FINALIZED" << std::endl;

        ifherk.CSR2Grid<u_num_inv_type, p_num_inv_type>(x, forcing_num_inv);

        float_type tmp_sum = 0;
        for (int i = 0; i < loc_size; i++) {
            tmp_sum += x[i]*x[i];
        }

        for (int i = 1; i < world.size();i++) {
            if (i == world.rank()) {
                std::cout << "sum of results at Rank " << i << " is " << tmp_sum << std::endl;
            }
            world.barrier();
        }

        

        /*float_type*/ u1_inf = this->compute_errors<u_num_inv_type, u_ref_type, error_u_type>(
			std::string("u_0_"), 0);
		/*float_type*/ u2_inf = this->compute_errors<u_num_inv_type, u_ref_type, error_u_type>(
			std::string("u_1_"), 1);

		/*float_type*/ p_inf = this->compute_errors<p_num_inv_type, p_ref_type, error_p_type>(
			std::string("p_0_"), 0);

		//force_type errVec;

		//real_coordinate_type tmp_coord(0.0);
        errVec.resize(domain_->ib().size());
        std::fill(errVec.begin(), errVec.end(), tmp_coord);

		for (int i=0; i<domain_->ib().size(); ++i)
        {
            if (domain_->ib().rank(i)!=world.rank())
                errVec[i]=0;
            else
                errVec[i]=forcing_ref[i]-forcing_num_inv[i];
        }
		
		/*float_type*/ err_forcing = ifherk.dotVec(errVec, errVec);

		
		if (world.rank() == 1)
			std::cout << "L2 Error of forcing is " << std::sqrt(err_forcing) << std::endl;

		simulation_.write("final.hdf5");

        MPI_Barrier(MPI_COMM_WORLD);

        if (rank  != 0)
        {
            MKL_free(ia);
            MKL_free(ja);
            MKL_free(a);
            MKL_free(x);
            MKL_free(b);
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

				/*int rand_val = rand();

				float_type v_1 = static_cast<float_type>(rand_val)/static_cast<float_type>(RAND_MAX) * pert_mag*2 - pert_mag;
				rand_val = rand();
				float_type v_2 = static_cast<float_type>(rand_val)/static_cast<float_type>(RAND_MAX) * pert_mag*2 - pert_mag;
				rand_val = rand();
				float_type v_3 = static_cast<float_type>(rand_val)/static_cast<float_type>(RAND_MAX) * pert_mag*2 - pert_mag;*/

				//node(u,0) = tmpf0*(4*xf0*xf0 - 2.0 + 4*yf0*yf0 - 2.0);
				//node(u,1) = tmpf1*(4*xf1*xf1 - 2.0 + 4*yf1*yf1 - 2.0);
				//node(p,0) = tmpc*(4*xc*xc - 2.0 + 4*yc*yc - 2.0);

                //node(u_ref,0) = tmpf0;
                //node(u_ref,1) = tmpf1;
                //node(p_ref,0) = tmpc;
                node(u_ref,0) = 10.0;
                node(u_ref,1) = 10.0;
                node(p_ref,0) = 10.0;
                //node(p_ref,0) = 0;

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
    int vortexType = 0;

    int row_to_print = 0;

    bool clean_p_tar;
    bool testing_smearing = false;

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
};

double vortex_run(std::string input, int argc = 0, char** argv = nullptr);

} // namespace iblgf

#endif
