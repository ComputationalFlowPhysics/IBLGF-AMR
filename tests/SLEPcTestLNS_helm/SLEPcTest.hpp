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
         (error_uz         , float_type, 1,    1,       1,     cell,true  ),
		 (error_p          , float_type, 1,    1,       1,     cell,true  ),
         (error_N          , float_type, 2,    1,       1,     face,true  ),
         (error_Nz         , float_type, 1,    1,       1,     cell,true  ),
		 (error_cs         , float_type, 1,    1,       1,     cell,true  ),
         (error_w          , float_type, 3,    1,       1,     edge,true  ),
         (test             , float_type, 1,    1,       1,     cell,false ),
        //IF-HERK
		 (u                , float_type, 2,    1,       1,     face,true  ),
         (uz               , float_type, 1,    1,       1,     cell,true  ),
		 (p                , float_type, 1,    1,       1,     cell,true  ),
         (u_num_inv        , float_type, 2,    1,       1,     face,true  ),
         (N_num_inv        , float_type, 2,    1,       1,     face,true  ),
         (uz_num_inv       , float_type, 1,    1,       1,     cell,true  ),
         (Nz_num_inv       , float_type, 1,    1,       1,     cell,true  ),
		 (p_num_inv        , float_type, 1,    1,       1,     cell,true  ),
         (w_num_inv        , float_type, 3,    1,       1,     edge,true  ),
		 (u_ref            , float_type, 2,    1,       1,     face,true  ),
         (uz_ref           , float_type, 1,    1,       1,     cell,true  ),
		 (p_ref            , float_type, 1,    1,       1,     cell,true  ),
         (cs_ref           , float_type, 1,    1,       1,     cell,true  ),
         (N_ref            , float_type, 2,    1,       1,     face,true  ),
         (Nz_ref           , float_type, 1,    1,       1,     cell,true  ),
         (w_ref            , float_type, 3,    1,       1,     edge,true  ),
         (u_tar            , float_type, 2,    1,       1,     face,true  ),
         (uz_tar           , float_type, 1,    1,       1,     cell,true  ),
		 (p_tar            , float_type, 1,    1,       1,     cell,true  ),
         (cs_tar           , float_type, 1,    1,       1,     cell,true  ),
         (N_tar            , float_type, 2,    1,       1,     face,true  ),
         (Nz_tar           , float_type, 1,    1,       1,     cell,true  ),
         (w_tar            , float_type, 3,    1,       1,     edge,true  ),
		 (u_num            , float_type, 2,    1,       1,     face,true  ),
         (uz_num           , float_type, 1,    1,       1,     cell,true  ),
		 (p_num            , float_type, 1,    1,       1,     cell,true  ),
         (cs_num           , float_type, 1,    1,       1,     cell,true  ),
         (N_num            , float_type, 2,    1,       1,     face,true  ),
         (Nz_num           , float_type, 1,    1,       1,     cell,true  ),
         (w_num            , float_type, 3,    1,       1,     edge,true  )
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
        target_real = simulation_.dictionary()->template get_or<float_type>("target_real", 1.0);
        target_imag = simulation_.dictionary()->template get_or<float_type>("target_imag", 1.0);
        testing_smearing = simulation_.dictionary()->template get_or<bool>("testing_smearing", false);
        num_input = simulation_.dictionary()->template get_or<bool>("num_input", false);
        check_mat_res = simulation_.dictionary()->template get_or<bool>("check_mat_res", false);
        addImagBC = simulation_.dictionary()->template get_or<bool>("addImagBC", false);

		auto domain_range = domain_->bounding_box().max() - domain_->bounding_box().min();
		Lx = domain_range[0] * dx_;



		ctr_dis_x = 0.0*dx_; //this is setup as the center of the vortex in the unit of grid spacing
		ctr_dis_y = 0.0*dx_;


		



		bool use_fat_ring = simulation_.dictionary()->template get_or<bool>("fat_ring", false);

        mode_c = simulation_.dictionary()->template get_or<float_type>("mode_c", 0.1);
        if (mode_c < 0.0) {
            mode_c = -mode_c;
        }

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

        if (check_mat_res)
        {
            if (world.rank() != 0)
                ifherk.pad_velocity<u_ref_type, uz_ref_type, u_ref_type,
                    uz_ref_type>(true);
            if (world.rank() != 0)
                ifherk.pad_velocity<u_type, uz_type, u_type, uz_ref_type>(true);

            world.barrier();
            if (world.rank() == 1) std::cout << "Curl" << std::endl;

            //if (world.rank() != 0) ifherk.Curl_access<u_ref_type, w_ref_type>();

            ifherk.Jacobian<u_ref_type, uz_ref_type, p_ref_type, u_tar_type,
                uz_tar_type, p_tar_type>(forcing_ref, fz, forcing_tar, fz_tar);

            world.barrier();
            if (world.rank() == 1)
                std::cout << "Upward interpolation" << std::endl;

            //if (world.rank() != 0) ifherk.Upward_interpolation<w_ref_type, w_ref_type>();
            if (world.rank() != 0) ifherk.up_and_down<u_ref_type>();
            if (world.rank() != 0) ifherk.up_and_down<uz_ref_type>();
            world.barrier();
            if (world.rank() == 1)
                std::cout << "up and down finished" << std::endl;
            if (world.rank() != 0)
                ifherk.Curl_access<u_ref_type, uz_ref_type, w_ref_type>();
            world.barrier();
            if (world.rank() == 1)
                std::cout << "Curl access finished" << std::endl;
            if (world.rank() != 0)
                ifherk.nonlinear_Jac_access<u_type, u_ref_type, uz_type,
                    uz_ref_type, N_ref_type, Nz_ref_type>();
            world.barrier();
            if (world.rank() == 1)
                std::cout << "nonlinear Jac access finished" << std::endl;
            if (world.rank() != 0) ifherk.AddSmearing<N_ref_type>(forcing_ref);
            world.barrier();
            if (world.rank() == 1)
                std::cout << "Smearing finished" << std::endl;
            if (world.rank() != 0) ifherk.AddSmearingUz<Nz_ref_type>(fz);
            world.barrier();
            if (world.rank() == 1)
                std::cout << "Smearing Uz finished" << std::endl;
            if (world.rank() != 0)
                ifherk.div_access<N_ref_type, Nz_ref_type, cs_ref_type>();
            world.barrier();
            if (world.rank() == 1)
                std::cout << "Div access finished" << std::endl;
            if (world.rank() != 0) ifherk.up_and_down<cs_ref_type>();
            world.barrier();
            if (world.rank() == 1)
                std::cout << "up and down cs ref finished" << std::endl;
            if (world.rank() != 0)
                ifherk.pad_pressure<cs_ref_type, p_ref_type>();
            world.barrier();
            if (world.rank() == 1)
                std::cout << "pressure pad finished" << std::endl;
            if (world.rank() != 0)
                ifherk.Upward_interpolation<cs_ref_type, cs_ref_type>();
            world.barrier();
            if (world.rank() == 1)
                std::cout << "cs interpolation finished" << std::endl;

            if (world.rank() != 0)
                ifherk.Upward_interpolation<w_ref_type, w_ref_type>();
            world.barrier();
            if (world.rank() == 1) std::cout << "interpolation" << std::endl;
            if (world.rank() != 0) ifherk.up_and_down<p_ref_type>();
            world.barrier();
            if (world.rank() == 1) std::cout << "Clean" << std::endl;
            if (world.rank() != 0) ifherk.clean<w_tar_type>();
            if (clean_p_tar)
            {
                if (world.rank() != 0) ifherk.clean<p_tar_type>();
            }

            for (int i = 0; i < forcing_ref.size(); i++)
            {
                if (domain_->ib().rank(i) != world.rank())
                {
                    /*if (forcing_ref[i][0]!=0 || forcing_ref[i][1]!=0) {
                    std::cout << "Now local forcing got reassigned" << std::endl;
                }*/
                    forcing_ref[i] = 0;
                    continue;
                }

                float_type t = static_cast<float_type>(i) /
                               static_cast<float_type>(forcing_ref.size());
                float_type val_u = std::sin(2 * M_PI * t);
                float_type val_v = std::cos(2 * M_PI * t);

                /*float_type val_u = 1.0; 
			float_type val_v = 1.0;*/

                forcing_ref[i][0] = val_u;
                forcing_ref[i][1] = val_v;
            }
        }
        world.barrier();
        if (world.rank() == 1) std::cout << "constructing matrix" << std::endl;
        
		world.barrier();
		simulation_.write("init.hdf5");

        ifherk.construct_linear_mat<u_type, uz_type>();
        ifherk.construction_imaginary();
        if (addImagBC) ifherk.construction_BCMat_u_imag();
        ifherk.construction_B_matrix();
        ifherk.Jac.clean_entry(1e-10);
        ifherk.Imag.clean_entry(1e-10);

        world.barrier();
        if (world.rank() == 1) {
            std::cout << "finishing constructing matrix" << std::endl;
        }

		if (world.rank() == 1) {
            std::cout << "including zero size is " << ifherk.Jac.tot_size(true) << std::endl;
            std::cout << "not including zero size is " << ifherk.Jac.tot_size(false) << std::endl;
        }

		int ndim = ifherk.total_dim();
        int loc_size = ifherk.Jac.numRow_loc();

        if (check_mat_res) {

        float_type* allVec = NULL;
        float_type* loc_vec = NULL;

        float_type* res_tmp = NULL;
        float_type* res = NULL;

        float_type* BC_diff = NULL;

        float_type* res_num = NULL;
        float_type* res_tar = NULL;

        float_type* errvec = NULL;
        

        std::vector<int> allSize;

        

        if (world.rank() == 0) {

            PetscMalloc(sizeof(float_type) * ndim, &allVec);
            PetscMalloc(sizeof(float_type) * 0, &loc_vec);
        }
        else {
            PetscMalloc(sizeof(float_type) * ndim, &allVec);
            PetscMalloc(sizeof(float_type) * loc_size, &loc_vec);
            ifherk.Grid2CSR<u_ref_type, p_ref_type, w_ref_type, N_ref_type, cs_ref_type, uz_ref_type, Nz_ref_type>(loc_vec, forcing_ref, fz, false);
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

        
        PetscMalloc(sizeof(float_type) * loc_size, &res_tmp);
        PetscMalloc(sizeof(float_type) * loc_size, &res);

        

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

        if (world.rank() != 0) ifherk.clean<edge_aux_type>();

        

        

        PetscMalloc(sizeof(float_type) * loc_size, &BC_diff);

        for (int i = 0; i < loc_size;i++) {
            BC_diff[i] = 0.0;
        }

        ifherk.CSR2CSR_correction(res, BC_diff, 1.0); 

        float_type err_BC = 0.0;
        if (world.rank() != 0)
        {
            for (int i = 0; i < loc_size; i++)
            {
                err_BC += std::abs(BC_diff[i]);
            }
        }

        float_type BC_diff_sum = 0.0;

        boost::mpi::all_reduce(world, err_BC, BC_diff_sum, std::plus<float_type>());

        if (world.rank() == 1) {
            std::cout << "BC error is " << BC_diff_sum << std::endl;
        }

        world.barrier();

        if (world.rank() == 0) {
            std::cout << "start CSR2Grid" << std::endl;
        }

        world.barrier();

        ifherk.CSR2Grid<u_num_type, p_num_type, w_num_type, N_num_type, cs_num_type, uz_num_type, Nz_num_type>(res, forcing_num, fz);

        world.barrier();

        if (world.rank() == 0) {
            std::cout << "computing error nonleaves" << std::endl;
        }

        world.barrier();

        ifherk.compute_error_nonleaf<edge_aux_type, w_num_type, idx_w_type>("edge_num_", false, 0);
        ifherk.compute_error_nonleaf<edge_aux_type, w_ref_type, idx_w_type>("edge_ref_", false, 0);

        ifherk.clean<face_aux_type>();
        ifherk.compute_error_nonleaf<face_aux_type, u_ref_type, idx_u_type>("face_ref_0_", false, 0);
        ifherk.compute_error_nonleaf<face_aux_type, u_num_type, idx_u_type>("face_num_0_", false, 0);

        ifherk.compute_error_nonleaf<face_aux_type, u_ref_type, idx_u_type>("face_ref_1_", false, 1);
        ifherk.compute_error_nonleaf<face_aux_type, u_num_type, idx_u_type>("face_num_1_", false, 1);

        ifherk.compute_error_nonleaf<face_aux_type, N_ref_type, idx_N_type>("N_ref_0_", false, 0);
        ifherk.compute_error_nonleaf<face_aux_type, N_num_type, idx_N_type>("N_num_0_", false, 0);

        ifherk.compute_error_nonleaf<face_aux_type, N_ref_type, idx_N_type>("N_ref_1_", false, 1);
        ifherk.compute_error_nonleaf<face_aux_type, N_num_type, idx_N_type>("N_num_1_", false, 1);

        ifherk.clean<cell_aux_type>();
        ifherk.compute_error_nonleaf<cell_aux_type, p_ref_type, idx_p_type>("cell_ref_", false);
        ifherk.compute_error_nonleaf<cell_aux_type, p_num_type, idx_p_type>("cell_num_", false);

        ifherk.compute_error_nonleaf<cell_aux_type, cs_ref_type, idx_cs_type>("cell_source_ref_", false);
        ifherk.compute_error_nonleaf<cell_aux_type, cs_num_type, idx_cs_type>("cell_source_num_", false);

        world.barrier();

        if (world.rank() == 0) {
            std::cout << "finished computing error nonleaves" << std::endl;
        }

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
        float_type uz_inf = this->compute_errors<uz_num_type, uz_tar_type, error_uz_type>(
			std::string("u_z_"), 0);

		float_type p_inf = this->compute_errors<p_num_type, p_tar_type, error_p_type>(
			std::string("p_0_"), 0);

        float_type w0_inf = this->compute_errors<w_num_type, w_tar_type, error_w_type>(
			std::string("w_0_"), 0);
        float_type w1_inf = this->compute_errors<w_num_type, w_tar_type, error_w_type>(
			std::string("w_1_"), 1);
        float_type w2_inf = this->compute_errors<w_num_type, w_tar_type, error_w_type>(
			std::string("w_2_"), 2);

        float_type cs_inf = this->compute_errors<cs_num_type, cs_tar_type, error_cs_type>(
			std::string("cs_0_"), 0);

        float_type N0_inf = this->compute_errors<N_num_type, N_tar_type, error_N_type>(
			std::string("N_0_"), 0);

        float_type N1_inf = this->compute_errors<N_num_type, N_tar_type, error_N_type>(
			std::string("N_1_"), 1);
        float_type Nz_inf = this->compute_errors<Nz_num_type, Nz_tar_type, error_Nz_type>(
			std::string("N_z_"), 0);

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

        world.barrier();

        if (world.rank() == 1) {
            std::cout << "finished writing" << std::endl;
        }

        world.barrier();



        //ifherk.clean<u_num_type>();
        if (clean_p_tar) ifherk.clean<p_num_type>();
        ifherk.clean<w_num_type>();
        ifherk.clean<w_tar_type>();
        if (clean_p_tar) ifherk.clean<p_tar_type>();

        ifherk.clean<error_u_type>();
        ifherk.clean<error_p_type>();
        ifherk.clean<error_w_type>();

        //std::fill(forcing_num.begin(), forcing_num.end(), tmp_coord);

        

        PetscMalloc(sizeof(float_type) * loc_size, &res_num);
        PetscMalloc(sizeof(float_type) * loc_size, &res_tar);

        
        PetscMalloc(sizeof(float_type) * loc_size, &errvec);

        ifherk.Grid2CSR<u_tar_type, p_tar_type, w_tar_type, N_tar_type, cs_tar_type, uz_tar_type, Nz_tar_type>(res_tar, forcing_tar, fz_tar);
        ifherk.Grid2CSR<u_num_type, p_num_type, w_num_type, N_num_type, cs_num_type, uz_num_type, Nz_num_type>(res_num, forcing_num, fz_num);

        float_type diff_sum = 0.0;
        if (world.rank() != 0)
        {
            for (int i = 0; i < loc_size; i++)
            {
                float_type tmp_err = res_tar[i] - res_num[i];
                diff_sum += tmp_err*tmp_err;
                errvec[i] = tmp_err;
            }
        }

        for (int i=0; i<domain_->ib().size(); ++i)
        {
            
            errVec[i]=0;
            
        }

        ifherk.CSR2Grid<error_u_type, error_p_type, error_w_type, error_N_type, error_cs_type, error_uz_type, error_Nz_type>(errvec, errVec, error_fz);

        //ifherk.compute_error_nonleaf<edge_aux_type, error_w_type>("error_copy_", true);

        //simulation_.write("final_copy.hdf5");



        float_type diff_sum_all = 0.0;

        boost::mpi::all_reduce(world, diff_sum, diff_sum_all, std::plus<float_type>());

        world.barrier();

        if (world.rank() == 1) {
            std::cout << "Diff error is " << diff_sum_all << std::endl;
        }
        }

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

        Vec            x, b, b1, u;          /* approx solution, RHS, exact solution */
        //Vec*           Cv;               /*deflation of 
        Mat            A, B;             /* linear system matrix */

        Mat            K, K1;
        EPS            eps;              /* eigenproblem solver context */
        EPSType        type;
        PetscBool      flg,terse;
        PetscReal      tol;
        PetscInt       nev, maxit, its;
        char           filename[PETSC_MAX_PATH_LEN];
        PetscViewer    viewer;
        KSP            ksp;              /* linear solver context */
        PC             pc;               /* preconditioner context */
        ST             st;
        PetscReal      norm;  /* norm of solution error */
        PetscInt       i,n = ndim,col[3],rstart,rend,nlocal;

        
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
        //PetscCall(VecDuplicate(x, &b1));
        PetscCall(VecDuplicate(x, &u));

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

        int counter_zero_diag = 0;
        std::vector<int> zero_diag_idx;

        for (i = rstart; i < rend; i++)
        {
            //int range = rend - rstart;
            //int range_nonzero = range/2;
            int i_loc = i - rstart;


            /*if (world.rank() == 1) {
                std::cout << "Setting value for " << i << "th row" << std::endl;
            }*/
            /*float_type loc_v = std::sqrt(static_cast<float_type>(i) + 0.5) - 90;

            PetscScalar valComplex = loc_v + 0.0*PETSC_i;

            PetscCall(MatSetValues(A, 1, &i, 1, &i, &valComplex, INSERT_VALUES));*/

            //float_type v = 1;

            //if (i_loc < range_nonzero) PetscCall(MatSetValues(B, 1, &i, 1, &i, &v, INSERT_VALUES));
            
            //int i_s = i -1;
            //if (i!= 0) PetscCall(MatSetValues(A, 1, &i, 1, &i_s, &loc_v, INSERT_VALUES));
            std::map<int, float_type> row = ifherk.Jac.mat[i_loc+1];
            std::map<int, float_type> row_c = ifherk.Imag.mat[i_loc+1];
            std::vector<PetscScalar> row_val;
            std::vector<PetscInt> row_idx;
            row_val.resize(row.size() + row_c.size());
            row_idx.resize(row.size() + row_c.size());
            int counter = 0;

            /*if (world.rank() == 2) {
                std::cout << "finishde feeding matrix entries at " << i << std::endl;
            }*/
            
            

            
            /*if (row_c.size() > 1) {
                std::cout << "size of row_c bigger than 1 at " << i << std::endl;
            }*/
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

            for (const auto& [key, val] : row) {
                row_val[counter] = val;
                row_idx[counter] = key - 1;
                counter++;
            }

            /*if (world.rank() == 2) {
                std::cout << "finishde put into vectors at " << i << std::endl;
            }*/

            PetscCall(MatSetValues(A, 1, &i, counter, row_idx.data(), row_val.data(), INSERT_VALUES));

            /*if (world.rank() == 2) {
                std::cout << "finishde setting " << i << std::endl;
            }*/

            /*std::map<int, float_type> row_c = ifherk.Imag.mat[i_loc+1];
            for (const auto& [key, val] : row_c)
            {
                PetscInt loc_col = key-1;
                PetscScalar valComplex = val*PETSC_i;
                PetscCall(MatSetValue(A, i, loc_col, valComplex, INSERT_VALUES));
                
            }*/

            PetscInt diagonal_mod = 1; //if there is no entry in the diagonal, set the diagonal entry to be zero

            for (const auto& [key, val] : row) {
                PetscInt key1 = key;
                if (key1 == (i + 1)) {
                    diagonal_mod = 0;
                    break;
                }
            }

            

            /*if (diagonal_mod == 1) {
                counter_zero_diag++;
                zero_diag_idx.emplace_back(i);
                float_type val = 0.0;
                PetscScalar valComplex = val;
                PetscCall(MatSetValue(A, i, i, valComplex, INSERT_VALUES));
            }*/


            
            

            std::map<int, float_type> row1 = ifherk.B.mat[i_loc+1];
            if (row1.size() == 0) {
                float_type val = 0.0;
                PetscScalar valComplex = val;
                //PetscCall(MatSetValues(B, 1, &i, 1, &i, &val, INSERT_VALUES));
                PetscCall(MatSetValue(B, i, i, valComplex, INSERT_VALUES));
                //PetscCall(MatSetValue(A, i, i, valA, INSERT_VALUES));
            }
            for (const auto& [key, val] : row1)
            {
                PetscInt loc_col = key-1;
                if (!std::isfinite(val)) {
                    std::cout << "mat B rank " << world.rank() << " row " << i << " loc_col " << loc_col << std::endl;
                }
                PetscScalar valComplex = val;
                PetscCall(MatSetValue(B, i, loc_col, valComplex, INSERT_VALUES));
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

        MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);


        if (world.rank() == 1) {
            std::cout << "finished Assemblying" << std::endl;
        }

        PetscCall(EPSCreate(PETSC_COMM_WORLD, &eps));

        /*
     Set operators. In this case, it is a Generalized eigenvalue problem
  */
        PetscCall(EPSSetOperators(eps, A, B));
        EPSSetProblemType(eps,EPS_PGNHEP);
        EPSSetDimensions(eps,5,PETSC_DEFAULT,PETSC_DEFAULT);

        
        
        EPSSetWhichEigenpairs(eps,	EPS_TARGET_MAGNITUDE);
        PetscScalar valComplex = target_real + target_imag*PETSC_i;
        PetscCall(EPSSetTarget(eps,valComplex));
        //EPSSetWhichEigenpairs(eps,EPS_SMALLEST_MAGNITUDE);

        PetscCall(EPSGetST(eps, &st));
        PetscCall(STSetType(st, STSINVERT));

        PetscCall(STGetKSP(st, &ksp));
        PetscCall(KSPSetType(ksp, KSPPREONLY));
        PetscCall(KSPGetPC(ksp, &pc));
        PetscCall(PCSetType(pc, PCLU));

#if defined(PETSC_HAVE_MUMPS)
       if (world.rank() == 1) {
           std::cout << "using MUMPS" << std::endl;
       }
       PCFactorSetMatSolverType(pc,MATSOLVERMUMPS);
       /* the next line is required to force the creation of the ST operator and its passing to KSP */
       STGetOperator(st,NULL);
       PCFactorSetUpMatSolverType(pc);
       PCFactorGetMatrix(pc,&K);
       MatMumpsSetIcntl(K,14,200);
       MatMumpsSetCntl(K,3,1e-12);
#endif
#if defined(PETSC_HAVE_MKL_PARDISO)
       PCFactorSetMatSolverType(pc,MATSOLVERMKL_CPARDISO);
       if (world.rank() == 1) {
           std::cout << "using Pardiso" << std::endl;
       }
       /* the next line is required to force the creation of the ST operator and its passing to KSP */
       STGetOperator(st,NULL);
       PCFactorSetUpMatSolverType(pc);
       PCFactorGetMatrix(pc,&K);
       MatMkl_CPardisoSetCntl(K,iparm[0],1);
       MatMkl_CPardisoSetCntl(K,iparm[1],2);
       MatMkl_CPardisoSetCntl(K,iparm[5],6);
       MatMkl_CPardisoSetCntl(K,iparm[7],8);
       MatMkl_CPardisoSetCntl(K,iparm[9],10);
       MatMkl_CPardisoSetCntl(K,iparm[10],11);
       MatMkl_CPardisoSetCntl(K,iparm[12],13);
       MatMkl_CPardisoSetCntl(K,iparm[17],18);
       MatMkl_CPardisoSetCntl(K,iparm[18],19);
       MatMkl_CPardisoSetCntl(K,iparm[26],27);
       MatMkl_CPardisoSetCntl(K,1,        68); //display all info
       //MatMkl_CPardisoSetCntl(K,iparm[39],40);
       //MatMumpsSetIcntl(K,14,50);
       //MatMumpsSetCntl(K,3,1e-12);
#endif

        if (world.rank() == 2) {
            std::cout << "set up of pardiso finished" << std::endl;
        }

        /*
     Set solver parameters at runtime
  */
        PetscCall(EPSSetFromOptions(eps));

        if (world.rank() == 2) {
            std::cout << "set up of eps finished" << std::endl;
        }

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Solve the eigensystem
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        if (set_deflation) PetscCall(EPSSetDeflationSpace(eps, 1, &b));

        PetscCall(EPSSolve(eps));

        if (world.rank() == 2) {
            std::cout << "EPS solved" << std::endl;
        }

        PetscCall(EPSGetIterationNumber(eps, &its));

        if (world.rank() == 2) {
            std::cout << "Iteration number got" << std::endl;
        } 
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

        std::vector<float_type> re_v, im_v, err_vec;

        PetscInt nconv;
        PetscCall(EPSGetConverged(eps,&nconv));

        for (PetscInt ll = 0; ll < nconv;  ll++) {
            PetscScalar EV;
            PetscScalar EVi;
            PetscReal err_v;
            PetscCall(EPSGetEigenvalue(eps, ll, &EV, &EVi));
            PetscCall(EPSComputeError(eps,ll,EPS_ERROR_RELATIVE,&err_v));
            float_type evr = PetscRealPart(EV);
            float_type evi = PetscImaginaryPart(EV);
            float_type err_vv = err_v;

            re_v.emplace_back(evr);
            im_v.emplace_back(evi);
            err_vec.emplace_back(err_vv);    
        }

        if (world.rank() == 1) {
            
            int ss = mode_c*100 + 0.1;
            std::string mode_c_str = std::to_string(ss);
            if (mode_c < 1) mode_c_str = "0"+mode_c_str;
            std::string EVs_name = "EVs_"+mode_c_str+".txt";
            std::ofstream outfile;
            int width = 20;
            outfile.open(EVs_name, std::ios_base::app);
            for (int ll = 0; ll < nconv;  ll++) {
                float_type evr = re_v[ll];
                float_type evi = im_v[ll];
                float_type err_vv = err_vec[ll];
                std::string sign = "+";
                if (evi < 0) sign = "-";
                outfile << std::setprecision(9) << std::setw(width) << std::fixed << evr << sign << std::fabs(evi) << "i";
                outfile << std::setprecision(9) << std::setw(width) << std::scientific << err_vv;
                outfile << std::endl;     
            }
            outfile.close();
        }

        EPSGetEigenvector(eps,0, x, NULL);

        for (i = rstart; i < rend;i++) {
            int i_loc = i - rstart;
            PetscScalar tmp;
            PetscCall(VecGetValues(x, 1, &i, &tmp));
            x0_real[i_loc] = PetscRealPart(tmp);
            x0_imag[i_loc] = PetscImaginaryPart(tmp);
        }

        EPSGetEigenvector(eps,1, x, NULL);

        for (i = rstart; i < rend;i++) {
            int i_loc = i - rstart;
            PetscScalar tmp;
            PetscCall(VecGetValues(x, 1, &i, &tmp));
            x1_real[i_loc] = PetscRealPart(tmp);
            x1_imag[i_loc] = PetscImaginaryPart(tmp);
        }

        EPSGetEigenvector(eps,2, x, NULL);

        for (i = rstart; i < rend;i++) {
            int i_loc = i - rstart;
            PetscScalar tmp;
            PetscCall(VecGetValues(x, 1, &i, &tmp));
            x2_real[i_loc] = PetscRealPart(tmp);
            x2_imag[i_loc] = PetscImaginaryPart(tmp);
        }

        EPSGetEigenvector(eps,3, x, NULL);

        for (i = rstart; i < rend;i++) {
            int i_loc = i - rstart;
            PetscScalar tmp;
            PetscCall(VecGetValues(x, 1, &i, &tmp));
            x3_real[i_loc] = PetscRealPart(tmp);
            x3_imag[i_loc] = PetscImaginaryPart(tmp);
        }

        EPSGetEigenvector(eps,4, x, NULL);

        for (i = rstart; i < rend;i++) {
            int i_loc = i - rstart;
            PetscScalar tmp;
            PetscCall(VecGetValues(x, 1, &i, &tmp));
            x4_real[i_loc] = PetscRealPart(tmp);
            x4_imag[i_loc] = PetscImaginaryPart(tmp);
        }

        MPI_Barrier(PETSC_COMM_WORLD);
        PetscCall(EPSDestroy(&eps));
        PetscCall(MatDestroy(&A));
        PetscCall(MatDestroy(&B));
        

        

        PetscCall(VecDestroy(&x));
        PetscCall(VecDestroy(&u));
        PetscCall(VecDestroy(&b));
        
        //PetscCall(KSPDestroy(&ksp));

        PetscCall(SlepcFinalize());
        }


        ifherk.CSR2Grid<u_num_inv_type, p_num_inv_type, w_num_inv_type, N_num_inv_type, cs_num_type, uz_num_inv_type, Nz_num_type>(x0_real, forcing_ref, fz);
        //if (world.rank() != 0) ifherk.pad_velocity<u_num_inv_type, u_num_inv_type>(true);
        if (world.rank() != 0) ifherk.Curl_access<u_num_inv_type, uz_num_inv_type, w_ref_type>();
        simulation_.write("x0_real.hdf5");
        ifherk.CSR2Grid<u_num_inv_type, p_num_inv_type, w_num_inv_type, N_num_inv_type, cs_num_type, uz_num_inv_type, Nz_num_type>(x0_imag, forcing_ref, fz);
        //if (world.rank() != 0) ifherk.pad_velocity<u_num_inv_type, u_num_inv_type>(true);
        if (world.rank() != 0) ifherk.Curl_access<u_num_inv_type, uz_num_inv_type, w_ref_type>();
        simulation_.write("x0_imag.hdf5");
        ifherk.CSR2Grid<u_num_inv_type, p_num_inv_type, w_num_inv_type, N_num_inv_type, cs_num_type, uz_num_inv_type, Nz_num_type>(x1_real, forcing_ref, fz);
        //if (world.rank() != 0) ifherk.pad_velocity<u_num_inv_type, u_num_inv_type>(true);
        if (world.rank() != 0) ifherk.Curl_access<u_num_inv_type, uz_num_inv_type, w_ref_type>();
        simulation_.write("x1_real.hdf5");
        ifherk.CSR2Grid<u_num_inv_type, p_num_inv_type, w_num_inv_type, N_num_inv_type, cs_num_type, uz_num_inv_type, Nz_num_type>(x1_imag, forcing_ref, fz);
        //if (world.rank() != 0) ifherk.pad_velocity<u_num_inv_type, u_num_inv_type>(true);
        if (world.rank() != 0) ifherk.Curl_access<u_num_inv_type, uz_num_inv_type, w_ref_type>();
        simulation_.write("x1_imag.hdf5");
        ifherk.CSR2Grid<u_num_inv_type, p_num_inv_type, w_num_inv_type, N_num_inv_type, cs_num_type, uz_num_inv_type, Nz_num_type>(x2_real, forcing_ref, fz);
        //if (world.rank() != 0) ifherk.pad_velocity<u_num_inv_type, u_num_inv_type>(true);
        if (world.rank() != 0) ifherk.Curl_access<u_num_inv_type, uz_num_inv_type, w_ref_type>();
        simulation_.write("x2_real.hdf5");
        ifherk.CSR2Grid<u_num_inv_type, p_num_inv_type, w_num_inv_type, N_num_inv_type, cs_num_type, uz_num_inv_type, Nz_num_type>(x2_imag, forcing_ref, fz);
        //if (world.rank() != 0) ifherk.pad_velocity<u_num_inv_type, u_num_inv_type>(true);
        if (world.rank() != 0) ifherk.Curl_access<u_num_inv_type, uz_num_inv_type, w_ref_type>();
        simulation_.write("x2_imag.hdf5");
        ifherk.CSR2Grid<u_num_inv_type, p_num_inv_type, w_num_inv_type, N_num_inv_type, cs_num_type, uz_num_inv_type, Nz_num_type>(x3_real, forcing_ref, fz);
        //if (world.rank() != 0) ifherk.pad_velocity<u_num_inv_type, u_num_inv_type>(true);
        if (world.rank() != 0) ifherk.Curl_access<u_num_inv_type, uz_num_inv_type, w_ref_type>();
        simulation_.write("x3_real.hdf5");
        ifherk.CSR2Grid<u_num_inv_type, p_num_inv_type, w_num_inv_type, N_num_inv_type, cs_num_type, uz_num_inv_type, Nz_num_type>(x3_imag, forcing_ref, fz);
        //if (world.rank() != 0) ifherk.pad_velocity<u_num_inv_type, u_num_inv_type>(true);
        if (world.rank() != 0) ifherk.Curl_access<u_num_inv_type, uz_num_inv_type, w_ref_type>();
        simulation_.write("x3_imag.hdf5");

        ifherk.CSR2Grid<u_num_inv_type, p_num_inv_type, w_num_inv_type, N_num_inv_type, cs_num_type, uz_num_inv_type, Nz_num_type>(x4_real, forcing_ref, fz);
        if (world.rank() != 0) ifherk.Curl_access<u_num_inv_type, uz_num_inv_type, w_ref_type>();
        simulation_.write("x4_real.hdf5");
        ifherk.CSR2Grid<u_num_inv_type, p_num_inv_type, w_num_inv_type, N_num_inv_type, cs_num_type, uz_num_inv_type, Nz_num_type>(x4_imag, forcing_ref, fz);
        //if (world.rank() != 0) ifherk.pad_velocity<u_num_inv_type, u_num_inv_type>(true);
        if (world.rank() != 0) ifherk.Curl_access<u_num_inv_type, uz_num_inv_type, w_ref_type>();
        simulation_.write("x4_imag.hdf5");

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
    int vortexType = 0;

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

    float_type mode_c;

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
