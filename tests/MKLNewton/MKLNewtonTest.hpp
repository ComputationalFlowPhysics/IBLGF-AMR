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


#include <iblgf/solver/DirectSolver/MKLPardiso_solve.hpp>
#include <iblgf/solver/Stability_solver/Stability_solver.hpp>
//need c2f
//#include "mpi.h"
//#include "/home/root/intel-oneAPI/oneAPI/mkl/latest/include/mkl.h"
//#include "/home/root/intel-oneAPI/oneAPI/mkl/latest/include/mkl_cluster_sparse_solver.h"

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
         //(error_u          , float_type, 2,    1,       1,     face,true  ),
		 //(error_p          , float_type, 1,    1,       1,     cell,true  ),
         //(error_w          , float_type, 1,    1,       1,     edge,true  ),
         (test             , float_type, 1,    1,       1,     cell,false ),
        //IF-HERK
		 (u                , float_type, 2,    1,       1,     face,true  ),
		 (p                , float_type, 1,    1,       1,     cell,true  ),
         //(u_num_inv        , float_type, 2,    1,       1,     face,true  ),
		 //(p_num_inv        , float_type, 1,    1,       1,     cell,true  ),
         //(w_num_inv        , float_type, 1,    1,       1,     edge,true  ),
		 //(u_ref            , float_type, 2,    1,       1,     face,true  ),
		 //(p_ref            , float_type, 1,    1,       1,     cell,true  ),
         //(w_ref            , float_type, 1,    1,       1,     edge,true  ),
         //(u_tar            , float_type, 2,    1,       1,     face,true  ),
		 //(p_tar            , float_type, 1,    1,       1,     cell,true  ),
         //(w_tar            , float_type, 1,    1,       1,     edge,true  ),
		 (u_num            , float_type, 2,    1,       1,     face,true  ),
		 (p_num            , float_type, 1,    1,       1,     cell,true  ),
         (w_num            , float_type, 1,    1,       1,     edge,true  )
    ))
    // clang-format on
};

struct NS_AMR_LGF : public SetupNewton<NS_AMR_LGF, parameters>
{
    using super_type =SetupNewton<NS_AMR_LGF,parameters>;
    using vr_fct_t = std::function<float_type(float_type x, float_type y, int field_idx, bool perturbation)>;
    using solver_t = iblgf::solver::IntelPardisoSolve<float_type>;
    using stability_t = typename super_type::stability_t;

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
        , stab_solve(&this->simulation_) 
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

        Omega = simulation_.dictionary()->template get_or<float_type>("Omega", 0.0);

		smooth_start_ = simulation_.dictionary()->template get_or<bool>("smooth_start", false);

		vortexType = simulation_.dictionary()->template get_or<int>("Vort_type", 0);

		pert_mag = simulation_.dictionary()->template get_or<float_type>("pert_mag", 0.1);

		simulation_.bc_vel() =
			[this](std::size_t idx, float_type t, auto coord = {0, 0})
			{
				float_type T0 = 0.5;

				float_type r = std::sqrt((coord[0] * coord[0] + coord[1] * coord[1]));
                float_type f_alpha = 0.0;
				if (r < 0.25 || r > 0.75) {
                    f_alpha = 0.0;
                    /*if (r < 1e-12) {
                        f_alpha = 0.0;
                    }
                    else {
                        if (idx == 0) {
                            f_alpha =  coord[1]/r/0.1*Omega;
                        }
                        else if (idx == 1) {
                            f_alpha = -coord[0]/r/0.1*Omega;
                        }
                        else {
                            f_alpha = 0.0;
                        }
                    }*/
                }
                else {
                    if (idx == 0) { f_alpha = coord[1] / r / r * Omega; }
                    else if (idx == 1)
                    {
                        f_alpha = -coord[0] / r / r * Omega;
                    }
                    else { f_alpha = 0.0; }
                }

				if (t<=0.0 && smooth_start_)
					return 0.0;
				else if (t<T0-1e-10 && smooth_start_)
				{
					float_type h1 = exp(-1/(t/T0));
					float_type h2 = exp(-1/(1 - t/T0));

					return -(U_[idx] + f_alpha) * (h1/(h1+h2));
				}
				else
				{
					return -(U_[idx] + f_alpha);
				}
			};


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

					return -(U_[idx]) * (h1/(h1+h2));
				}
				else
				{
					return -(U_[idx]);
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
        num_input = simulation_.dictionary()->template get_or<bool>("num_input", false);

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

        domain_->register_adapt_condition()=
			[this]( std::vector<float_type> source_max, auto& octs, std::vector<int>& level_change ) 
			{return this->template adapt_level_change(source_max, octs, level_change);};

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

            //this->randomize();

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

        //if (world.rank() != 0) ifherk.clean_up_initial_velocity<u_type>();

        ifherk.Assigning_idx();

        ifherk.ComputeForcing<u_type, p_type, u_num_type, p_num_type>(forcing_num);


        ifherk.NewtonRHS<u_type, p_type, u_num_type, p_num_type>(forcing_num, forcing_tar);
        


        simulation_.write("NewtonRHS.hdf5");

        //stab_solve.template Init_Construct_Newton_Matrix<u_type>();
        stab_solve.template NewtonIteration<u_type, p_type, w_num_type>(forcing_num);
        if (world.rank() == 0)
			std::cout << "WriteFinal\n" << std::endl;
        simulation_.write("final.hdf5");
        if (world.rank() == 0)
			std::cout << "Done \n" << std::endl;

        return 0;

	}

    /** @brief  Refienment conditon for octants.  */
    template<class OctantType>
    bool refinement(
        OctantType* it, int diff_level, bool use_all = false) const noexcept
    {
        return false;
    }




    template< class key_t >
	void adapt_level_change(std::vector<float_type> source_max,
			        std::vector<key_t>& octs,
				        std::vector<int>&   level_change )
	{
		octs.clear();
		level_change.clear();
		for (auto it = domain_->begin(); it != domain_->end(); ++it)
		{

			if (!it->locally_owned()) continue;
			if (!it->is_leaf() && !it->is_correction()) continue;
			if (it->is_leaf() && it->is_correction()) continue;




			int l1=-1;
			int l2=-1;
			int l3=-1;

			if (!it->is_correction() && it->is_leaf())
				l1=this->template adapt_levle_change_for_field<edge_aux_type>(it, source_max[1], true);

			if (it->is_correction() && !it->is_leaf())
				l2=this->template adapt_levle_change_for_field<edge_aux_type>(it, source_max[1], false);

			if (!it->is_correction() && it->is_leaf())
				l3=this->template adapt_levle_change_for_field<edge_aux_type>(it, source_max[1], false);

			int l=std::max(std::max(l1,l2),l3);
			//int l = std::max(l1,l2);


			auto dx_level = domain_->dx_base() / std::pow(2, it->refinement_level());

			bool outOfWake = false;


			if( l!=0)
			{
				if (it->is_leaf()&&!it->is_correction())
				{
					octs.emplace_back(it->key());
					level_change.emplace_back(l);
				}
				else if (it->is_correction() && !it->is_leaf() && l2>0)
				{
					octs.emplace_back(it->key().parent());
					level_change.emplace_back(l2);
				}
			}
		}
	}

    template<class cell_aux, class u, class OctantType >
    int adapt_level_change(OctantType* it, std::vector<float_type> source_max)
    {
        // no source in correction part by default
        if (it->is_correction())
            return -1;

        int l1=this->template adapt_levle_change_for_field<cell_aux>(it, source_max[1], true);
        int l2=-1;

        return std::max(l1,l2);
    }


	template<class Field, class OctantType>
	int adapt_levle_change_for_field(OctantType it, float_type source_max, bool use_base_level_threshold)
	{
		if (vortexType != 0) return 0;
		if (it->is_ib() && it->is_leaf())
			if (it->refinement_level()<(nLevelRefinement_))
				return 1;
			else
				return 0;

		source_max *=1.05;

		float_type field_max=
			domain::Operator::maxnorm<Field>(it->data());

		if (field_max<1e-10) return -1;

		float_type deletion_factor=0.7;

		int l_aim = static_cast<int>( ceil(nLevelRefinement_-log(field_max/source_max) / log(refinement_factor_)));
		int l_delete_aim = static_cast<int>( ceil(nLevelRefinement_-((log(field_max/source_max) - log(deletion_factor)) / log(refinement_factor_))));

		if (l_aim>hard_max_level_)
			l_aim=hard_max_level_;
		if (l_aim > nLevelRefinement_) l_aim = nLevelRefinement_;

		if (it->refinement_level()==0 && use_base_level_threshold)
		{
			if (field_max>source_max*base_threshold_)
				l_aim = std::max(l_aim,0);

			if (field_max>source_max*base_threshold_*deletion_factor)
				l_delete_aim = std::max(l_delete_aim,0);
		}

		if (it->is_ib())
			return 0;

		int l_change = l_aim - it->refinement_level();
		if (l_change>0)
			return 1;

		l_change = l_delete_aim - it->refinement_level();
		if (l_change<0) return -1;

		return 0;
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
                node(u,0) = 10.0;
                node(u,1) = 10.0;
                node(p,0) = 10.0;
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

    void randomize()
	{
		

		boost::mpi::communicator world;
		if (domain_->is_server()) return;
		

		//if (ic_filename_ != "null") return;

		// Voriticity IC
		for (auto it = domain_->begin(); it != domain_->end(); ++it)
		{
			if (!it->locally_owned()) continue;

			

			for (auto& node : it->data())
			{
                int rand_num = rand() % 1000;
                
                float_type u_0 = static_cast<float_type>(rand_num)/static_cast<float_type>(1000) - 0.5;

				node(u,0) /= 2.0;

                rand_num = rand() % 1000;

                float_type u_1 = static_cast<float_type>(rand_num)/static_cast<float_type>(1000) - 0.5;

				node(u,1) /= 2.0;

                rand_num = rand() % 1000;

                float_type p_0 = static_cast<float_type>(rand_num)/static_cast<float_type>(1000) - 0.5;

				node(p,0) /= 2.0;
			}

		}
    }


    private:

    boost::mpi::communicator client_comm_;
    stability_t stab_solve;

    bool single_ring_=true;
    bool perturbation_=false;
    bool hard_max_refinement_=false;
    bool smooth_start_;
    bool print_mat;
    bool num_input;
    int vortexType = 0;

    int row_to_print = 0;

    bool clean_p_tar;
    bool testing_smearing = false;

    std::vector<float_type> U_;
    float_type Omega; //rotation rate of the cylinder (if needed)
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



} // namespace iblgf

#endif
