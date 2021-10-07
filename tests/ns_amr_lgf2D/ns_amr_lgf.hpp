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

#ifndef IBLGF_INCLUDED_NS_AMR_LGF_HPP
#define IBLGF_INCLUDED_NS_AMR_LGF_HPP

#ifndef DEBUG_IFHERK
#define DEBUG_IFHERK
#endif
#define DEBUG_POISSON

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

#include "../../setups/setup_base.hpp"
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
         (error_p          , float_type, 1,    1,       1,     cell,false ),
         (test             , float_type, 1,    1,       1,     cell,false ),
        //IF-HERK
         (u                , float_type, 2,    1,       1,     face,true  ),
         (u_ref            , float_type, 2,    1,       1,     face,true  ),
         (p_ref            , float_type, 1,    1,       1,     cell,true  ),
         (p                , float_type, 1,    1,       1,     cell,true  ),
         (w_num            , float_type, 1,    1,       1,     edge,false ),
         (w_exact          , float_type, 1,    1,       1,     edge,false ),
         (error_w          , float_type, 1,    1,       1,     edge,false ),
         //for radial velocity
         (exact_u_theta    , float_type, 2,    1,       1,     edge,false ),
         (num_u_theta      , float_type, 2,    1,       1,     edge,false ),
         (error_u_theta    , float_type, 2,    1,       1,     edge,false )
    ))
    // clang-format on
};

struct NS_AMR_LGF : public SetupBase<NS_AMR_LGF, parameters>
{
    using super_type =SetupBase<NS_AMR_LGF,parameters>;
    using vr_fct_t = std::function<float_type(float_type x, float_type y, int field_idx, bool perturbation)>;

    //Timings
    using clock_type = std::chrono::high_resolution_clock;
    using milliseconds = std::chrono::milliseconds;
    using duration_type = typename clock_type::duration;
    using time_point_type = typename clock_type::time_point;

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
		if (use_fat_ring)
			vr_fct_ =
			[this](float_type x, float_type y, int field_idx, bool perturbation) {return this->vortex_ring_vor_fat_ic(x, y, field_idx, perturbation); };
		else
			vr_fct_ =
			[this](float_type x, float_type y, int field_idx, bool perturbation) {return this->vortex_ring_vor_ic(x, y, field_idx, perturbation); };

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

		//bool subtract_non_leaf_ = simulation_.dictionaty()->template get_or<bool>("subtract_non_leaf", true);

		if (dt_ < 0) dt_ = dx_ * cfl_;

		dt_ /= pow(2.0, nLevelRefinement_);
		tot_steps_ *= pow(2, nLevelRefinement_);

		pcout << "\n Setup:  Test - Simple IC \n" << std::endl;
		pcout << "Number of refinement levels: " << nLevelRefinement_
			<< std::endl;

		//domain_->decomposition().subtract_non_leaf() = true;

		domain_->register_adapt_condition()=
			[this]( std::vector<float_type> source_max, auto& octs, std::vector<int>& level_change ) 
			{return this->template adapt_level_change(source_max, octs, level_change);};

		/*domain_->register_adapt_condition() =
			[this](auto octant, std::vector<float_type> source_max) {return this->template adapt_level_change<cell_aux_type, u_type>(octant, source_max); };*/

		/*domain_->register_adapt_condition()=
			[this](auto octant, std::vector<float_type> source_max){return this->template adapt_level_change<edge_aux_type, edge_aux_type>(octant, source_max);};*/


		if (vortexType != 0) {
		domain_->register_refinement_condition() = [this](auto octant,
			int diff_level) {
				return this->refinement(octant, diff_level);
		};
		}

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
		if (!use_restart()) { this->initialize(); }
		else
		{
			simulation_.template read_h5<u_type>(simulation_.restart_field_dir(), "u");
		}

		boost::mpi::communicator world;
		if (world.rank() == 0)
			std::cout << "on Simulation: \n" << simulation_ << std::endl;
	}

	float_type run()
	{
		boost::mpi::communicator world;

		time_integration_t ifherk(&this->simulation_);

		if (ic_filename_ != "null")
			simulation_.template read_h5<u_type>(ic_filename_, "u");

		mDuration_type ifherk_duration(0);
		TIME_CODE(ifherk_duration, SINGLE_ARG(
			ifherk.time_march(use_restart());
		))
			pcout_c << "Time to solution [ms] " << ifherk_duration.count() << std::endl;

		ifherk.clean_leaf_correction_boundary<u_type>(domain_->tree()->base_level(), true, 1);

		float_type maxNumVort = -1;


		if (ref_filename_ != "null")
		{
			if (vortexType == 0) {
			simulation_.template read_h5<u_ref_type>(ref_filename_, "u");
			simulation_.template read_h5<p_ref_type>(ref_filename_, "p");
			}

			auto center = (domain_->bounding_box().max() -
				domain_->bounding_box().min() + 1) / 2.0 +
				domain_->bounding_box().min();


			for (auto it = domain_->begin_leaves();
				it != domain_->end_leaves(); ++it)
			{
				if (!it->locally_owned()) continue;

				auto dx_level = domain_->dx_base() / std::pow(2, it->refinement_level());
				auto scaling = std::pow(2, it->refinement_level());

				for (auto& node : it->data())
				{
					const auto& coord = node.level_coordinate();
					float_type x = static_cast<float_type>
						(coord[0] - center[0] * scaling) * dx_level;
					float_type y = static_cast<float_type>
						(coord[1] - center[1] * scaling) * dx_level;
					//float_type z = static_cast<float_type>
					//    (coord[2]-center[2]*scaling)*dx_level;

					float_type r2 = x * x + y * y;
					if (r2 > 4 * R_ * R_)
					{
						node(u_ref, 0) = 0.0;
						node(u_ref, 1) = 0.0;
						//node(u_ref, 2)=0.0;
					}
					float_type r__ = std::sqrt(x * x + y * y);
					float_type t_final = dt_ * tot_steps_;
					node(w_exact) = w_taylor_vort(r__, t_final);
					node(w_num) = (node(u, 1) - node.at_offset(u, -1, 0, 1) -
						node(u, 0) + node.at_offset(u, 0, -1, 0)) / dx_level;
					if (vortexType != 0) {
					x = static_cast<float_type>
						(coord[0] - center[0] * scaling) * dx_level;
					y = static_cast<float_type>
						(coord[1] - center[1] * scaling + 0.5) * dx_level;
					node(u_ref, 0) = u_vort(x, y, t_final, 0);
					x = static_cast<float_type>
						(coord[0] - center[0] * scaling + 0.5) * dx_level;
					y = static_cast<float_type>
						(coord[1] - center[1] * scaling) * dx_level;
					node(u_ref, 1) = u_vort(x, y, t_final, 1);


					//compute analytical u_theta
					x = static_cast<float_type>(
						coord[0] - center[0] * scaling) *
						dx_level;
					y = static_cast<float_type>(
						coord[1] - center[1] * scaling) *
						dx_level;
					float_type u = u_vort(x, y, t_final, 0);
					float_type v = u_vort(x, y, t_final, 1);
					float_type u_theta = std::sqrt(u * u + v * v);
					node(exact_u_theta, 0) = u_theta;
					node(exact_u_theta, 1) = 0.0;     //0 is u_theta, 1 is u_r
					}

					
				}

			}
		}
		getUtheta<u_type, num_u_theta_type>();
		float_type t_final = dt_ * tot_steps_;
		pcout << "the final time is " << t_final << std::endl;
		pcout << "the max numerical vorticity is " << maxNumVort << std::endl;
		ifherk.clean_leaf_correction_boundary<u_type>(domain_->tree()->base_level(), true, 1);

		float_type u1_inf = this->compute_errors<u_type, u_ref_type, error_u_type>(
			std::string("u1_"), 0);
		float_type u2_inf = this->compute_errors<u_type, u_ref_type, error_u_type>(
			std::string("u2_"), 1);
		float_type u_t_inf = this->compute_errors<num_u_theta_type, exact_u_theta_type, error_u_theta_type>(
			std::string("u_t_"), 0);
		float_type u_r_inf = this->compute_errors<num_u_theta_type, exact_u_theta_type, error_u_theta_type>(
			std::string("u_r_"), 1);

		float_type w_inf = this->compute_errors<edge_aux_type, w_exact_type, error_w_type>(std::string("w_"), 0);
		//float_type u3_linf=this->compute_errors<u_type, u_ref_type, error_u_type>(
		//        std::string("u3_"), 2);

		simulation_.write("final.hdf5");
		return u1_inf;

	}


	template<class Source, class Target>
	void getUtheta() noexcept
	{
		//up_and_down<Source>();
		boost::mpi::communicator world;
		if (domain_->is_server()) return;
		auto center = (domain_->bounding_box().max() -
			domain_->bounding_box().min() + 1) / 2.0 +
			domain_->bounding_box().min();

		for (int l = domain_->tree()->base_level();
			l < domain_->tree()->depth(); ++l)
		{
			auto client = domain_->decomposition().client();
			client->template buffer_exchange<Source>(l);
			const auto dx_base = domain_->dx_base();
			for (auto it = domain_->begin(l); it != domain_->end(l); ++it)
			{
				auto scaling = std::pow(2, it->refinement_level());
				if (!it->locally_owned() || !it->has_data()) continue;
				const auto dx_level =
					dx_base / math::pow2(it->refinement_level());
				for (auto& n : it->data()) {
					const auto& coord = n.level_coordinate();
					float_type x = static_cast<float_type>
						(coord[0] - center[0] * scaling) * dx_level;
					float_type y = static_cast<float_type>
						(coord[1] - center[1] * scaling) * dx_level;

					float_type avg_u = n(Source::tag(), 0) / 2.0 + n.at_offset(Source::tag(), 0, -1, 0) / 2.0;
					float_type avg_v = n(Source::tag(), 1) / 2.0 + n.at_offset(Source::tag(), -1, 0, 1) / 2.0;

					float_type u_theta = (x * avg_v - y * avg_u) / std::sqrt(x * x + y * y);
					float_type u_r = (x * avg_u + y * avg_v) / std::sqrt(x * x + y * y);

					n(Target::tag(), 0) = u_theta;
					n(Target::tag(), 1) = u_r;

				}
				/*for (std::size_t field_idx = 0; field_idx < Target::nFields();
					 ++field_idx)
				{
					auto& lin_data =
						it->data_r(Target::tag(), field_idx).linalg_data();

					lin_data *= _scale;
				}*/
			}
			client->template buffer_exchange<Target>(l);
		}
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
				l1=this->template adapt_levle_change_for_field<cell_aux_type>(it, source_max[0], true);

			if (it->is_correction() && !it->is_leaf())
				l2=this->template adapt_levle_change_for_field<correction_tmp_type>(it, source_max[0], false);

			if (!it->is_correction() && it->is_leaf())
				l3=this->template adapt_levle_change_for_field<edge_aux_type>(it, source_max[1], false);

			int l=std::max(std::max(l1,l2),l3);

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

        int l1=this->template adapt_levle_change_for_field<cell_aux>(it, source_max[0], true);
	//int l1 = -1;
        //int l2=this->template adapt_levle_change_for_field<u>(it, 1.0, true);
        int l2=-1;

        return std::max(l1,l2);
    }


	template<class Field, class OctantType>
	int adapt_levle_change_for_field(OctantType it, float_type source_max, bool use_base_level_threshold)
	{
		if (vortexType != 0) return 0;
		if (it->is_ib() && it->is_leaf())
			if (it->refinement_level()<nLevelRefinement_)
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

	template<class Field, class OctantType>
	int adapt_levle_change_for_field_old(OctantType it, float_type source_max, bool use_base_level_threshold)
	{
		return 0; //for constant mesh testing
		source_max *= 1.1; // to avoid constant change
		// ----------------------------------------------------------------

		float_type field_max =
			domain::Operator::maxabs<Field>(it->data());

		// to refine and harder to delete
		// This prevent rapid change of level refinement
		float_type deletion_factor = refinement_factor_ * 0.75;

		int l_aim = static_cast<int>(
			ceil(nLevelRefinement_ -
				log(field_max / source_max) / log(refinement_factor_)));
		int l_delete_aim = static_cast<int>(
			ceil(nLevelRefinement_ -
				log(field_max / source_max) / log(deletion_factor)));

		if (hard_max_refinement_ && (l_aim > nLevelRefinement_))
			l_aim = nLevelRefinement_;

		//change the adaptive condition to refinement condition
		int diff_level = it->refinement_level();
		bool ref_cond = refinement<OctantType>(it, diff_level);
		//if (ref_cond) return 1;
		//else return 0;


		if (it->refinement_level() == 0 && use_base_level_threshold)
		{
			if (field_max > source_max * base_threshold_)
				l_aim = std::max(l_aim, 0);

			if (field_max > source_max * base_threshold_ * deletion_factor)
				l_delete_aim = std::max(l_delete_aim, 0);
		}

		int l_change = l_aim - it->refinement_level();

		if ((l_delete_aim < 0) || (!use_base_level_threshold && l_aim == 0)) return -1;
		if (l_change > 0) return 1;
		return 0;
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

		if (ic_filename_ != "null") return;

		// Voriticity IC
		for (auto it = domain_->begin(); it != domain_->end(); ++it)
		{
			if (!it->locally_owned()) continue;

			auto dx_level = dx_base / std::pow(2, it->refinement_level());
			auto scaling = std::pow(2, it->refinement_level());

			for (auto& node : it->data())
			{

				const auto& coord = node.level_coordinate();

				/*float_type x = static_cast<float_type>
				(coord[0]-center[0]*scaling+0.5)*dx_level;
				float_type y = static_cast<float_type>
				(coord[1]-center[1]*scaling)*dx_level;
				//float_type z = static_cast<float_type>
				//(coord[2]-center[2]*scaling)*dx_level;

				node(edge_aux,0) = vor(x,y,0);*/

				/***********************************************************/
				/*x = static_cast<float_type>
				(coord[0]-center[0]*scaling)*dx_level;
				y = static_cast<float_type>
				(coord[1]-center[1]*scaling+0.5)*dx_level;
				//z = static_cast<float_type>
				//(coord[2]-center[2]*scaling)*dx_level;

				node(edge_aux,1) = vor(x,y,1);*/

				/***********************************************************/
				float_type x = static_cast<float_type>
					(coord[0] - center[0] * scaling) * dx_level;
				float_type y = static_cast<float_type>
					(coord[1] - center[1] * scaling) * dx_level;
				//z = static_cast<float_type>
				//(coord[2]-center[2]*scaling+0.5)*dx_level;

				//node(edge_aux,0) = vor(x,y-0.5*vort_sep,0)+ vor(x,y+0.5*vort_sep,0);
				node(edge_aux, 0) = vor(x, y, 0);
				
				x = static_cast<float_type>(coord[0]-center[0]*scaling)*dx_level;
				y = static_cast<float_type>(coord[1]-center[1]*scaling+0.5)*dx_level;
				node(u, 0) = u_vort(x,y,0,0);
				x = static_cast<float_type>(coord[0]-center[0]*scaling+0.5)*dx_level;
				y = static_cast<float_type>(coord[1]-center[1]*scaling)*dx_level;
				node(u, 1) = u_vort(x,y,0,1);
			}

		}

		//psolver.template apply_lgf<edge_aux_type, stream_f_type>();
		auto client = domain_->decomposition().client();

		for (int l = domain_->tree()->base_level();
			l < domain_->tree()->depth(); ++l)
		{
			//client->template buffer_exchange<stream_f_type>(l);

			/*for (auto it = domain_->begin(l); it != domain_->end(l); ++it)
			{
				if (!it->locally_owned() || !it->has_data()) continue;
				const auto dx_level =
					dx_base / std::pow(2, it->refinement_level());
				domain::Operator::curl_transpose<stream_f_type, u_type>(
					it->data(), dx_level, -1.0);
			}*/
			client->template buffer_exchange<u_type>(l);
		}

	}


	float_type vortex_ring_vor_fat_ic(float_type x, float_type y, int field_idx, bool perturbation)
	{
		//return 1;
		//const float_type alpha = 0.54857674;
		//float_type       gam = 1;
		float_type       R2 = R_ * R_;
		//float_type       nu = Lx / Re_; //assuming U = 1
		//float_type       t0 = 10.0*Re_;    //t0d = 1.0

		float_type r2 = (x * x + y * y);
		float_type r = sqrt(r2);


		float_type rd = (static_cast <float_type> (rand()) / static_cast <float_type> (RAND_MAX)) - 0.5;
		float_type prtub = 0.000;
		rd *= prtub * perturbation;

		float_type vort_ic = w_oseen_vort(r, 0);
		return vort_ic;
		//return 0;
	}

    float_type vor(float_type x, float_type y, int field_idx) const
    {
		float_type x_loc = x - ctr_dis_x;
		float_type y_loc = y - ctr_dis_y;
        return vr_fct_(x_loc,y_loc,field_idx,perturbation_)/* - vr_fct_(x_loc,y_loc-vort_sep,field_idx,perturbation_))*/; 
        //else
        //return -vr_fct_(x,y,z-d2v_/2,field_idx,perturbation_)+vr_fct_(x,y,z+d2v_/2,field_idx,perturbation_);
    }


	float_type vortex_ring_vor_ic(float_type x, float_type y, int field_idx, bool perturbation)
	{
		//return 1;
		//float_type gam = 1.0;


		float_type       R2 = R_ * R_;

		//float_type       nu = Lx / Re_; //assuming U = 1
		//float_type       t0 = 10.0*Re_;    //t0d = 1.0

		float_type r2 = (x * x + y * y);
		float_type r = sqrt(r2);
		float_type rd = (static_cast <float_type> (rand()) / static_cast <float_type> (RAND_MAX)) - 0.5;
		float_type prtub = 0.000;
		rd *= prtub * perturbation;

		float_type vort_ic = w_oseen_vort(r, 0);
		return vort_ic;
		//return 0;
	}

    /** @brief  Refienment conditon for octants.  */
	template<class OctantType>
	bool refinement(OctantType it, int diff_level, bool use_all = false) const
		noexcept
	{
		auto b = it->data().descriptor();
		b.level() = it->refinement_level();
		const float_type dx_base = domain_->dx_base();

		auto center = (domain_->bounding_box().max() -
			domain_->bounding_box().min() + 1) / 2.0 +
			domain_->bounding_box().min();

		auto scaling = std::pow(2, b.level());
		center *= scaling;
		auto dx_level = dx_base / std::pow(2, b.level());

		b.grow(2, 2);
		auto corners = b.get_corners();

		float_type w_max = std::abs(vr_fct_(float_type(0.0), float_type(0.0), 0, perturbation_));

		for (int i = b.base()[0]; i <= b.max()[0]; ++i)
		{
			for (int j = b.base()[1]; j <= b.max()[1]; ++j)
			{

				//float_type x = static_cast<float_type>(i - center[0]) * dx_level;
				//float_type y = static_cast<float_type>(j - center[1]) * dx_level;
				
				float_type x = static_cast<float_type>(i) * dx_level;
				float_type y = static_cast<float_type>(j) * dx_level;

				float_type half_block = static_cast<float_type>(b.extent()[0]) * dx_level / 2.0;
				//z = static_cast<float_type>(k - center[2] + 0.5) * dx_level;

				float_type tmp_w = vor(x, y, 0);
				//tmp_w =  vor(x,y-0.5*vort_sep,0)+ vor(x,y+0.5*vort_sep,0);


				float_type maxLevel = 4.0;
				float_type max_c = std::max(std::fabs(x), std::fabs(y));
				//float_type max_c = std::fabs(x) + std::fabs(y);
				float_type rd = std::sqrt(x * x + y * y);
				float_type bd = 4.8 / pow(2, b.level()) - half_block;

				//float_type bd = 4.8 - 1.2*b.level() - half_block;
				if (max_c < bd)
					return true;

				/*if (std::fabs(tmp_w) > 1.0 * pow(refinement_factor_, diff_level))
					return true;*/
				//}
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
		auto res =
			_domain->construct_basemesh_blocks(_d, _domain->block_extent());
		domain_->read_parameters(_d);

		return res;
	}

	float_type u_vort(float_type x, float_type y, float_type td, int idx = 0) {
		float_type x_loc = x - ctr_dis_x;
		float_type y_loc = y - ctr_dis_y;
		if (vortexType == 1) {
		return u_taylor_vort(x_loc, y_loc, td, idx);
		}
		else if (vortexType == 2) {
		return u_oseen_vort(x_loc, y_loc, td, idx);
		}
		else if (vortexType == 0) {
		return 0.0;
		}
		else {return 0.0;}
	}

	float_type w_taylor_vort(float_type rd, float_type td) {
		float_type t_0 = Re_ / 2.0 / R_ / R_;
		//float_type gam = 1.0;
		float_type t1 = td / R_ / R_ / R_ / R_;
		float_type t = t1 + t_0;
		float_type r = rd;
		float_type r2 = r * r;
		//float_type nu = Lx / Re_;
		float_type exponent = 0.5 * (1.0 - r * r * t_0 / t / R_ / R_);
		float_type expval = std::exp(exponent);
		float_type w = 2.0 * (t_0 / t) * (t_0 / t) / R_ * (1.0 - r2 / 2.0 / R_ / R_ * (t_0 / t)) * expval;
		//float_type w = gam/4.0/M_PI*Re_/t*std::exp(-r2/4.0*Re_/t);
		//float_type w = 1.0/4.0/M_PI/nu/t/t*(1.0 - r2/4.0/nu/t)*std::exp(-r2/4.0/nu/t);
		return w;
	}

	float_type u_taylor_vort(float_type x, float_type y, float_type td, int idx = 0) {
		float_type t_0 = Re_ / 2.0 / R_ / R_;
		float_type t_1 = td / R_ / R_ / R_ / R_;
		float_type t = t_0 + t_1;
		float_type r = std::sqrt(x * x + y * y);
		float_type r2 = r * r;
		float_type exponent = 0.5 * (1.0 - r * r * t_0 / t / R_ / R_);
		float_type expval = std::exp(exponent);
		float_type u_theta = (t_0 / t) * (t_0 / t) * r / R_ * expval;
		float_type theta = std::atan2(y, x);
		float_type multiplier = 0;
		if (idx == 0) {
			multiplier = -std::sin(theta);
		}
		else if (idx == 1) {
			multiplier = std::cos(theta);
		}
		float_type u_val = u_theta * multiplier;
		return u_val;
	}

	float_type w_oseen_vort(float_type rd, float_type td) {
		float_type mean_c = 2.24181; //if using non-dim in Panton, max vel happens at eta = 2.24181
		float_type fac = 2.0 * mean_c * mean_c / (mean_c * mean_c + 2); //factor to make maxvelocity to be 1
		float_type t0 = Re_ / mean_c / mean_c;
		float_type tc = t0 + td;
		float_type nu = 1.0 / Re_;
		float_type eta = rd / std::sqrt(tc * nu);
		float_type vort = 1.0 / nu / tc * std::exp(-eta * eta / 4.0) / fac;
		return vort;

	}

	float_type u_oseen_vort(float_type x, float_type y, float_type td, int idx = 0) {
		float_type mean_c = 2.24181; //if using non-dim in Panton, max vel happens at eta = 2.24181
		float_type fac = 2.0 * mean_c * mean_c / (mean_c * mean_c + 2); //factor to make maxvelocity to be 1
		float_type t0 = Re_ / mean_c / mean_c;
		float_type tc = t0 + td;
		float_type rd = sqrt(x * x + y * y);
		float_type nu = 1.0 / Re_;
		float_type eta = rd / std::sqrt(tc * nu);
		float_type expVal = std::exp(-eta * eta / 4.0);

		float_type denom = std::sqrt(tc * nu);

		float_type u_theta = 2.0 / denom / eta * (1.0 - expVal) / fac;
		float_type theta = std::atan2(y, x);
		float_type multiplier = 0;
		if (idx == 0) {
			multiplier = -std::sin(theta);
		}
		else if (idx == 1) {
			multiplier = std::cos(theta);
		}
		float_type u_val = u_theta * multiplier;
		return u_val;
	}

	/*template<class field>
	void assignWRef() {
		if (!doamin_->is_server()) {
			for (auto it_t = domain_->begin_leaves(); it_t != domain_->end_leaves(); ++it_t) {
				if (it_t->locally_owned() && it_t->has_data()) {
					int refinement_level = it_t->refinement_level();
					float_type dx = dx_ / std::pow(2.0, refinement_level);
					auto scaling = std::pow(2, refinement_level);

					for (auto& node : it_t->data()) {
						const auto& coord = node.level_coordinate();
						float_type x = static_cast<float_type>(coord[0] - center[0] * scaling + 0.5) * dx_level;
						float_type y = static_cast<float_type>(coord[1] - center[1] * scaling + 0.5) * dx_level;
						float_type r = std::sqrt(x * x + y * y);
						float_type t_final = dx_ * cfl_ * tot_steps_;
						node(field::tag()) = w_taylor_vort(r, t_final);
					}
				}
			}
		}
	}*/


    private:

    boost::mpi::communicator client_comm_;

    bool single_ring_=true;
    bool perturbation_=false;
    bool hard_max_refinement_=false;
    bool smooth_start_;
    int vortexType = 0;

    std::vector<float_type> U_;
    //bool subtract_non_leaf_  = true;
    float_type R_;
    float_type v_delta_;
    float_type d2v_;
    float_type source_max_;
    float_type vort_sep;

	float_type ctr_dis_x = 0.0;
	float_type ctr_dis_y = 0.0;

    float_type rmin_ref_;
    float_type rmax_ref_;
    float_type rz_ref_;
    float_type c1=0;
    float_type c2=0;
    float_type eps_grad_=1.0e6;;
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
