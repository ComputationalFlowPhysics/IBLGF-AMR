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

#ifndef IBLGF_INCLUDED_OPERATORLINEARTEST_HPP
#define IBLGF_INCLUDED_OPERATORLINEARTEST_HPP

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
#include <iblgf/solver/time_integration/ifherk_linear.hpp>

#include "../../setups/setup_linear.hpp"
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
        // (error_u          , float_type, 2, 1, 1, face, true  ),
        // (error_p          , float_type, 1, 1, 1, cell, false ),
        (test             , float_type, 1, 1, 1, cell, false ),
        (u                , float_type, 2, 1, 1, face, true  ),
        (u_base           , float_type, 2, 1, 1, face, true  ),
        // (u_ref            , float_type, 2, 1, 1, face, true  ),
        // (p_ref            , float_type, 1, 1, 1, cell, true  ),
        (p                , float_type, 1, 1, 1, cell, true  ),
        // (w_num            , float_type, 1, 1, 1, edge, false ),
        // (w_exact          , float_type, 1, 1, 1, edge, false ),
        // (error_w          , float_type, 1, 1, 1, edge, false ),
        // (exact_u_theta    , float_type, 2, 1, 1, edge, false ),
        // (num_u_theta      , float_type, 2, 1, 1, edge, false ),
        // (error_u_theta    , float_type, 2, 1, 1, edge, false ),
        // (u_mean           , float_type, 2, 1, 1, face, true  ),
        (face_aux_tmp   , float_type, 2, 1, 1, face, true  ),
        (nonlinear_tmp  , float_type, 2, 1, 1, face, true  ),
        (edge_aux2      , float_type, 1, 1, 1, edge, true  ),
        // (curl_source      , float_type, 2, 1, 1, face, true  ),
        // (curl_target      , float_type, 1, 1, 1, edge, true  ),
        // (curl_exact       , float_type, 1, 1, 1, edge, true  ),
        // (curl_error       , float_type, 1, 1, 1, edge, true  ),
        // (nonlinear_source  , float_type, 2, 1, 1, face, true  ),
        // (nonlinear_target  , float_type, 2, 1, 1, face, true  ),
        // (nonlinear_exact   , float_type, 2, 1, 1, face, true  ),
        // (nonlinear_error   , float_type, 2, 1, 1, face, true  ),
        (u_s              , float_type, 2, 1, 1, face, true  ),
        (u_p              , float_type, 2, 1, 1, face, true  ),
        (nonlinear_jac_target , float_type, 2, 1, 1, face, true  ),
        (nonlinear_jac_exact  , float_type, 2, 1, 1, face, true  ),
        (nonlinear_jac_error   , float_type, 2, 1, 1, face, true  ),
        (nonlinear_jac_T_target , float_type, 2, 1, 1, face, true  ),
        (nonlinear_jac_T_exact  , float_type, 2, 1, 1, face, true  ),
        (nonlinear_jac_T_error   , float_type, 2, 1, 1, face, true  )
    ))
    // clang-format on
};

struct OperatorTest : public SetupBase<OperatorTest, parameters>
{
    using super_type =SetupBase<OperatorTest,parameters>;

    //Timings
    using clock_type = std::chrono::high_resolution_clock;
    using milliseconds = std::chrono::milliseconds;
    using duration_type = typename clock_type::duration;
    using time_point_type = typename clock_type::time_point;

	OperatorTest(Dictionary* _d)
		: super_type(_d, [this](auto _d, auto _domain) {
		return this->initialize_domain(_d, _domain);
			})
	{
		if (domain_->is_client()) client_comm_ = client_comm_.split(1);
		else
			client_comm_ = client_comm_.split(0);

		//smooth_start_ = simulation_.dictionary()->template get_or<bool>("smooth_start", false);
		U_.resize(domain_->dimension());
		U_[0] = simulation_.dictionary()->template get_or<float_type>("Ux", 0.0);
		U_[1] = simulation_.dictionary()->template get_or<float_type>("Uy", 0.0);
		if (domain_->dimension()>2)
			U_[2] = simulation_.dictionary()->template get_or<float_type>("Uz", 0.0);

        smooth_start_ = simulation_.dictionary()->template get_or<bool>("smooth_start", false);


        Omega = simulation_.dictionary()->template get_or<float_type>("Omega", 0.0);

        vortexType = simulation_.dictionary()->template get_or<int>("Vort_type", 0);

		//IntrpIBLevel = simulation_.dictionary()->template get_or<bool>("IntrpIBLevel", false); //intrp IB level, used if restart file is from coarser mesh

		simulation_.bc_vel() =
			[this](std::size_t idx, float_type t, auto coord = {0, 0})
			{
				return 0.0;
			};


		simulation_.frame_vel() =
			[this](std::size_t idx, float_type t, auto coord = {0, 0})
			{
				// float_type T0 = 0.5;

				// if (t<=0.0 && smooth_start_)
				// 	return 0.0;
				// else if (t<T0-1e-10 && smooth_start_)
				// {
				// 	float_type h1 = exp(-1/(t/T0));
				// 	float_type h2 = exp(-1/(1 - t/T0));

				// 	return -(U_[idx]) * (h1/(h1+h2));
				// }
				// else
				// {
				// 	return -(U_[idx]);
				// }
                return 0.0;
			};
        /////added stuff
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
		non_base_level_update = simulation_.dictionary()->template get_or<bool>("no_base_level_update", false);
		NoMeshUpdate = simulation_.dictionary()->template get_or<bool>("no_mesh_update", false);

		auto domain_range = domain_->bounding_box().max() - domain_->bounding_box().min();
		Lx = domain_range[0] * dx_;



		ctr_dis_x = 0.0*dx_; //this is setup as the center of the vortex in the unit of grid spacing
		ctr_dis_y = 0.0*dx_;

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
        //////
		/*simulation_.frame_vel() =
			[this](std::size_t idx, float_type t, auto coord = {0, 0, 0})
			{return 0.0;};*/

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

		// domain_->register_adapt_condition()=
		// 	[this]( std::vector<float_type> source_max, auto& octs, std::vector<int>& level_change ) 
		// 	{return this->template adapt_level_change(source_max, octs, level_change);};

		/*domain_->register_adapt_condition() =
			[this](auto octant, std::vector<float_type> source_max) {return this->template adapt_level_change<cell_aux_type, u_type>(octant, source_max); };*/

		/*domain_->register_adapt_condition()=
			[this](auto octant, std::vector<float_type> source_max){return this->template adapt_level_change<edge_aux_type, edge_aux_type>(octant, source_max);};*/


		
		domain_->register_refinement_condition() = [this](auto octant,
			int diff_level) {
				return this->refinement(octant, diff_level);
		};
		

		nIB_add_level_ = _d->get_dictionary("simulation_parameters")->template get_or<int>("nIB_add_level", 0);

		domain_->init_refine(nLevelRefinement_, global_refinement_, nIB_add_level_);


		domain_->distribute<fmm_mask_builder_t, fmm_mask_builder_t>();

		this->initialize();

		boost::mpi::communicator world;
		if (world.rank() == 0)
			std::cout << "on Simulation: \n" << simulation_ << std::endl;
	}

    float_type run()
    {
        boost::mpi::communicator world;

        time_integration_t ifherk(&this->simulation_);

        // if (domain_->is_client())
        // {
        //     const float_type dx_base = domain_->dx_base();
        //     const float_type base_level = domain_->tree()->base_level();

        //     //Bufffer exchange of some fields
        //     auto client = domain_->decomposition().client();
        //     // client->buffer_exchange<lap_source_type>(base_level);
        //     // client->buffer_exchange<div_source_type>(base_level);
        //     client->buffer_exchange<curl_source_type>(base_level);
        //     // client->buffer_exchange<grad_source_type>(base_level);
        //     client->buffer_exchange<curl_exact_type>(base_level);
        //     client->buffer_exchange<nonlinear_source_type>(base_level);
        //     client->buffer_exchange<u_s_type>(base_level);
        //     client->buffer_exchange<u_p_type>(base_level);


        //     for (auto it = domain_->begin_leaves(); it != domain_->end_leaves();
        //          ++it)
        //     {
        //         if (!it->locally_owned() || !it->has_data()) continue;

        //         auto dx_level = dx_base / std::pow(2, it->refinement_level());

        //         // domain::Operator::laplace<lap_source_type, lap_target_type>(
        //         //     it->data(), dx_level);
        //         // domain::Operator::divergence<div_source_type, div_target_type>(
        //         //     it->data(), dx_level);
        //         domain::Operator::curl<curl_source_type, curl_target_type>(
        //             it->data(), dx_level);
        //         // domain::Operator::gradient<grad_source_type, grad_target_type>(
        //         //     it->data(), dx_level);
        //     }
        //     client->buffer_exchange<curl_target_type>(base_level);
        //     for (auto it = domain_->begin_leaves(); it != domain_->end_leaves();
        //          ++it)
        //     {
        //         if (!it->locally_owned() || !it->has_data()) continue;

        //         domain::Operator::nonlinear<nonlinear_source_type,
        //             curl_target_type, nonlinear_target_type>(it->data());
        //     }
        //     // ifherk.nonlinear_jac_access<u_s_type, u_p_type, nonlinear_jac_target_type>();
        // }
        // ifherk.Assign_idx<nonlinear_target_type, nonlinear_jac_source_type>();
        if (domain_->is_client())
        {
            ifherk.nonlinear_jac_access<u_s_type, u_p_type, nonlinear_jac_target_type>();
            ifherk.nonlinear_jac_adjoint_access<u_s_type, u_p_type, nonlinear_jac_T_target_type>();
        }
        //ifherk.nonlinear_jac_access<u_s_type, u_p_type, nonlinear_jac_target_type>();
       

        // this->compute_errors<lap_target_type, lap_exact_type, lap_error_type>(
        //     "Lap_");
        // this->compute_errors<grad_target_type, grad_exact_type,
        //     grad_error_type>("Grad_");
        // this->compute_errors<div_target_type, div_exact_type, div_error_type>(
        //     "Div_");
        // this->compute_errors<curl_target_type, curl_exact_type,
        //     curl_error_type>("Curl_");
        // this->compute_errors<nonlinear_target_type, nonlinear_exact_type,
        //     nonlinear_error_type>("Nonlin_");
        world.barrier();
        float_type u1_inf = this->compute_errors<nonlinear_jac_target_type, nonlinear_jac_exact_type,
            nonlinear_jac_error_type>("Nonlin_jac_",0);
        u1_inf = this->compute_errors<nonlinear_jac_T_target_type, nonlinear_jac_T_exact_type,
            nonlinear_jac_T_error_type>("Nonlin_jac_T_",0);
        u1_inf=this->compute_errors<nonlinear_jac_target_type, nonlinear_jac_exact_type,
            nonlinear_jac_error_type>("Nonlin_jac_",1);
        u1_inf=this->compute_errors<nonlinear_jac_T_target_type, nonlinear_jac_T_exact_type,
            nonlinear_jac_T_error_type>("Nonlin_jac_T_",1);
        world.barrier();
        // (&this->simulation_)->write("mesh.hdf5");
        ifherk.up_and_down<nonlinear_jac_target_type>();
        // simulation_.write("final.hdf5");
        for (int i = 0; i < world.size(); i++)
        {
            if (world.rank() == i)
            {
                std::cout << "rank " << world.rank() << std::endl;
            }
            world.barrier();
        }
        world.barrier();
        simulation_.write_test("final.hdf5");
        ifherk.up_and_down<nonlinear_jac_target_type>();
        return 0.0;
    }

	void initialize()
    {
        poisson_solver_t psolver(&this->simulation_);

        boost::mpi::communicator world;
        if (domain_->is_server()) return;
        auto center =
            (domain_->bounding_box().max() - domain_->bounding_box().min()) /
                2.0 +
            domain_->bounding_box().min();

        // Adapt center to always have peak value in a cell-center
        //center+=0.5/std::pow(2,nRef);
        const float_type dx_base = domain_->dx_base();

        for (auto it = domain_->begin(); it != domain_->end();
             ++it)
        {
            if (!it->locally_owned()) continue;
            // if (!(*it && it->has_data())) continue;
            auto dx_level = dx_base / std::pow(2, it->refinement_level());
            auto scaling = std::pow(2, it->refinement_level());

            for (auto& node : it->data())
            {
                const auto& coord = node.level_coordinate();

                //Cell centered coordinates
                //This can obviously be made much less verbose
                float_type xc = static_cast<float_type>(
                                    coord[0] - center[0] * scaling + 0.5) *
                                dx_level;
                float_type yc = static_cast<float_type>(
                                    coord[1] - center[1] * scaling + 0.5) *
                                dx_level;
                /* float_type zc = static_cast<float_type>(
                                    coord[2] - center[2] * scaling + 0.5) *
                                dx_level;*/

                //Face centered coordinates
                float_type xf0 =
                    static_cast<float_type>(coord[0] - center[0] * scaling) *
                    dx_level;
                float_type yf0 = yc;
                // float_type zf0 = zc;

                float_type xf1 = xc;
                float_type yf1 =
                    static_cast<float_type>(coord[1] - center[1] * scaling) *
                    dx_level;
                // float_type zf1 = zc;

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
                /* float_type ze0 =
                    static_cast<float_type>(coord[2] - center[2] * scaling) *
                    dx_level;*/
                float_type xe1 =
                    static_cast<float_type>(coord[0] - center[0] * scaling) *
                    dx_level;
                float_type ye1 = static_cast<float_type>(
                                     coord[1] - center[1] * scaling + 0.5) *
                                 dx_level;
                /* float_type ze1 =
                    static_cast<float_type>(coord[2] - center[2] * scaling) *
                    dx_level; */
                float_type xe2 =
                    static_cast<float_type>(coord[0] - center[0] * scaling) *
                    dx_level;
                float_type ye2 =
                    static_cast<float_type>(coord[1] - center[1] * scaling) *
                    dx_level;
                /* float_type ze2 = static_cast<float_type>(
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

                node(u_s, 0) = tmpf0*yf0;
				node(u_s, 1) = tmpf1*xf1;

				node(u_p, 0) = tmpf0*cos(yf0);
				node(u_p, 1) = tmpf1*cos(xf1);
                //Gradient
                // node(grad_source) = fct;
                // node(grad_exact, 0) = -2 * a_ * xf0 * tmpf0;
                // node(grad_exact, 1) = -2 * a_ * yf1 * tmpf1;

                // //Laplace
                // node(lap_source, 0) = tmpc;
                // node(lap_exact) = -4 * a_ * tmpc + 4 * a2 * xc2 * tmpc +
                //                   4 * a2 * yc2 * tmpc;

                // //Divergence
                // node(div_source, 0) = tmpf0;
                // node(div_source, 1) = tmpf1;
                // node(div_exact, 0) = -2 * a_ * xc * tmpc - 2 * a_ * yc * tmpc;

                //Curl
                // node(curl_source, 0) = tmpf0;
                // node(curl_source, 1) = tmpf1;

                // node(curl_exact) =
                //     2 * a_ * ye2 * tmpe2 - 2 * a_ * xe2 * tmpe2;

                // //non_linear
                // node(nonlinear_source, 0) = tmpf0;
                // node(nonlinear_source, 1) = tmpf1;
                

                // node(nonlinear_exact, 0) =
                //     tmpf0 * (2 * a_ * xf0 * tmpf0 - 2 * a_ * yf0 * tmpf0);

                // node(nonlinear_exact, 1) =
                //     -tmpf1 * (2 * a_ * xf1 * tmpf1 - 2 * a_ * yf1 * tmpf1);

                node(nonlinear_jac_exact, 0) = 4*tmpf0*tmpf0*xf0*xf0*cos(xf0) - 2*tmpf0*tmpf0*yf0*yf0*cos(xf0)
				 - 2*tmpf0*tmpf0*xf0*yf0*cos(yf0) + tmpf0*tmpf0*xf0*sin(xf0) - tmpf0*tmpf0*xf0*sin(yf0);

				//node(nl_jac_exact, 0) = 1.0;

				node(nonlinear_jac_exact, 1) = 4*tmpf1*tmpf1*yf1*yf1*cos(yf1) - 2*tmpf1*tmpf1*xf1*xf1*cos(yf1)
				 - 2*tmpf1*tmpf1*xf1*yf1*cos(xf1) + tmpf1*tmpf1*yf1*sin(yf1) - tmpf1*tmpf1*yf1*sin(xf1);
                
                node(nonlinear_jac_T_exact, 0) = 4*tmpf0*tmpf0*xf0*yf0*cos(yf0) - 2*tmpf0*tmpf0*xf0*xf0*cos(xf0)
				 - 2*tmpf0*tmpf0*yf0*yf0*cos(xf0) + tmpf0*tmpf0*cos(xf0) + tmpf0*tmpf0*xf0*sin(yf0);

				node(nonlinear_jac_T_exact, 1) = 4*tmpf1*tmpf1*xf1*yf1*cos(xf1) - 2*tmpf1*tmpf1*xf1*xf1*cos(yf1)
				 - 2*tmpf1*tmpf1*yf1*yf1*cos(yf1) + tmpf1*tmpf1*yf1*sin(xf1) + tmpf1*tmpf1*cos(yf1);

            }
        }
    }

    /** @brief  Refienment conditon for octants.  */
	template<class OctantType>
	bool refinement(OctantType it, int diff_level, bool use_all = false) const
		noexcept
	{
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


    private:

    boost::mpi::communicator client_comm_;

    bool single_ring_=true;
    bool perturbation_=false;
    bool hard_max_refinement_=false;
    bool smooth_start_;
	bool non_base_level_update = false;
    int vortexType = 0;
	bool NoMeshUpdate = false;
	//bool IntrpIBLevel = false;

    std::vector<float_type> U_;
	float_type Omega; //rotational rate
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
	int nIB_add_level_ = 0;
    int hard_max_level_ = 0;
    int global_refinement_=0;
    fcoord_t offset_;

    float_type a_ = 1.0;

    float_type dt_,dx_;
    float_type cfl_;
    float_type Re_;
    int tot_steps_;
    float_type refinement_factor_=1./8;
    float_type base_threshold_=1e-4;

    std::string ic_filename_, ref_filename_;

    float_type Lx;
};

double vortex_run(std::string input, int argc = 0, char** argv = nullptr);

} // namespace iblgf

#endif
