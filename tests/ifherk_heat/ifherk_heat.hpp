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

#ifndef IBLGF_INCLUDED_IFHERK_HEAT_HPP
#define IBLGF_INCLUDED_IFHERK_HEAT_HPP

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
#include <iblgf/utilities/interpolation.hpp>
#include <iblgf/solver/poisson/poisson.hpp>
#include <iblgf/solver/time_integration/ifherk.hpp>

#include "../../setups/setup_base.hpp"
#include <iblgf/operators/operators.hpp>

namespace iblgf
{
const int Dim = 3;

struct parameters
{
    static constexpr std::size_t Dim = 3;
    REGISTER_FIELDS(Dim,
        (
            //name               type        Dim   lBuffer  hBuffer, storage type
            (error_u, float_type, 3, 1, 1, face, true),
            (decomposition, float_type, 1, 1, 1, cell, true),
            //IF-HERK
            (u, float_type, 3, 1, 1, face, true),
            (u_ref, float_type, 3, 1, 1, face, true),
            (p, float_type, 1, 1, 1, cell, true)))
};

struct IfherkHeat : public SetupBase<IfherkHeat, parameters>
{
    using super_type = SetupBase<IfherkHeat, parameters>;

    //Timings
    using clock_type = std::chrono::high_resolution_clock;
    using milliseconds = std::chrono::milliseconds;
    using duration_type = typename clock_type::duration;
    using time_point_type = typename clock_type::time_point;

    IfherkHeat(Dictionary* _d)
    : super_type(_d, [this](auto _d, auto _domain) {
        return this->initialize_domain(_d, _domain);
    })
    {
        if (domain_->is_client()) client_comm_ = client_comm_.split(1);
        else
            client_comm_ = client_comm_.split(0);

        dx_ = domain_->dx_base();
        cfl_ =
            simulation_.dictionary()->template get_or<float_type>("cfl", 0.2);
        dt_ = simulation_.dictionary()->template get_or<float_type>("dt", -1.0);

        tot_steps_ =
            simulation_.dictionary()->template get<int>("nBaseLevelTimeSteps");
        Re_ = simulation_.dictionary()->template get<float_type>("Re");
        R_ = simulation_.dictionary()->template get<float_type>("R");
        d2v_ = simulation_.dictionary()->template get_or<float_type>(
            "DistanceOfVortexRings", R_);
        v_delta_ = simulation_.dictionary()->template get_or<float_type>(
            "vDelta", 0.2 * R_);

        ic_filename_ = simulation_.dictionary_->template get_or<std::string>(
            "hdf5_ic_name", "null");

        ref_filename_ = simulation_.dictionary_->template get_or<std::string>(
            "hdf5_ref_name", "null");

        source_max_ = simulation_.dictionary_->template get_or<float_type>(
            "source_max", 1.0);

        refinement_factor_ = simulation_.dictionary_->template get<float_type>(
            "refinement_factor");

        nLevelRefinement_ =
            simulation_.dictionary_->template get_or<int>("nLevels", 0);

        global_refinement_ = simulation_.dictionary_->template get_or<int>(
            "global_refinement", 0);

        if (dt_ < 0) dt_ = dx_ * cfl_;

        dt_ /= pow(2.0, nLevelRefinement_);
        tot_steps_ *= pow(2, nLevelRefinement_);

        pcout << "\n Setup:  Test - Simple IC \n" << std::endl;
        pcout << "Number of refinement levels: " << nLevelRefinement_
              << std::endl;

        domain_->register_adapt_condition() = [this](auto    octant,
                                                  float_type source_max) {
            return this->template adapt_level_change<cell_aux_type>(
                octant, source_max);
        };

        domain_->register_refinement_condition() = [this](auto octant,
                                                       int     diff_level) {
            return this->refinement(octant, diff_level);
        };

        if (!use_restart())
        {
            domain_->init_refine(_d->get_dictionary("simulation_parameters")
                                     ->template get_or<int>("nLevels", 0),
                global_refinement_);
        }
        else
        {
            domain_->restart_list_construct();
        }

        domain_->distribute<fmm_mask_builder_t, fmm_mask_builder_t>();
        if (!use_restart()) { this->initialize(); }
        else
        {
            simulation_.template read_h5<u_type>(
                simulation_.restart_field_dir());
        }

        boost::mpi::communicator world;
        if (world.rank() == 0)
            std::cout << "on Simulation: \n" << simulation_ << std::endl;
    }

    void run()
    {
        boost::mpi::communicator world;

        time_integration_t ifherk(&this->simulation_);

        if (ic_filename_ != "null")
            simulation_.template read_h5<u_type>(ic_filename_);

        mDuration_type ifherk_duration(0);
        TIME_CODE(
            ifherk_duration, SINGLE_ARG(ifherk.time_march(use_restart());))
        pcout_c << "Time to solution [ms] " << ifherk_duration.count()
                << std::endl;

        if (ref_filename_ != "null")
            simulation_.template read_h5<u_ref_type>(ref_filename_);

        this->compute_errors<u_type, u_ref_type, error_u_type>(
            std::string("u1_"), 0);
        this->compute_errors<u_type, u_ref_type, error_u_type>(
            std::string("u2_"), 1);
        this->compute_errors<u_type, u_ref_type, error_u_type>(
            std::string("u3_"), 2);

        simulation_.write2("final.hdf5");
    }

    template<class Field, class OctantType>
    int adapt_level_change(OctantType* it, float_type source_max)
    {
        auto& nodes_domain = it->data_ref().nodes_domain();

        // ----------------------------------------------------------------
        float_type field_max = 1e-14;

        for (auto it2 = nodes_domain.begin(); it2 != nodes_domain.end(); ++it2)
        {
            if (std::fabs(it2->template get<Field>()) > field_max)
                field_max = std::fabs(it2->template get<Field>());
        }

        //int count=0;
        //field_max=0;
        //for(auto it2=nodes_domain.begin();it2!=nodes_domain.end();++it2 )
        //{
        //    field_max+=it2->template get<Field>();
        //    count++;
        //}
        //field_max /= count;

        // ----------------------------------------------------------------

        //float_type base_threshold=1e-3;
        // set deletion_factor to be half of refinement_factor_ so it's easier
        // to refine and harder to delete
        // This prevent rapid change of level refinement
        float_type deletion_factor = refinement_factor_ * 0.75;

        int l_aim = static_cast<int>(
            ceil(nLevelRefinement_ -
                 log(field_max / source_max) / log(refinement_factor_)));
        int l_delete_aim = static_cast<int>(
            ceil(nLevelRefinement_ -
                 log(field_max / source_max) / log(deletion_factor)));

        if (l_aim > nLevelRefinement_) l_aim = nLevelRefinement_;

        //if (it->refinement_level()==0)
        //{
        //    if (field_max>source_max*base_threshold)
        //        l_delete_aim = std::max(l_delete_aim,0);
        //}

        int l_change = l_aim - it->refinement_level();
        //int l_delete_change = l_delete_aim - it->refinement_level();
        if (l_delete_aim < 0) return -1;
        if (l_change > 0) return 1;
        return 0;
        //return l_change<0 ? l_change:0;
        //return l_change>0 ? l_change:0;
        //return l_change;
    }

    /** @brief Initialization of poisson problem.
     *  @detail Testing poisson with manufactured solutions: exp
     */
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

        if (ic_filename_ != "null") return;

        // Voriticity IC
        for (auto it = domain_->begin(); it != domain_->end(); ++it)
        {
            if (!it->locally_owned()) continue;

            auto dx_level = dx_base / std::pow(2, it->refinement_level());
            auto scaling = std::pow(2, it->refinement_level());

            auto  view(it->data_ref().node_field().domain_view());
            auto& nodes_domain = it->data_ref().nodes_domain();

            //float_type T = dt_*tot_steps_;
            for (auto it2 = nodes_domain.begin(); it2 != nodes_domain.end();
                 ++it2)
            {
                //it2->get<source>() = 0.0;

                const auto& coord = it2->level_coordinate();

                /***********************************************************/
                float_type x = static_cast<float_type>(
                                   coord[0] - center[0] * scaling + 0.5) *
                               dx_level;
                float_type y =
                    static_cast<float_type>(coord[1] - center[1] * scaling) *
                    dx_level;
                float_type z =
                    static_cast<float_type>(coord[2] - center[2] * scaling) *
                    dx_level;

                it2->template get<edge_aux_type>(0) =
                    -vortex_ring_vor_ic(x, y, z - d2v_ / 2, 0) +
                    vortex_ring_vor_ic(x, y, z + d2v_ / 2, 0);
                /***********************************************************/
                x = static_cast<float_type>(coord[0] - center[0] * scaling) *
                    dx_level;
                y = static_cast<float_type>(
                        coord[1] - center[1] * scaling + 0.5) *
                    dx_level;
                z = static_cast<float_type>(coord[2] - center[2] * scaling) *
                    dx_level;

                it2->template get<edge_aux_type>(1) =
                    -vortex_ring_vor_ic(x, y, z - d2v_ / 2, 1) +
                    vortex_ring_vor_ic(x, y, z + d2v_ / 2, 1);
                /***********************************************************/
                x = static_cast<float_type>(coord[0] - center[0] * scaling) *
                    dx_level;
                y = static_cast<float_type>(coord[1] - center[1] * scaling) *
                    dx_level;
                z = static_cast<float_type>(
                        coord[2] - center[2] * scaling + 0.5) *
                    dx_level;

                it2->template get<edge_aux_type>(2) =
                    -vortex_ring_vor_ic(x, y, z - d2v_ / 2, 2) +
                    vortex_ring_vor_ic(x, y, z + d2v_ / 2, 2);

                /***********************************************************/
                it2->template get<decomposition_type>() = world.rank();
            }
        }

        psolver.template apply_lgf<edge_aux_type, stream_f_type>();
        auto client = domain_->decomposition().client();

        for (int l = domain_->tree()->base_level();
             l < domain_->tree()->depth(); ++l)
        {
            //client->template buffer_exchange<stream_f_type>(l);

            for (auto it = domain_->begin(l); it != domain_->end(l); ++it)
            {
                if (!it->locally_owned() || !it->has_data()) continue;
                const auto dx_level =
                    dx_base / std::pow(2, it->refinement_level());
                domain::Operator::curl_transpose<stream_f_type, u_type>(
                    it->data_ref(), dx_level, -1.0);
            }
            client->template buffer_exchange<u_type>(l);
        }
    }

    float_type vortex_ring_vor_fat_ic(
        float_type x, float_type y, float_type z, int field_idx) const
    {
        const float_type alpha = 0.54857674;
        float_type       R2 = R_ * R_;

        float_type r2 = x * x + y * y;
        float_type r = sqrt(r2);
        float_type s2 = z * z + (r - R_) * (r - R_);

        float_type theta = std::atan2(y, x);
        float_type w_theta = alpha * 1.0 / R2 * std::exp(-4.0 * s2 / (R2 - s2));

        if (s2 >= R2) return 0.0;

        if (field_idx == 0) return -w_theta * std::sin(theta);
        else if (field_idx == 1)
            return w_theta * std::cos(theta);
        else
            return 0.0;
    }

    float_type vortex_ring_vor_ic(float_type x, float_type y, float_type z,
        int field_idx, bool perturbation = true) const
    {
        float_type delta_2 = v_delta_ * v_delta_;

        float_type r2 = x * x + y * y;
        float_type r = sqrt(r2);
        float_type s2 = z * z + (r - R_) * (r - R_);

        float_type theta = std::atan2(y, x);
        float_type w_theta = 1.0 / M_PI / delta_2 * std::exp(-s2 / delta_2);

        float_type rd = (static_cast<float_type>(rand()) /
                            static_cast<float_type>(RAND_MAX)) -
                        0.5;
        float_type prtub = 0.001;
        rd *= prtub;

        //if (s2>=delta_2) return 0.0;

        if (field_idx == 0) return -w_theta * std::sin(theta) * (1 + rd);
        else if (field_idx == 1)
            return w_theta * std::cos(theta) * (1 + rd);
        else
            return 0.0;
    }

    /** @brief  Refienment conditon for octants.  */
    template<class OctantType>
    bool refinement(OctantType* it, int diff_level, bool use_all = false) const
        noexcept
    {
        auto b = it->data_ref().descriptor();
        b.level() = it->refinement_level();
        const float_type dx_base = domain_->dx_base();

        auto center =
            (domain_->bounding_box().max() - domain_->bounding_box().min()) /
                2.0 +
            domain_->bounding_box().min();

        auto scaling = std::pow(2, b.level());
        center *= scaling;
        auto dx_level = dx_base / std::pow(2, b.level());

        b.grow(2, 2);
        auto corners = b.get_corners();

        float_type w_max = std::abs(vortex_ring_vor_ic(
            float_type(R_), float_type(0.0), float_type(0.0), 1));

        for (int i = b.base()[0]; i <= b.max()[0]; ++i)
        {
            for (int j = b.base()[1]; j <= b.max()[1]; ++j)
            {
                for (int k = b.base()[2]; k <= b.max()[2]; ++k)
                {
                    float_type x =
                        static_cast<float_type>(i - center[0] + 0.5) * dx_level;
                    float_type y =
                        static_cast<float_type>(j - center[1]) * dx_level;
                    float_type z =
                        static_cast<float_type>(k - center[2]) * dx_level;

                    float_type tmp_w =
                        -vortex_ring_vor_ic(x, y, z - d2v_ / 2, 0) +
                        vortex_ring_vor_ic(x, y, z + d2v_ / 2, 0);

                    if (std::fabs(tmp_w) >
                        w_max * pow(refinement_factor_, diff_level))
                        return true;

                    x = static_cast<float_type>(i - center[0]) * dx_level;
                    y = static_cast<float_type>(j - center[1] + 0.5) * dx_level;
                    z = static_cast<float_type>(k - center[2]) * dx_level;

                    tmp_w = -vortex_ring_vor_ic(x, y, z - d2v_ / 2, 1) +
                            vortex_ring_vor_ic(x, y, z + d2v_ / 2, 1);
                    if (std::fabs(tmp_w) >
                        w_max * pow(refinement_factor_, diff_level))
                        return true;

                    x = static_cast<float_type>(i - center[0]) * dx_level;
                    y = static_cast<float_type>(j - center[1]) * dx_level;
                    z = static_cast<float_type>(k - center[2] + 0.5) * dx_level;

                    tmp_w = -vortex_ring_vor_ic(x, y, z - d2v_ / 2, 2) +
                            vortex_ring_vor_ic(x, y, z + d2v_ / 2, 2);
                    if (std::fabs(tmp_w) >
                        w_max * pow(refinement_factor_, diff_level))
                        return true;
                }
            }
        }

        return false;
    }

    /** @brief  Initialization of the domain blocks. This is registered in the
     *          domain through the base setup class, passing it to the domain ctor.
     */
    std::vector<extent_t> initialize_domain(Dictionary* _d, domain_t* _domain)
    {
        auto res =
            _domain->construct_basemesh_blocks(_d, _domain->block_extent());
        domain_->read_parameters(_d);

        return res;
    }

  private:
    boost::mpi::communicator client_comm_;
    float_type               R_;
    float_type               v_delta_;
    float_type               d2v_;
    float_type               source_max_;

    float_type rmin_ref_;
    float_type rmax_ref_;
    float_type rz_ref_;
    float_type c1 = 0;
    float_type c2 = 0;
    float_type eps_grad_ = 1.0e6;
    ;
    int      nLevelRefinement_ = 0;
    int      global_refinement_ = 0;
    fcoord_t offset_;

    float_type a_ = 10.0;

    float_type dt_, dx_;
    float_type cfl_;
    float_type Re_;
    int        tot_steps_;
    float_type refinement_factor_ = 1. / 8;

    std::string ic_filename_, ref_filename_;
};

} // namespace iblgf

#endif
