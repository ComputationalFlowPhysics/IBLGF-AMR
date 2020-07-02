#ifndef IBLGF_INCLUDED_OPERATORTEST_HPP
#define IBLGF_INCLUDED_OPERATORTEST_HPP

#include <iostream>
#include <vector>
#include <limits>
#include <algorithm>
#include <vector>
#include <fftw3.h>
#include <boost/mpi.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>

// IBLGF-specific
#include <global.hpp>
#include <simulation.hpp>
#include <domain/domain.hpp>
#include <domain/dataFields/dataBlock.hpp>
#include <domain/dataFields/datafield.hpp>
#include <domain/octree/tree.hpp>
#include <chrono>
#include <IO/parallel_ostream.hpp>
#include <lgf/lgf.hpp>
#include <fmm/fmm.hpp>

#include <utilities/convolution.hpp>
#include <utilities/interpolation.hpp>
#include <solver/poisson/poisson.hpp>

#include "../../setups/setup_base.hpp"
#include <operators/operators.hpp>

const int Dim = 3;

struct parameters
{
    static constexpr std::size_t Dim = 3;
    REGISTER_FIELDS(Dim,
        (
            //name               type        Dim   lBuffer  hBuffer, storage type
            (grad_source, float_type, 1, 1, 1, cell),
            (grad_target, float_type, 3, 1, 1, face),
            (grad_exact, float_type, 3, 1, 1, face),
            (grad_error, float_type, 3, 1, 1, face),

            (lap_source, float_type, 1, 1, 1, cell),
            (lap_target, float_type, 1, 1, 1, cell),
            (lap_exact, float_type, 1, 1, 1, cell),
            (lap_error, float_type, 1, 1, 1, cell),

            (div_source, float_type, 3, 1, 1, face),
            (div_target, float_type, 1, 1, 1, cell),
            (div_exact, float_type, 1, 1, 1, cell),
            (div_error, float_type, 1, 1, 1, cell),

            (curl_source, float_type, 3, 1, 1, face),
            (curl_target, float_type, 3, 1, 1, edge),
            (curl_exact, float_type, 3, 1, 1, edge),
            (curl_error, float_type, 3, 1, 1, edge),

            (nonlinear_source, float_type, 3, 1, 1, face),
            (nonlinear_target, float_type, 3, 1, 1, face),
            (nonlinear_exact, float_type, 3, 1, 1, face),
            (nonlinear_error, float_type, 3, 1, 1, face)))
};

struct OperatorTest : public SetupBase<OperatorTest, parameters>
{
    using super_type = SetupBase<OperatorTest, parameters>;

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

        global_refinement_ = simulation_.dictionary_->template get_or<int>(
            "global_refinement", 0);

        pcout << "\n Setup:  Test - Vortex ring \n" << std::endl;
        pcout << "Number of refinement levels: " << nLevels_ << std::endl;

        domain_->register_refinement_condition() = [this](auto octant,
                                                       int     diff_level) {
            return this->refinement(octant, diff_level);
        };
        domain_->init_refine(_d->get_dictionary("simulation_parameters")
                                 ->template get_or<int>("nLevels", 0),
            global_refinement_);
        domain_->distribute<fmm_mask_builder_t, fmm_mask_builder_t>();
        this->initialize();

        boost::mpi::communicator world;
        if (world.rank() == 0)
            std::cout << "on Simulation: \n" << simulation_ << std::endl;
    }

    void run()
    {
        boost::mpi::communicator world;
        if (domain_->is_client())
        {
            const float_type dx_base = domain_->dx_base();

            //Bufffer exchange of some fields
            auto client = domain_->decomposition().client();
            client->buffer_exchange<lap_source>();
            client->buffer_exchange<div_source>();
            client->buffer_exchange<curl_source>();
            client->buffer_exchange<grad_source>();
            client->buffer_exchange<curl_exact>();
            client->buffer_exchange<nonlinear_source>();

            mDuration_type lap_duration(0);
            TIME_CODE(lap_duration,
                SINGLE_ARG(
                    for (auto it = domain_->begin_leafs();
                         it != domain_->end_leafs(); ++it) {
                        if (!it->locally_owned() || !it->has_data()) continue;
                        auto dx_level =
                            dx_base / std::pow(2, it->refinement_level());
                        domain::Operator::laplace<lap_source, lap_target>(
                            *(it->has_data()), dx_level);
                        domain::Operator::divergence<div_source, div_target>(
                            *(it->has_data()), dx_level);
                        domain::Operator::curl<curl_source, curl_target>(
                            *(it->has_data()), dx_level);
                        domain::Operator::gradient<grad_source, grad_target>(
                            *(it->has_data()), dx_level);
                    } client->buffer_exchange<curl_target>();
                    for (auto it = domain_->begin_leafs();
                         it != domain_->end_leafs(); ++it) {
                        if (!it->locally_owned() || !it->has_data()) continue;
                        auto dx_level =
                            dx_base / std::pow(2, it->refinement_level());
                        domain::Operator::nonlinear<nonlinear_source,
                            curl_target, nonlinear_target>(
                            *(it->has_data()), dx_level);
                    }))

            pcout_c << "Total time: " << lap_duration.count() << " on "
                    << world.size() << std::endl;
        }

        this->compute_errors<lap_target, lap_exact, lap_error>("Lap_");
        this->compute_errors<grad_target, grad_exact, grad_error>("Grad_");
        this->compute_errors<div_target, div_exact, div_error>("Div_");
        this->compute_errors<curl_target, curl_exact, curl_error>("Curl_");
        this->compute_errors<nonlinear_target, nonlinear_exact,
            nonlinear_error>("Nonlin_");
        simulation_.write("mesh.hdf5");
    }

    /** @brief Initialization of poisson problem.
     *  @detail Testing poisson with manufactured solutions: exp
     */
    void initialize()
    {
        boost::mpi::communicator world;
        if (domain_->is_server()) return;
        auto center =
            (domain_->bounding_box().max() - domain_->bounding_box().min()) /
                2.0 +
            domain_->bounding_box().min();

        // Adapt center to always have peak value in a cell-center
        //center+=0.5/std::pow(2,nRef);
        const float_type dx_base = domain_->dx_base();

        for (auto it = domain_->begin_leafs(); it != domain_->end_leafs(); ++it)
        {
            if (!it->locally_owned()) continue;
            if (!(*it && it->has_data())) continue;
            auto dx_level = dx_base / std::pow(2, it->refinement_level());
            auto scaling = std::pow(2, it->refinement_level());

            auto  view(it->data().node_field().domain_view());
            auto& nodes_domain = it->data().nodes_domain();
            for (auto it2 = nodes_domain.begin(); it2 != nodes_domain.end();
                 ++it2)
            {
                const auto& coord = it2->level_coordinate();

                //Cell centered coordinates
                //This can obviously be made much less verbose
                float_type xc = static_cast<float_type>(
                                    coord[0] - center[0] * scaling + 0.5) *
                                dx_level;
                float_type yc = static_cast<float_type>(
                                    coord[1] - center[1] * scaling + 0.5) *
                                dx_level;
                float_type zc = static_cast<float_type>(
                                    coord[2] - center[2] * scaling + 0.5) *
                                dx_level;

                //Face centered coordinates
                float_type xf0 =
                    static_cast<float_type>(coord[0] - center[0] * scaling) *
                    dx_level;
                float_type yf0 = yc;
                float_type zf0 = zc;

                float_type xf1 = xc;
                float_type yf1 =
                    static_cast<float_type>(coord[1] - center[1] * scaling) *
                    dx_level;
                float_type zf1 = zc;

                float_type xf2 = xc;
                float_type yf2 = yc;
                float_type zf2 =
                    static_cast<float_type>(coord[2] - center[2] * scaling) *
                    dx_level;

                //Edge centered coordinates
                float_type xe0 = static_cast<float_type>(
                                     coord[0] - center[0] * scaling + 0.5) *
                                 dx_level;
                float_type ye0 =
                    static_cast<float_type>(coord[1] - center[1] * scaling) *
                    dx_level;
                float_type ze0 =
                    static_cast<float_type>(coord[2] - center[2] * scaling) *
                    dx_level;
                float_type xe1 =
                    static_cast<float_type>(coord[0] - center[0] * scaling) *
                    dx_level;
                float_type ye1 = static_cast<float_type>(
                                     coord[1] - center[1] * scaling + 0.5) *
                                 dx_level;
                float_type ze1 =
                    static_cast<float_type>(coord[2] - center[2] * scaling) *
                    dx_level;
                float_type xe2 =
                    static_cast<float_type>(coord[0] - center[0] * scaling) *
                    dx_level;
                float_type ye2 =
                    static_cast<float_type>(coord[1] - center[1] * scaling) *
                    dx_level;
                float_type ze2 = static_cast<float_type>(
                                     coord[2] - center[2] * scaling + 0.5) *
                                 dx_level;

                const float_type r = std::sqrt(xc * xc + yc * yc + zc * zc);
                const float_type rf0 =
                    std::sqrt(xf0 * xf0 + yf0 * yf0 + zf0 * zf0);
                const float_type rf1 =
                    std::sqrt(xf1 * xf1 + yf1 * yf1 + zf1 * zf1);
                const float_type rf2 =
                    std::sqrt(xf2 * xf2 + yf2 * yf2 + zf2 * zf2);
                const float_type re0 =
                    std::sqrt(xe0 * xe0 + ye0 * ye0 + ze0 * ze0);
                const float_type re1 =
                    std::sqrt(xe1 * xe1 + ye1 * ye1 + ze1 * ze1);
                const float_type re2 =
                    std::sqrt(xe2 * xe2 + ye2 * ye2 + ze2 * ze2);
                const float_type a2 = a_ * a_;
                const float_type xc2 = xc * xc;
                const float_type yc2 = yc * yc;
                const float_type zc2 = zc * zc;
                /***********************************************************/

                float_type r_2 = r * r;
                const auto fct = std::exp(-a_ * r_2);
                const auto tmpc = std::exp(-a_ * r_2);

                const auto tmpf0 = std::exp(-a_ * rf0 * rf0);
                const auto tmpf1 = std::exp(-a_ * rf1 * rf1);
                const auto tmpf2 = std::exp(-a_ * rf2 * rf2);

                const auto tmpe0 = std::exp(-a_ * re0 * re0);
                const auto tmpe1 = std::exp(-a_ * re1 * re1);
                const auto tmpe2 = std::exp(-a_ * re2 * re2);

                //Gradient
                it2->get<grad_source>() = fct;
                it2->get<grad_exact>(0) = -2 * a_ * xf0 * tmpf0;
                it2->get<grad_exact>(1) = -2 * a_ * yf1 * tmpf1;
                it2->get<grad_exact>(2) = -2 * a_ * zf2 * tmpf2;

                //Laplace
                it2->get<lap_source>(0) = tmpc;
                it2->get<lap_exact>() = -6 * a_ * tmpc + 4 * a2 * xc2 * tmpc +
                                        4 * a2 * yc2 * tmpc +
                                        4 * a2 * zc2 * tmpc;

                //Divergence
                it2->get<div_source>(0) = tmpf0;
                it2->get<div_source>(1) = tmpf1;
                it2->get<div_source>(2) = tmpf2;
                it2->get<div_exact>(0) = -2 * a_ * xc * tmpc -
                                         2 * a_ * yc * tmpc -
                                         2 * a_ * zc * tmpc;

                //Curl
                it2->get<curl_source>(0) = tmpf0;
                it2->get<curl_source>(1) = tmpf1;
                it2->get<curl_source>(2) = tmpf2;

                it2->get<curl_exact>(0) =
                    2 * a_ * ze0 * tmpe0 - 2 * a_ * ye0 * tmpe0;
                it2->get<curl_exact>(1) =
                    2 * a_ * xe1 * tmpe1 - 2 * a_ * ze1 * tmpe1;
                it2->get<curl_exact>(2) =
                    2 * a_ * ye2 * tmpe2 - 2 * a_ * xe2 * tmpe2;

                //non_linear
                it2->get<nonlinear_source>(0) = tmpf0;
                it2->get<nonlinear_source>(1) = tmpf1;
                it2->get<nonlinear_source>(2) = tmpf2;

                it2->get<nonlinear_exact>(0) =
                    tmpf0 * (2 * a_ * xf0 * tmpf0 - 2 * a_ * yf0 * tmpf0) +
                    tmpf0 * (2 * a_ * xf0 * tmpf0 - 2 * a_ * zf0 * tmpf0);

                it2->get<nonlinear_exact>(1) =
                    tmpf1 * (2 * a_ * yf1 * tmpf1 - 2 * a_ * zf1 * tmpf1) -
                    tmpf1 * (2 * a_ * xf1 * tmpf1 - 2 * a_ * yf1 * tmpf1);

                it2->get<nonlinear_exact>(2) =
                    -tmpf2 * (2 * a_ * xf2 * tmpf2 - 2 * a_ * zf2 * tmpf2) -
                    tmpf2 * (2 * a_ * yf2 * tmpf2 - 2 * a_ * zf2 * tmpf2);
            }
        }
    }

    /** @brief Compute L2 and LInf errors */
    template<class Numeric, class Exact, class Error>
    void compute_errors(std::string _output_prefix = "")
    {
        const float_type dx_base = domain_->dx_base();
        float_type       L2 = 0.;
        float_type       LInf = -1.0;
        int              count = 0;
        float_type       L2_exact = 0;
        float_type       LInf_exact = -1.0;

        std::vector<float_type> L2_perLevel(
            nLevels_ + 1 + global_refinement_, 0.0);
        std::vector<float_type> L2_exact_perLevel(
            nLevels_ + 1 + global_refinement_, 0.0);
        std::vector<float_type> LInf_perLevel(
            nLevels_ + 1 + global_refinement_, 0.0);
        std::vector<float_type> LInf_exact_perLevel(
            nLevels_ + 1 + global_refinement_, 0.0);

        std::vector<int> counts(nLevels_ + 1 + global_refinement_, 0);

        if (domain_->is_server()) return;

        for (auto it_t = domain_->begin_leafs(); it_t != domain_->end_leafs();
             ++it_t)
        {
            if (!it_t->locally_owned() || !it_t->has_data()) continue;

            int    refinement_level = it_t->refinement_level();
            double dx = dx_base / std::pow(2.0, refinement_level);

            auto& nodes_domain = it_t->data().nodes_domain();
            for (auto it2 = nodes_domain.begin(); it2 != nodes_domain.end();
                 ++it2)
            {
                for (std::size_t i = 0; i < Exact::nFields(); ++i)
                {
                    float_type tmp_exact = it2->template get<Exact>(i);
                    float_type tmp_num = it2->template get<Numeric>(i);

                    float_type error_tmp = tmp_num - tmp_exact;

                    it2->template get<Error>(i) = error_tmp;

                    L2 += error_tmp * error_tmp * (dx * dx * dx);
                    L2_exact += tmp_exact * tmp_exact * (dx * dx * dx);

                    L2_perLevel[refinement_level] +=
                        error_tmp * error_tmp * (dx * dx * dx);
                    L2_exact_perLevel[refinement_level] +=
                        tmp_exact * tmp_exact * (dx * dx * dx);
                    ++counts[refinement_level];

                    if (std::fabs(tmp_exact) > LInf_exact)
                        LInf_exact = std::fabs(tmp_exact);

                    if (std::fabs(error_tmp) > LInf)
                        LInf = std::fabs(error_tmp);

                    if (std::fabs(error_tmp) > LInf_perLevel[refinement_level])
                        LInf_perLevel[refinement_level] = std::fabs(error_tmp);

                    if (std::fabs(tmp_exact) >
                        LInf_exact_perLevel[refinement_level])
                        LInf_exact_perLevel[refinement_level] =
                            std::fabs(tmp_exact);

                    ++count;
                }
            }
        }

        float_type L2_global(0.0);
        float_type LInf_global(0.0);

        float_type L2_exact_global(0.0);
        float_type LInf_exact_global(0.0);

        boost::mpi::all_reduce(
            client_comm_, L2, L2_global, std::plus<float_type>());
        boost::mpi::all_reduce(
            client_comm_, L2_exact, L2_exact_global, std::plus<float_type>());

        boost::mpi::all_reduce(client_comm_, LInf, LInf_global,
            [&](const auto& v0, const auto& v1) { return v0 > v1 ? v0 : v1; });
        boost::mpi::all_reduce(client_comm_, LInf_exact, LInf_exact_global,
            [&](const auto& v0, const auto& v1) { return v0 > v1 ? v0 : v1; });

        pcout_c << "Glabal " << _output_prefix
                << "L2_exact = " << std::sqrt(L2_exact_global) << std::endl;
        pcout_c << "Global " << _output_prefix
                << "LInf_exact = " << LInf_exact_global << std::endl;

        pcout_c << "Glabal " << _output_prefix
                << "L2 = " << std::sqrt(L2_global) << std::endl;
        pcout_c << "Global " << _output_prefix << "LInf = " << LInf_global
                << std::endl;

        //Level wise errros
        std::vector<float_type> L2_perLevel_global(
            nLevels_ + 1 + global_refinement_, 0.0);
        std::vector<float_type> LInf_perLevel_global(
            nLevels_ + 1 + global_refinement_, 0.0);

        std::vector<float_type> L2_exact_perLevel_global(
            nLevels_ + 1 + global_refinement_, 0.0);
        std::vector<float_type> LInf_exact_perLevel_global(
            nLevels_ + 1 + global_refinement_, 0.0);

        std::vector<int> counts_global(nLevels_ + 1 + global_refinement_, 0);
        for (std::size_t i = 0; i < LInf_perLevel_global.size(); ++i)
        {
            boost::mpi::all_reduce(client_comm_, counts[i], counts_global[i],
                std::plus<float_type>());
            boost::mpi::all_reduce(client_comm_, L2_perLevel[i],
                L2_perLevel_global[i], std::plus<float_type>());
            boost::mpi::all_reduce(client_comm_, LInf_perLevel[i],
                LInf_perLevel_global[i], [&](const auto& v0, const auto& v1) {
                    return v0 > v1 ? v0 : v1;
                });

            boost::mpi::all_reduce(client_comm_, L2_exact_perLevel[i],
                L2_exact_perLevel_global[i], std::plus<float_type>());
            boost::mpi::all_reduce(client_comm_, LInf_exact_perLevel[i],
                LInf_exact_perLevel_global[i],
                [&](const auto& v0, const auto& v1) {
                    return v0 > v1 ? v0 : v1;
                });

            pcout_c << _output_prefix << "L2_" << i << " "
                    << std::sqrt(L2_perLevel_global[i]) << std::endl;
            pcout_c << _output_prefix << "LInf_" << i << " "
                    << LInf_perLevel_global[i] << std::endl;
            pcout_c << "count_" << i << " " << counts_global[i] << std::endl;
        }
    }

    /** @brief  Refienment conditon for octants.  */
    template<class OctantType>
    bool refinement(OctantType* it, int diff_level, bool use_all = false) const
        noexcept
    {
        return false;
    }

    /** @brief  Initialization of the domain blocks. This is registered in the
     *          domain through the base setup class, passing it to the domain ctor.
     */
    std::vector<extent_t> initialize_domain(Dictionary* _d, domain_t* _domain)
    {
        return _domain->construct_basemesh_blocks(_d, _domain->block_extent());
    }

  private:
    boost::mpi::communicator client_comm_;

    float_type eps_grad_ = 1.0e6;
    ;
    int        nLevels_ = 0;
    int        global_refinement_;
    float_type a_ = 100.0;
};

#endif // IBLGF_INCLUDED_POISSON_HPP
