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

#ifndef IBLGF_INCLUDED_OPERATORTEST_HPP
#define IBLGF_INCLUDED_OPERATORTEST_HPP

#include <iostream>
#include <iblgf/dictionary/dictionary.hpp>

#include "../../setups/setup_helmholtz.hpp"
#include <iblgf/operators/operators.hpp>

namespace iblgf
{
using namespace domain;
using namespace types;
using namespace dictionary;

const int Dim = 2;

struct parameters
{
    static constexpr std::size_t Dim = 2;
    static constexpr std::size_t N_modes= 32;
    static constexpr std::size_t PREFAC  = 1;
    // clang-format off
    REGISTER_FIELDS
    (
    Dim,
        (
            //name, type, nFields, l/h-buf,mesh_obj, output(optional)
            /*(grad_source,       float_type, PREFAC*N_modes, 1, 1, cell, true),
            (grad_target,       float_type, 3*PREFAC*N_modes, 1, 1, face, true),
            (grad_exact,        float_type, 3*PREFAC*N_modes, 1, 1, face, true),
            (grad_error,        float_type, 3*PREFAC*N_modes, 1, 1, face, true),

            (lap_source,        float_type, PREFAC*N_modes, 1, 1, cell, true),
            (lap_target,        float_type, PREFAC*N_modes, 1, 1, cell, true),
            (lap_exact,         float_type, PREFAC*N_modes, 1, 1, cell, true),
            (lap_error,         float_type, PREFAC*N_modes, 1, 1, cell, true),

            (div_source,        float_type, 3*PREFAC*N_modes, 1, 1, face, true),
            (div_target,        float_type, PREFAC*N_modes, 1, 1, cell, true),
            (div_exact,         float_type, PREFAC*N_modes, 1, 1, cell, true),
            (div_error,         float_type, PREFAC*N_modes, 1, 1, cell, true),*/

            (curl_source,       float_type, 3*PREFAC*N_modes, 1, 1, face, true),
            (curl_target,       float_type, 3*PREFAC*N_modes, 1, 1, edge, true),
            (curl_exact,        float_type, 3*PREFAC*N_modes, 1, 1, edge, true),
            (curl_error,        float_type, 3*PREFAC*N_modes, 1, 1, edge, true),

            (nonlinear_source,  float_type, 3*PREFAC*N_modes, 1, 1, face, true),
            (nonlinear_target,  float_type, 3*PREFAC*N_modes, 1, 1, face, true),
            (nonlinear_exact,   float_type, 3*PREFAC*N_modes, 1, 1, face, true),
            (nonlinear_error,   float_type, 3*PREFAC*N_modes, 1, 1, face, true)
        )
    )
    // clang-format on
};

struct OperatorTest : public Setup_helmholtz<OperatorTest, parameters>
{
    using super_type = Setup_helmholtz<OperatorTest, parameters>;

    //Timings
    using clock_type = std::chrono::high_resolution_clock;
    using milliseconds = std::chrono::milliseconds;
    using duration_type = typename clock_type::duration;
    using time_point_type = typename clock_type::time_point;

    static constexpr std::size_t N_modes = parameters::N_modes;

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

        c_z = simulation_.dictionary_->template get_or<int>(
            "c_z", 1.0);

        L_z = simulation_.dictionary_->template get_or<int>(
            "L_z", 1.0);

        PREFAC = 1;

        dz = L_z/static_cast<float_type>(N_modes)/static_cast<float_type>(PREFAC);

        pcout << "value of dz is " << dz << std::endl;

        pcout << "\n Setup:  Test - Vortex ring \n" << std::endl;
        pcout << "Number of refinement levels: " << nLevels_ << std::endl;

        domain_->register_refinement_condition() = [this](auto octant,
                                                       int     diff_level) {
            return this->refinement(octant, diff_level);
        };
        domain_->init_refine(_d->get_dictionary("simulation_parameters")
                                 ->template get_or<int>("nLevels", 0),
            global_refinement_, 0);
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
            const float_type base_level = domain_->tree()->base_level();

            //Bufffer exchange of some fields
            auto client = domain_->decomposition().client();
            //client->buffer_exchange<lap_source_type>(base_level);
            //client->buffer_exchange<div_source_type>(base_level);
            client->buffer_exchange<curl_source_type>(base_level);
            //client->buffer_exchange<grad_source_type>(base_level);
            //client->buffer_exchange<curl_exact_type>(base_level);
            //client->buffer_exchange<nonlinear_source_type>(base_level);

            for (auto it = domain_->begin_leaves(); it != domain_->end_leaves();
                 ++it)
            {
                if (!it->locally_owned() || !it->has_data()) continue;

                auto dx_level = dx_base / std::pow(2, it->refinement_level());

                /*domain::Operator::laplace<lap_source_type, lap_target_type>(
                    it->data(), dx_level);*/
                /*domain::Operator::divergence<div_source_type, div_target_type>(
                    it->data(), dx_level);*/
                domain::Operator::curl_helmholtz<curl_source_type, curl_target_type>(
                    it->data(), dx_level, N_modes, dz, PREFAC);
                /*domain::Operator::gradient<grad_source_type, grad_target_type>(
                    it->data(), dx_level);*/
            }
            /*client->buffer_exchange<curl_target_type>(base_level);
            for (auto it = domain_->begin_leaves(); it != domain_->end_leaves();
                 ++it)
            {
                if (!it->locally_owned() || !it->has_data()) continue;

                domain::Operator::nonlinear<nonlinear_source_type,
                    curl_target_type, nonlinear_target_type>(it->data());
            }*/

        }

        /*this->compute_errors<lap_target_type, lap_exact_type, lap_error_type>(
            "Lap_");*/
        /*this->compute_errors<grad_target_type, grad_exact_type,
            grad_error_type>("Grad_");*/
        /*this->compute_errors<div_target_type, div_exact_type, div_error_type>(
            "Div_");*/
        /*for (int i = 0; i < curl_target_type::nFields();i++) {
            //pcout << i << " ";
            this->compute_errors<curl_target_type, curl_exact_type,
                curl_error_type>("Curl_", i);
        }*/
        this->compute_errors_for_all<curl_target_type, curl_exact_type,
                curl_error_type>(dz, "Curl_", 0, N_modes, L_z);
        this->compute_errors_for_all<curl_target_type, curl_exact_type,
                curl_error_type>(dz, "Curl_", 1, N_modes, L_z);
        this->compute_errors_for_all<curl_target_type, curl_exact_type,
                curl_error_type>(dz, "Curl_", 2, N_modes, L_z);
        /*this->compute_errors<curl_target_type, curl_exact_type,
            curl_error_type>("Curl_");*/
        /*this->compute_errors<nonlinear_target_type, nonlinear_exact_type,
            nonlinear_error_type>("Nonlin_");*/
        //simulation_.write("mesh.hdf5");
    }

    /** @brief Initialization of poisson problem.
     *  @detail Testing poisson with manufactured solutions: exp(-x^2-y^2)sin(2pi*z/c)
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

        for (auto it = domain_->begin_leaves(); it != domain_->end_leaves();
             ++it)
        {
            if (!it->locally_owned()) continue;
            if (!(*it && it->has_data())) continue;
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
                //const float_type zc2 = zc * zc;
                /***********************************************************/

                float_type r_2 = r * r;
                const auto fct = std::exp(-r_2);
                const auto tmpc = std::exp(-r_2);

                const auto tmpf0 = std::exp(-rf0 * rf0);
                const auto tmpf1 = std::exp(-rf1 * rf1);
                const auto tmpf2 = std::exp(-rf2 * rf2);

                const auto tmpe0 = std::exp(-re0 * re0);
                const auto tmpe1 = std::exp(-re1 * re1);
                const auto tmpe2 = std::exp(-re2 * re2);

                //Gradient
                //node(grad_source) = fct;
                for (int i = 0 ; i < PREFAC*N_modes ; i++) {
                    float_type tmp_z = static_cast<float_type>(i)*dz;
                    node(curl_source, i)                  = tmpf0*std::sin(2.0*M_PI*(tmp_z+0.5*dz)/c_z);
                    node(curl_source, PREFAC*N_modes  +i) = tmpf1*std::sin(2.0*M_PI*(tmp_z+0.5*dz)/c_z);
                    node(curl_source, PREFAC*N_modes*2+i) = tmpf2*std::sin(2.0*M_PI*(tmp_z)/c_z);
                    //node(curl_source, PREFAC*N_modes*2+i) = 0;

                    node(curl_exact, i)                   = -tmpe0*std::sin(2.0*M_PI*tmp_z/c_z)*ye0*2.0
                                               -2.0*M_PI/c_z*tmpe0*std::cos(2.0*M_PI*tmp_z/c_z);
                    node(curl_exact, PREFAC*N_modes  +i)  =  tmpe1*std::sin(2.0*M_PI*tmp_z/c_z)*xe1*2.0
                                               +2.0*M_PI/c_z*tmpe1*std::cos(2.0*M_PI*tmp_z/c_z);
                    node(curl_exact, PREFAC*N_modes*2+i)  = 2.0*(ye2 - xe2)*tmpe2*std::sin(2.0*M_PI*(tmp_z+0.5*dz)/c_z);

                    /*node(curl_source, i)                  = tmpf0;
                    node(curl_source, PREFAC*N_modes  +i) = tmpf1;
                    //node(curl_source, PREFAC*N_modes*2+i) = tmpf2*std::cos(2.0*M_PI*(tmp_z+0.5*dz)/c_z);
                    node(curl_source, PREFAC*N_modes*2+i) = tmpf2;

                    node(curl_exact, i)    = -tmpe0*ye0*2.0;
                    node(curl_exact, PREFAC*N_modes  +i)  =  tmpe1*xe1*2.0;
                    node(curl_exact, PREFAC*N_modes*2+i)  =  2.0*(ye2 - xe2)*tmpe2;*/
                }
                /*node(grad_exact, 0) = -2 * a_ * xf0 * tmpf0;
                node(grad_exact, 1) = -2 * a_ * yf1 * tmpf1;
                node(grad_exact, 2) = -2 * a_ * zf2 * tmpf2;

                //Laplace
                node(lap_source, 0) = tmpc;
                node(lap_exact) = -6 * a_ * tmpc + 4 * a2 * xc2 * tmpc +
                                  4 * a2 * yc2 * tmpc + 4 * a2 * zc2 * tmpc;

                //Divergence
                node(div_source, 0) = tmpf0;
                node(div_source, 1) = tmpf1;
                node(div_source, 2) = tmpf2;
                node(div_exact, 0) = -2 * a_ * xc * tmpc - 2 * a_ * yc * tmpc -
                                     2 * a_ * zc * tmpc;

                //Curl
                node(curl_source, 0) = tmpf0;
                node(curl_source, 1) = tmpf1;
                node(curl_source, 2) = tmpf2;

                node(curl_exact, 0) =
                    2 * a_ * ze0 * tmpe0 - 2 * a_ * ye0 * tmpe0;
                node(curl_exact, 1) =
                    2 * a_ * xe1 * tmpe1 - 2 * a_ * ze1 * tmpe1;
                node(curl_exact, 2) =
                    2 * a_ * ye2 * tmpe2 - 2 * a_ * xe2 * tmpe2;

                //non_linear
                node(nonlinear_source, 0) = tmpf0;
                node(nonlinear_source, 1) = tmpf1;
                node(nonlinear_source, 2) = tmpf2;

                node(nonlinear_exact, 0) =
                    tmpf0 * (2 * a_ * xf0 * tmpf0 - 2 * a_ * yf0 * tmpf0) +
                    tmpf0 * (2 * a_ * xf0 * tmpf0 - 2 * a_ * zf0 * tmpf0);

                node(nonlinear_exact, 1) =
                    tmpf1 * (2 * a_ * yf1 * tmpf1 - 2 * a_ * zf1 * tmpf1) -
                    tmpf1 * (2 * a_ * xf1 * tmpf1 - 2 * a_ * yf1 * tmpf1);

                node(nonlinear_exact, 2) =
                    -tmpf2 * (2 * a_ * xf2 * tmpf2 - 2 * a_ * zf2 * tmpf2) -
                    tmpf2 * (2 * a_ * yf2 * tmpf2 - 2 * a_ * zf2 * tmpf2);*/
            }
        }
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

  private:
    boost::mpi::communicator client_comm_;
    float_type               eps_grad_ = 1.0e6;
    int                      nLevels_ = 0;
    int                      global_refinement_;
    float_type               a_ = 100.0;
    float_type               c_z = 1.0;
    float_type               dz;
    float_type               L_z = 1.0;  //length of z dimension 
    int                      PREFAC = 3; //3 FOR THE 3/2 RULE IN REAL DOMAIN, 2 ON THE OPERATORS FOR THE COMPLEX VARIABLES
};

} // namespace iblgf

#endif // IBLGF_INCLUDED_POISSON_HPP
