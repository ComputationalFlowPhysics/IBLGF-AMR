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

#ifndef IBLGF_INCLUDED_POD2D_HPP
#define IBLGF_INCLUDED_POD2D_HPP

#include <iostream>
#include <algorithm>
#include <array>
#include <cmath>
#include <iblgf/dictionary/dictionary.hpp>

#include "../../setups/setup_base.hpp"
#include <iblgf/operators/operators.hpp>
#include <iblgf/solver/modal_analysis/pod_petsc.hpp>
#include <slepceps.h>
#include <slepcsys.h>
namespace iblgf
{
using namespace domain;
using namespace types;
using namespace dictionary;

const int Dim = 3;

struct parameters
{
    static constexpr std::size_t Dim = 3;
    // clang-format off
    REGISTER_FIELDS
    (
    Dim,
        (
            //name, type, nFields, l/h-buf,mesh_obj, output(optional)
            (tlevel,        float_type, 1, 1, 1, cell, true),
            (u,             float_type, 3, 1, 1, face, true),
            (u_s,             float_type, Dim, 1, 1, face, true),
            (u_a,             float_type, Dim, 1, 1, face, true),
            (u_sym,             float_type, Dim, 1, 1, face, true),
            (p,             float_type, 1, 1, 1, cell, true),
            (test,          float_type, 1, 1, 1, cell,true ),
            (idx_u,         float_type, 3, 1, 1, face, true),
            (u_mean,      float_type, 3, 1, 1, face, true)

        )
    )
    // clang-format on
};
struct POD2D : public SetupBase<POD2D, parameters>
{
    using super_type = SetupBase<POD2D, parameters>;

    //Timings
    using clock_type = std::chrono::high_resolution_clock;
    using milliseconds = std::chrono::milliseconds;
    using duration_type = typename clock_type::duration;
    using time_point_type = typename clock_type::time_point;
    using key_id_t = typename domain_t::tree_t::key_type::value_type;

    POD2D(Dictionary* _d, std::string restart_tree_dir, std::string restart_field_dir)
    : super_type(
          _d, [this](auto _d, auto _domain) { return this->initialize_domain(_d, _domain); }, restart_tree_dir)
    , restart_tree_dir_(restart_tree_dir)
    , restart_field_dir_(restart_field_dir)
    {
        if (domain_->is_client()) client_comm_ = client_comm_.split(1);
        else
            client_comm_ = client_comm_.split(0);
        nLevelRefinement_ = simulation_.dictionary_->template get_or<int>("nLevels", 0);
        // std::cout << "Number of refinement levels: " << nLevelRefinement_ << std::endl;
        // std::cout << "Restarting list construction..." << std::endl;
        // domain_->register_adapt_condition() =
        //     [this](std::vector<float_type> source_max, auto& octs, std::vector<int>& level_change)
        // { return this->template adapt_level_change(source_max, octs, level_change); };

        domain_->register_refinement_condition() = [this](auto octant, int diff_level) { return false; };
        // domain_->init_refine(nLevelRefinement_, 0, 0);

        domain_->restart_list_construct();
        domain_->distribute<fmm_mask_builder_t, fmm_mask_builder_t>();
        simulation_.template read_h5<u_type>(restart_field_dir, "u");

        // simulation_.read(restart_tree_dir,"tree");
        // simulation_.read(restart_field_dir,"fields");
    }
    POD2D(Dictionary* _d)
    : super_type(_d)
    {
        if (domain_->is_client()) client_comm_ = client_comm_.split(1);
        else
            client_comm_ = client_comm_.split(0);
        nLevelRefinement_ = simulation_.dictionary_->template get_or<int>("nLevels", 0);

        domain_->register_refinement_condition() = [this](auto octant, int diff_level) { return false; };
        domain_->ib().init(_d->get_dictionary("simulation_parameters"), domain_->dx_base(), nLevelRefinement_, 100);
        domain_->init_refine(nLevelRefinement_, 0, 0);
        domain_->distribute<fmm_mask_builder_t, fmm_mask_builder_t>();
        this->initialize();
    }

    float_type run(int argc, char* argv[])
    {
        boost::mpi::communicator world;
        PetscBool petsc_initialized = PETSC_FALSE;
        PetscCall(PetscInitialized(&petsc_initialized));
        const bool owns_slepc = (petsc_initialized == PETSC_FALSE);
        if (owns_slepc) { PetscCall(SlepcInitialize(&argc, &argv, (char*)0, NULL)); }
        simulation_.write("init");
        solver::POD<super_type> pod(&this->simulation_);
        pod.run_vec_test();
        pod.run_MOS<u_s_type>("u_s","_sym_");
        pod.run_MOS<u_a_type>("u_a","_asym_");
        // pod.run_MOS<u_type>("u");
        if (owns_slepc) { PetscCall(SlepcFinalize()); }
        simulation_.write("final");

        return 0.0;

    }

    void write_fake_snapshots_known_modes(int idxStart, int nTotal, int nskip)
    {
        boost::mpi::communicator world;
        constexpr float_type pi = static_cast<float_type>(3.14159265358979323846);
        const auto bb_min = domain_->bounding_box().min();
        const auto bb_max = domain_->bounding_box().max();
        const float_type denom_x = std::max<float_type>(1e-12, static_cast<float_type>(bb_max[0] - bb_min[0]));
        const float_type denom_y = std::max<float_type>(1e-12, static_cast<float_type>(bb_max[1] - bb_min[1]));
        const float_type denom_z = std::max<float_type>(1e-12, static_cast<float_type>(bb_max[2] - bb_min[2]));

        const std::array<float_type, 5> r1 = {-2.0, -1.0, 0.0, 1.0, 2.0};
        const std::array<float_type, 5> r2 = {1.0, -2.0, 0.0, 2.0, -1.0};
        const std::array<float_type, 5> r3 = {1.0, 0.0, -2.0, 0.0, 1.0};
        const float_type n1 = std::sqrt(static_cast<float_type>(10.0));
        const float_type n2 = std::sqrt(static_cast<float_type>(10.0));
        const float_type n3 = std::sqrt(static_cast<float_type>(6.0));
        const float_type s1 = static_cast<float_type>(3.0);
        const float_type s2 = static_cast<float_type>(2.0);
        const float_type s3 = static_cast<float_type>(1.0);

        auto mode_value = [&](int mode_id, float_type X, float_type Y, float_type Z) -> float_type {
            if (mode_id == 0) return std::sin(1.0 * pi * X) * std::sin(1.0 * pi * Y) * std::sin(1.0 * pi * Z);
            if (mode_id == 1) return std::sin(2.0 * pi * X) * std::sin(2.0 * pi * Y) * std::sin(2.0 * pi * Z);
            return std::sin(3.0 * pi * X) * std::sin(3.0 * pi * Y) * std::sin(3.0 * pi * Z);
        };

        auto pod_mode_norm = [&](int mode_id) -> float_type {
            for (auto it = domain_->begin(); it != domain_->end(); ++it)
            {
                if (!it->locally_owned() || !it->has_data()) continue;
                if (!it->is_leaf() || it->is_correction()) continue;

                for (auto& n : it->data())
                {
                    const auto gc = n.global_coordinate();
                    const float_type X = 2.0 * (static_cast<float_type>(gc[0] - bb_min[0]) / denom_x) - 1.0;
                    const float_type Y = 2.0 * (static_cast<float_type>(gc[1] - bb_min[1]) / denom_y) - 1.0;
                    const float_type Z = 2.0 * (static_cast<float_type>(gc[2] - bb_min[2]) / denom_z) - 1.0;
                    const float_type mv = mode_value(mode_id, X, Y, Z);
                    for (std::size_t f = 0; f < Dim; ++f) { n(u_s, static_cast<int>(f)) = mv; }
                }
            }
            world.barrier();

            solver::POD<super_type> pod_helper(&this->simulation_);
            return pod_helper.template grid2vec_weighted_l2_norm<idx_u_type, u_s_type>();
        };

        const float_type nphi1 = pod_mode_norm(0);
        const float_type nphi2 = pod_mode_norm(1);
        const float_type nphi3 = pod_mode_norm(2);

        for (int s = 0; s < nTotal; ++s)
        {
            const int timeIdx = idxStart + s * nskip;
            const int k = s % 5;
            const float_type c1 = s1 * r1[static_cast<std::size_t>(k)] / n1;
            const float_type c2 = s2 * r2[static_cast<std::size_t>(k)] / n2;
            const float_type c3 = s3 * r3[static_cast<std::size_t>(k)] / n3;

            for (auto it = domain_->begin(); it != domain_->end(); ++it)
            {
                if (!it->locally_owned() || !it->has_data()) continue;
                if (!it->is_leaf() || it->is_correction()) continue;

                for (auto& n : it->data())
                {
                    const auto gc = n.global_coordinate();
                    const float_type X = 2.0 * (static_cast<float_type>(gc[0] - bb_min[0]) / denom_x) - 1.0;
                    const float_type Y = 2.0 * (static_cast<float_type>(gc[1] - bb_min[1]) / denom_y) - 1.0;
                    const float_type Z = 2.0 * (static_cast<float_type>(gc[2] - bb_min[2]) / denom_z) - 1.0;

                    const float_type phi1 = mode_value(0, X, Y, Z) / nphi1;
                    const float_type phi2 = mode_value(1, X, Y, Z) / nphi2;
                    const float_type phi3 = mode_value(2, X, Y, Z) / nphi3;
                    const float_type val = c1 * phi1 + c2 * phi2 + c3 * phi3;

                    for (std::size_t f = 0; f < Dim; ++f)
                    {
                        n(u_s, static_cast<int>(f)) = val;
                        n(u_a, static_cast<int>(f)) = val;
                    }
                }
            }

            world.barrier();
            simulation_.write("adapted_to_ref_" + std::to_string(timeIdx));
            world.barrier();
        }
    }
    
    void initialize()
    {
        boost::mpi::communicator world;
        if (domain_->is_server()) return;
        auto center =
            (domain_->bounding_box().max() - domain_->bounding_box().min() + 1) / 2.0 + domain_->bounding_box().min();

        for (auto it = domain_->begin(); it != domain_->end(); ++it)
        {
            if (!it->locally_owned()) continue;
            int ref_level_ = it->refinement_level();
            for (auto& n : it->data())
            {
                n(tlevel) = ref_level_ + 0.5;
                n(u, 0) = ref_level_ + 0.5;
                n(u, 1) = ref_level_ + 0.5;
            }
        }
    }

    std::vector<coordinate_t> initialize_domain(Dictionary* _d, domain_t* _domain)
    {
        auto res = _domain->construct_basemesh_blocks(_d, _domain->block_extent());
        domain_->read_parameters(_d);

        return res;
    }

  private:
    int                      nLevelRefinement_ = 0; // Number of refinement levels
    boost::mpi::communicator client_comm_;
    std::vector<key_id_t>    ref_keys_;  //referfecne keys local to rank
    std::vector<int>         ref_leafs_; //reference leafs local to rank
    std::string              restart_tree_dir_;
    std::string              restart_field_dir_;
};
} // namespace iblgf
#endif
