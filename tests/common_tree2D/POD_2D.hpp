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
#include <cmath>
#include <algorithm>
#include <type_traits>
#include <utility>
#include <boost/filesystem.hpp>
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

const int Dim = 2;

struct parameters
{
    static constexpr std::size_t Dim = 2;
    // clang-format off
    REGISTER_FIELDS
    (
    Dim,
        (
            //name, type, nFields, l/h-buf,mesh_obj, output(optional)
            (tlevel,        float_type, 1, 1, 1, cell, true),
            (u,             float_type, 2, 1, 1, face, true),
            (u_ref,         float_type, 2, 1, 1, face, true),
            (u_s,             float_type, Dim, 1, 1, face, true),
            (u_a,             float_type, Dim, 1, 1, face, true),
            (u_sym,             float_type, Dim, 1, 1, face, true),
            (p,             float_type, 1, 1, 1, cell, true),
            (test,          float_type, 1, 1, 1, cell,true ),
            (idx_u,         float_type, 2, 1, 1, face, true),
            (u_mean,      float_type, 2, 1, 1, face, true)

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
        domain_->init_refine(nLevelRefinement_, 0, 0);
        domain_->distribute<fmm_mask_builder_t, fmm_mask_builder_t>();
        this->initialize();
    }

    float_type run(int argc, char* argv[])
    {
        boost::mpi::communicator world;
        PetscCall(SlepcInitialize(&argc, &argv, (char*)0, NULL));
        simulation_.write("init");
        this->cleanup_pod_mode_outputs("_sym_");
        this->cleanup_pod_mode_outputs("_asym_");
        solver::POD<SetupBase> pod(&this->simulation_);
        pod.run_vec_test();
        pod.run_MOS<u_s_type>("u_s","_sym_");
        this->template write_actual_modes_to_u_ref<u_s_type>("_sym_", "u_s");
        pod.run_MOS<u_a_type>("u_a","_asym_");
        this->template write_actual_modes_to_u_ref<u_a_type>("_asym_", "u_a");
        PetscCall(SlepcFinalize());
        simulation_.write("final");

        return 0.0;

    }

    float_type mode0_error_sym() const { return mode0_error_sym_; }
    float_type mode0_error_asym() const { return mode0_error_asym_; }

    void write_fake_snapshots_known_modes(int idxStart, int nTotal, int nskip)
    {
        boost::mpi::communicator world;
        constexpr float_type     pi = static_cast<float_type>(3.14159265358979323846);
        const auto               bb_min = domain_->bounding_box().min();
        const auto               bb_max = domain_->bounding_box().max();
        const float_type         denom_x = std::max<float_type>(1e-12, static_cast<float_type>(bb_max[0] - bb_min[0]));
        const float_type         denom_y = std::max<float_type>(1e-12, static_cast<float_type>(bb_max[1] - bb_min[1]));
        const std::array<float_type, 5> r1 = {-2.0, -1.0, 0.0, 1.0, 2.0};
        const std::array<float_type, 5> r2 = {1.0, -2.0, 0.0, 2.0, -1.0};
        const std::array<float_type, 5> r3 = {1.0, 0.0, -2.0, 0.0, 1.0};
        const float_type n1 = std::sqrt(static_cast<float_type>(10.0));
        const float_type n2 = std::sqrt(static_cast<float_type>(10.0));
        const float_type n3 = std::sqrt(static_cast<float_type>(6.0));
        const float_type s1 = static_cast<float_type>(3.0);
        const float_type s2 = static_cast<float_type>(2.0);
        const float_type s3 = static_cast<float_type>(1.0);

        auto write_visual_grid = [&](float_type c1, float_type c2, float_type c3, const std::string& name) {
            for (auto it = domain_->begin(); it != domain_->end(); ++it)
            {
                if (!it->locally_owned() || !it->has_data()) continue;
                if (!it->is_leaf() || it->is_correction()) continue;

                for (auto& n : it->data())
                {
                    const auto gc = n.global_coordinate();
                    const float_type X = 2.0 * (static_cast<float_type>(gc[0] - bb_min[0]) / denom_x) - 1.0;
                    const float_type Y = 2.0 * (static_cast<float_type>(gc[1] - bb_min[1]) / denom_y) - 1.0;

                    const float_type mode1 = std::sin(1.0 * pi * X) * std::sin(1.0 * pi * Y);
                    const float_type mode2 = std::sin(2.0 * pi * X) * std::sin(2.0 * pi * Y);
                    const float_type mode3 = std::sin(3.0 * pi * X) * std::sin(3.0 * pi * Y);
                    const float_type val = c1 * mode1 + c2 * mode2 + c3 * mode3;

                    for (std::size_t f = 0; f < Dim; ++f)
                    {
                        n(u_s, static_cast<int>(f)) = val;
                        n(u_a, static_cast<int>(f)) = val;
                    }
                }
            }
            world.barrier();
            simulation_.write(name);
            world.barrier();
        };

        // Write 3 basis grids + one combined reference grid for easy visual inspection.
        write_visual_grid(1.0, 0.0, 0.0, "pod_true_phi1");
        write_visual_grid(0.0, 1.0, 0.0, "pod_true_phi2");
        write_visual_grid(0.0, 0.0, 1.0, "pod_true_phi3");
        // "Final" grid uses the first snapshot coefficients (k=0).
        write_visual_grid(s1 * r1[0] / n1, s2 * r2[0] / n2, s3 * r3[0] / n3, "pod_true_final_grid");

        for (int s = 0; s < nTotal; ++s)
        {
            const int timeIdx = idxStart + s * nskip;
            const int        k = s % 5;
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

                    const float_type mode1 = std::sin(1.0 * pi * X) * std::sin(1.0 * pi * Y);
                    const float_type mode2 = std::sin(2.0 * pi * X) * std::sin(2.0 * pi * Y);
                    const float_type mode3 = std::sin(3.0 * pi * X) * std::sin(3.0 * pi * Y);
                    const float_type val = c1 * mode1 + c2 * mode2 + c3 * mode3;

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

    void write_fake_snapshots_single_mode(int idxStart, int nTotal, int nskip)
    {
        boost::mpi::communicator world;
        constexpr float_type     pi = static_cast<float_type>(3.14159265358979323846);

        for (int s = 0; s < nTotal; ++s)
        {
            const int timeIdx = idxStart + s * nskip;
            const float_type c = (s % 2 == 0) ? 1.0 : -1.0; // zero temporal mean

            for (auto it = domain_->begin(); it != domain_->end(); ++it)
            {
                if (!it->locally_owned() || !it->has_data()) continue;
                if (!it->is_leaf() || it->is_correction()) continue;

                for (auto& n : it->data())
                {
                    const auto gc = n.global_coordinate();
                    const float_type x = gc[0] - 0.5;
                    const float_type mode = std::sin(2.0 * pi * x);

                    for (std::size_t f = 0; f < Dim; ++f)
                    {
                        n(u_s, static_cast<int>(f)) = c * mode;
                        n(u_a, static_cast<int>(f)) = c * mode;
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

    void cleanup_pod_mode_outputs(const std::string& prefix)
    {
        boost::mpi::communicator world;
        const std::string        out_dir =
            simulation_.dictionary_->get_dictionary("output")->template get<std::string>("directory");

        if (world.rank() == 0)
        {
            const boost::filesystem::path dir_path("./" + out_dir);
            if (boost::filesystem::exists(dir_path) && boost::filesystem::is_directory(dir_path))
            {
                const std::string needle = "flow_podmode" + prefix;
                for (const auto& entry : boost::filesystem::directory_iterator(dir_path))
                {
                    if (!boost::filesystem::is_regular_file(entry.path())) continue;
                    const std::string name = entry.path().filename().string();
                    if (name.find(needle) == 0 && entry.path().extension() == ".hdf5")
                    {
                        boost::filesystem::remove(entry.path());
                    }
                }
            }
        }
        world.barrier();
    }

    float_type compute_mode_error_sym(const std::string& mode_file, const std::string& out_prefix, int field_idx = 0)
    {
        simulation_.template read_h5<u_s_type>(mode_file, "u_s");
        this->template fit_reference_mode_to_u_ref<u_s_type>();
        return this->template compute_errors<u_s_type, u_ref_type, test_type>(out_prefix, field_idx);
    }

    float_type compute_mode_error_asym(const std::string& mode_file, const std::string& out_prefix, int field_idx = 0)
    {
        simulation_.template read_h5<u_a_type>(mode_file, "u_a");
        this->template fit_reference_mode_to_u_ref<u_a_type>();
        return this->template compute_errors<u_a_type, u_ref_type, test_type>(out_prefix, field_idx);
    }

    template<class NumericField>
    void fit_reference_mode_to_u_ref()
    {
        boost::mpi::communicator world;
        constexpr float_type     pi = static_cast<float_type>(3.14159265358979323846);
        const auto               bb_min = domain_->bounding_box().min();
        const auto               bb_max = domain_->bounding_box().max();
        const float_type         denom_x = std::max<float_type>(1e-12, static_cast<float_type>(bb_max[0] - bb_min[0]));
        const float_type         denom_y = std::max<float_type>(1e-12, static_cast<float_type>(bb_max[1] - bb_min[1]));

        float_type dot_mode1_local = 0.0, dot_mode2_local = 0.0, dot_mode3_local = 0.0;
        float_type nrm_mode1_local = 0.0, nrm_mode2_local = 0.0, nrm_mode3_local = 0.0;

        for (auto it = domain_->begin_leaves(); it != domain_->end_leaves(); ++it)
        {
            if (!it->locally_owned() || !it->has_data()) continue;
            if (it->is_correction()) continue;

            const int refinement_level = it->refinement_level();
            const float_type dx = domain_->dx_base() / std::pow(2.0, refinement_level);
            const float_type weight = std::pow(dx, domain_->dimension());

            for (auto& n : it->data())
            {
                const auto gc = n.global_coordinate();
                const float_type X = 2.0 * (static_cast<float_type>(gc[0] - bb_min[0]) / denom_x) - 1.0;
                const float_type Y = 2.0 * (static_cast<float_type>(gc[1] - bb_min[1]) / denom_y) - 1.0;
                const float_type mode1 = std::sin(1.0 * pi * X) * std::sin(1.0 * pi * Y);
                const float_type mode2 = std::sin(2.0 * pi * X) * std::sin(2.0 * pi * Y);
                const float_type mode3 = std::sin(3.0 * pi * X) * std::sin(3.0 * pi * Y);

                for (std::size_t f = 0; f < Dim; ++f)
                {
                    const float_type num = n(NumericField::tag(), static_cast<int>(f));
                    dot_mode1_local += num * mode1 * weight;
                    dot_mode2_local += num * mode2 * weight;
                    dot_mode3_local += num * mode3 * weight;
                    nrm_mode1_local += mode1 * mode1 * weight;
                    nrm_mode2_local += mode2 * mode2 * weight;
                    nrm_mode3_local += mode3 * mode3 * weight;
                }
            }
        }

        float_type dot_mode1 = 0.0, dot_mode2 = 0.0, dot_mode3 = 0.0;
        float_type nrm_mode1 = 0.0, nrm_mode2 = 0.0, nrm_mode3 = 0.0;
        boost::mpi::all_reduce(world, dot_mode1_local, dot_mode1, std::plus<float_type>());
        boost::mpi::all_reduce(world, dot_mode2_local, dot_mode2, std::plus<float_type>());
        boost::mpi::all_reduce(world, dot_mode3_local, dot_mode3, std::plus<float_type>());
        boost::mpi::all_reduce(world, nrm_mode1_local, nrm_mode1, std::plus<float_type>());
        boost::mpi::all_reduce(world, nrm_mode2_local, nrm_mode2, std::plus<float_type>());
        boost::mpi::all_reduce(world, nrm_mode3_local, nrm_mode3, std::plus<float_type>());

        const std::array<float_type, 3> dots = {dot_mode1, dot_mode2, dot_mode3};
        const std::array<float_type, 3> nrms = {nrm_mode1, nrm_mode2, nrm_mode3};
        int best_mode = 0;
        if (std::abs(dots[1]) > std::abs(dots[best_mode])) best_mode = 1;
        if (std::abs(dots[2]) > std::abs(dots[best_mode])) best_mode = 2;
        const float_type denom = nrms[static_cast<std::size_t>(best_mode)];
        const float_type dot = dots[static_cast<std::size_t>(best_mode)];
        const float_type scale = (denom > 1e-30) ? (dot / denom) : 0.0;

        for (auto it = domain_->begin_leaves(); it != domain_->end_leaves(); ++it)
        {
            if (!it->locally_owned() || !it->has_data()) continue;
            if (it->is_correction()) continue;

            for (auto& n : it->data())
            {
                const auto gc = n.global_coordinate();
                const float_type X = 2.0 * (static_cast<float_type>(gc[0] - bb_min[0]) / denom_x) - 1.0;
                const float_type Y = 2.0 * (static_cast<float_type>(gc[1] - bb_min[1]) / denom_y) - 1.0;
                const float_type mode1 = std::sin(1.0 * pi * X) * std::sin(1.0 * pi * Y);
                const float_type mode2 = std::sin(2.0 * pi * X) * std::sin(2.0 * pi * Y);
                const float_type mode3 = std::sin(3.0 * pi * X) * std::sin(3.0 * pi * Y);
                float_type exact_basis = mode1;
                if (best_mode == 1) exact_basis = mode2;
                if (best_mode == 2) exact_basis = mode3;
                const float_type exact_val = scale * exact_basis;

                for (std::size_t f = 0; f < Dim; ++f)
                {
                    n(u_ref, static_cast<int>(f)) = exact_val;
                }
            }
        }
    }

    template<class NumericField>
    std::pair<float_type, float_type> compute_error_against_u_ref(int field_idx = 0)
    {
        boost::mpi::communicator world;
        float_type               l2_local = 0.0;
        float_type               linf_local = 0.0;

        for (auto it = domain_->begin_leaves(); it != domain_->end_leaves(); ++it)
        {
            if (!it->locally_owned() || !it->has_data()) continue;
            if (it->is_correction()) continue;

            const int refinement_level = it->refinement_level();
            const float_type dx = domain_->dx_base() / std::pow(2.0, refinement_level);
            const float_type weight = std::pow(dx, domain_->dimension());

            for (auto& n : it->data())
            {
                const float_type num = n(NumericField::tag(), field_idx);
                const float_type ref = n(u_ref, field_idx);
                const float_type e = num - ref;
                l2_local += e * e * weight;
                linf_local = std::max(linf_local, std::abs(e));
            }
        }

        float_type l2_global = 0.0;
        float_type linf_global = 0.0;
        boost::mpi::all_reduce(world, l2_local, l2_global, std::plus<float_type>());
        boost::mpi::all_reduce(world, linf_local, linf_global, boost::mpi::maximum<float_type>());
        return {std::sqrt(l2_global), linf_global};
    }

    template<class NumericField>
    void write_actual_modes_to_u_ref(const std::string& prefix, const std::string& field_name)
    {
        boost::mpi::communicator world;
        const std::string        out_dir =
            simulation_.dictionary_->get_dictionary("output")->template get<std::string>("directory");
        const int n_total = simulation_.dictionary_->template get_or<int>("nTotal", 0);
        const int n_modes = std::min(n_total, 10);
        const std::string mode_err_file = "./" + out_dir + "/mode_error" + prefix + ".txt";
        if (world.rank() == 0)
        {
            std::ofstream ofs(mode_err_file);
            ofs << std::scientific << std::setprecision(16);
            ofs << "# mode L2 LInf\n";
        }
        world.barrier();

        for (int i = 0; i < n_modes; ++i)
        {
            const std::string mode_name = "podmode" + prefix + std::to_string(i);
            const std::string flow_file = "./" + out_dir + "/flow_" + mode_name + ".hdf5";
            if (!boost::filesystem::exists(flow_file)) { continue; }

            simulation_.template read_h5<NumericField>(flow_file, field_name);
            world.barrier();

            this->template fit_reference_mode_to_u_ref<NumericField>();
            const auto mode_err = this->template compute_error_against_u_ref<NumericField>(0);
            if (world.rank() == 0)
            {
                std::ofstream ofs(mode_err_file, std::ios::app);
                ofs << i << " " << mode_err.first << " " << mode_err.second << "\n";
            }

            if (i == 0)
            {
                if constexpr (std::is_same<NumericField, u_s_type>::value)
                {
                    mode0_error_sym_ = mode_err.second;
                }
                else if constexpr (std::is_same<NumericField, u_a_type>::value)
                {
                    mode0_error_asym_ = mode_err.second;
                }
            }

            world.barrier();
            simulation_.write(mode_name);
            world.barrier();
        }
    }

  private:
    int                      nLevelRefinement_ = 0; // Number of refinement levels
    boost::mpi::communicator client_comm_;
    std::vector<key_id_t>    ref_keys_;  //referfecne keys local to rank
    std::vector<int>         ref_leafs_; //reference leafs local to rank
    std::string              restart_tree_dir_;
    std::string              restart_field_dir_;
    float_type               mode0_error_sym_ = -1.0;
    float_type               mode0_error_asym_ = -1.0;
};
} // namespace iblgf
#endif
