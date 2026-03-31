#include <mpi.h>
#include <gtest/gtest.h>

#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/filesystem.hpp>

#include <cmath>
#include <algorithm>
#include <iostream>
#include <memory>
#include <vector>
#include <stdexcept>

#include <iblgf/dictionary/dictionary.hpp>
#include <iblgf/lgf/lgf_gl.hpp>

#include "../../setups/setup_base.hpp"

using namespace iblgf;
using namespace dictionary;

namespace iblgf
{
struct fmm_test_parameters
{
    static constexpr std::size_t Dim = 3;

    REGISTER_FIELDS
    (
    Dim,
        (
            (fmm_exact, float_type, 1, 1, 1, cell, true),
            (fmm_error, float_type, 1, 1, 1, cell, true)
        )
    )
};

struct FmmTestSetup : public SetupBase<FmmTestSetup, fmm_test_parameters>
{
    using super_type = SetupBase<FmmTestSetup, fmm_test_parameters>;
    using coordinate_t = typename super_type::coordinate_t;
    using domain_t = typename super_type::domain_t;
    using Fmm_t = typename super_type::Fmm_t;
    using fmm_mask_builder_t = typename super_type::fmm_mask_builder_t;
    using source_tmp_type = typename super_type::source_tmp_type;
    using target_tmp_type = typename super_type::target_tmp_type;
    using fmm_exact_type = typename super_type::fmm_exact_type;
    using fmm_error_type = typename super_type::fmm_error_type;
    using octant_t = typename domain_t::octant_t;
    using MASK_TYPE = typename octant_t::MASK_TYPE;
    using lgf_t = lgf::LGF_GL<Dim>;

    FmmTestSetup(Dictionary* d)
    : super_type(d, [this](auto dict, auto domain) {
        return this->initialize_domain(dict, domain);
    })
    {
        if (domain_->is_client()) client_comm_ = client_comm_.split(1);
        else client_comm_ = client_comm_.split(0);

        global_refinement_ =
            simulation_.dictionary_->template get_or<int>("global_refinement", 0);

        domain_->register_refinement_condition() =
            [this](auto octant, int diff_level) {
                return this->refinement(octant, diff_level);
            };

        domain_->init_refine(
            d->get_dictionary("simulation_parameters")->template get_or<int>("nLevels", 0),
            global_refinement_,
            0);

        domain_->distribute<fmm_mask_builder_t, fmm_mask_builder_t>();

        initialize_fields();
    }

    float_type run()
    {
        if (domain_->is_server()) return 0.0;

        fmm_mask_builder_t::fmm_clean_load(domain_.get());
        fmm_mask_builder_t::fmm_lgf_mask_build(domain_.get(), false);

        const int Nb = domain_->block_extent()[0] + 2;
        Fmm_t fmm(domain_.get(), Nb);
        lgf_t lgf_kernel;

        const int level = domain_->tree()->base_level();

        fmm.template apply<source_tmp_type, target_tmp_type>(
            domain_.get(),
            &lgf_kernel,
            level,
            false,
            1.0,
            MASK_TYPE::AMR2AMR);

        float_type l2_error = 0.0;
        float_type linf_error = 0.0;
        std::size_t count = 0;

        for (auto it = domain_->begin_leaves(); it != domain_->end_leaves(); ++it)
        {
            if (!it->locally_owned() || !it->has_data()) continue;
            if (it->is_correction()) continue;

            auto& numerical_data = it->data_r(target_tmp_type::tag()).linalg_data();
            auto& exact_data = it->data_r(fmm_exact_type::tag()).linalg_data();
            auto& error_data = it->data_r(fmm_error_type::tag()).linalg_data();

            const auto block = it->data_r(target_tmp_type::tag()).real_block();
            const auto base = block.base();
            const auto extent = block.extent();

            for (int k = 1; k < extent[2] - 1; ++k)
            for (int j = 1; j < extent[1] - 1; ++j)
            for (int i = 1; i < extent[0] - 1; ++i)
            {
                coordinate_t x(0);
                x[0] = base[0] + i;
                x[1] = base[1] + j;
                x[2] = base[2] + k;

                coordinate_t rel(0);
                rel[0] = x[0] - source_coord_[0];
                rel[1] = x[1] - source_coord_[1];
                rel[2] = x[2] - source_coord_[2];

                const float_type exact = lgf_kernel.get(rel);
                const float_type numerical = numerical_data(i, j, k);
                const float_type err = std::abs(numerical - exact);

                exact_data(i, j, k) = exact;
                error_data(i, j, k) = numerical - exact;

                l2_error += err * err;
                linf_error = std::max(linf_error, err);
                ++count;
            }
        }

        float_type l2_global = 0.0;
        float_type linf_global = 0.0;
        std::size_t count_global = 0;

        boost::mpi::all_reduce(client_comm_, l2_error, l2_global, std::plus<float_type>());
        boost::mpi::all_reduce(client_comm_, linf_error, linf_global, boost::mpi::maximum<float_type>());
        boost::mpi::all_reduce(client_comm_, count, count_global, std::plus<std::size_t>());

        l2_global = std::sqrt(l2_global / static_cast<float_type>(count_global));

        std::cout << "FMM L2 error norm: " << l2_global << std::endl;
        std::cout << "FMM Linf error norm: " << linf_global << std::endl;

        return linf_global;
    }

    void initialize_fields()
    {
        if (domain_->is_server()) return;

        for (auto it = domain_->begin(); it != domain_->end(); ++it)
        {
            if (!it->has_data() || !it->data().is_allocated()) continue;

            std::fill(it->data_r(source_tmp_type::tag()).begin(),
                      it->data_r(source_tmp_type::tag()).end(), 0.0);
            std::fill(it->data_r(target_tmp_type::tag()).begin(),
                      it->data_r(target_tmp_type::tag()).end(), 0.0);
            std::fill(it->data_r(fmm_exact_type::tag()).begin(),
                      it->data_r(fmm_exact_type::tag()).end(), 0.0);
            std::fill(it->data_r(fmm_error_type::tag()).begin(),
                      it->data_r(fmm_error_type::tag()).end(), 0.0);
        }

        bool source_set = false;

        for (auto it = domain_->begin_leaves(); it != domain_->end_leaves(); ++it)
        {
            if (!it->locally_owned() || !it->has_data()) continue;
            if (it->is_correction()) continue;
            if (source_set) break;

            auto& src = it->data_r(source_tmp_type::tag()).linalg_data();
            const auto block = it->data_r(source_tmp_type::tag()).real_block();
            const auto base = block.base();

            src(1, 1, 1) = 1.0;

            source_coord_[0] = base[0] + 1;
            source_coord_[1] = base[1] + 1;
            source_coord_[2] = base[2] + 1;

            source_set = true;
        }

        if (!source_set)
            throw std::runtime_error("FmmTestSetup: failed to place source");
    }

    template<class OctantType>
    bool refinement(OctantType* it, int diff_level, bool use_all = false) const noexcept
    {
        return false;
    }

    std::vector<coordinate_t> initialize_domain(Dictionary* d, domain_t* domain)
    {
        return domain->construct_basemesh_blocks(d, domain->block_extent());
    }

private:
    boost::mpi::communicator client_comm_;
    int global_refinement_ = 0;
    coordinate_t source_coord_ = coordinate_t(0);
};
}

static int g_argc = 0;
static char** g_argv = nullptr;

TEST(FMMTest, SingleSourceMatchesDirectLGF)
{
    boost::mpi::communicator world;

    for (auto& entry : boost::filesystem::directory_iterator("./"))
    {
        auto s = entry.path();

        if (s.filename().string().rfind("config", 0) == 0)
        {
            if (world.rank() == 0)
                std::cout << "------------- Testing on config file "
                          << s.filename() << " -------------" << std::endl;

            auto dict = std::make_shared<Dictionary>(s.string(), g_argc, g_argv);
            iblgf::FmmTestSetup test(dict.get());

            const float_type linf_error = test.run();
            world.barrier();

            EXPECT_LT(linf_error, 1e-8);
        }
    }
}

int main(int argc, char** argv)
{
    g_argc = argc;
    g_argv = argv;

    ::testing::InitGoogleTest(&argc, argv);
    MPI_Init(&argc, &argv);

    int rc = RUN_ALL_TESTS();

    MPI_Finalize();
    return rc;
}