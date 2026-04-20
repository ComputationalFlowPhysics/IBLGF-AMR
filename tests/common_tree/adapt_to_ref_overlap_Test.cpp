#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include <boost/mpi/collectives.hpp>

#include "common_tree.hpp"
#include <iblgf/dictionary/dictionary.hpp>
#include <iblgf/solver/modal_analysis/merge_trees.hpp>

namespace iblgf
{
namespace
{

using key_id_t = CommonTree::key_id_t;

std::set<key_id_t> collect_physical_leaf_keys(CommonTree& domain)
{
    std::set<key_id_t> keys;
    auto               tree = domain.tree();
    for (auto it = tree->begin(); it != tree->end(); ++it)
    {
        if (!it->has_data()) continue;
        if (!it->is_leaf()) continue;
        if (it->is_correction()) continue;
        if (!it->physical()) continue;
        keys.insert(it->key().id());
    }
    return keys;
}

std::set<key_id_t> globalize_key_set(const boost::mpi::communicator& world,
                                     const std::set<key_id_t>&       local)
{
    std::vector<key_id_t> local_vec(local.begin(), local.end());
    std::vector<std::vector<key_id_t>> gathered;
    boost::mpi::gather(world, local_vec, gathered, 0);

    std::vector<key_id_t> global_vec;
    if (world.rank() == 0)
    {
        std::set<key_id_t> merged;
        for (const auto& v : gathered)
        {
            merged.insert(v.begin(), v.end());
        }
        global_vec.assign(merged.begin(), merged.end());
    }
    boost::mpi::broadcast(world, global_vec, 0);
    return std::set<key_id_t>(global_vec.begin(), global_vec.end());
}

void initialize_linear_u_field(CommonTree& domain)
{
    auto tree = domain.tree();
    for (auto it = tree->begin(); it != tree->end(); ++it)
    {
        if (!it->locally_owned()) continue;
        if (!it->has_data()) continue;
        for (auto& node : it->data())
        {
            const auto gc = node.global_coordinate();
            node(CommonTree::u, 0) = gc[0];
            node(CommonTree::u, 1) = gc[1];
            node(CommonTree::u, 2) = gc[2];
        }
    }
}

double max_linear_u_error(CommonTree& domain)
{
    double local_max = 0.0;
    auto   tree = domain.tree();
    for (auto it = tree->begin(); it != tree->end(); ++it)
    {
        if (!it->locally_owned()) continue;
        if (!it->has_data()) continue;
        if (!it->is_leaf()) continue;
        if (it->is_correction()) continue;
        if (!it->physical()) continue;

        for (auto& node : it->data())
        {
            const auto gc = node.global_coordinate();
            local_max = std::max(local_max, std::abs(node(CommonTree::u, 0) - gc[0]));
            local_max = std::max(local_max, std::abs(node(CommonTree::u, 1) - gc[1]));
            local_max = std::max(local_max, std::abs(node(CommonTree::u, 2) - gc[2]));
        }
    }
    return local_max;
}

} // namespace

TEST(CommonTree3DUnitTest, AdaptToRefInterpolatesLinearFieldCorrectly)
{
    boost::mpi::communicator world;
    if (world.size() < 2)
    {
        GTEST_SKIP() << "adapt_to_ref overlap test requires at least 2 MPI ranks; got "
                     << world.size() << " ranks.";
    }

    Dictionary dict_ref("./configs/common_tree_overlap_1.cfg", 0, nullptr);
    auto       ref_domain = std::make_unique<CommonTree>(&dict_ref);
    const int  ref_levels =
        dict_ref.get_dictionary("simulation_parameters")->template get_or<int>("nLevels", 0);

    const auto ref_keys_global = globalize_key_set(world, collect_physical_leaf_keys(*ref_domain));
    ASSERT_FALSE(ref_keys_global.empty()) << "Reference tree must contain leaf keys.";

    const std::vector<std::string> candidates = {
        "./configs/common_tree_overlap_2.cfg",
        "./configs/common_tree_overlap_3.cfg"};

    for (const auto& cfg : candidates)
    {
        SCOPED_TRACE(cfg);
        Dictionary dict_i(cfg, 0, nullptr);
        auto       domain_i = std::make_unique<CommonTree>(&dict_i);

        initialize_linear_u_field(*domain_i);

        std::vector<CommonTree::domain_t::key_t> octs;
        std::vector<int>                         level_change;

        for (int j = ref_levels; j >= 0; --j)
        {
            auto ref_tree = ref_domain->tree();
            auto tree_i = domain_i->tree();
            octs.clear();
            level_change.clear();
            MergeTrees<CommonTree>::get_level_changes(ref_tree, tree_i, j, octs, level_change);
            boost::mpi::broadcast(world, octs, 0);
            boost::mpi::broadcast(world, level_change, 0);
            domain_i->run_adapt_from_keys<CommonTree::u_type>(-1, octs, level_change);
        }

        const auto final_keys_global =
            globalize_key_set(world, collect_physical_leaf_keys(*domain_i));
        EXPECT_EQ(final_keys_global, ref_keys_global)
            << "Adapted tree keys must match reference keys.";

        const double local_max_err = max_linear_u_error(*domain_i);
        double       global_max_err = 0.0;
        boost::mpi::all_reduce(
            world, local_max_err, global_max_err, boost::mpi::maximum<double>());
        EXPECT_LT(global_max_err, 1e-10)
            << "Interpolated u field must preserve linear profile.";
    }
}

} // namespace iblgf

