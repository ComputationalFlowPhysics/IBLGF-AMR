#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <set>

#include "common_tree.hpp"
#include <iblgf/dictionary/dictionary.hpp>
#include <iblgf/solver/modal_analysis/merge_trees.hpp>

namespace iblgf
{
namespace
{

using key_id_t = CommonTree::key_id_t;

std::set<key_id_t> set_intersection(const std::set<key_id_t>& a,
                                    const std::set<key_id_t>& b)
{
    std::set<key_id_t> out;
    for (const auto& k : a)
    {
        if (b.find(k) != b.end()) { out.insert(k); }
    }
    return out;
}

} // namespace

TEST(CommonTree3DUnitTest, OverlapFromThreeDifferentDomainsIsCorrect)
{
    boost::mpi::communicator world;
    if (world.size() < 2)
    {
        GTEST_SKIP() << "common_tree overlap test requires at least 2 MPI ranks; got "
                     << world.size() << " ranks.";
    }

    Dictionary dict1("./configs/common_tree_overlap_1.cfg", 0, nullptr);
    Dictionary dict2("./configs/common_tree_overlap_2.cfg", 0, nullptr);
    Dictionary dict3("./configs/common_tree_overlap_3.cfg", 0, nullptr);

    auto domain1 = std::make_unique<CommonTree>(&dict1);
    auto domain2 = std::make_unique<CommonTree>(&dict2);
    auto domain3 = std::make_unique<CommonTree>(&dict3);

    const auto keys1 = MergeTrees<CommonTree>::globalize_key_set(
        world, MergeTrees<CommonTree>::collect_physical_leaf_key_ids(*domain1));
    const auto keys2 = MergeTrees<CommonTree>::globalize_key_set(
        world, MergeTrees<CommonTree>::collect_physical_leaf_key_ids(*domain2));
    const auto keys3 = MergeTrees<CommonTree>::globalize_key_set(
        world, MergeTrees<CommonTree>::collect_physical_leaf_key_ids(*domain3));

    const auto overlap12 = set_intersection(keys1, keys2);
    const auto overlap123 = set_intersection(overlap12, keys3);

    ASSERT_FALSE(overlap123.empty())
        << "Expected non-empty overlap region for the 3 AMR configs.";

    std::vector<CommonTree::domain_t::key_t> octs_to_delete;
    std::vector<int>                         level_change;

    auto tree1 = domain1->tree();
    auto tree2 = domain2->tree();
    MergeTrees<CommonTree>::getDelOcts_distributed(
        tree1, tree2, octs_to_delete, level_change, world);
    MergeTrees<CommonTree>::run_adapt_del_distributed(
        *domain1, octs_to_delete, level_change, world);

    tree1 = domain1->tree();
    auto tree3 = domain3->tree();
    octs_to_delete.clear();
    level_change.clear();
    MergeTrees<CommonTree>::getDelOcts_distributed(
        tree1, tree3, octs_to_delete, level_change, world);
    MergeTrees<CommonTree>::run_adapt_del_distributed(
        *domain1, octs_to_delete, level_change, world);

    const auto keys_final = MergeTrees<CommonTree>::globalize_key_set(
        world, MergeTrees<CommonTree>::collect_physical_leaf_key_ids(*domain1));
    EXPECT_EQ(keys_final, overlap123)
        << "Common tree leaves must exactly match the 3-way overlap.";

    EXPECT_TRUE(std::includes(keys1.begin(), keys1.end(), keys_final.begin(), keys_final.end()))
        << "Final grid must be a subset of config-1 grid.";
    EXPECT_TRUE(std::includes(keys2.begin(), keys2.end(), keys_final.begin(), keys_final.end()))
        << "Final grid must be a subset of config-2 grid.";
    EXPECT_TRUE(std::includes(keys3.begin(), keys3.end(), keys_final.begin(), keys_final.end()))
        << "Final grid must be a subset of config-3 grid.";
}

} // namespace iblgf
