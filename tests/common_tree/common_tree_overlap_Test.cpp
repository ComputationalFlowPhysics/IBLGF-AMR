#include <gtest/gtest.h>

#include <memory>
#include <set>

#include <boost/mpi/environment.hpp>

#include "common_tree.hpp"
#include <iblgf/dictionary/dictionary.hpp>
#include <iblgf/solver/modal_analysis/merge_trees.hpp>

namespace iblgf
{
namespace
{

using key_id_t = CommonTree::key_id_t;

std::set<key_id_t> set_intersection(const std::set<key_id_t>& a, const std::set<key_id_t>& b)
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
        GTEST_SKIP() << "common_tree overlap 3D test requires at least 2 MPI ranks; got "
                     << world.size() << " ranks.";
    }

    Dictionary dict1("./configs/common_tree_1.cfg", 0, nullptr);
    Dictionary dict2("./configs/common_tree_2.cfg", 0, nullptr);
    Dictionary dict3("./configs/common_tree_3.cfg", 0, nullptr);

    auto domain1 = std::make_unique<CommonTree>(&dict1);
    auto domain2 = std::make_unique<CommonTree>(&dict2);
    auto domain3 = std::make_unique<CommonTree>(&dict3);

    domain1->run(101, false);
    domain2->run(102, false);
    domain3->run(103, false);
    world.barrier();

    Dictionary dict_ref("./configs/common_tree_driver.cfg", 0, nullptr);
    auto       merger = MergeTrees<CommonTree>(&dict_ref);
    auto       ref_domain = merger.get_common_tree();

    auto keys1 = MergeTrees<CommonTree>::get_tree_keys(*domain1);
    auto keys2 = MergeTrees<CommonTree>::get_tree_keys(*domain2);
    auto keys3 = MergeTrees<CommonTree>::get_tree_keys(*domain3);

    auto overlap12 = set_intersection(keys1, keys2);
    auto overlap123 = set_intersection(overlap12, keys3);

    EXPECT_FALSE(overlap123.empty())
        << "Expected non-empty overlap region for the 3 AMR configs.";

    auto final_keys = MergeTrees<CommonTree>::get_tree_keys(*ref_domain);
    EXPECT_EQ(overlap123.size(), final_keys.size())
        << "Final tree should have same number of leaves as the 3-way overlap.";
    EXPECT_EQ(overlap123, final_keys)
        << "Final tree leaves must exactly match the 3-way overlap.";

    ref_domain->run(199, false);
}

} // namespace iblgf

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    boost::mpi::environment env(argc, argv);
    return RUN_ALL_TESTS();
}
