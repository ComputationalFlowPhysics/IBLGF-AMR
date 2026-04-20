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

TEST(CommonTree2DUnitTest, OverlapFromThreeDifferentDomainsIsCorrect)
{
    boost::mpi::communicator world;
    if (world.size() < 2)
    {
        GTEST_SKIP() << "common_tree_overlap_Test requires at least 2 MPI ranks; got "
                     << world.size() << " ranks.";
    }

    Dictionary dict1("./configs/common_tree_1.cfg", 0, nullptr);
    Dictionary dict2("./configs/common_tree_2.cfg", 0, nullptr);
    Dictionary dict3("./configs/common_tree_3.cfg", 0, nullptr);

    auto domain1 = std::make_unique<CommonTree>(&dict1);
    auto domain2 = std::make_unique<CommonTree>(&dict2);
    auto domain3 = std::make_unique<CommonTree>(&dict3);

    // Dump the 3 original grids for visual inspection:
    // output_overlap_1/flow_common_tree_101.hdf5
    // output_overlap_2/flow_common_tree_102.hdf5
    // output_overlap_3/flow_common_tree_103.hdf5
    domain1->run(101, false);
    domain2->run(102, false);
    domain3->run(103, false);
    world.barrier();

    Dictionary dict_ref("./configs/common_tree_driver.cfg", 0, nullptr);
    auto merger=MergeTrees<CommonTree>(&dict_ref);
    auto ref_domain_=merger.get_common_tree();
    
    auto keys1= MergeTrees<CommonTree>::get_tree_keys(*domain1);
    auto keys2= MergeTrees<CommonTree>::get_tree_keys(*domain2);
    auto keys3= MergeTrees<CommonTree>::get_tree_keys(*domain3);
    auto overlap12 = set_intersection(keys1, keys2);
    auto overlap123 = set_intersection(overlap12, keys3);
    EXPECT_FALSE(overlap123.empty())
        << "Expected non-empty overlap region for the 3 AMR configs.";
    auto final_keys= MergeTrees<CommonTree>::get_tree_keys(*ref_domain_);
    EXPECT_EQ(overlap123.size(), final_keys.size())
        << "Final tree should have same number of leaves as the 3-way overlap.";
    EXPECT_EQ(overlap123, final_keys)        << "Final tree leaves must exactly match the 3-way overlap.";

    ref_domain_->run(199, false);

       

}

} // namespace iblgf
int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    // Initialize MPI before any tests run
    boost::mpi::environment env(argc, argv);
    boost::mpi::communicator world;  // optional, can use in main if needed

    return RUN_ALL_TESTS(); // now MPI is already initialized
}
