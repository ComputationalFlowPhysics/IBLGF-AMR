#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>

#include "common_tree.hpp"
#include <iblgf/dictionary/dictionary.hpp>
#include <iblgf/solver/modal_analysis/merge_trees.hpp>

namespace iblgf
{

TEST(Symfield3DIntegrationTest, ProducesConsistentSymmetricAndAntisymmetricFields)
{
    boost::mpi::communicator world;
    if (world.size() < 2)
    {
        GTEST_SKIP() << "symfield 3D integration test requires at least 2 MPI ranks; got "
                     << world.size() << " ranks.";
    }

    Dictionary dictionary("./configs/config_symgrid3D", 0, nullptr);
    auto ref_domain = std::make_unique<CommonTree>(&dictionary);
    MergeTrees<CommonTree>::initialize_linear_u_field(*ref_domain);
    ref_domain->symfield<CommonTree::u_type, CommonTree::u_type>(499);

    auto tree = ref_domain->tree();
    for (auto it = tree->begin(); it != tree->end(); ++it)
    {
        if (!it->locally_owned()) continue;
        if (!it->has_data()) continue;
        if (!it->is_leaf()) continue;
        if (it->is_correction()) continue;

        for (auto& node : it->data())
        {
            const auto gc = node.global_coordinate();
            EXPECT_NEAR(node(CommonTree::u_s, 0), 2 * gc[0], 1e-12);
            EXPECT_NEAR(node(CommonTree::u_s, 1), 2 * gc[1], 1e-12);
            EXPECT_NEAR(node(CommonTree::u_s, 2), 2 * gc[2], 1e-12);

            EXPECT_NEAR(node(CommonTree::u_a, 0), 0.0, 1e-12);
            EXPECT_NEAR(node(CommonTree::u_a, 1), 0.0, 1e-12);
            EXPECT_NEAR(node(CommonTree::u_a, 2), 0.0, 1e-12);
        }
    }
}

} // namespace iblgf
