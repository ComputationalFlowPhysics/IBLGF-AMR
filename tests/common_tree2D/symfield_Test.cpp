#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <functional>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>

#include "common_tree.hpp"
#include <iblgf/dictionary/dictionary.hpp>
#include <iblgf/solver/modal_analysis/merge_trees.hpp>

namespace iblgf
{

TEST(Symfield2DIntegrationTest, ProducesConsistentSymmetricAndAntisymmetricFields)
{
    boost::mpi::communicator world;
    if (world.size() < 2)
    {
        GTEST_SKIP() << "symfield 2D integration test requires at least 2 MPI ranks; got "
                     << world.size() << " ranks.";
    }

    Dictionary dictionary("./configs/config_symgrid2D", 0, nullptr);
    auto       ref_domain = std::make_unique<CommonTree>(&dictionary);
    MergeTrees<CommonTree>::initialize_linear_u_field(*ref_domain);
    ref_domain->symfield<CommonTree::u_type, CommonTree::u_type>(499);

    double local_max_err_sym = 0.0;
    double local_max_err_anti = 0.0;
    double local_norm_u = 0.0;
    double local_norm_u_sym = 0.0;

    auto tree = ref_domain->tree();
    for (auto it = tree->begin(); it != tree->end(); ++it)
    {
        if (!it->locally_owned()) continue;
        if (!it->has_data()) continue;
        if (!it->is_leaf()) continue;
        if (it->is_correction()) continue;

        for (auto& node : it->data())
        {
            auto u_s_x=node(CommonTree::u_s, 0);
            auto u_s_y=node(CommonTree::u_s, 1);
            auto u_a_x=node(CommonTree::u_a, 0);
            auto u_a_y=node(CommonTree::u_a, 1);
            auto gc=node.global_coordinate();
            EXPECT_NEAR(u_s_x, 2*gc[0], 1e-12) //since u_s=(u+u_sym) and u is initialized to gc, u_s should equal 2*gc at all nodes
                << "u_s_x must equal x coordinate at all checked nodes.";
            EXPECT_NEAR(u_s_y, 2*gc[1], 1e-12)
                << "u_s_y must equal 0 coordinate at all checked nodes.";
            EXPECT_NEAR(u_a_x, 0, 1e-12)
                << "u_a_x must equal zero at all checked nodes.";
            EXPECT_NEAR(u_a_y, 0, 1e-12)
                << "u_a_y must equal y coordinate at all checked nodes.";
        }
    }

}

} // namespace iblgf
