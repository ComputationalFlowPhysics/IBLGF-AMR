#include <gtest/gtest.h>

#include <functional>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>

#include "common_tree.hpp"
#include <iblgf/dictionary/dictionary.hpp>
#include <iblgf/solver/modal_analysis/merge_trees.hpp>

namespace iblgf
{

TEST(Symgrid3DUnitTest, ProducesYMirrorSymmetricLeafGrid)
{
    boost::mpi::communicator world;
    if (world.size() < 2)
    {
        GTEST_SKIP() << "symgrid 3D test requires at least 2 MPI ranks; got "
                     << world.size() << " ranks.";
    }

    {
        Dictionary dictionary("./configs/common_tree_3.cfg", 0, nullptr);
        auto domain = std::make_unique<CommonTree>(&dictionary);
        domain->run(301, false);
    }

    Dictionary dictionary("./configs/config_symgrid3D", 0, nullptr);
    auto merger = MergeTrees<CommonTree>(&dictionary);
    auto ref_domain = merger.ref_to_symmetric_ref();

    auto tree = ref_domain->tree();
    int local_missing_pairs = 0;
    int local_checked_leafs = 0;

    auto domain_dict =
        dictionary.get_dictionary("simulation_parameters")->get_dictionary("domain");
    const auto bd_base = domain_dict->template get<int, Dim>("bd_base");
    const auto block_extent = domain_dict->template get<int>("block_extent");
    const int mirror_span = (-2 * bd_base[1]) / block_extent;
    if(world.rank() == 0) {
        std::cout << "bd_base: " << bd_base[0] << ", " << bd_base[1] << ", " << bd_base[2] << std::endl;
        std::cout << "block_extent: " << block_extent << std::endl;
        std::cout << "mirror_span: " << mirror_span << std::endl;

    for (auto it = tree->begin(); it != tree->end(); ++it)
    {
        if (!it->has_data()) continue;
        if (!it->is_leaf()) continue;

        const auto coord = it->tree_coordinate();
        const auto level = it->key().level();
        const int ref_level = static_cast<int>(level) - tree->base_level();
        if (ref_level < 0) continue;

        auto opposite_coord = coord;
        opposite_coord[1] = mirror_span * (1 << ref_level) - (coord[1] + 1);

        auto opposite = tree->find_octant(CommonTree::domain_t::key_t(opposite_coord, level));
        if (!opposite || !opposite->has_data() || !opposite->is_leaf())
        {
            ++local_missing_pairs;
        }
        ++local_checked_leafs;
    }

    ASSERT_GT(local_checked_leafs, 0) << "Test did not check any leaf octants.";
    EXPECT_EQ(local_missing_pairs, 0)
        << "Final grid is not symmetric in y; found leaf octants without mirrored partners.";
    }
    ref_domain->symfield<CommonTree::u_type, CommonTree::u_type>(399);
}

} // namespace iblgf
