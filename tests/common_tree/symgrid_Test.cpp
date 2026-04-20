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

    Dictionary dictionary("./configs/common_tree_symgrid.cfg", 0, nullptr);
    auto       sim_dict = dictionary.get_dictionary("simulation_parameters");
    const int  ref_levels = sim_dict->template get_or<int>("nLevels", 0);

    auto ref_domain = std::make_unique<CommonTree>(&dictionary);

    std::vector<CommonTree::domain_t::key_t> octs;
    std::vector<int>                         level_change;
    auto domain_dict = dictionary.get_dictionary("simulation_parameters")->get_dictionary("domain");
    const auto bd_base = domain_dict->template get<int, Dim>("bd_base");
    const auto block_extent = domain_dict->template get<int>("block_extent");
    const int  mirror_span = (-2 * bd_base[1]) / block_extent;
    for (int i = ref_levels; i >= 0; --i)
    {
        octs.clear();
        level_change.clear();
        auto ref_tree = ref_domain->tree();
        MergeTrees<CommonTree>::symgrid(ref_tree, i, octs, level_change, mirror_span, false);
        const int tt = i == 0 ? 1 : -1;
        MergeTrees<CommonTree>::run_adapt_from_keys_distributed(
            *ref_domain, tt, octs, level_change, world);
    }

    for (int i = ref_levels; i >= 0; --i)
    {
        octs.clear();
        level_change.clear();
        auto ref_tree = ref_domain->tree();
        MergeTrees<CommonTree>::symgrid(ref_tree, i, octs, level_change, mirror_span, false);
        const int tt = i == 0 ? 1 : -1;
        MergeTrees<CommonTree>::run_adapt_from_keys_distributed(
            *ref_domain, tt, octs, level_change, world);
    }

    auto tree = ref_domain->tree();
    int  local_missing_pairs = 0;
    int  local_checked_leafs = 0;

    for (auto it = tree->begin(); it != tree->end(); ++it)
    {
        if (!it->has_data()) continue;
        if (!it->is_leaf()) continue;
        if (it->is_correction()) continue;

        const auto coord = it->tree_coordinate();
        const auto level = it->key().level();
        const int  ref_level = static_cast<int>(level) - tree->base_level();
        if (ref_level < 0) continue;

        auto opposite_coord = coord;
        opposite_coord[1] = mirror_span * (1 << ref_level) - (coord[1] + 1);

        auto opposite = tree->find_octant(CommonTree::domain_t::key_t(opposite_coord, level));
        if (!opposite || !opposite->has_data() || !opposite->is_leaf() ||
            opposite->is_correction())
        {
            ++local_missing_pairs;
        }
        ++local_checked_leafs;
    }

    int global_missing_pairs = 0;
    int global_checked_leafs = 0;
    boost::mpi::all_reduce(
        world, local_missing_pairs, global_missing_pairs, std::plus<int>());
    boost::mpi::all_reduce(
        world, local_checked_leafs, global_checked_leafs, std::plus<int>());

    ASSERT_GT(global_checked_leafs, 0) << "Test did not check any leaf octants.";
    EXPECT_EQ(global_missing_pairs, 0)
        << "Final grid is not symmetric in y; found leaf octants without mirrored partners.";
}

} // namespace iblgf
