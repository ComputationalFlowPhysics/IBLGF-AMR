#include <gtest/gtest.h>

#include <cstdio>
#include <fstream>
#include <memory>
#include <set>
#include <string>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>

#include "common_tree.hpp"
#include <iblgf/dictionary/dictionary.hpp>
#include <iblgf/solver/modal_analysis/merge_trees.hpp>

namespace iblgf
{
namespace
{

void remove_file_if_exists(const std::string& path)
{
    std::remove(path.c_str());
}

void write_adapt_driver_cfg(const std::string& path, int n_total,
    const std::string& restart_file, bool resume)
{
    std::ofstream out(path, std::ios::trunc);
    out << "simulation_parameters\n";
    out << "{\n";
    out << "    nLevels=3;\n";
    out << "    refinement_factor=0.125;\n";
    out << "    geometry=2circles;\n";
    out << "    nStart=201;\n";
    out << "    nTotal=" << n_total << ";\n";
    out << "    nskip=1;\n";
    out << "\n";
    out << "    output\n";
    out << "    {\n";
    out << "        directory=output_test2;\n";
    out << "        field_dir=output_test1;\n";
    out << "    }\n";
    out << "    tree_file_prefix = tree_info_common_tree_;\n";
    out << "    flow_file_prefix = flow_common_tree_;\n";
    out << "    resume=" << (resume ? "true" : "false") << ";\n";
    out << "    restart_file=" << restart_file << ";\n";
    out << "    do_symfield=false;\n";
    out << "\n";
    out << "    tree_ref_file = ./output_test2/tree_info_common_tree_299.bin;\n";
    out << "    flow_ref_file = ./output_test2/flow_common_tree_299.hdf5;\n";
    out << "\n";
    out << "    domain\n";
    out << "    {\n";
    out << "        bd_base = (-64,-64);\n";
    out << "        bd_extent = (128,128);\n";
    out << "        dx_base=0.25;\n";
    out << "        block_extent=8;\n";
    out << "\n";
    out << "        block\n";
    out << "        {\n";
    out << "            base=(-24,-24);\n";
    out << "            extent=(80,80);\n";
    out << "        }\n";
    out << "    }\n";
    out << "}\n";
}

void build_overlap_snapshots_201_to_203_and_ref_299()
{
    {
        Dictionary dict1("./configs/common_tree_1.cfg", 0, nullptr);
        Dictionary dict2("./configs/common_tree_2.cfg", 0, nullptr);
        Dictionary dict3("./configs/common_tree_3.cfg", 0, nullptr);

        auto domain1 = std::make_unique<CommonTree>(&dict1);
        auto domain2 = std::make_unique<CommonTree>(&dict2);
        auto domain3 = std::make_unique<CommonTree>(&dict3);

        domain1->run(201, false);
        domain2->run(202, false);
        domain3->run(203, false);
    }

    Dictionary dict_ref("./configs/adapt_to_ref_driver.cfg", 0, nullptr);
    auto       domain_ref = std::make_unique<CommonTree>(&dict_ref);
    domain_ref->run(299, false);
}

} // namespace

TEST(CommonTree2DRestartTest, AdaptDriverAPIWritesAndResumesRestartState)
{
    boost::mpi::communicator world;
    if (world.size() < 2)
    {
        GTEST_SKIP() << "adapt_to_ref restart 2D test requires at least 2 MPI ranks; got "
                     << world.size() << " ranks.";
    }

    build_overlap_snapshots_201_to_203_and_ref_299();
    world.barrier();

    const std::string cfg_phase1 = "./configs/adapt_to_ref_driver_restart_phase1_test.cfg";
    const std::string cfg_phase2 = "./configs/adapt_to_ref_driver_restart_phase2_test.cfg";
    const std::string restart_file = "./adapt_to_ref_restart_test_2d.txt";

    if (world.rank() == 0)
    {
        remove_file_if_exists(restart_file);
        write_adapt_driver_cfg(cfg_phase1, 1, restart_file, true);
        write_adapt_driver_cfg(cfg_phase2, 3, restart_file, true);
    }
    world.barrier();

    Dictionary dict_phase1(cfg_phase1, 0, nullptr);
    auto       merger_phase1 = MergeTrees<CommonTree>(&dict_phase1);
    merger_phase1.adapt_to_ref_with_restart();

    world.barrier();
    MergeTrees<CommonTree>::AdaptRestartState state1;
    bool                                      ok1 = false;
    if (world.rank() == 0)
    {
        ok1 = MergeTrees<CommonTree>::load_adapt_restart_state(restart_file, state1);
    }
    boost::mpi::broadcast(world, ok1, 0);
    boost::mpi::broadcast(world, state1.next_idx, 0);
    ASSERT_TRUE(ok1);
    EXPECT_EQ(state1.next_idx, 202);

    Dictionary dict_phase2(cfg_phase2, 0, nullptr);
    auto       merger_phase2 = MergeTrees<CommonTree>(&dict_phase2);
    merger_phase2.adapt_to_ref_with_restart();

    world.barrier();
    MergeTrees<CommonTree>::AdaptRestartState state2;
    bool                                      ok2 = false;
    if (world.rank() == 0)
    {
        ok2 = MergeTrees<CommonTree>::load_adapt_restart_state(restart_file, state2);
    }
    boost::mpi::broadcast(world, ok2, 0);
    boost::mpi::broadcast(world, state2.next_idx, 0);
    ASSERT_TRUE(ok2);
    EXPECT_EQ(state2.next_idx, 204);

    if (world.rank() == 0)
    {
        remove_file_if_exists(restart_file);
        remove_file_if_exists(cfg_phase1);
        remove_file_if_exists(cfg_phase2);
    }
}

} // namespace iblgf

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    boost::mpi::environment env(argc, argv);
    return RUN_ALL_TESTS();
}
