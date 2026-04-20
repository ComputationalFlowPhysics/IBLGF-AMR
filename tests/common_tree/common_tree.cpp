#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>

#include <iostream>
#include <string>
#include <vector>

#include "common_tree.hpp"
#include <iblgf/dictionary/dictionary.hpp>
#include <iblgf/solver/modal_analysis/merge_trees.hpp>

namespace
{

std::string get_input_path(int argc, char* argv[])
{
    if (argc > 1 && argv[1][0] != '-') { return argv[1]; }
    return "./configFile";
}

std::string flow_file_path(const std::string& dir, const int idx)
{
    return "./" + dir + "/flowTime_" + std::to_string(idx) + ".hdf5";
}

std::string tree_file_path(const std::string& dir, const int idx)
{
    return "./" + dir + "/tree_info_" + std::to_string(idx) + ".bin";
}

} // namespace

int main(int argc, char* argv[])
{
    boost::mpi::environment  env(argc, argv);
    boost::mpi::communicator world;

    iblgf::Dictionary dictionary(get_input_path(argc, argv), argc, argv);
    auto              sim_dict = dictionary.get_dictionary("simulation_parameters");
    auto              dict_out = sim_dict->get_dictionary("output");

    const std::string pp_dir = dict_out->template get<std::string>("directory");
    const std::string dir = dict_out->template get_or<std::string>("field_dir", pp_dir);

    const int idx_start = sim_dict->template get_or<int>("nStart", 100);
    const int n_total = sim_dict->template get_or<int>("nTotal", 100);
    const int nskip = sim_dict->template get_or<int>("nskip", 100);

    const std::string flow_file_0 = flow_file_path(dir, idx_start);
    const std::string tree_file_0 = tree_file_path(dir, idx_start);

    if (world.rank() == 0)
    {
        std::cout << "Initial snapshot: " << tree_file_0 << " / " << flow_file_0 << std::endl;
        std::cout << "nTotal=" << n_total << ", nskip=" << nskip << ", nStart=" << idx_start
                  << std::endl;
    }

    auto ref_domain = std::make_unique<iblgf::CommonTree>(&dictionary, tree_file_0, flow_file_0);

    std::vector<iblgf::CommonTree::domain_t::key_t> octs;
    std::vector<int>                                level_change;

    for (int i = 1; i < n_total; ++i)
    {
        const int idx = idx_start + i * nskip;
        const std::string flow_file_i = flow_file_path(dir, idx);
        const std::string tree_file_i = tree_file_path(dir, idx);

        auto ref_tree = ref_domain->tree();
        auto domain_i = std::make_unique<iblgf::CommonTree>(&dictionary, tree_file_i, flow_file_i);
        auto tree_i = domain_i->tree();

        octs.clear();
        level_change.clear();
        iblgf::MergeTrees<iblgf::CommonTree>::getDelOcts_distributed(
            ref_tree, tree_i, octs, level_change, world);
        iblgf::MergeTrees<iblgf::CommonTree>::run_adapt_del_distributed(
            *ref_domain, octs, level_change, world);
    }

    return 0;
}
