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

} // namespace

int main(int argc, char* argv[])
{
    boost::mpi::environment  env(argc, argv);
    boost::mpi::communicator world;

    iblgf::Dictionary dictionary(get_input_path(argc, argv), argc, argv);
    auto              sim_dict = dictionary.get_dictionary("simulation_parameters");
    auto              domain_dict = sim_dict->get_dictionary("domain");

    const std::string tree_ref_file = sim_dict->template get<std::string>("tree_ref_file");
    const std::string flow_ref_file = sim_dict->template get<std::string>("flow_ref_file");
    const int         ref_levels = sim_dict->template get_or<int>("nLevels", 0);

    const auto bd_base = domain_dict->template get<int, iblgf::Dim>("bd_base");
    const auto block_extent = domain_dict->template get<int>("block_extent");
    const int  mirror_span = (-2 * bd_base[1]) / block_extent;

    auto ref_domain = std::make_unique<iblgf::CommonTree>(&dictionary, tree_ref_file, flow_ref_file);

    std::vector<iblgf::CommonTree::domain_t::key_t> octs;
    std::vector<int>                                level_change;

    for (int pass = 0; pass < 2; ++pass)
    {
        for (int level = ref_levels; level >= 0; --level)
        {
            octs.clear();
            level_change.clear();

            auto ref_tree = ref_domain->tree();
            iblgf::MergeTrees<iblgf::CommonTree>::symgrid(
                ref_tree, level, octs, level_change, mirror_span, true);

            const int output_tag = level == 0 ? 1 : -1;
            iblgf::MergeTrees<iblgf::CommonTree>::run_adapt_from_keys_distributed(
                *ref_domain, output_tag, octs, level_change, world);
        }

        if (world.rank() == 0 && pass == 0)
        {
            std::cout << "Adaptation to reference levels completed." << std::endl;
        }
    }

    if (world.rank() == 0) { std::cout << "Symmetry adaptation completed." << std::endl; }

    ref_domain->initialize();
    ref_domain->symfield<iblgf::CommonTree::u_type, iblgf::CommonTree::u_type>(2);

    if (world.rank() == 0) { std::cout << "Symmetry completed." << std::endl; }

    return 0;
}
