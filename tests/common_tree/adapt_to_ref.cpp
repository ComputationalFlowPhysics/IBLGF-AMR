#include <boost/mpi.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>

#include "common_tree.hpp"
#include <iostream>
#include <vector>
#include <iblgf/dictionary/dictionary.hpp>
#include "merge_trees.hpp"
using namespace iblgf;
int main(int argc, char* argv[])
{
    boost::mpi::environment  env(argc, argv);
    boost::mpi::communicator world;

    std::string input = "./";
    input += std::string("configFile");

    if (argc > 1 && argv[1][0] != '-') { input = argv[1]; }

    // Read in dictionary
    Dictionary dictionary(input, argc, argv);

    // Find output directory
    auto        dict_out = dictionary.get_dictionary("simulation_parameters")->get_dictionary("output");
    std::string dir = dict_out->template get<std::string>("directory");

    int idxStart = dictionary.get_dictionary("simulation_parameters")->get_or<int>("nStart", 100);
    int nTotal = dictionary.get_dictionary("simulation_parameters")->get_or<int>("nTotal", 100);
    int nskip = dictionary.get_dictionary("simulation_parameters")->get_or<int>("nskip", 100);

    std::string tree_ref_file_=dictionary.get_dictionary("simulation_parameters")->get<std::string>("tree_ref_file");
    std::string flow_ref_file_=dictionary.get_dictionary("simulation_parameters")->get<std::string>("flow_ref_file");

    int ref_levels_=dictionary.get_dictionary("simulation_parameters")->get_or<int>("nLevels", 0);
    std::vector<CommonTree::domain_t::key_t> octs;
    std::vector<int>                  level_change;
    // // std::unique_ptr<CommonTree>       last_tree;
    // // std::unique_ptr<CommonTree>       tree_1;
    // // std::unique_ptr<CommonTree>       new_tree;
    // std::string                       flow_file_1 = "./" + dir + "/flowTime_" + std::to_string(idxStart) + ".hdf5";
    // std::string                       tree_file_1 = "./" + dir + "/tree_info_" + std::to_string(idxStart) + ".bin";

    auto ref_domain_=std::make_unique<CommonTree>(&dictionary,tree_ref_file_, flow_ref_file_);
    octs.clear();
    level_change.clear();

    for (int i=0; i<nTotal;i++)
    {
        int timeIdx= idxStart + i * nskip;
        std::string flow_file_i= "./" + dir + "/flowTime_" + std::to_string(timeIdx) + ".hdf5";
        std::string tree_file_i= "./" + dir + "/tree_info_" + std::to_string(timeIdx) + ".bin";
        auto ref_tree= ref_domain_->tree();
        auto domain_i= std::make_unique<CommonTree>(&dictionary, tree_file_i, flow_file_i);
        auto tree_i= domain_i->tree();
        
        for(int j=ref_levels_; j>=0; --j)
        {
            octs.clear();
            level_change.clear();
            MergeTrees<CommonTree>::get_level_changes(ref_tree, tree_i, j, octs, level_change);
            // if(octs.empty()) continue;
            std::cout << "Adapting to ref level " << j << " with " << octs.size() << " octs." << std::endl;
            domain_i->run_adapt_from_keys<CommonTree::u_type>(timeIdx, octs, level_change);


        }
        // we want to adapt domain_i to ref_domain and have snapshot of new grid at that time

        // MergeTrees<CommonTree>::getDelOcts(ref_tree, tree_i, octs, level_change);


        // ref_domain_->run_adapt_del(octs, level_change);


    }
}