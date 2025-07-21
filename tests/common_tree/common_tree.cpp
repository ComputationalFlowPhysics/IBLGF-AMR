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

    std::vector<CommonTree::domain_t::key_t> octs;
    std::vector<int>                  level_change;
    // std::unique_ptr<CommonTree>       last_tree;
    // std::unique_ptr<CommonTree>       tree_1;
    // std::unique_ptr<CommonTree>       new_tree;
    std::string                       flow_file_1 = "./" + dir + "/flowTime_" + std::to_string(idxStart) + ".hdf5";
    std::string                       tree_file_1 = "./" + dir + "/tree_info_" + std::to_string(idxStart) + ".bin";

    auto ref_domain_=std::make_unique<CommonTree>(&dictionary,tree_file_1, flow_file_1);
    octs.clear();
    level_change.clear();

    for (int i=1; i<nTotal;i++)
    {
        std::string flow_file_i= "./" + dir + "/flowTime_" + std::to_string(idxStart + i * nskip) + ".hdf5";
        std::string tree_file_i= "./" + dir + "/tree_info_" + std::to_string(idxStart + i * nskip) + ".bin";
        auto ref_tree= ref_domain_->tree();
        auto domain_i= std::make_unique<CommonTree>(&dictionary, tree_file_i, flow_file_i);
        auto tree_i= domain_i->tree();
        octs.clear();
        level_change.clear();
        MergeTrees<CommonTree>::getDelOcts(ref_tree, tree_i, octs, level_change);


        ref_domain_->run_adapt_del(octs, level_change);


    }







//     last_tree = std::make_unique<CommonTree>(&dictionary, tree_file_1, flow_file_1);
//     int i = 1;

//     std::string flow_file_2 = "./" + dir + "/flowTime_" + std::to_string(idxStart + i * nskip) + ".hdf5";
//     // std::string flow_file_2="./"+dir+"/flowTime_"+std::to_string(idxStart+(i+1)*nskip)+".hdf5";
//     std::string tree_file_2 = "./" + dir + "/tree_info_" + std::to_string(idxStart + i * nskip) + ".bin";
//     // std::string tree_file_2="./"+dir+"/tree_info_"+std::to_string(idxStart+(i+1)*nskip)+".bin";
// //     tree_1 = std::make_unique<CommonTree>(&dictionary, tree_file_2, flow_file_2);
//     for (int i = 1; i < nTotal; i++)
//     {
//         std::string flow_file_3 = "./" + dir + "/flowTime_" + std::to_string(idxStart + i * nskip) + ".hdf5";
//         // std::string flow_file_2="./"+dir+"/flowTime_"+std::to_string(idxStart+(i+1)*nskip)+".hdf5";
//         std::string tree_file_3 = "./" + dir + "/tree_info_" + std::to_string(idxStart + i * nskip) + ".bin";
//         // std::string tree_file_2="./"+dir+"/tree_info_"+std::to_string(idxStart+(i+1)*nskip)+".bin";
//         tree_1 = std::make_unique<CommonTree>(&dictionary, tree_file_3, flow_file_3);
//         octs.clear();
//         leafs.clear();
//         auto tree0 = last_tree->tree();
//         auto tree1 = tree_1->tree();
//         MergeTrees<CommonTree>::merger(tree0, tree1, octs, leafs);
//         tree0=nullptr; // Clear tree0 to avoid dangling pointer issues
//         tree1=nullptr; // Clear tree1 to avoid dangling pointer issues
//         // MergeTrees<CommonTree>::get_ranks(
//         //         last_tree->tree(), octs, leafs);
//         new_tree = std::make_unique<CommonTree>(&dictionary, octs, leafs);
//         // new_tree=std::make_unique<CommonTree>(&dictionary, tree_file_3, flow_file_3);
//         last_tree = std::move(new_tree); // Now swap after it's ready
//         last_tree->run(nskip, false);
//     }
}