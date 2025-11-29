#include <boost/mpi.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>

#include "common_tree.hpp"
#include <iostream>
#include <vector>
#include <iblgf/dictionary/dictionary.hpp>
#include <iblgf/solver/modal_analysis/merge_trees.hpp>
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

   
    // int idxStart = dictionary.get_dictionary("simulation_parameters")->get_or<int>("nStart", 100);
    // int nTotal = dictionary.get_dictionary("simulation_parameters")->get_or<int>("nTotal", 100);
    // int nskip = dictionary.get_dictionary("simulation_parameters")->get_or<int>("nskip", 100);
    std::string f_suffix=dictionary.get_dictionary("simulation_parameters")->get<std::string>("read_write_filename");

    // std::vector<CommonTree::domain_t::key_t> octs;
    // std::vector<int>                  level_change;
    // std::unique_ptr<CommonTree>       last_tree;
    // std::unique_ptr<CommonTree>       tree_1;
    // std::unique_ptr<CommonTree>       new_tree;
    // std::string                       flow_file_1 = "./" + "flowTime_" + f_suffix + ".hdf5";
    // std::string                       tree_file_1 = "./" + "tree_info_" + f_suffix + ".bin";
    std::string flow_file_1 = "./flowTime_" + f_suffix + ".hdf5";
    std::string tree_file_1 = "./tree_info_" + f_suffix + ".bin";
    std::cout<< "flow_file_1: " << flow_file_1 << std::endl;
    std::cout<< "tree_file_1: " << tree_file_1 << std::endl;
    // std::cout<< "nTotal: " << nTotal << std::endl;
    // // std::cout<< "nTotal: " << nTotal << std::endl;
    // std::cout<< "nskip: " << nskip << std::endl;
    // std::cout<< "idxStart: " << idxStart << std::endl;
    CommonTree setup(&dictionary, tree_file_1, flow_file_1);
    setup.read_write_test();

    // auto ref_domain_=std::make_unique<CommonTree>(&dictionary,tree_file_1, flow_file_1);


    return 0;

}