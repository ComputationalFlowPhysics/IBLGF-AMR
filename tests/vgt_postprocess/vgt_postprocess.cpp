#include <boost/mpi.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>

#include "vgt_postprocess.hpp"
#include <iostream>
#include <vector>
#include <iblgf/dictionary/dictionary.hpp>
#include <iblgf/solver/modal_analysis/merge_trees.hpp>
using namespace iblgf;
int
main(int argc, char* argv[])
{
    boost::mpi::environment  env(argc, argv);
    boost::mpi::communicator world;

    std::string input = "./";
    input += std::string("configFile");

    if (argc > 1 && argv[1][0] != '-') { input = argv[1]; }

    // Read in dictionary
    Dictionary dictionary(input, argc, argv);

    std::string tree_ref_file_ = dictionary.get_dictionary("simulation_parameters")->get<std::string>("tree_ref_file");
    std::string flow_ref_file_ = dictionary.get_dictionary("simulation_parameters")->get<std::string>("flow_ref_file");
    std::cout << "tree_ref_file_: " << tree_ref_file_ << std::endl;
    VGT_PostProcess setup(&dictionary, tree_ref_file_, flow_ref_file_);
    setup.run();
    return 0;
}
