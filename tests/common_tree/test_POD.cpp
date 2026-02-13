#include <boost/mpi.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>

#include <iostream>
#include <vector>
#include <iblgf/dictionary/dictionary.hpp>
#include "POD_2D.hpp"
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
    // std::string dir = dict_out->template get<std::string>("directory");



    std::string tree_ref_file_=dictionary.get_dictionary("simulation_parameters")->get<std::string>("tree_ref_file");
    std::string flow_ref_file_=dictionary.get_dictionary("simulation_parameters")->get<std::string>("flow_ref_file");


    POD2D pod(&dictionary, tree_ref_file_, flow_ref_file_);
    pod.run(argc, argv);
    

}