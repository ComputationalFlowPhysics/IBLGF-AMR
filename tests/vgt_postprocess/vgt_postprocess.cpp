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
    auto        dict_out = dictionary.get_dictionary("simulation_parameters")->get_dictionary("output");
    std::string pp_dir = dict_out->template get<std::string>("directory"); // where post processed files will be saved
    std::string dir=dict_out->template get_or<std::string>("field_dir", pp_dir); // where data is loaded from

    int idxStart = dictionary.get_dictionary("simulation_parameters")->get<int>("nStart");
    int nEnd = dictionary.get_dictionary("simulation_parameters")->get<int>("nEnd");
    int nskip = dictionary.get_dictionary("simulation_parameters")->get_or<int>("nskip", 100);
    int idx_cur=idxStart;
    while(idx_cur<=nEnd){
        if(world.rank()==0) std::cout << "Processing time index: " << idx_cur << std::endl;
        std::string flow_file_i= "./" + dir + "/flowTime_" + std::to_string(idx_cur) + ".hdf5";
        std::string tree_file_i= "./" + dir + "/tree_info_" + std::to_string(idx_cur) + ".bin";
        VGT_PostProcess setup(&dictionary, tree_file_i, flow_file_i);
        setup.run_batch(idx_cur);
        idx_cur+=nskip;
    }
    // std::string tree_ref_file_ = dictionary.get_dictionary("simulation_parameters")->get<std::string>("tree_ref_file");
    // std::string flow_ref_file_ = dictionary.get_dictionary("simulation_parameters")->get<std::string>("flow_ref_file");
    // if(world.rank()==0) std::cout << "tree_ref_file_: " << tree_ref_file_ << std::endl;
    // VGT_PostProcess setup(&dictionary, tree_ref_file_, flow_ref_file_);
    // setup.run();
    return 0;
}
