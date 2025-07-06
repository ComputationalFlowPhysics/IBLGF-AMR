#include <boost/mpi.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>

#include "common_tree.hpp"
#include <iostream>
#include <vector>
#include <iblgf/dictionary/dictionary.hpp>
#include "merge_trees.hpp"
using namespace iblgf;
int main(int argc, char *argv[])
{
    boost::mpi::environment env(argc, argv);
	boost::mpi::communicator world;

    std::string input="./";
    input += std::string("configFile");

    if (argc>1 && argv[1][0] != '-')
    {
        input = argv[1];
    }

    // Read in dictionary
    Dictionary dictionary(input, argc, argv);

    // Find output directory
    auto dict_out=dictionary
            .get_dictionary("simulation_parameters")->get_dictionary("output");
    std::string dir = dict_out->template get<std::string>("directory");

    int idxStart=60000;
    // int idxEnd=10//64000;
    int nskip=100;
    int nTotal=100;
    // CommonTree ref_tree;
    std::vector<CommonTree::key_id_t> octs;
    std::vector<int> leafs;
    std::unique_ptr<CommonTree> last_tree;
    std::string flow_file_1="./"+dir+"/flowTime_"+std::to_string(idxStart)+".hdf5";
    std::string tree_file_1="./"+dir+"/tree_info_"+std::to_string(idxStart)+".bin";
    last_tree = std::make_unique<CommonTree>(&dictionary,
            tree_file_1, flow_file_1);

    for (int i=1; i<nTotal; i++)
    {

        std::string flow_file_1="./"+dir+"/flowTime_"+std::to_string(idxStart+i*nskip)+".hdf5";
        // std::string flow_file_2="./"+dir+"/flowTime_"+std::to_string(idxStart+(i+1)*nskip)+".hdf5";
        std::string tree_file_1="./"+dir+"/tree_info_"+std::to_string(idxStart+i*nskip)+".bin";
        // std::string tree_file_2="./"+dir+"/tree_info_"+std::to_string(idxStart+(i+1)*nskip)+".bin";
        auto tree_1 = std::make_unique<CommonTree>(&dictionary,
                tree_file_1, flow_file_1);
        octs.clear();
        leafs.clear();
        auto tree0= last_tree->tree();
        auto tree1= tree_1->tree();
        MergeTrees<CommonTree>::merger(
                tree0,tree1, octs, leafs);
        // MergeTrees<CommonTree>::get_ranks(
        //         last_tree->tree(), octs, leafs);
        last_tree= std::make_unique<CommonTree>(
                &dictionary, octs, leafs);
    }

    last_tree->run(0, false);


}