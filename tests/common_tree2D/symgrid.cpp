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

    // Find output directory
    auto        dict_out = dictionary.get_dictionary("simulation_parameters")->get_dictionary("output");
    std::string dir = dict_out->template get<std::string>("directory");

    std::string tree_ref_file_ = dictionary.get_dictionary("simulation_parameters")->get<std::string>("tree_ref_file");
    std::string flow_ref_file_ = dictionary.get_dictionary("simulation_parameters")->get<std::string>("flow_ref_file");

    int ref_levels_ = dictionary.get_dictionary("simulation_parameters")->get_or<int>("nLevels", 0);
    std::vector<CommonTree::domain_t::key_t> octs;
    std::vector<int>                         level_change;


    auto ref_domain_ = std::make_unique<CommonTree>(&dictionary, tree_ref_file_, flow_ref_file_);
   
    octs.clear();
    level_change.clear();
    for(auto i=ref_levels_; i>=0; --i)
    {
        octs.clear();
        level_change.clear();
        auto ref_tree = ref_domain_->tree();
        MergeTrees<CommonTree>::symgrid(ref_tree, i,octs, level_change,true);
        int tt = i == 0 ? 1 : -1; //only save after all levels have been adapted
        ref_domain_->run_adapt_from_keys<CommonTree::u_type>(tt, octs, level_change);

    }
    if(world.rank() == 0)
    {
        std::cout << "Adaptation to reference levels completed." << std::endl;
    }
    for(auto i=ref_levels_; i>=0; --i)
    {
        octs.clear();
        level_change.clear();
        auto ref_tree = ref_domain_->tree();
        MergeTrees<CommonTree>::symgrid(ref_tree, i,octs, level_change,true);
        int tt = i == 0 ? 1 : -1; //only save after all levels have been adapted
        ref_domain_->run_adapt_from_keys<CommonTree::u_type>(tt, octs, level_change);
    }
    // for(auto i=ref_levels_; i>=0; --i)
    // {
    //     octs.clear();
    //     level_change.clear();
    //     auto ref_tree = ref_domain_->tree();
    //     MergeTrees<CommonTree>::symgrid(ref_tree, i,octs, level_change,true);
    //     int tt = i == 0 ? 1 : -1; //only save after all levels have been adapted
    //     ref_domain_->run_adapt_from_keys<CommonTree::u_type>(tt, octs, level_change);
// 
    // }
    std::cout<< "Symmetry adaptation completed." << std::endl;
    ref_domain_->initialize();
    ref_domain_->symfield<CommonTree::u_type, CommonTree::u_type>(2);
    std::cout<< "Symmetry completed." << std::endl;
    

    return 0;
}