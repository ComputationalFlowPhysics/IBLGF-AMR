#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>

#include <iostream>
#include <string>
#include <vector>

#include "common_tree.hpp"
#include <iblgf/dictionary/dictionary.hpp>
#include <iblgf/solver/modal_analysis/merge_trees.hpp>

using namespace iblgf;
int main(int argc, char* argv[])
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
    dictionary::Dictionary dictionary(input, argc, argv);
    auto merger=MergeTrees<CommonTree>(&dictionary);
    auto ref_domain=merger.ref_to_symmetric_ref();


    if (world.rank() == 0) { std::cout << "Symmetry adaptation completed." << std::endl; }

    ref_domain->initialize();
    ref_domain->symfield<iblgf::CommonTree::u_type, iblgf::CommonTree::u_type>(2);

    if (world.rank() == 0) { std::cout << "Symmetry completed." << std::endl; }

    return 0;
}
