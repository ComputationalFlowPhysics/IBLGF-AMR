#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>

#include <string>

#include <iblgf/dictionary/dictionary.hpp>
#include <iblgf/solver/modal_analysis/merge_trees.hpp>

#include "common_tree.hpp"

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
    boost::mpi::environment env(argc, argv);

    iblgf::Dictionary dictionary(get_input_path(argc, argv), argc, argv);
    auto merger = iblgf::MergeTrees<iblgf::CommonTree>(&dictionary);
    merger.get_common_tree();

    return 0;
}
