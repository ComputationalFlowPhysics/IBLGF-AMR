
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
    // vector of extents to try
    std::vector<int> extents = {2,3,4};
    std::string configFile0 = "                        \
        simulation_parameters                         \
        {                                             \
                                                      \
            nLevels=1;                                \
            global_refinement=0;                      \
            refinement_factor=0.125;                  \
            correction=true;                          \
            subtract_non_leaf=true;                   \
            nBaseLevelTimeSteps=10;                   \
            Re=100;                   \
            output_frequency=10;        \
            write_restart=false;        \
                                                      \
            output                                    \
            {                                         \
                directory=vortexRings;                \
            }                                         \
                                                      \
            domain                                    \
            {                                         \
                Lx=2.5;                               \
                bd_base=(-96,-96);                \
                bd_extent=(192,192);              \
                block_extent=6;                       \
                block                                 \
                {                                     \
                    base=(-96,-48);               \
                    extent=(192,96);                \
                }                                     \
            }                                         \
        }                                             \
    ";
    std::string configFile1 = "                        \
        simulation_parameters                         \
        {                                             \
                                                      \
            nLevels=1;                                \
            global_refinement=0;                      \
            refinement_factor=0.125;                  \
            correction=true;                          \
            subtract_non_leaf=true;                   \
            nBaseLevelTimeSteps=10;                   \
            Re=100;                              \
            output_frequency=10;              \
            write_restart=false;                \
                                                      \
            output                                    \
            {                                         \
                directory=vortexRings;                \
            }                                         \
                                                      \
            domain                                    \
            {                                         \
                Lx=2.5;                               \
                bd_base=(-96,-96);                \
                bd_extent=(192,192);              \
                block_extent=6;                       \
                block                                 \
                {                                     \
                    base=(-48,-96);               \
                    extent=(96,192);                \
                }                                     \
            }                                         \
        }                                             \
        ";


    //first config file dict and tree
    dictionary::Dictionary dictionary0("simulation_parameters", configFile0);
    CommonTree ct0(&dictionary0);
    // ct0.print_keys();
    
    //second config file dict and tree
    dictionary::Dictionary dictionary1("simulation_parameters", configFile1);
    CommonTree ct1(&dictionary1);
    ct0.run(0);
    ct1.run(1);
    // ct1.print_keys();
    auto tree0= ct0.tree();
    auto tree1= ct1.tree();
    

    std::vector< CommonTree::key_id_t> octs;

    std::vector<int> leafs;
    MergeTrees<CommonTree>::merger(tree0, tree1,octs,leafs);


    CommonTree ct2(&dictionary0, octs, leafs);
    ct2.run(2, false);

    auto tree2 = ct2.tree();
     // send list octs of tree2 to correct rank of tree0 parents

    std::vector<CommonTree::key_id_t> recv_octs;
    std::vector<int> recv_leafs;
    if (world.rank()==0)
    {
        recv_leafs=leafs;
        recv_octs=octs;
    }
    MergeTrees<CommonTree>::get_ranks(tree0, recv_octs, recv_leafs);


    ct0.run_adapt_to_ref(recv_octs, recv_leafs);
    // ct0.run_adapt_to_ref(recv_octs, recv_leafs);


    return 0;
}