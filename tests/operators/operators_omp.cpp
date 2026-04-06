#ifndef IBLGF_OMP
#define IBLGF_OMP
#endif

#include <iblgf/dictionary/dictionary.hpp>
#include "operatorTest.hpp"
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>

using namespace iblgf;

int
main(int argc, char* argv[])
{
    boost::mpi::environment  env(argc, argv);
    boost::mpi::communicator world;

    // clang-format off
    std::string configFile = "                        \
        simulation_parameters                         \
        {                                             \
                                                      \
            nLevels=0;                                \
            global_refinement=0;                      \
            refinement_factor=0.125;                  \
            correction=true;                          \
            subtract_non_leaf=true;                   \
                                                      \
            output                                    \
            {                                         \
                directory=vortexRings;                \
            }                                         \
                                                      \
            domain                                    \
            {                                         \
                Lx=2.5;                               \
                bd_base=(-96,-96,-96);                \
                bd_extent=(192,192,192);              \
                block_extent=6;                      \
                block                                 \
                {                                     \
                    base=(-48,-48,-48);               \
                    extent=(96,96,96);                \
                }                                     \
            }                                         \
        }                                             \
    ";
    // clang-format on
    //     // clang-format off
    // std::string configFile = "                        \
    //     simulation_parameters                         \
    //     {                                             \
    //                                                   \
    //         nLevels=0;                                \
    //         global_refinement=0;                      \
    //         refinement_factor=0.125;                  \
    //         correction=true;                          \
    //         subtract_non_leaf=true;                   \
    //                                                   \
    //         output                                    \
    //         {                                         \
    //             directory=vortexRings;                \
    //         }                                         \
    //                                                   \
    //         domain                                    \
    //         {                                         \
    //             Lx=2.5;                               \
    //             bd_base=(-60,-60,-60);                \
    //             bd_extent=(120,120,120);              \
    //             block_extent=30;                      \
    //             block                                 \
    //             {                                     \
    //                 base=(-60,-60,-60);               \
    //                 extent=(120,120,120);                \
    //             }                                     \
    //         }                                         \
    //     }                                             \
    // ";
    // // clang-format on

    //Read in dictionary
    dictionary::Dictionary dictionary("simulation_parameters", configFile);

    //Instantiate setup
    OperatorTest setup(&dictionary);

    // run setup
    setup.run();
    return 0;
}