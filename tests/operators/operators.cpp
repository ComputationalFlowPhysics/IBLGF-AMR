//      ▄▄▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄▄▄   ▄            ▄▄▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄▄▄▄
//     ▐░░░░░░░░░░░▌▐░░░░░░░░░░▌ ▐░▌          ▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌
//      ▀▀▀▀█░█▀▀▀▀ ▐░█▀▀▀▀▀▀▀█░▌▐░▌          ▐░█▀▀▀▀▀▀▀▀▀ ▐░█▀▀▀▀▀▀▀▀▀
//          ▐░▌     ▐░▌       ▐░▌▐░▌          ▐░▌          ▐░▌
//          ▐░▌     ▐░█▄▄▄▄▄▄▄█░▌▐░▌          ▐░▌ ▄▄▄▄▄▄▄▄ ▐░█▄▄▄▄▄▄▄▄▄
//          ▐░▌     ▐░░░░░░░░░░▌ ▐░▌          ▐░▌▐░░░░░░░░▌▐░░░░░░░░░░░▌
//          ▐░▌     ▐░█▀▀▀▀▀▀▀█░▌▐░▌          ▐░▌ ▀▀▀▀▀▀█░▌▐░█▀▀▀▀▀▀▀▀▀
//          ▐░▌     ▐░▌       ▐░▌▐░▌          ▐░▌       ▐░▌▐░▌
//      ▄▄▄▄█░█▄▄▄▄ ▐░█▄▄▄▄▄▄▄█░▌▐░█▄▄▄▄▄▄▄▄▄ ▐░█▄▄▄▄▄▄▄█░▌▐░▌
//     ▐░░░░░░░░░░░▌▐░░░░░░░░░░▌ ▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░▌
//      ▀▀▀▀▀▀▀▀▀▀▀  ▀▀▀▀▀▀▀▀▀▀   ▀▀▀▀▀▀▀▀▀▀▀  ▀▀▀▀▀▀▀▀▀▀▀  ▀

#include <gtest/gtest.h>

#include <iblgf/dictionary/dictionary.hpp>
#include "operatorTest.hpp"

// *Added*
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <memory>

namespace iblgf
{
using namespace types;

TEST(operator_test, convergence)
{
    // *Added*
    // ensure MPI is initialized once per process (same pattern as server_client)
    static std::unique_ptr<boost::mpi::environment> env_holder;
    if (!env_holder) { env_holder = std::make_unique<boost::mpi::environment>(); }

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
                block_extent=6;                       \
                block                                 \
                {                                     \
                    base=(-48,-48,-48);               \
                    extent=(96,96,96);                \
                }                                     \
            }                                         \
        }                                             \
    ";
    // clang-format on

    //Read in dictionary
    dictionary::Dictionary dictionary("simulation_parameters", configFile);

    //Instantiate setup
    OperatorTest setup(&dictionary);

    // run setup
    setup.run();
}

} // namespace iblgf