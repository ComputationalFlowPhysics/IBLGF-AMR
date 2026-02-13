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

// #include <gtest/gtest.h>

#include <iblgf/dictionary/dictionary.hpp>
#include "operators_jacTest.hpp"

using namespace iblgf;

using namespace types;
int main(int argc, char *argv[])
{
    boost::mpi::environment env(argc, argv);
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
    OperatorJacTest setup(&dictionary);

    // run setup
    setup.run();
    return 0;
}

 // namespace iblgf
