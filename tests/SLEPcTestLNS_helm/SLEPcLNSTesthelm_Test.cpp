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
#include <boost/filesystem.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>

#include "ns_amr_lgf.hpp"
#include <iblgf/dictionary/dictionary.hpp>


namespace iblgf{
double vortex_run(const std::string input, int argc, char** argv)
{
    // Read in dictionary
    dictionary::Dictionary dictionary(input, argc, argv);

    //Instantiate setup
    NS_AMR_LGF setup(&dictionary);

    // run setup
    double L_inf_error = setup.run();

    double EXP_LInf = dictionary.get_dictionary("simulation_parameters")
                          ->template get_or<double>("EXP_LInf", 0);

    return L_inf_error - EXP_LInf;
}

TEST(PoissonSolverTest, VortexRing_1)
{
    boost::mpi::communicator world;

    for (auto& entry : boost::filesystem::directory_iterator( "./"))
    {
        auto s = entry.path();

        if (s.filename().string().rfind("config", 0) == 0)
        {
            if (world.rank() == 0)
                std::cout << "------------- Testing on config file "
                          << s.filename() << " -------------" << std::endl;

            double L_inf_error = vortex_run(s.string());
            world.barrier();

            EXPECT_LT(L_inf_error, 0.0);
        }
    }
}
} //namespace iblgf
