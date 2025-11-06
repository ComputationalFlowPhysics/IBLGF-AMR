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

#include "vortexrings.hpp" 
#include <iblgf/dictionary/dictionary.hpp>

namespace iblgf {
double poisson3d_run(const std::string input, int argc = 0, char** argv = nullptr)
{
    // Read in dictionary
    dictionary::Dictionary dictionary(input, argc, argv);

    //Instantiate setup
    VortexRingTest setup(&dictionary);

    const double measured = setup.run();

    const double EXP_LInf = dictionary.get_dictionary("simulation_parameters")
                                ->template get_or<double>("EXP_LInf", 0.0);

    return measured - EXP_LInf;
}

TEST(Poisson3DAnalyticTest, ConfigsInCurrentDir)
{
    boost::mpi::communicator world;

    for (auto& entry : boost::filesystem::directory_iterator("./"))
    {
        auto s = entry.path();

        if (s.filename().string().rfind("config", 0) == 0)
        {
            if (world.rank() == 0)
                std::cout << "------------- Poisson-3D test on "
                          << s.filename() << " -------------" << std::endl;

            const double result = poisson3d_run(s.string());
            world.barrier();

            EXPECT_LT(result, 0.0);
        }
    }
}

} // namespace iblgf

// Standard gtest+MPI main, consistent with your other tests
int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    boost::mpi::environment env(argc, argv);
    boost::mpi::communicator world;

    return RUN_ALL_TESTS();
}