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
#ifndef IBLGF_COMPILE_CUDA
#define IBLGF_COMPILE_CUDA
#endif
#include <gtest/gtest.h>
#ifndef __CUDACC__
#include <boost/filesystem.hpp>
#endif
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>

#include "vortexrings.hpp"
#include <iblgf/dictionary/dictionary.hpp>


namespace iblgf {

double poisson3d_run(const std::string input, int argc = 0, char** argv = nullptr)
{
    // Read in dictionary
    dictionary::Dictionary dictionary(input, argc, argv);

    // Instantiate setup
    VortexRingTest setup(&dictionary);

    const double measured = setup.run();

    const double EXP_LInf =
        dictionary.get_dictionary("simulation_parameters")
                  ->template get_or<double>("EXP_LInf", 0.0);

    return measured - EXP_LInf;
}

TEST(Poisson3DAnalyticTest_GPU, ConfigsInCurrentDir)
{
    boost::mpi::communicator world;

    for (auto& entry : boost::filesystem::directory_iterator("./"))
    {
        auto s = entry.path();

        // Only process files named config
        if (s.filename().string().rfind("config", 0) == 0)
        {
            if (world.rank() == 0)
            {
                std::cout << "------------- Poisson-3D test on "
                          << s.filename() << " -------------" << std::endl;
            }

            const double result = poisson3d_run(s.string());

            world.barrier();

            EXPECT_LT(result, 0.0);
        }
    }
}

} 

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    boost::mpi::environment env(argc, argv);
    boost::mpi::communicator world;
    int rank = world.rank();
    // get number of GPUs and set device
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    std::cout<<"Rank "<<rank<<" found "<<deviceCount<<" GPUs"<<std::endl;
    cudaSetDevice(rank % deviceCount);
    return RUN_ALL_TESTS();
}
