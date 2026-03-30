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
    L_inf_error = setup.u1_Linf_fine();

    double EXP_LInf = dictionary.get_dictionary("simulation_parameters")
                          ->template get_or<double>("EXP_LInf", 0);

    return L_inf_error - EXP_LInf;
}

TEST(PoissonSolverTest_GPU, VortexRing_1)
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
int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    // Initialize MPI before any tests run
    boost::mpi::environment env(argc, argv);
    boost::mpi::communicator world;  // optional, can use in main if needed
    int rank = world.rank();
    // get number of GPUs and set device
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    std::cout<<"Rank "<<rank<<" found "<<deviceCount<<" GPUs"<<std::endl;
    cudaSetDevice(rank % deviceCount);
    return RUN_ALL_TESTS(); // now MPI is already initialized
}
