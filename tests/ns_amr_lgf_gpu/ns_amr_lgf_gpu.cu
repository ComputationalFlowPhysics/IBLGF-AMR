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

#include <boost/mpi.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <cuda_runtime.h>

#include "ns_amr_lgf.hpp"
#include <iblgf/dictionary/dictionary.hpp>


using namespace iblgf;

int main(int argc, char *argv[])
{
    boost::mpi::environment env(argc, argv);
    boost::mpi::communicator world;

    std::string input = "./";
    input += std::string("configFile");

    if (argc > 1 && argv[1][0] != '-')
    {
        input = argv[1];
    }

    int rank = world.rank();
    
    // Get number of GPUs and set device based on MPI rank
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Error on rank " << rank << ": " 
                  << cudaGetErrorString(err) << std::endl;
        return 1;
    }
    
    if (deviceCount == 0)
    {
        std::cerr << "No CUDA devices found on rank " << rank << std::endl;
        return 1;
    }
    
    int device = rank % deviceCount;
    cudaSetDevice(device);
    
    if (rank == 0)
    {
        std::cout << "Running GPU-enabled Navier-Stokes solver with " 
                  << world.size() << " MPI ranks and " 
                  << deviceCount << " GPU(s) per node" << std::endl;
    }
    
    std::cout << "Rank " << rank << " using GPU device " << device 
              << " of " << deviceCount << " available" << std::endl;

    // Read in dictionary
    dictionary::Dictionary dictionary(input, argc, argv);

    // Instantiate setup
    NS_AMR_LGF setup(&dictionary);

    // Run setup
    double L_inf_error = setup.run();
    L_inf_error = setup.u1_Linf_fine();

    double EXP_LInf = dictionary.get_dictionary("simulation_parameters")
                          ->template get_or<double>("EXP_LInf", 0);

    if (rank == 0)
    {
        std::cout << "L_inf error: " << L_inf_error << std::endl;
        std::cout << "Expected L_inf: " << EXP_LInf << std::endl;
        std::cout << "Difference: " << (L_inf_error - EXP_LInf) << std::endl;
    }

    return 0;
}
