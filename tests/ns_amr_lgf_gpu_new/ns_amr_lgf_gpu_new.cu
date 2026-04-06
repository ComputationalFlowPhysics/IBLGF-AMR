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

#include "ns_amr_lgf_gpu_debug.hpp"
#include <iblgf/dictionary/dictionary.hpp>
#include <cstring>

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

    bool debug_init = false;
    bool debug_kernels = false;
    bool debug_lgf = false;
    for (int i = 1; i < argc; ++i)
    {
        if (std::strcmp(argv[i], "--debug-init") == 0) debug_init = true;
        if (std::strcmp(argv[i], "--debug-kernels") == 0) debug_kernels = true;
        if (std::strcmp(argv[i], "--debug-lgf") == 0) debug_lgf = true;
    }

    const int rank = world.rank();
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    std::cout << "Rank " << rank << " found " << deviceCount << " GPUs" << std::endl;
    if (err == cudaSuccess && deviceCount > 0)
    {
        cudaSetDevice(rank % deviceCount);
    }

    Dictionary dictionary(input, argc, argv);
    iblgf::debug::NS_AMR_LGF_Debug setup(&dictionary);
    if (debug_init) iblgf::debug::debug_init_stats(setup);
    if (debug_kernels) iblgf::debug::debug_kernel_microtests(setup);
    if (debug_lgf) iblgf::debug::debug_run_lgf(setup);
    if (debug_init || debug_kernels || debug_lgf) return 0;

    setup.run();

    return 0;
}
