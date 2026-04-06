// CPU debug driver for LGF pipeline

#include "ns_amr_lgf_gpu_debug.hpp"
#include <iblgf/dictionary/dictionary.hpp>
#include <cstring>

using namespace iblgf;

int main(int argc, char* argv[])
{
    boost::mpi::environment env(argc, argv);
    boost::mpi::communicator world;

    std::string input = "./";
    input += std::string("configFile");
    if (argc > 1 && argv[1][0] != '-')
    {
        input = argv[1];
    }

    bool debug_lgf = false;
    bool debug_lgf_levels = false;
    for (int i = 1; i < argc; ++i)
    {
        if (std::strcmp(argv[i], "--debug-lgf") == 0) debug_lgf = true;
        if (std::strcmp(argv[i], "--debug-lgf-levels") == 0) debug_lgf_levels = true;
    }

    Dictionary dictionary(input, argc, argv);
    iblgf::debug::NS_AMR_LGF_Debug setup(&dictionary);

    if (debug_lgf) iblgf::debug::debug_run_lgf(setup);
    if (debug_lgf_levels) iblgf::debug::debug_run_lgf_levels(setup);

    if (debug_lgf || debug_lgf_levels) return 0;

    setup.run();
    return 0;
}
