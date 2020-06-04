#include <boost/mpi.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>

#define IBLGF_VORTEX_RUN_ALL

#include "vortexrings.hpp"
#include <dictionary/dictionary.hpp>


double vortex_run(const std::string input, int argc, char **argv)
{
    // Read in dictionary
    Dictionary dictionary(input, argc, argv);

    //Instantiate setup
    VortexRingTest setup(&dictionary);

    // run setup
    double L_inf_error=setup.run();

    double EXP_LInf=dictionary.get_dictionary("simulation_parameters")
                            ->template get_or<double>("EXP_LInf", 0);

    return L_inf_error-EXP_LInf;
}
