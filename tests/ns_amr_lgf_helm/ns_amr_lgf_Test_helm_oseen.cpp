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

// Lamb-Oseen vortex analytical-verification test for the Helmholtz (2D + Fourier)
// solver. Initializes a 2D Lamb-Oseen vortex in Fourier mode 0, marches a few
// steps, and asserts the numerical velocity matches the analytic time-dependent
// Oseen solution u_oseen_vort(x, y, t_final) to within EXP_LInf.
//
// CONVERGENCE_TEST turns on the analytic reference / error machinery in
// ns_amr_lgf.hpp and MUST be defined before the header is included. It is a
// whole-translation-unit flag, so this is its own test binary (the plain
// ns_amr_lgf_Test_helm smoke test does not define it and is unaffected).

#define CONVERGENCE_TEST

#include <algorithm>

#include <gtest/gtest.h>
#include <boost/filesystem.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>

#include "ns_amr_lgf.hpp"
#include <iblgf/dictionary/dictionary.hpp>

namespace iblgf
{
double oseen_run(const std::string input, int argc = 0, char** argv = nullptr)
{
    // Read in dictionary
    dictionary::Dictionary dictionary(input, argc, argv);

    // Instantiate setup (ctor calls initialize() for Vort_type != 0)
    NS_AMR_LGF setup(&dictionary);

    // run setup: returns max L_inf( u_numeric - u_oseen(t_final) ) over components
    float_type L_inf_error = setup.run(argc, argv);

    double EXP_LInf = dictionary.get_dictionary("simulation_parameters")
                          ->template get_or<double>("EXP_LInf", 0);

    return L_inf_error - EXP_LInf;
}

TEST(HelmholtzOseenTest, LambOseenVortex)
{
    boost::mpi::communicator world;

    for (auto& entry : boost::filesystem::directory_iterator("./"))
    {
        auto s = entry.path();

        if (s.filename().string().rfind("config", 0) == 0)
        {
            if (world.rank() == 0)
                std::cout << "------------- Oseen test on config file "
                          << s.filename() << " -------------" << std::endl;

            double L_inf_error = oseen_run(s.string());
            world.barrier();

            EXPECT_LT(L_inf_error, 0.0);
        }
    }
}
} // namespace iblgf
