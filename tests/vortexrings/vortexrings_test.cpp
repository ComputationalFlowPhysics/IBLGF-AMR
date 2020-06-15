#include <gtest/gtest.h>
#include <googletest/gtest-mpi-listener.hpp>

#include <boost/mpi.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/filesystem.hpp>


#include "vortexrings.hpp"

namespace filesystem = boost::filesystem;


int main(int argc, char** argv)
{
    // Filter out Google Test arguments
    ::testing::InitGoogleTest(&argc, argv);

    // Initialize MPI
    MPI_Init(&argc, &argv);

    // Add object that will finalize MPI on exit; Google Test owns this pointer
    ::testing::AddGlobalTestEnvironment(new GTestMPIListener::MPIEnvironment);

    // Get the event listener list.
    ::testing::TestEventListeners& listeners =
            ::testing::UnitTest::GetInstance()->listeners();

    // Remove default listener: the default printer and the default XML printer
    ::testing::TestEventListener *l =
            listeners.Release(listeners.default_result_printer());

    // Adds MPI listener; Google Test owns this pointer
    listeners.Append(
            new GTestMPIListener::MPIWrapperPrinter(l,
                MPI_COMM_WORLD)
            );
    // Run tests, then clean up and exit.
    return RUN_ALL_TESTS();
}

TEST(PoissonSolverTest, VortexRing_1)
{
    boost::mpi::communicator world;

    for (auto& entry : filesystem::directory_iterator("./configs"))
    {
        auto s=entry.path();

        if (s.filename().string().rfind("config", 0) == 0)
        {
            if (world.rank()==0)
                std::cout<< "------------- Testing on config file " << s.filename()<< " -------------" <<std::endl;

            double L_inf_error=vortex_run(s.string());
            world.barrier();

            EXPECT_LT(L_inf_error,0.0);
        }
    }
}
