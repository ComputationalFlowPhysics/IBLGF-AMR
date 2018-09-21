#include <boost/mpi.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>

#include <setups/poissonProblem/poissonProblem.hpp>
#include <dictionary/dictionary.hpp>


int main(int argc, char *argv[])
{

	boost::mpi::environment env(argc, argv);
	boost::mpi::communicator world;

	std::string input="./";
    input += std::string("configFile");
    
    if (argc>1 && argv[1][0] != '-')
    {
        input = argv[1];
    }
    
    // Read in dictionary
    Dictionary dictionary(input);
    
    // run setup
    PoissonProblem setup(&dictionary);

    setup.run();

    return 0;
}
