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

#include <boost/mpi.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <omp.h>
#include <time.h>
#include <stdlib.h>

//#include "ns_amr_lgf.hpp"
//#include <iblgf/dictionary/dictionary.hpp>


//using namespace iblgf;

int main(int argc, char *argv[])
{

	boost::mpi::environment env(argc, argv);
	boost::mpi::communicator world;

	int rank = world.rank();
    int sumV = 0;
    struct timespec tstart,tend;

    int n_pts =  100000000;

    std::vector<double> vals;
    vals.resize(24);
    std::fill(vals.begin(), vals.end(),0);

    double a,b,c,pi,dt,mflops;

    clock_gettime(CLOCK_REALTIME,&tstart);
    for (int j = 0; j < 1000; j++) {
        #pragma omp parallel for
        for (auto it = vals.begin(); it != vals.end();it++) {
            for (int k = 0; k < 1000; k++) {
                *it += 1;
            }
        }
    }
    clock_gettime(CLOCK_REALTIME,&tend);
    dt = (tend.tv_sec+tend.tv_nsec/1e9)-(tstart.tv_sec+tstart.tv_nsec/1e9);
    world.barrier();

    for (int i = 0; i < world.size();i++) {
        if (i == rank) {
            std::cout << "Sum is " << sumV << " at " << rank << " with time " << dt << "s" << std::endl;
        }
        world.barrier();
    }


    clock_gettime(CLOCK_REALTIME,&tstart);
    //#pragma omp master
    for (auto it = vals.begin(); it != vals.end();it++) {
        std::cout << *it << " ";
    }
    std::cout << std::endl;
    clock_gettime(CLOCK_REALTIME,&tend);
    dt = (tend.tv_sec+tend.tv_nsec/1e9)-(tstart.tv_sec+tstart.tv_nsec/1e9);

    for (int i = 0; i < world.size();i++) {
        if (i == rank) {
            std::cout << "Sum is " << sumV << " at " << rank << " with time " << dt << "s" << std::endl;
        }
        world.barrier();
    }

    return 0;
}