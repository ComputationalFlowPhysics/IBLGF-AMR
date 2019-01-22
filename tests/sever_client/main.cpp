#include <iostream>
#include <vector>

#include <boost/mpi.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <stdlib.h>
#include <chrono>
#include <thread>

#include <global.hpp>
#include "client.hpp"
#include "server.hpp"


int main(int argc, char *argv[])
{
	boost::mpi::environment env(argc, argv);
	boost::mpi::communicator world;

    int severRank=0;
    if(world.rank()==severRank)
    {
        Server server;
        server.test();
    }
    else
    {
       const int nQueries=1;
       Client client(severRank);
       client.connect();
       client.test() ;
       for(int i =1;i<=nQueries;++i)
       {
           if(world.rank()==1  )
           {
               client.test() ;
           }
       }
       client.disconnect();
    }
    world.barrier();

    return 0;
}
