#include <iostream>
#include <vector>

#include <boost/mpi.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <stdlib.h>
#include <chrono>
#include <thread>

#include <global.hpp>
#include <mpi/client.hpp>
#include <mpi/server.hpp>
#include <mpi/tag_generator.hpp>


using namespace sr_mpi;
int main(int argc, char *argv[])
{
	boost::mpi::environment env(argc, argv);
	boost::mpi::communicator world;

    if(world.rank()==0)
    {
        Server server;
        server.run();
    }
    else
    {
       const int nQueries=1;

       Client client;
       client.test_query() ;
       for(int i =1;i<=nQueries;++i)
       {
           //int query_rank = rand() % world.size() ;
           if(world.rank()==1  )
           {
               client.test_query() ;
           }
       }
       client.disconnect();
    }
    world.barrier();

    return 0;
}
