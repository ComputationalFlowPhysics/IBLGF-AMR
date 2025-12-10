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

#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#include <boost/mpi.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <stdlib.h>
#include <chrono>
#include <thread>
#include <memory>

#include <iblgf/global.hpp>
#include "client.hpp"
#include "server.hpp"


namespace iblgf
{

TEST(server_client_tests, queries)
{

    //Test to assure no deadlock, even with multple queries at the same 
    //time (this test the tag_generator as well )

    boost::mpi::communicator world;
    
    const int serverRank=0;
    const int rank=world.rank();

    if(rank==serverRank)
    {
        Server server;
        server.test();
    }
    else
    {
       const int nQueries=10;
       Client client(serverRank);
       client.connect();
       auto res=client.test() ;
       for(auto& e : res) { EXPECT_EQ(e, -rank ); }
       for(int i =1;i<=nQueries;++i)
       {
           if(world.rank()==1  )
           {
               auto res=client.test() ;
               for(auto& e : res) { EXPECT_EQ(e, -rank ); }
           }
       }
       client.disconnect();
    }

     world.barrier();
}
}
