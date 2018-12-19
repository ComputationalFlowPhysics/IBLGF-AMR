#ifndef DOMAIN_INCLUDED_CLIENT_HPP
#define DOMAIN_INCLUDED_CLIENT_HPP

#include <vector>
#include <stdexcept>

#include <global.hpp>
#include <boost/mpi/communicator.hpp>
#include <domain/mpi/task_manager.hpp>

namespace domain
{

//FIXME: Do better
using namespace sr_mpi;

/** @brief ProcessType Client 
 *  Worker process, who stores only a sub-domain 
 */
template<class Domain>
class Client
{

public:
    using domain_t = Domain;
    using communicator_type  = typename  domain_t::communicator_type;
    using octant_t  = typename  domain_t::octant_t;
    using datablock_t  = typename  domain_t::datablock_t;
    using key_t  = typename  domain_t::key_t;

    using key_query_t = Task<tags::key_query,std::vector<int>>;
    using task_manager_t = TaskManager<key_query_t>;


public:
    Client(const Client&  other) = default;
    Client(      Client&& other) = default;
    Client& operator=(const Client&  other) & = default;
    Client& operator=(      Client&& other) & = default;
    ~Client() = default;

    Client(Domain* _d, communicator_type _comm =communicator_type())
    :domain_(_d), comm_(_comm)
    {
        boost::mpi::communicator world;
        std::cout<<"I am a client on rank: "<<world.rank()<<std::endl;
    }

public:
    void receive_keys()
    {
        std::vector<key_t> keys;
        comm_.recv(0,0,keys);

        //Init trees and allocate all memory
        domain_->tree()->init(keys, [&](octant_t* _o){
            auto bbase=domain_->tree()->octant_to_level_coordinate(
                    _o->tree_coordinate());
            _o->data()=std::make_shared<datablock_t>(bbase, 
                    domain_->block_extent(),_o->refinement_level(), true);
        });
    }

    //Silly test for aks and answer
    void test_query()
    {
        std::vector<int> task_dat(3,comm_.rank());
        std::vector<int> task_dat2(3,comm_.rank());

        recv_dat.resize(3);


        auto& send_comm=task_manager_.send_communicator<key_query_t>();
        auto& recv_comm=task_manager_.recv_communicator<key_query_t>();

        //Send random queries:
        auto task= send_comm.post(&task_dat, 0);
        if(comm_.rank()==1)
        {
            auto task= send_comm.post(&task_dat2, 0);
        }

        int count=0;
        while(true)
        {
            send_comm.send();
            auto finished_tasks=send_comm.check();
            for(auto& e : finished_tasks )
            {
                recv_dat.push_back(std::vector<int>(3,0));
                auto answer=recv_comm.post_answer(e,&recv_dat[count++]);
            }
            if(send_comm.done())
                break;
        }
        
        //Busy wait:
        while(!recv_comm.done())
        {
            recv_comm.receive();
            auto ft=recv_comm.check();
            for(auto& e : ft)
            {
            std::cout<<"Received answer: ";
                for(auto& d: e->data()) std::cout<<d<<" ";
                std::cout<<std::endl;
            }
        }
    }
    void disconnect()
    {
        const auto tag=tag_gen().get<tags::connection>(comm_.rank());
        comm_.send(0,tag, false);
    }

private:
    Domain* domain_;
    communicator_type comm_;
    task_manager_t task_manager_;
    std::vector<std::vector<int>> recv_dat; //Dummy
};

}

#endif
