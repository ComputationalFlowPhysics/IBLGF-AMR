#ifndef DOMAIN_INCLUDED_CLIENT_HPP
#define DOMAIN_INCLUDED_CLIENT_HPP

#include <vector>
#include <stdexcept>

#include <global.hpp>
#include <boost/mpi/communicator.hpp>
#include <domain/mpi/client_base.hpp>
#include <domain/mpi/query_registry.hpp>
#include <domain/mpi/task_manager.hpp>

namespace domain
{

//FIXME: Do better with the namespace 
using namespace sr_mpi;


template<class Domain>
struct ClientTraits 
{
    using domain_t = Domain;
    using key_t  = typename  domain_t::key_t;
    using key_query_t = Task<tags::key_query,std::vector<key_t>>;
    using rank_query_t = Task<tags::key_query,std::vector<int>>;
    using task_manager_t = TaskManager<key_query_t, rank_query_t>;
};

/** @brief ProcessType Client 
 *  Worker process, who stores only a sub-domain 
 */
template<class Domain>
class Client : public ClientBase<ClientTraits<Domain>>
{

public:
    using domain_t = Domain;
    using communicator_type  = typename  domain_t::communicator_type;
    using octant_t  = typename  domain_t::octant_t;
    using datablock_t  = typename  domain_t::datablock_t;
    using key_t  = typename  domain_t::key_t;

    using trait_t =  ClientTraits<Domain>;
    using key_query_t = typename trait_t::key_query_t;
    using rank_query_t = typename trait_t::rank_query_t;
    using task_manager_t =typename trait_t::task_manager_t;
 


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
    void test()
    {
        auto& send_comm=
            this->task_manager_.template send_communicator<key_query_t>();

        std::vector<key_t> task_dat(3);
        std::vector<int> recvData;
        auto task= send_comm.post_task(&task_dat, 0);
        QueryRegistry<key_query_t, rank_query_t> mq;
        mq.register_recvMap([&recvData](int i){return &recvData;} );
        this->wait(mq);
    }


private:
    Domain* domain_;
    communicator_type comm_;
};

}

#endif
