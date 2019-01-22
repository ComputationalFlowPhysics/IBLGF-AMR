#ifndef DOMAIN_INCLUDED_CLIENT_HPP
#define DOMAIN_INCLUDED_CLIENT_HPP

#include <vector>
#include <stdexcept>

#include <global.hpp>
#include <boost/mpi/communicator.hpp>
#include <domain/mpi/client_base.hpp>
#include <domain/mpi/query_registry.hpp>
#include <domain/mpi/task_manager.hpp>
#include <domain/mpi/server_base.hpp>
#include "serverclient_traits.hpp"

namespace domain
{

//FIXME: Do better with the namespace 
using namespace sr_mpi;


/** @brief ProcessType Client 
 *  Worker process, who stores only a sub-domain 
 */
template<class Domain>
class Client : public ClientBase<ServerClientTraits<Domain>>
{

public:
    using domain_t = Domain;
    using communicator_type  = typename  domain_t::communicator_type;
    using octant_t  = typename  domain_t::octant_t;
    using datablock_t  = typename  domain_t::datablock_t;
    using key_t  = typename  domain_t::key_t;
    using key_coord_t = typename key_t::coordinate_type;

    using trait_t =  ServerClientTraits<Domain>;
    using super_type = ClientBase<trait_t>;

    using key_query_t = typename trait_t::key_query_t;
    using rank_query_t = typename trait_t::rank_query_t;
    using task_manager_t =typename trait_t::task_manager_t;

    using intra_client_server_t = ServerBase<trait_t>;

 protected:
    using super_type::comm_;
    using super_type::task_manager_;


public:
    Client(const Client&  other) = default;
    Client(      Client&& other) = default;
    Client& operator=(const Client&  other) & = default;
    Client& operator=(      Client&& other) & = default;
    ~Client() = default;

    Client(Domain* _d, communicator_type _comm =communicator_type())
    :domain_(_d), intra_server(this->task_manager_)
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

    //Some dummy test to send to quey some ranks for sme octants
    void rank_query()
    {
        auto& send_comm=
            task_manager_->template send_communicator<key_query_t>();


        key_coord_t c(comm_.rank()*4);
        std::vector<key_t> task_dat;
        for(int i =0;i<3;++i)
        {
            task_dat.emplace_back(c, domain_->tree()->base_level());
        }
        
        auto task= send_comm.post_task(&task_dat, 0);
        QueryRegistry<key_query_t, rank_query_t> mq;

        std::vector<int> recvData;
        mq.register_recvMap([&recvData](int i){return &recvData;} );
        this->wait(mq);

        std::cout<<"Query results: key-rank"<<std::endl;
        for(std::size_t i =0; i< task_dat.size();++i)
        {
            std::cout<<"key: "<<task_dat[i]<<" rank "<<recvData[i]<<std::endl; 
        }
    }

    template<class Field>
    void communicate_induced_fields()
    {
        //send message:
        
        auto& send_comm=
            task_manager_->template send_communicator<key_query_t>();

        int rank=comm_.rank();
        int rank_other=rank==1?2:1;

        std::vector<int> sendDat(3,rank);
        std::vector<int> recvData;

        auto task= send_comm.post_task(&sendDat, rank_other);

        QueryRegistry<key_query_t, rank_query_t> mq;
        mq.register_recvMap([&recvData](int i){return &recvData;} );

        for (auto it  = domain_.begin_leafs();
                  it != domain_.end_leafs(); ++it)
        {

           auto view(it->data()->node_field().domain_view());
           auto& nodes_domain=it->data()->nodes_domain();
           for(auto it2=nodes_domain.begin();it2!=nodes_domain.end();++it2 )
           {

           }
        }

        //Setup server to:
        //inta_server.d
        //InlineQueryRegistry<rank_query_t, key_query_t> mq(comm_.size());
        //mq.register_completeFunc([this](auto _task, auto _answerData)
        //{
        //    this->get_octant_rank(_task, _answerData);
        //});

        //this->run_query(mq);

        


    }




private:
    Domain* domain_;
    intra_client_server_t intra_server;
};

}

#endif
