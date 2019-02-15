#ifndef DOMAIN_INCLUDED_CLIENT_HPP
#define DOMAIN_INCLUDED_CLIENT_HPP

#include <vector>
#include <stdexcept>

#include <global.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/serialization/vector.hpp>
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
    using induced_fields_task_t = typename trait_t::induced_fields_task_t;
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
            _o->rank()=comm_.rank();
        });
    }

    auto rank_query(std::vector<key_t>& task_dat)
    {
        auto& send_comm=
            task_manager_->template send_communicator<key_query_t>();

        auto task= send_comm.post_task(&task_dat, 0);
        QueryRegistry<key_query_t, rank_query_t> mq;

        std::vector<int> recvData;
        mq.register_recvMap([&recvData](int i){return &recvData;} );
        this->wait(mq);
        return recvData;
    }


    /** @brief Communicate induced fields per level 
     */
    template<class SendField,class RecvField>
    void communicate_induced_fields( int _level )
    {
        std::cout<<"Comunicating induced fields of "
                 <<SendField::name()<<" to "<<RecvField::name()<<std::endl;

        auto& send_comm=
            task_manager_->template send_communicator<induced_fields_task_t>();
        auto& recv_comm=
            task_manager_->template recv_communicator<induced_fields_task_t>();

        boost::mpi::communicator  w; 
        const int myRank=w.rank();
         
        for (auto it  = domain_->begin(_level);
                  it != domain_->end(_level); ++it)
        {
            const auto cc=it->tree_coordinate();
            const auto idx=cc.x()+100*cc.y()+100*100*cc.z() + 100*100*100*it->level();
            if(it->rank()!=myRank )
            {
                bool is_influenced=false;
                //This should probably be stored
                for(std::size_t i = 0; i< it->influence_number(); ++i)
                {
                    auto inf=it->influence(i);
                    if(inf && inf->rank()==myRank)
                    { is_influenced=true ; break;} 
                }

               if(is_influenced )
               {
                   auto send_ptr=it->data()->
                       template get<SendField>().date_ptr();
                   auto task= send_comm.post_task(send_ptr, it->rank(), true, idx);
                   task->requires_confirmation()=false;

                   //std::cout<<"Sending field from "<<myRank<<" to "<<it->rank()
                   //    <<" index " <<it->key()._index
                   //    <<" c "<<it->global_coordinate()
                   //    <<" tag: "<<task->id()
                   //    <<std::endl;
               }
            }
            else
            {
                std::set<int> unique_inflRanks;
                for(std::size_t i = 0; i< it->influence_number(); ++i)
                {
                    auto inf=it->influence(i);
                    if(inf && inf->rank()!=myRank )
                    {
                        unique_inflRanks.insert(inf->rank());
                    }
                }
                for(auto& r: unique_inflRanks)
                {
                    const auto recv_ptr=it->data()->
                                template get<RecvField>().date_ptr();
                    auto task = recv_comm.post_task( recv_ptr, r, true,  idx);
                    task->requires_confirmation()=false;

                    //std::cout<<"Recv field from "<<r<<" to "<<myRank
                    //    <<" index " <<it->key()._index
                    //    <<" iindex " <<static_cast<int>(it->key()._index)
                    //    <<" c "<<it->global_coordinate()
                    //    <<" tag: "<<task->id()
                    //    <<std::endl;
                }
            }

            //Try starting the communication 
            send_comm.start_communication();
            recv_comm.start_communication();
        }
                    

        //Start communications
        while(true)
        {
            //buffer and send it 
            send_comm.start_communication();
            recv_comm.start_communication();

            //Check if something has finished
            send_comm.finish_communication();
            const auto recv_tasks=recv_comm.finish_communication();
            if(send_comm.done() && recv_comm.done() )
                break;
        }
    }


    template<class SendField,class RecvField=SendField>
    void communicate_updownward_pass(bool _upward)
    {
        std::cout<<"Comunicating fields for downward pass of "
                 <<SendField::name()<<" to "<<RecvField::name()<<std::endl;

        auto& send_comm=
            task_manager_->
            template send_communicator<induced_fields_task_t>();
        auto& recv_comm=
            task_manager_->
            template recv_communicator<induced_fields_task_t>();

        boost::mpi::communicator  w; 
        const int myRank=w.rank();

        for (int ls = domain_->tree()->base_level();
                 ls >= domain_->tree()->depth(); ++ls)
        {
            for (auto it  = domain_->begin(ls);
                      it != domain_->end(ls); ++it)
            {
                const auto idx=get_octant_idx(it);

                //Check if there are ghost children
                const auto unique_ranks=it->unique_child_ranks();
                for(auto r : unique_ranks)
                {
                    auto data_ptr=it->data()->
                        template get<SendField>().date_ptr();
                    if(_upward) 
                    {
                        auto task=recv_comm.post_task( data_ptr,r, true,idx);
                        task->requires_confirmation()=false;
                    } 
                    else
                    {
                        auto task= send_comm.post_task(data_ptr,r,true,idx);
                        task->requires_confirmation()=false;
                    }
                }

                //Check if ghost has locally_owned children 
                if(!it->locally_owned() && it->data() && !it->is_leaf())
                {
                    if(it->has_locally_owned_children())
                    {
                        const auto recv_ptr=it->data()->
                            template get<RecvField>().date_ptr();
                        if(_upward)
                        {
                            auto task = 
                                send_comm.post_task(recv_ptr, it->rank(),
                                                    true,idx);
                            task->requires_confirmation()=false;
                        }
                        else
                        {
                            auto task = 
                                recv_comm.post_task(recv_ptr, it->rank(), 
                                                    true,idx);
                            task->requires_confirmation()=false;
                        }

                    }
                }
            }
        }
       
        //Start communications
        while(true)
        {
            //buffer and send it 
            send_comm.start_communication();
            recv_comm.start_communication();

            //Check if something has finished
            send_comm.finish_communication();
            const auto recv_tasks=recv_comm.finish_communication();
            if(send_comm.done() && recv_comm.done() )
                break;
        }
    }
  /*************************************************************************/
    //TODO put in octant
    template<class T>
    auto get_octant_idx(T it) const noexcept
    {
        const auto cc=it->tree_coordinate();
        return cc.x()+100*cc.y()+100*100*cc.z() +
            100*100*100*it->level();
    }

    /*************************************************************************/


    /** @brief Query ranks of the neighbors, influence octants, children and
     *         parents which do not belong to this processor.
     *
     */
    void query_octants()
    {
        //Octant initialization function
        auto f =[&](octant_t* _o){
            auto bbase=domain_->tree()->octant_to_level_coordinate(
                    _o->tree_coordinate());
            _o->data()=std::make_shared<datablock_t>(bbase, 
                    domain_->block_extent(),_o->refinement_level(), true);
        };
        domain_->tree()->construct_maps(this,f);
    }
    
    auto domain()const{return domain_;}


private:
    Domain* domain_;
    intra_client_server_t intra_server;
};

}

#endif
