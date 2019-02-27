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
    using MASK_LIST = typename domain_t::octant_t::MASK_LIST;
    using communicator_type  = typename  domain_t::communicator_type;
    using octant_t  = typename  domain_t::octant_t;
    using datablock_t  = typename  domain_t::datablock_t;
    using key_t  = typename  domain_t::key_t;
    using key_coord_t = typename key_t::coordinate_type;

    using trait_t =  ServerClientTraits<Domain>;
    using super_type = ClientBase<trait_t>;

    using key_query_t  = typename trait_t::key_query_t;
    using rank_query_t = typename trait_t::rank_query_t;

    template<template<class>class BufferPolicy=OrAssignRecv>
    using mask_query_t = typename trait_t::template
                                        mask_query_t<BufferPolicy>;

    template<template<class>class BufferPolicy=AddAssignRecv>
    using induced_fields_task_t = typename trait_t::template
                                        induced_fields_task_t<BufferPolicy>;

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
        //boost::mpi::communicator world;
        //std::cout<<"I am a client on rank: "<<world.rank()<<std::endl;
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

    void check_induced_field_communication()
    {
        auto& send_comm=
            task_manager_-> template
            send_communicator<induced_fields_task_t<AddAssignRecv>>();
        auto& recv_comm=
            task_manager_->template
            recv_communicator<induced_fields_task_t<AddAssignRecv>>();
            send_comm.start_communication();
            recv_comm.start_communication();
            send_comm.finish_communication();
            recv_comm.finish_communication();
    }

    void finish_induced_field_communication()
    {
        auto& send_comm=
            task_manager_-> template
            send_communicator<induced_fields_task_t<AddAssignRecv>>();
        auto& recv_comm=
            task_manager_->template
            recv_communicator<induced_fields_task_t<AddAssignRecv>>();
        while(true)
        {
            send_comm.start_communication();
            recv_comm.start_communication();
            send_comm.finish_communication();
            recv_comm.finish_communication();
            if(send_comm.done() && recv_comm.done() )
                break;
        }
    }


    void communicate_mask_single_level_inf_sync(int level, int mask_id, bool _neighbors)
    {
        boost::mpi::communicator  w;

        auto& send_comm=
            task_manager_-> template
                send_communicator<mask_query_t<OrAssignRecv>>();

        auto& recv_comm=
            task_manager_-> template
                recv_communicator<mask_query_t<OrAssignRecv>>();

        const int myRank=w.rank();

        for (auto it = domain_->begin(level);
                it != domain_->end(level);
                ++it)
        {
            const auto idx=get_octant_idx(it);

            if(it->locally_owned())
            {
                //Check if there are ghost children

                std::set<int> unique_inflRanks;

                for(std::size_t i = 0; i< it->influence_number(); ++i)
                {
                    const auto inf=it->influence(i);
                    if(inf && inf->rank()!=myRank)
                    {
                        unique_inflRanks.insert(inf->rank());
                    }
                }

                if(_neighbors)
                {
                    for(int i = 0; i< it->nNeighbors(); ++i)
                    {
                        const auto inf=it->neighbor(i);
                        if(inf && inf->rank()!=myRank)
                        {
                            unique_inflRanks.insert(inf->rank());
                        }
                    }
                }

                for(auto& r: unique_inflRanks)
                {
                    auto mask_ptr=it->mask_ptr(mask_id);

                    auto task = send_comm.post_task( mask_ptr, r, true,  idx);
                    task->requires_confirmation()=false;

                }

            } else
            {
                bool is_influenced=false;

                //Check influence list
                for(std::size_t i = 0; i< it->influence_number(); ++i)
                {
                  const auto inf=it->influence(i);
                  if(inf && inf->rank()==myRank)
                  { is_influenced=true ; break;}

                }

                if(_neighbors)
                {
                    for(int i = 0; i< it->nNeighbors(); ++i)
                    {
                        const auto inf=it->neighbor(i);
                        if(inf && inf->rank()==myRank)
                        { is_influenced=true ; break;}
                    }
                }

                if( is_influenced )
                {
                    auto mask_ptr=it->mask_ptr(mask_id);
                    auto task= recv_comm.post_task(mask_ptr, it->rank(), true, idx);

                    task->requires_confirmation()=false;
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

    void communicate_mask_single_level_updownward_OR(int level, int mask_id, bool _upward)
    {

        auto& send_comm=
            task_manager_-> template
                send_communicator<mask_query_t<OrAssignRecv>>();

        auto& recv_comm=
            task_manager_-> template
                recv_communicator<mask_query_t<OrAssignRecv>>();

        boost::mpi::communicator  w;
        const int myRank=w.rank();

        for (auto it = domain_->begin(level);
                it != domain_->end(level);
                ++it)
        {
            const auto idx=get_octant_idx(it);

            if(it->locally_owned())
            {
                //Check if there are ghost children

                const auto unique_ranks=it->unique_child_ranks();
                for(auto r : unique_ranks)
                {
                    if(_upward)
                    {
                        auto mask_ptr=it->mask_ptr(mask_id);

                        auto task=recv_comm.post_task( mask_ptr, r, true, idx );
                        task->requires_confirmation()=false;

                    } else
                    {
                        auto mask_ptr=it->mask_ptr(mask_id);

                        auto task=send_comm.post_task( mask_ptr, r, true, idx );
                        task->requires_confirmation()=false;
                    }

                }
            } else
            {
                //Check if ghost has locally_owned children

                if(it->has_locally_owned_children())
                {
                    if(_upward)
                    {
                        auto mask_ptr=it->mask_ptr(mask_id);

                        auto task=send_comm.post_task(mask_ptr, it->rank(), true,idx);
                        task->requires_confirmation()=false;
                    } else
                    {
                        auto mask_ptr=it->mask_ptr(mask_id);

                        auto task=recv_comm.post_task(mask_ptr, it->rank(), true,idx);
                        task->requires_confirmation()=false;
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

    void communicate_mask_single_level_child_sync(int level, int mask_id)
    {

        auto& send_comm=
            task_manager_-> template
                send_communicator<mask_query_t<OrAssignRecv>>();

        auto& recv_comm=
            task_manager_-> template
                recv_communicator<mask_query_t<OrAssignRecv>>();

        boost::mpi::communicator  w;
        const int myRank=w.rank();

        for (auto it = domain_->begin(level);
                it != domain_->end(level);
                ++it)
        {

            if(it->locally_owned())
            {
                //Check if there are ghost children
                for(int c=0;c<it->num_children();++c)
                {
                    auto child = it->child(c);

                    if(!child) continue;
                    if(!child->locally_owned())
                    {
                        auto idx=get_octant_idx(child);

                        auto r = child->rank();
                        auto mask_ptr=child->mask_ptr(mask_id);

                        auto task=recv_comm.post_task( mask_ptr, r, true, idx );
                        task->requires_confirmation()=false;
                    }
                }
            } else
            {
                for(int c=0;c<it->num_children();++c)
                {
                    auto child = it->child(c);

                    if(!child) continue;
                    if(child->locally_owned())
                    {
                        auto idx=get_octant_idx(child);
                        auto r = it->rank();
                        auto mask_ptr=child->mask_ptr(mask_id);

                        auto task=send_comm.post_task( mask_ptr, r, true, idx );
                        task->requires_confirmation()=false;
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

    template<class SendField,class RecvField, class Octant_t>
    int communicate_induced_fields_recv_m_send_count( Octant_t it, bool _neighbors=false )
    {
        int count = 0;
        boost::mpi::communicator  w;
        const int myRank=w.rank();

        if( !it->locally_owned() )
        {

            //Check if this ghost octant influenced by octants of this rank
            bool is_influenced=false;

            //Check influence list
            for(std::size_t i = 0; i< it->influence_number(); ++i)
            {
                const auto inf=it->influence(i);
                if(inf && inf->rank()==myRank && inf->mask(MASK_LIST::Mask_FMM_Source))
                {return -1;}

            }

            if(_neighbors)
            {
                for(int i = 0; i< it->nNeighbors(); ++i)
                {
                    const auto inf=it->neighbor(i);
                    if(inf && inf->rank()==myRank && inf->mask(MASK_LIST::Mask_FMM_Source))
                    {return -1;}
                }
            }

        } else
        {

            std::set<int> unique_inflRanks;

            for(std::size_t i = 0; i< it->influence_number(); ++i)
            {
                const auto inf=it->influence(i);
                if(inf && inf->rank()!=myRank && inf->mask(MASK_LIST::Mask_FMM_Source))
                {
                    count++;
                }
            }

            if(_neighbors)
            {
                for(int i = 0; i< it->nNeighbors(); ++i)
                {
                    const auto inf=it->neighbor(i);
                    if(inf && inf->rank()!=myRank && inf->mask(MASK_LIST::Mask_FMM_Source))
                    {
                        count++;
                    }
                }
            }

        }
    return count;
    }

    template<class SendField,class RecvField, class Octant_t>
    void communicate_induced_fields( Octant_t it, bool _neighbors=false )
    {

        if (!it->mask(MASK_LIST::Mask_FMM_Target)) return;

        boost::mpi::communicator w;

        auto& send_comm=
            task_manager_-> template
            send_communicator<induced_fields_task_t<AddAssignRecv>>();
        auto& recv_comm=
            task_manager_->template
            recv_communicator<induced_fields_task_t<AddAssignRecv>>();

        const int myRank=w.rank();

        const auto idx=get_octant_idx(it);

        if( !it->locally_owned() )
        {

            //Check if this ghost octant influenced by octants of this rank
            bool is_influenced=false;

            //Check influence list
            for(std::size_t i = 0; i< it->influence_number(); ++i)
            {
                const auto inf=it->influence(i);
                if(inf && inf->rank()==myRank && inf->mask(MASK_LIST::Mask_FMM_Source))
                { is_influenced=true ; break;}

            }

            if(_neighbors)
            {
                for(int i = 0; i< it->nNeighbors(); ++i)
                {
                    const auto inf=it->neighbor(i);
                    if(inf && inf->rank()==myRank && inf->mask(MASK_LIST::Mask_FMM_Source))
                    { is_influenced=true ; break;}
                }
            }

            if( is_influenced )
            {
                auto send_ptr=it->data()->
                template get<SendField>().date_ptr();
                auto task= send_comm.post_task(send_ptr, it->rank(), true, idx);
                task->requires_confirmation()=false;

            }

        } else
        {

            std::set<int> unique_inflRanks;

            for(std::size_t i = 0; i< it->influence_number(); ++i)
            {
                const auto inf=it->influence(i);
                if(inf && inf->rank()!=myRank && inf->mask(MASK_LIST::Mask_FMM_Source))
                {
                    unique_inflRanks.insert(inf->rank());
                }
            }

            if(_neighbors)
            {
                for(int i = 0; i< it->nNeighbors(); ++i)
                {
                    const auto inf=it->neighbor(i);
                    if(inf && inf->rank()!=myRank && inf->mask(MASK_LIST::Mask_FMM_Source))
                    {
                        unique_inflRanks.insert(inf->rank());
                    }
                }
            }

            for(auto& r: unique_inflRanks)
            {
                const auto recv_ptr=it->data()->
                template get<RecvField>().date_ptr();
                auto task = recv_comm.post_task( recv_ptr, r, true, idx);
                task->requires_confirmation()=false;

            }
        }

        //Start communications

        send_comm.start_communication();
        recv_comm.start_communication();
        send_comm.finish_communication();
        recv_comm.finish_communication();

    }

    /** @brief Communicate induced fields per level */
    template<class SendField,class RecvField>
    void communicate_induced_fields_old( int level, bool _neighbors=false )
    {

        boost::mpi::communicator w;

        auto& send_comm=
            task_manager_-> template
            send_communicator<induced_fields_task_t<AddAssignRecv>>();
        auto& recv_comm=
            task_manager_->template
            recv_communicator<induced_fields_task_t<AddAssignRecv>>();

        const int myRank=w.rank();

        for (auto it  = domain_->begin(level); it != domain_->end(level); ++it)
        {

            if (!it->mask(MASK_LIST::Mask_FMM_Target)) continue;

            const auto idx=get_octant_idx(it);

            if( !it->locally_owned() )
            {

                //Check if this ghost octant influenced by octants of this rank
                bool is_influenced=false;

                //Check influence list
                for(std::size_t i = 0; i< it->influence_number(); ++i)
                {
                  const auto inf=it->influence(i);
                  if(inf && inf->rank()==myRank && inf->mask(MASK_LIST::Mask_FMM_Source))
                  { is_influenced=true ; break;}

                }

                if(_neighbors)
                {
                    for(int i = 0; i< it->nNeighbors(); ++i)
                    {
                        const auto inf=it->neighbor(i);
                        if(inf && inf->rank()==myRank && inf->mask(MASK_LIST::Mask_FMM_Source))
                        { is_influenced=true ; break;}
                    }
                }

                if( is_influenced )
                {
                  auto send_ptr=it->data()->
                    template get<SendField>().date_ptr();
                  auto task= send_comm.post_task(send_ptr, it->rank(), true, idx);
                  task->requires_confirmation()=false;

                }
            } else
            {

                std::set<int> unique_inflRanks;

                for(std::size_t i = 0; i< it->influence_number(); ++i)
                {
                    const auto inf=it->influence(i);
                    if(inf && inf->rank()!=myRank && inf->mask(MASK_LIST::Mask_FMM_Source))
                    {
                        unique_inflRanks.insert(inf->rank());
                    }
                }

                if(_neighbors)
                {
                    for(int i = 0; i< it->nNeighbors(); ++i)
                    {
                        const auto inf=it->neighbor(i);
                        if(inf && inf->rank()!=myRank && inf->mask(MASK_LIST::Mask_FMM_Source))
                        {
                            unique_inflRanks.insert(inf->rank());
                        }
                    }
                }

                for(auto& r: unique_inflRanks)
                {
                    const auto recv_ptr=it->data()->
                                template get<RecvField>().date_ptr();
                    auto task = recv_comm.post_task( recv_ptr, r, true,  idx);
                    task->requires_confirmation()=false;

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
            auto tts= recv_comm.finish_communication();

            if(send_comm.done() && recv_comm.done() )
                break;
        }
    }


    /** @brief communicate fields for up/downward pass of fmm */
    template<class SendField,class RecvField, template<class>class BufferPolicy>
    void communicate_updownward_pass(int level, bool _upward)
    {
        int mask_id=(_upward) ?
                MASK_LIST::Mask_FMM_Source : MASK_LIST::Mask_FMM_Target;

        boost::mpi::communicator w;

        auto& send_comm=
            task_manager_-> template
                send_communicator<induced_fields_task_t<BufferPolicy>>();
        auto& recv_comm=
            task_manager_-> template
                recv_communicator<induced_fields_task_t<BufferPolicy>>();

        for (auto it  = domain_->begin(level); it != domain_->end(level); ++it)
        {

            if (!it->mask(mask_id)) continue;

            const auto idx=get_octant_idx(it);

            if(it->locally_owned() && it->data() )
            {

                const auto unique_ranks=it->unique_child_ranks(mask_id);
                for(auto r : unique_ranks)
                {
                    if(_upward)
                    {
                        auto data_ptr=it->data()->
                            template get<RecvField>().date_ptr();
                        auto task=recv_comm.post_task( data_ptr, r, true, idx);
                        task->requires_confirmation()=false;

                    } else
                    {
                        auto data_ptr=it->data()->
                            template get<SendField>().date_ptr();
                        auto task= send_comm.post_task(data_ptr,r,true,idx);
                        task->requires_confirmation()=false;

                    }
                }
            }

            //Check if ghost has locally_owned children
            if(!it->locally_owned() && it->data())
            {
                if(it->has_locally_owned_children(mask_id))
                {
                    if(_upward)
                    {
                        const auto data_ptr=it->data()->
                            template get<SendField>().date_ptr();
                        auto task =
                            send_comm.post_task(data_ptr, it->rank(),
                                    true,idx);
                        task->requires_confirmation()=false;

                    } else
                    {
                        const auto data_ptr=it->data()->
                            template get<RecvField>().date_ptr();
                        auto task =
                            recv_comm.post_task(data_ptr, it->rank(),
                                    true,idx);
                        task->requires_confirmation()=false;
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

    /** @brief communicate fields for up/downward pass of fmm */
    template<class SendField,class RecvField >
    void communicate_updownward_add(int level, bool _upward)
    {
        communicate_updownward_pass<SendField,RecvField,AddAssignRecv>
        (level, _upward);
    }

    template<class SendField,class RecvField >
    void communicate_updownward_assign(int level, bool _upward)
    {
        communicate_updownward_pass<SendField,RecvField,CopyAssign>
        (level,_upward);
    }

    /** @brief communicate fields for up/downward pass of fmm */
    //TODO: Make it better and put in octant
    template<class T>
    auto get_octant_idx(T it) const noexcept
    {
        const auto cc=it->tree_coordinate();
        return cc.x()+100*cc.y()+100*100*cc.z() +
            100*100*100*(it->level()) + 1;
    }

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
