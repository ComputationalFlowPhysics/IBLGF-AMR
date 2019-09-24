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
#include <domain/mpi/haloCommunicator.hpp>
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
    using fields_tuple_t = typename datablock_t::fields_tuple_t;
    using key_t  = typename  domain_t::key_t;
    using key_coord_t = typename key_t::coordinate_type;
    using fmm_mask_type = typename octant_t::fmm_mask_type;

    using trait_t =  ServerClientTraits<Domain>;
    using super_type = ClientBase<trait_t>;

    //QueryTypes
    using key_query_t             = typename trait_t::key_query_t;
    using rank_query_t            = typename trait_t::rank_query_t;
    using mask_init_query_send_t  = typename trait_t::mask_init_query_send_t;
    using mask_init_query_recv_t  = typename trait_t::mask_init_query_recv_t;
    using leaf_query_send_t       = typename trait_t::leaf_query_send_t;
    using leaf_query_recv_t       = typename trait_t::leaf_query_recv_t;
    using correction_query_send_t = typename trait_t::correction_query_send_t;
    using correction_query_recv_t = typename trait_t::correction_query_recv_t;

    template<template<class>class BufferPolicy=OrAssignRecv>
    using mask_query_t = typename trait_t::template
                                        mask_query_t<BufferPolicy>;

    //TaskTypes
    template<template<class>class BufferPolicy=AddAssignRecv>
    using induced_fields_task_t = typename trait_t::template
                                        induced_fields_task_t<BufferPolicy>;
    using acc_induced_fields_task_t = typename trait_t::acc_induced_fields_task_t;
    using halo_task_t = typename trait_t::halo_task_t;

    template<class Field>
    using halo_communicator_t=HaloCommunicator<halo_task_t, Field, domain_t>;
    template<class... Fields>
    using halo_communicator_template_t =
        std::tuple<halo_communicator_t<Fields>...>;
    using halo_communicators_tuple_t =
        typename tuple_utils::make_from_tuple<
            halo_communicator_template_t, fields_tuple_t>::type;


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
    :domain_(_d)
    {
        //boost::mpi::communicator world;
        //std::cout<<"I am a client on rank: "<<world.rank()<<std::endl;
        send_tasks_.resize(comm_.size());
        recv_tasks_.resize(comm_.size());
    }

public:
    void receive_keys()
    {
        std::vector<key_t> keys;
        comm_.recv(0,0,keys);

        //Init trees and allocate all memory
        domain_->tree()->init(keys, [&](octant_t* _o){
            auto level = _o->refinement_level();
            level=level>=0?level:0;
            auto bbase=domain_->tree()->octant_to_level_coordinate(
                    _o->tree_coordinate(),level);
            _o->data()=std::make_shared<datablock_t>(bbase,
                    domain_->block_extent(),level, true);
            _o->rank()=comm_.rank();
        });

        //Receive global depth of the tree
        int depth=domain_->tree()->depth();
        comm_.recv(0,0,depth);
        domain_->tree()->depth()=depth;
    }

    auto mask_query(std::vector<key_t>& task_dat)
    {
        auto& send_comm=
            task_manager_->template send_communicator<mask_init_query_send_t>();

        auto task= send_comm.post_task(&task_dat, 0);
        QueryRegistry<mask_init_query_send_t, mask_init_query_recv_t> mq;

        std::vector<fmm_mask_type> recvData;
        mq.register_recvMap([&recvData](int i){return &recvData;} );
        this->wait(mq);
        return recvData;
    }

    auto correction_query(std::vector<key_t>& task_dat)
    {
        auto& send_comm=
            task_manager_->template send_communicator<correction_query_send_t>();

        auto task= send_comm.post_task(&task_dat, 0);
        QueryRegistry<correction_query_send_t, correction_query_recv_t> mq;

        std::vector<bool> recvData;
        mq.register_recvMap([&recvData](int i){return &recvData;} );
        this->wait(mq);
        return recvData;
    }

    auto leaf_query(std::vector<key_t>& task_dat)
    {
        auto& send_comm=
            task_manager_->template send_communicator<leaf_query_send_t>();

        auto task= send_comm.post_task(&task_dat, 0);
        QueryRegistry<leaf_query_send_t, leaf_query_recv_t> mq;

        std::vector<bool> recvData;
        mq.register_recvMap([&recvData](int i){return &recvData;} );
        this->wait(mq);
        return recvData;
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
            send_communicator<induced_fields_task_t<InfluenceFieldBuffer>>();
        auto& recv_comm=
            task_manager_->template
            recv_communicator<induced_fields_task_t<>>();
            send_comm.start_communication();
            recv_comm.start_communication();
            send_comm.finish_communication();
            recv_comm.finish_communication();
    }

    void finish_induced_field_communication()
    {
        auto& send_comm=
            task_manager_-> template
            send_communicator<induced_fields_task_t<InfluenceFieldBuffer>>();
        auto& recv_comm=
            task_manager_->template
            recv_communicator<induced_fields_task_t<InfluenceFieldBuffer>>();

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


    void communicate_mask_single_level_inf_sync(int level, int mask_id, bool _neighbors, int fmm_mask_idx)
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
                    auto mask_ptr=it->fmm_mask_ptr(fmm_mask_idx,mask_id);

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
                    auto mask_ptr=it->fmm_mask_ptr(fmm_mask_idx,mask_id);
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
            recv_comm.finish_communication();
            if(send_comm.done() && recv_comm.done() )
                break;
        }

    }

    void communicate_mask_single_level_updownward_OR(int level, int mask_id, bool _upward, int fmm_mask_idx)
    {
        auto& send_comm=
            task_manager_-> template
                send_communicator<mask_query_t<OrAssignRecv>>();

        auto& recv_comm=
            task_manager_-> template
                recv_communicator<mask_query_t<OrAssignRecv>>();

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
                        auto mask_ptr=it->fmm_mask_ptr(fmm_mask_idx,mask_id);
                        auto task=recv_comm.post_task( mask_ptr, r, true, idx );
                        task->requires_confirmation()=false;

                    } else
                    {
                        auto mask_ptr=it->fmm_mask_ptr(fmm_mask_idx,mask_id);
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
                        auto mask_ptr=it->fmm_mask_ptr(fmm_mask_idx,mask_id);

                        auto task=send_comm.post_task(mask_ptr, it->rank(), true,idx);
                        task->requires_confirmation()=false;
                    } else
                    {
                        auto mask_ptr=it->fmm_mask_ptr(fmm_mask_idx,mask_id);

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
            recv_comm.finish_communication();
            if(send_comm.done() && recv_comm.done() )
                break;
        }

    }

    void communicate_mask_single_level_child_sync(int level, int mask_id, int fmm_mask_idx)
    {
        auto& send_comm=
            task_manager_-> template
                send_communicator<mask_query_t<OrAssignRecv>>();

        auto& recv_comm=
            task_manager_-> template
                recv_communicator<mask_query_t<OrAssignRecv>>();

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
                        auto mask_ptr=child->fmm_mask_ptr(fmm_mask_idx,mask_id);

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
                        auto mask_ptr=child->fmm_mask_ptr(fmm_mask_idx,mask_id);

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
            recv_comm.finish_communication();
            if(send_comm.done() && recv_comm.done() )
                break;
        }

    }

    template<class SendField,class RecvField, class Octant_t>
    int communicate_induced_fields_recv_m_send_count( Octant_t it, bool _neighbors, int fmm_mask_idx )
    {
        int count =0;
        boost::mpi::communicator  w;
        const int myRank=w.rank();

        //sends
        if( !it->locally_owned() )
        {

            //Check if this ghost octant influenced by octants of this rank

            //Check influence list
            for(std::size_t i = 0; i< it->influence_number(); ++i)
            {
                const auto inf=it->influence(i);
                if(inf && inf->rank()==myRank &&
                   inf->fmm_mask(fmm_mask_idx,MASK_LIST::Mask_FMM_Source))
                {
                    //return (inf->rank()+1)*1000;
                    //++count;
                    return +1000000;
                }
            }

            if(_neighbors)
            {
                for(int i = 0; i< it->nNeighbors(); ++i)
                {
                    const auto inf=it->neighbor(i);
                    if(inf && inf->rank()==myRank &&
                       inf->fmm_mask(fmm_mask_idx,MASK_LIST::Mask_FMM_Source))
                    {
                        //return (inf->rank()+1)*1000;
                        //++count;
                        return +1000000;
                    }
                }
            }
        }
        else //Receivs
        {
            //int inf_rank=-1;
            std::set<int> unique_inflRanks;
            for(std::size_t i = 0; i< it->influence_number(); ++i)
            {
                const auto inf=it->influence(i);
                if(inf && inf->rank()!=myRank &&
                   inf->fmm_mask(fmm_mask_idx,MASK_LIST::Mask_FMM_Source))
                {
                    //inf_rank = inf->rank();
                    //return (inf->rank()+1)*1000+1;
                    //return 1000000;
                    //count+=1000;
                    ++count;
                }
            }

            if(_neighbors)
            {
                for(int i = 0; i< it->nNeighbors(); ++i)
                {
                    const auto inf=it->neighbor(i);
                    if(inf && inf->rank()!=myRank &&
                       inf->fmm_mask(fmm_mask_idx,MASK_LIST::Mask_FMM_Source))
                    {
                        //return (inf->rank()+1)*1000+1;
                        //return inf->rank()+1;
                        //inf_rank = inf->rank();
                        //count+=1000;
                        //return +1000000;
                         ++count;
                    }
                }
            }

            //if (count>0)
            //    count+=(inf_rank+1)*10000000;
        }
        //if (count>0)
        //    count += it->level() * 10000000;
        //std::cout<<count<<std::endl;
        //std::cout<<it->level()<<std::endl;
        return count;
    }


    template<class SendField,class RecvField, class FMMType, class Kernel>
    void communicate_induced_fields(octant_t* it, FMMType* _fmm, Kernel* _kernel,
                                    int _level_diff, float_type _dx_level,
                                    bool _neighbors,
                                    bool _start_communication,
                                    int fmm_mask_idx)
    {
        if (!it->fmm_mask(fmm_mask_idx,MASK_LIST::Mask_FMM_Target)) return;

        boost::mpi::communicator w;
        auto& send_comm=
            task_manager_-> template
            send_communicator<induced_fields_task_t<InfluenceFieldBuffer>>();
        auto& recv_comm=
            task_manager_->template
            recv_communicator<induced_fields_task_t<InfluenceFieldBuffer>>();

        const int myRank=w.rank();
        const auto idx=get_octant_idx(it);

        if( !it->locally_owned() )
        {

            //Check if this ghost octant influenced by octants of this rank
            bool is_influenced=false;

            //Check influence list
            if (!_kernel->neighbor_only())
            {

                for(std::size_t i = 0; i< it->influence_number(); ++i)
                {
                    const auto inf=it->influence(i);
                    if(inf && inf->rank()==myRank &&
                        inf->fmm_mask(fmm_mask_idx,MASK_LIST::Mask_FMM_Source))
                    { is_influenced=true ; break;}
                }
            }

            if(_neighbors)
            {
                for(int i = 0; i< it->nNeighbors(); ++i)
                {
                    const auto inf=it->neighbor(i);
                    if(inf && inf->rank()==myRank &&
                       inf->fmm_mask(fmm_mask_idx,MASK_LIST::Mask_FMM_Source))
                    { is_influenced=true ; break;}
                }
            }

            if( is_influenced )
            {
                auto send_ptr=it->data()->
                template get<SendField>().data_ptr();

                auto task= send_comm.post_task(send_ptr, it->rank(), true, idx);
                task->attach_data(send_ptr);
                task->rank_other()=it->rank();
                task->requires_confirmation()=false;
                task->octant()=it;
                auto size = it->data()->template get<SendField>().real_block().nPoints();

                auto send_callback = [it, _fmm, _kernel,  _neighbors,
                                     _level_diff,_dx_level,size](auto& buffer_vector)
                {
                    //1. Swap buffer with sendfield
                    buffer_vector.resize(size);
                    std::fill(buffer_vector.begin(), buffer_vector.end(),0);
                    buffer_vector.swap(it->data()->
                            template get<SendField>().data());

                    //2. Compute influence field
                    _fmm->compute_influence_field(it,_kernel,
                                    _level_diff, _dx_level, _neighbors);

                    //3. Swap sendfield with buffer
                    buffer_vector.swap(it->data()->
                            template get<SendField>().data());
                };
                task->register_sendCallback(send_callback);
            }
        }
        else
        {
            std::set<int> unique_inflRanks;
            if (!_kernel->neighbor_only())
            {
                for(std::size_t i = 0; i< it->influence_number(); ++i)
                {
                    const auto inf=it->influence(i);
                    if(inf && inf->rank()!=myRank &&
                            inf->fmm_mask(fmm_mask_idx,MASK_LIST::Mask_FMM_Source))
                    {
                        unique_inflRanks.insert(inf->rank());
                    }
                }
            }

            if(_neighbors)
            {
                for(int i = 0; i< it->nNeighbors(); ++i)
                {
                    const auto inf=it->neighbor(i);
                    if(inf && inf->rank()!=myRank &&
                       inf->fmm_mask(fmm_mask_idx,MASK_LIST::Mask_FMM_Source))
                    {
                        unique_inflRanks.insert(inf->rank());
                    }
                }
            }

            for(auto& r: unique_inflRanks)
            {
                const auto recv_ptr=it->data()->
                    template get<RecvField>().data_ptr();
                auto task = recv_comm.post_task( recv_ptr, r, true, idx);

                //auto task = std::make_shared<induced_fields_task_t<InfluenceFieldBuffer>>(idx);
                task->attach_data(recv_ptr);
                task->rank_other()=r;
                task->requires_confirmation()=false;
                task->octant()=it;
                //recv_tasks_[r].push_back(task);
            }
        }

        //Start communications
        if(_start_communication)
        {
            send_comm.start_communication();
            recv_comm.start_communication();
            send_comm.finish_communication();
            recv_comm.finish_communication();
        }
    }

    template<class SendField, class RecvField>
    void combine_induced_field_messages() noexcept
    {
        auto& send_comm=
            task_manager_-> template
            send_communicator<induced_fields_task_t<InfluenceFieldBuffer>>();
        auto& recv_comm=
            task_manager_->template
            recv_communicator<induced_fields_task_t<InfluenceFieldBuffer>>();
        
        send_comm.pack_messages();
        recv_comm.pack_messages();
    }

    template<class SendField, class RecvField>
    void combine_induced_field_messages_old() noexcept
    {
        std::cout<<"I need to edit this to account for new memory friendly version"<<std::endl;
        auto& acc_send_comm=
            task_manager_-> template send_communicator<acc_induced_fields_task_t>();
        auto& acc_recv_comm=
            task_manager_-> template recv_communicator<acc_induced_fields_task_t>();


        if(send_fields_.size()!= static_cast<std::size_t>(comm_.size()))
            send_fields_.resize(comm_.size());
        if(recv_fields_.size()!= static_cast<std::size_t>(comm_.size()))
            recv_fields_.resize(comm_.size());

        //SendField
        for(std::size_t rank_other=0; rank_other<send_tasks_.size();++rank_other)
        {
            auto& tasks=send_tasks_[rank_other];
            if (tasks.empty()) continue;
            std::sort(tasks.begin(),tasks.end(),
            [&](const auto& c0, const auto& c1)
            {
                return c0->octant()->key().id()< c1->octant()->key().id();
            });
            std::size_t size=0;
            for(auto& task : tasks )
            {
                const auto& dat= task->octant()->data()->template get<SendField>().data();
                size+=dat.size();
            }
            if(size!=send_fields_[rank_other].size())
                send_fields_[rank_other].resize(size);

            //Generate a task for each send/recv rank
            int count=0;
            int idx=-1;
            for(auto& task : tasks )
            {
                if(count==0) idx=get_octant_idx(task->octant());
                const auto& dat= task->octant()->data()->template get<SendField>().data();
                for(std::size_t i=0;i<dat.size();++i)
                {
                    send_fields_[rank_other][count++]=dat[i];
                }
            }
            if(idx>=0)
            {
                auto accumulated_task=
                    acc_send_comm.post_task(&send_fields_[rank_other], rank_other, true, idx);
            }
        }

        //RecvField
        for(std::size_t rank_other=0; rank_other<recv_tasks_.size();++rank_other)
        {
            auto& tasks=recv_tasks_[rank_other];
            if (tasks.empty()) continue;
            std::sort(tasks.begin(),tasks.end(),
                    [&](const auto& c0, const auto& c1)
                    {
                        return c0->octant()->key().id()< c1->octant()->key().id();
                    });
            int idx=-1;
            int count=0;
            for(auto& task : tasks )
            {
                if(count==0) idx=get_octant_idx(task->octant());
                ++count;
                break;
            }

            if(idx>=0)
            {
                auto accumulated_task =
                    acc_recv_comm.post_task(&recv_fields_[rank_other], rank_other, true, idx);
            }
        }

        acc_send_comm.start_communication();
        acc_recv_comm.start_communication();
    }

    template<class SendField, class RecvField>
    void check_combined_induced_field_communication(bool _finish=false) noexcept
    {
        auto& send_comm=
            task_manager_-> template
            send_communicator<induced_fields_task_t<InfluenceFieldBuffer>>();
        auto& recv_comm=
            task_manager_->template
            recv_communicator<induced_fields_task_t<InfluenceFieldBuffer>>();


        while(true)
        {
            send_comm.unpack_messages();
            recv_comm.unpack_messages();
            if( !_finish || (recv_comm.done() && send_comm.done())  )
                break;
        }
        
        if(_finish)
        {
            recv_comm.clear();
            send_comm.clear();
        }
    }


    template<class SendField, class RecvField>
    void check_combined_induced_field_communication_old(bool _finish=false) noexcept
    {
        auto& acc_send_comm=
            task_manager_-> template send_communicator<acc_induced_fields_task_t>();
        auto& acc_recv_comm=
            task_manager_-> template recv_communicator<acc_induced_fields_task_t>();

        while(true)
        {
            acc_send_comm.start_communication();
            acc_recv_comm.start_communication();
            acc_send_comm.finish_communication();
            auto finished_tasks=acc_recv_comm.finish_communication();

            for(auto& t : finished_tasks)
            {
                for(std::size_t i = 0; i<recv_tasks_[t->rank_other()].size(); ++i)
                {
                    auto& octant_field=recv_tasks_[t->rank_other()][i]->
                        octant()->data()->template get<RecvField>().data();

                    for(std::size_t j=0; j<octant_field.size();++j)
                    {
                        octant_field[j]+=recv_fields_[t->rank_other()][i*octant_field.size()+j];
                    }
                }
            }
            if( !_finish || (acc_recv_comm.done() && acc_send_comm.done())  )
                break;
        }
        if(_finish)
        {
            send_fields_.clear();
            recv_fields_.clear();
            send_tasks_.clear();
            recv_tasks_.clear();
            send_tasks_.resize(comm_.size());
            recv_tasks_.resize(comm_.size());
        }
    }


    /** @brief communicate fields for up/downward pass of fmm */
    template<class SendField,class RecvField,
             template<class>class BufferPolicy,
             class OctantPtr>
    void communicate_updownward_pass(OctantPtr it, bool _upward, int fmm_mask_idx) noexcept
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


        if (!it->fmm_mask(fmm_mask_idx,mask_id)) return;

        const auto idx=get_octant_idx(it);

        if(it->locally_owned() && it->data() )
        {

            const auto unique_ranks=it->unique_child_ranks(fmm_mask_idx, mask_id);
            for(auto r : unique_ranks)
            {
                if(_upward)
                {
                    auto data_ptr=it->data()->
                        template get<RecvField>().data_ptr();
                    auto task=recv_comm.post_task( data_ptr, r, true, idx);
                    task->requires_confirmation()=false;

                } else
                {
                    auto data_ptr=it->data()->
                        template get<SendField>().data_ptr();
                    auto task= send_comm.post_task(data_ptr,r,true,idx);
                    task->requires_confirmation()=false;

                }
            }
        }

        //Check if ghost has locally_owned children
        if(!it->locally_owned() && it->data())
        {
            if(it->has_locally_owned_children(fmm_mask_idx, mask_id))
            {
                if(_upward)
                {
                    const auto data_ptr=it->data()->
                        template get<SendField>().data_ptr();
                    auto task =
                        send_comm.post_task(data_ptr, it->rank(),
                                true,idx);
                    task->requires_confirmation()=false;

                } else
                {
                    const auto data_ptr=it->data()->
                        template get<RecvField>().data_ptr();
                    auto task =
                        recv_comm.post_task(data_ptr, it->rank(),
                                true,idx);
                    task->requires_confirmation()=false;
                }
            }
        }

        //Start communications
        //buffer and send it
        send_comm.start_communication();
        recv_comm.start_communication();

        //send_comm.finish_communication();
        //recv_comm.finish_communication();
    }

    template<class SendField,class RecvField,template<class>class BufferPolicy>
    void finish_updownward_pass_communication()
    {
        auto& send_comm=
            task_manager_-> template
                send_communicator<induced_fields_task_t<BufferPolicy>>();
        auto& recv_comm=
            task_manager_-> template
                recv_communicator<induced_fields_task_t<BufferPolicy>>();

        while(true)
        {
            //buffer and send it
            send_comm.start_communication();
            recv_comm.start_communication();

            //Check if something has finished
            send_comm.finish_communication();
            recv_comm.finish_communication();

            if(send_comm.done() && recv_comm.done() )
                break;
        }
    }

    /** @brief Finish communication for fields for up/downward pass of fmm */
    template<class SendField,class RecvField>
    void finish_updownward_pass_communication_add()
    {
        finish_updownward_pass_communication<SendField,RecvField,AddAssignRecv> ();
    }

    /** @brief Finish communication for fields for up/downward pass of fmm */
    template<class SendField,class RecvField>
    void finish_updownward_pass_communication_assign()
    {
        finish_updownward_pass_communication<SendField,RecvField,CopyAssign> ();
    }

    /** @brief communicate fields for up/downward pass of fmm */
    template<class SendField,class RecvField, class OctantPtr>
    void communicate_updownward_add(OctantPtr it, bool _upward, int fmm_mask_idx)
    {
        communicate_updownward_pass<SendField,RecvField,AddAssignRecv> (it, _upward, fmm_mask_idx);
    }

    template<class SendField,class RecvField, class OctantPtr >
    void communicate_updownward_assign(OctantPtr it, bool _upward, int fmm_mask_idx)
    {
        communicate_updownward_pass<SendField,RecvField,CopyAssign>
        (it,_upward, fmm_mask_idx);
    }

    template<class SendField,class RecvField, template<class>class BufferPolicy>
    void communicate_updownward_pass(int level, bool _upward, bool _use_masks, int fmm_mask_idx)
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

            if (_use_masks && !it->fmm_mask(fmm_mask_idx,mask_id) ) continue;

            const auto idx=get_octant_idx(it);

            if(it->locally_owned() && it->data() )
            {

                const auto unique_ranks=(_use_masks)?
                        it->unique_child_ranks(fmm_mask_idx, mask_id) :
                        it->unique_child_ranks();

                for(auto r : unique_ranks)
                {
                    if(_upward)
                    {
                        auto data_ptr=it->data()->
                            template get<RecvField>().data_ptr();
                        auto task=recv_comm.post_task( data_ptr, r, true, idx);
                        task->requires_confirmation()=false;

                    } else
                    {
                        auto data_ptr=it->data()->
                            template get<SendField>().data_ptr();
                        auto task= send_comm.post_task(data_ptr,r,true,idx);
                        task->requires_confirmation()=false;

                    }
                }
            }

            //Check if ghost has locally_owned children
            if(!it->locally_owned() && it->data())
            {
                if( (_use_masks && it->has_locally_owned_children(fmm_mask_idx, mask_id)) ||
                    (!_use_masks && it->has_locally_owned_children())
                    )
                {
                    if(_upward)
                    {
                        const auto data_ptr=it->data()->
                            template get<SendField>().data_ptr();
                        auto task =
                            send_comm.post_task(data_ptr, it->rank(),
                                    true,idx);
                        task->requires_confirmation()=false;

                    } else
                    {
                        const auto data_ptr=it->data()->
                            template get<RecvField>().data_ptr();
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
            recv_comm.finish_communication();

            if(send_comm.done() && recv_comm.done() )
                break;
        }
    }

    /** @brief communicate fields for up/downward pass of fmm */
    template<class SendField,class RecvField >
    void communicate_updownward_add(int level, bool _upward, bool _use_masks, int fmm_mask_idx)
    {
        communicate_updownward_pass<SendField,RecvField,AddAssignRecv>
        (level, _upward, _use_masks, fmm_mask_idx);
    }

    template<class SendField,class RecvField >
    void communicate_updownward_assign(int level, bool _upward, bool _use_masks, int fmm_mask_idx)
    {
        communicate_updownward_pass<SendField,RecvField,CopyAssign>
        (level,_upward, _use_masks, fmm_mask_idx);
    }



    /** @brief communicate fields for up/downward pass of fmm */
    template<class OctantPtr>
    int updownward_pass_mcount(OctantPtr it, bool _upward, int fmm_mask_idx)
    {
        int mask_id=(_upward) ?
                MASK_LIST::Mask_FMM_Source : MASK_LIST::Mask_FMM_Target;

        int count=0;
        if (!it->fmm_mask(fmm_mask_idx,mask_id)) return count;
        if(it->locally_owned() && it->data() )
        {
            const auto unique_ranks=it->unique_child_ranks(fmm_mask_idx, mask_id);
            if(_upward)
            {
                const auto unique_ranks=it->unique_child_ranks(fmm_mask_idx, mask_id);
                for(auto r : unique_ranks)
                {
                    if(_upward) { /*recv*/ ++count; }
                    else { /*send*/ ++count; }
                }
            }
            //for(auto r : unique_ranks)
            //{
            //    if(_upward) { /*recv*/ ++count; }
            //    else { /*send*/ ++count; }
            //}
        }
        if(!it->locally_owned() && it->data())
        {
            if(it->has_locally_owned_children(fmm_mask_idx, mask_id))
            {
                if(_upward) { /*send*/ return 10000; }
                else { /*recv*/ return 10000; }
            }
        }
        return count;
    }

    /** @brief communicate fields for up/downward pass of fmm */
    //TODO: Make it better and put in octant
    template<class T>
    auto get_octant_idx(T it) const noexcept
    {
        const auto cc=it->tree_coordinate();
        return static_cast<int>(
                (it->level()+cc.x()*25+cc.y()*25*300+ 25*300*300*cc.z()) %
                boost::mpi::environment::max_tag()
                );
    }

    /** @brief Testing function for buffer/halo exchange for a field.
     *         The  haloCommunicator will later be stored as a tuple for
     *         all fields.
     */
    template<class Field>
    void buffer_exchange(const int _level)
    {

        auto& send_comm=
            task_manager_-> template send_communicator<halo_task_t>();
        auto& recv_comm=
            task_manager_-> template recv_communicator<halo_task_t>();

        //Initialize Halo communicator
        //TODO: put this outside somewhere
        if (!halo_initialized_)
                initialize_halo_communicators();

        auto& hcomm=std::get<halo_communicator_t<Field>>
            (halo_communicators_[_level]);

        //Get the overlaps
        hcomm.template pack_messages();
        for(auto st:hcomm.send_tasks()) { if(st) { send_comm.post_task(st); } }
        for(auto& st:hcomm.recv_tasks()) { if(st) { recv_comm.post_task(st); } }

        //Blocking send/recv till all is done. To be changed
        while(true)
        {
            send_comm.start_communication();
            recv_comm.start_communication();
            send_comm.finish_communication();
            recv_comm.finish_communication();
            if(send_comm.done() && recv_comm.done() )
                break;
        }

        //Assign received message to field view
        hcomm.template unpack_messages();
    }

    void initialize_halo_communicators()noexcept
    {
        halo_communicators_.resize(domain_->tree()->depth());
        for (int l  = domain_->tree()->base_level();
                l < domain_->tree()->depth(); ++l)
        {
            tuple_utils::for_each(halo_communicators_[l], [&](auto& hcomm){
                    hcomm.compute_tasks(domain_, l);
                    });
        }

        halo_initialized_=true;
    }

    void query_corrections()
    {
        domain_->tree()->query_corrections(this);
    }

    void query_leafs()
    {
        domain_->tree()->query_leafs(this);
        domain_->tree()->construct_leaf_maps(true);
    }

    void query_masks()
    {
        domain_->tree()->query_masks(this);
    }

    void query_octants()
    {
        domain_->tree()->construct_maps(this);
    }


    auto domain()const{return domain_;}


private:
    Domain* domain_;

    //TODO: make this a InlineQuery

    std::vector<std::vector<float_type>> send_fields_;
    std::vector<std::vector<float_type>> recv_fields_;
    std::vector<std::vector<std::shared_ptr<induced_fields_task_t<AddAssignRecv>>>> send_tasks_;
    std::vector<std::vector<std::shared_ptr<induced_fields_task_t<AddAssignRecv>>>> recv_tasks_;
    std::vector<halo_communicators_tuple_t> halo_communicators_;

    bool halo_initialized_=false;
};

}

#endif
