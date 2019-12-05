#ifndef DOMAIN_INCLUDED_SERVER_HPP
#define DOMAIN_INCLUDED_SERVER_HPP

#include <vector>
#include <limits>
#include <stdexcept>
#include <boost/mpi.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/utility.hpp>
#include <boost/mpi/status.hpp>

#include <global.hpp>
#include <fmm/fmm.hpp>
#include <domain/decomposition/compute_task.hpp>

#include <domain/mpi/task_manager.hpp>
#include <domain/mpi/server_base.hpp>
#include <domain/mpi/query_registry.hpp>
#include "serverclient_traits.hpp"

namespace domain
{


/** @brief ProcessType Server
 *  Master/Server process.
 *  Stores the full tree structure without the data.
 *  Responsible for load balancing and listens to
 *  the client/worker processes.
 */
template<class Domain>
class Server : public ServerBase<ServerClientTraits<Domain>>
{

public:
    using domain_t = Domain;
    using communicator_type  = typename  domain_t::communicator_type;
    using octant_t  = typename  domain_t::octant_t;
    using key_t  = typename  domain_t::key_t;
    using ctask_t = ComputeTask<octant_t>;

    using trait_t =  ServerClientTraits<Domain>;
    using super_type = ServerBase<trait_t>;

    using rank_query_t   = typename trait_t::rank_query_t;
    using key_query_t    = typename trait_t::key_query_t;

    using mask_init_query_send_t  = typename trait_t::mask_init_query_send_t;
    using mask_init_query_recv_t  = typename trait_t::mask_init_query_recv_t;

    using correction_query_send_t = typename trait_t::correction_query_send_t;
    using correction_query_recv_t = typename trait_t::correction_query_recv_t;

    using leaf_query_send_t   = typename trait_t::leaf_query_send_t;
    using leaf_query_recv_t   = typename trait_t::leaf_query_recv_t;

    using task_manager_t = typename trait_t::task_manager_t;

public: //helper struct


struct DecompositionUpdate
{
    DecompositionUpdate(int _worldsize)
    : send_octs(_worldsize), dest_ranks(_worldsize),
      recv_octs(_worldsize), src_ranks(_worldsize)
    {
    }

    void insert(int _current_rank, int _new_rank, key_t _key)
    {

        send_octs [_current_rank].emplace_back(_key);
        dest_ranks[_current_rank].emplace_back(_new_rank);

        recv_octs[_new_rank].emplace_back(_key);
        src_ranks[_new_rank].emplace_back(_current_rank);

    }

    //octant key and dest rank,outer vector in current  rank
    std::vector<std::vector<key_t>> send_octs; 
    std::vector<std::vector<int>>   dest_ranks; 
    //octant key and src rank, outer vector in current  rank
    std::vector<std::vector<key_t>> recv_octs; 
    std::vector<std::vector<int>>   src_ranks; 
                                                
};

public: //Ctors

    using super_type::ServerBase;

    Server(const Server&  other) = default;
    Server(      Server&& other) = default;
    Server& operator=(const Server&  other) & = default;
    Server& operator=(      Server&& other) & = default;
    ~Server() = default;

    Server(Domain* _d, communicator_type _comm)
    :domain_(_d), comm_(_comm)
    {
        boost::mpi::communicator world;
        std::cout<<"I am the server on rank: "<<world.rank()<<std::endl;
    }


public:

    template<class Begin, class End, class Container, class Function,class Function1>
    void split(Begin _begin, End _end, Container& _tasks_perProc,
               std::vector<float_type>& _loads_perProc,
               Function& _exitCheck, Function1& _continueCheck) const noexcept
    {

        float_type total_load=0.0;
        const auto nProcs=comm_.size()-1;
        for( auto it = _begin; it!= _end;++it )
        {
            if(_exitCheck(it)) break;
            if(_continueCheck(it)) continue;
            total_load+=it->load();
        }

        auto it = _begin;
        float_type current_load= 0.;
        for(int crank=0;crank<nProcs;++crank)
        {
            float_type target_load= (static_cast<float_type>(crank+1)/nProcs)*total_load;
            target_load=std::min(target_load,total_load);
            int count=0;
            while(current_load<=target_load || count ==0)
            {
                if(_continueCheck(it))
                {
                     ++it;
                     if(it==_end) break;
                     if(_exitCheck(it)) break;
                     continue;
                }


                it->rank()=crank+1;
                auto load= it->load();

                ctask_t task(it.ptr(), it->rank(), load);
                _loads_perProc[crank]+=load;
                _tasks_perProc[crank].push_back(task);
                current_load+=load;
                ++it;
                ++count;
                if(it==_end) break;
                if(_exitCheck(it)) break;
            }
            if(it==_end) break;
            if(_exitCheck(it)) break;
        }
    }

    auto compute_distribution(bool _rand=false) const
    {
        std::cout<<"Computing domain decomposition for "<<comm_.size()<<" processors" <<std::endl;
        const auto nProcs=comm_.size()-1;

        std::mt19937_64 rng;
        std::uniform_real_distribution<float_type> dist(0.5, 1);
        float_type weight=1.0;
        for( auto it = domain_->begin_df(); it!= domain_->end_df();++it )
        {
            if(_rand)weight =dist(rng);
            it->load()=it->load()*weight;
        }

        const int blevel=domain_->tree()->base_level();
        std::vector<std::list<ctask_t>> tasks_perProc(nProcs);
        std::vector<float_type> total_loads_perProc(nProcs,0.0);

        // Split leafs according to morton order
        auto exitCondition1=[](auto& it){return false;};
        auto continueCondition1=[&blevel](auto& it){return !(it->is_leaf() || it->is_correction());};

        for (int l = domain_->tree()->depth()-1; l >= 0; --l)
        {

            split(domain_->begin(l),domain_->end(l),
                    tasks_perProc, total_loads_perProc,
                    exitCondition1,continueCondition1);

            for (auto it = domain_->begin(l); it != domain_->end(l); ++it)
            {
                if(it->is_leaf() || it->is_correction()) continue;

                int rank_tobe=-1;
                float_type min_load=std::numeric_limits<float_type>::max();
                for(int i=0;i<it->num_children();++i)
                {
                    const auto child = it->child(i);
                    if(!child) continue;

                    if(child->rank()==-1) throw std::runtime_error("Child not set ");
                    if(total_loads_perProc[child->rank()] < min_load)
                    {
                        rank_tobe=child->rank();
                    }
                }
                it->rank()=rank_tobe;
                ctask_t task(it.ptr(), rank_tobe, it->load());
                total_loads_perProc[rank_tobe-1]+=it->load();
                tasks_perProc[rank_tobe-1].push_back(task);
            }
        }

        for( auto it = domain_->begin_df(); it!= domain_->end_df();++it )
        {
            if(it->rank()==-1)
            {
                throw std::runtime_error("Domain decomposition (Server):"
                        " Some octant's rank was not set");
            }
        }
        std::cout<<"Done with initial load balancing"<<std::endl;
        return tasks_perProc;
    }


    /** @brief Recompute the balancing and return updates
     *  @return current_rank, target_rank, octant key
     */
    auto check_decomposition_updates()
    {
        std::vector<int> ranks_old;
        for (auto it = domain_->begin(); it != domain_->end(); ++it)
        {
            ranks_old.push_back(it->rank());
        }
        compute_distribution(true);

        int c=0;
        DecompositionUpdate updates(comm_.size());
        for (auto it = domain_->begin(); it != domain_->end(); ++it)
        {
            if(ranks_old[c] != it->rank())
            {
                updates.insert(ranks_old[c], it->rank(),it->key());
                std::cout<<"Found old, new "
                         <<it->key().id() <<" "
                         <<ranks_old[c]<<" "<<it->rank()<<std::endl;
            }
            ++c;
        }
        return updates;
    }

    void update_decomposition()
    {
        auto updates=this->check_decomposition_updates();
        for(int i=1;i<comm_.size();++i)
        {
            comm_.send(i,i+0*comm_.size(), updates.send_octs[i] );
            comm_.send(i,i+1*comm_.size(), updates.dest_ranks[i] );
                            
            comm_.send(i,i+2*comm_.size(), updates.recv_octs[i] );
            comm_.send(i,i+3*comm_.size(), updates.src_ranks[i] );
        }
    }

    

    void send_keys()
    {
        auto tasks=compute_distribution();
        for(auto& t:tasks)
        {
            std::vector<key_t> keys;
            for(auto& tt: t ) keys.push_back(tt.key());
            comm_.send(t.front().rank(),0, keys );
        }
        //Send global tree depth
        for(int i=1;i<comm_.size();++i)
        {
            comm_.send(i,0, domain_->tree()->depth()) ;
        }
    }

    void mask_query()
    {
        InlineQueryRegistry<mask_init_query_recv_t, mask_init_query_send_t> mq(comm_.size());
        mq.register_completeFunc([this](auto _task, auto _answerData)
        {
            this->get_octant_mask(_task, _answerData);
        });

        this->run_query(mq);
    }

    void correction_query()
    {
        InlineQueryRegistry<correction_query_recv_t, correction_query_send_t> mq(comm_.size());
        mq.register_completeFunc([this](auto _task, auto _answerData)
        {
            this->get_octant_correction(_task, _answerData);
        });

        this->run_query(mq);
    }

    void leaf_query()
    {
        InlineQueryRegistry<leaf_query_recv_t, leaf_query_send_t> mq(comm_.size());
        mq.register_completeFunc([this](auto _task, auto _answerData)
        {
            this->get_octant_leaf(_task, _answerData);
        });

        this->run_query(mq);
    }

    void rank_query()
    {
        InlineQueryRegistry<rank_query_t, key_query_t> mq(comm_.size());
        mq.register_completeFunc([this](auto _task, auto _answerData)
        {
            this->get_octant_rank(_task, _answerData);
        });

        this->run_query(mq);
    }

    template<class TaskPtr, class OutPtr>
    void get_octant_mask(TaskPtr _task, OutPtr _out)
    {
        _out->resize(_task->data().size());
        int count=0;
        for(auto& key : _task->data())
        {
            auto oct =domain_->tree()->find_octant(key);
            if(oct)
            { (*_out)[count++]=(oct->fmm_mask());
            }
            else
            {
                throw std::runtime_error(
                        "can't find octant for mask query");
            }
        }
    }

    template<class TaskPtr, class OutPtr>
    void get_octant_correction(TaskPtr _task, OutPtr _out)
    {
        _out->resize(_task->data().size());
        int count=0;
        for(auto& key : _task->data())
        {
            auto oct =domain_->tree()->find_octant(key);
            if(oct)
            { (*_out)[count++]=(oct->is_correction());
            }
            else
            {
                (*_out)[count++]=false;
            }
        }
    }

    template<class TaskPtr, class OutPtr>
    void get_octant_leaf(TaskPtr _task, OutPtr _out)
    {
        _out->resize(_task->data().size());
        int count=0;
        for(auto& key : _task->data())
        {
            auto oct =domain_->tree()->find_octant(key);
            if(oct)
            { (*_out)[count++]=(oct->is_leaf());
            }
            else
            {
                (*_out)[count++]=false;
            }
        }
    }

    template<class TaskPtr, class OutPtr>
    void get_octant_rank(TaskPtr _task, OutPtr _out)
    {
        _out->resize(_task->data().size());
        int count=0;
        for(auto& key :  _task->data())
        {
            auto oct =domain_->tree()->find_octant(key);
            if(oct)
                (*_out)[count++]=oct->rank();
            else
                (*_out)[count++]=-1;
        }
    }


private:
    Domain* domain_;
    communicator_type comm_;

};

}


#endif
