#ifndef INCLUDED_TAKS_BASE__HPP
#define INCLUDED_TAKS_BASE__HPP

#include "tags.hpp"
#include "task_buffer.hpp"

namespace sr_mpi
{

template<class BufferType, class ID=int>
class Task_base
{

public:
    using id_type =ID;
    using task_buffer_t = BufferType;
    using data_type = typename BufferType::data_type;

public: //Ctor:

    Task_base( id_type _id)
    :id_(_id)
    {}

    Task_base()=default;


public: //access:

    const id_type& id()const noexcept{return id_;}
    id_type& id()noexcept{return id_;}

    const int& rank_other()const noexcept{return rank_other_;}
    int& rank_other()noexcept{return rank_other_;}

    const auto& data()const noexcept{return *data_;}
    auto&  data()noexcept{return *data_;}

    const auto& comm_buffer()const noexcept{return *comm_buffer_;}
    auto&  comm_buffer()noexcept{return *comm_buffer_;}

    const auto& comm_data()const noexcept{return comm_buffer_->data();}
    auto&  comm_data()noexcept{return *comm_buffer_->data();}

    const auto& reuest()const noexcept{return request_;}
    auto& request()noexcept{return request_;}


public: //memebers:

    void isend( boost::mpi::communicator _comm)
    {
        request_= _comm.isend(  rank_other_, id_, comm_buffer_->data());
        if(request_confirmation_)
        {
            confirmation_request_=_comm.irecv(rank_other_, tags::confirmation);
        }
    } 

    void irecv( boost::mpi::communicator _comm)
    {
        request_ = _comm.irecv(rank_other_, id_, comm_buffer_->data());
        if(request_confirmation_)
        {
            confirmation_request_=_comm.isend(rank_other_, tags::confirmation);
        }
    }
    auto test() noexcept
    {
        return request_.test();
    }
    bool confirmed()
    {
        if(confirmed_) return true;
        if(!request_confirmation_)
        {
            confirmed_=true;
            return confirmed_;
        }else
        {
            if(auto opt=confirmation_request_.test())
            {
                confirmed_=true;
            }
        } 
        return confirmed_;
    }
    bool requires_confirmation() const noexcept { return request_confirmation_; }

    void attach_data( data_type* _s ) noexcept { data_=_s; }

    void attach_buffer( task_buffer_t* _b ) noexcept
    {
        comm_buffer_=_b;
        comm_buffer_->attach();
    }
    void assign_data2buffer() noexcept
    {
        comm_buffer_->data()=data();

    }
    void assign_buffer2data() noexcept
    {
        data()=comm_buffer_->data();
    }
    void deattach_buffer() noexcept
    {
        comm_buffer_->detach();
        comm_buffer_=nullptr;
    }
    void wait_confirmation()
    {
        if(confirmed_) return;
        confirmation_request_.wait();
    }


    

protected:
    int rank_other_=-1;
    id_type id_=0;
    data_type* data_=nullptr;
    task_buffer_t* comm_buffer_=nullptr;
    boost::mpi::request request_;

    //Confirmation mechanism
    boost::mpi::request confirmation_request_;
    bool request_confirmation_=true;

    bool confirmed_=false;
};



template<int Tag, class T, class ID=int>
class Task : public Task_base< TaskBuffer<Tag,T,ID>, ID >
{

public:

    using super_type = Task_base<TaskBuffer<Tag,T,ID>,ID>;
    using super_type::Task_base;
    using id_type =ID;
    using buffer_type = TaskBuffer<Tag,T,ID>;
    using buffer_container_type = typename buffer_type::container_t;

    using data_type = typename super_type::data_type;

    using answer_data_type=data_type;
    using answer_task_type=Task<Tag, answer_data_type>;

    static constexpr int tag(){return Tag;}


public: //memebers:

    void complete()
    {
        std::cout<<"using default complete"<<std::endl;
    }

    void generate()
    {
        std::cout<<"using default generate"<<std::endl;
    }
};

template<class ID>
class Task<tags::key_query,std::vector<int>,ID> 
: public Task_base< TaskBuffer<tags::key_query,std::vector<int>,ID>>
{

public:
    using super_type = Task_base< 
        TaskBuffer<tags::key_query,std::vector<int>,ID>>;
    using super_type::Task_base;
    using id_type =ID;
    using buffer_type =  TaskBuffer<tags::key_query,std::vector<int>,ID>;
    using task_buffer_t =  typename super_type::task_buffer_t;
    using buffer_container_type = typename buffer_type::container_t;

    using data_type = typename super_type::data_type;
    using answer_data_type=std::vector<int>;
    using answer_task_type=Task<tags::key_query, answer_data_type>;

    static constexpr int tag(){return tags::key_query;}

public: //memebers:

    void attach_data( data_type* _s ) noexcept { this->data_=_s; }

    //inplace
    void attach_buffer( task_buffer_t* _b ) noexcept { }
    void assign_data2buffer() noexcept {}
    void assign_buffer2data() noexcept { }
    void deattach_buffer() noexcept { }

    void isend( boost::mpi::communicator _comm)
    {

        //std::cout<<"Rank: "<<_comm.rank()<<" with count: "<<++count<<std::endl;
        this->request_= _comm.isend(  this->rank_other_, this->id_, *this->data_);
        if(this->request_confirmation_)
        {
            this->confirmation_request_=_comm.irecv(this->rank_other_, tags::confirmation);
        }
    } 
    void irecv( boost::mpi::communicator _comm)
    {
        this->request_ = _comm.irecv(this->rank_other_, this->id_, *this->data_);
        if(this->request_confirmation_)
        {
            this->confirmation_request_=_comm.isend(this->rank_other_, tags::confirmation);
        }
    }

    void complete( data_type* query_data, 
                   answer_data_type* answer_buffer ) noexcept
    {
        
        answer_data_type ans(count*10, this->rank_other_-count);
        *answer_buffer=ans;
        ++count;

    }
    void generate()
    {
        boost::mpi::communicator world;
    }

    static int count;
};


template<class ID>
int Task<tags::key_query,std::vector<int>,ID>::count=0;


//These should be specialized for different tasks

}
#endif 
