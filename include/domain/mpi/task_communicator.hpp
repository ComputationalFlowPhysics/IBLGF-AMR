#ifndef INCLUDED_TASK_COMMUNICATOR__HPP
#define INCLUDED_TASK_COMMUNICATOR__HPP

#include<vector>
#include<array>
#include<list>
#include<queue>
#include<deque>


namespace sr_mpi
{

/**
 * @brief: Task communicator.  Manages posted messages, buffers them and calls 
 *         non-blocking send/recvs and queries their status.
 *         Can be used in both Send and Recv mode.
 */
template<class TaskType, class Derived>
class TaskCommunicator
: public crtp::Crtps<Derived, TaskCommunicator<TaskType,Derived>>
{
public:

    using task_t = TaskType;
    using task_ptr_t = std::shared_ptr<task_t>;
    using vector_task_ptr_t = std::vector<task_ptr_t>;
    using id_type = typename task_t::id_type;
    using task_vector_t =std::vector<task_ptr_t>;
    using task_data_t = typename TaskType::data_type;
    using buffer_queue_t = std::deque<task_ptr_t>;
    using buffer_container_t =typename task_t::buffer_container_type;

    using request_list_t = std::list<boost::mpi::request>;
    using query_arr_t =std::list<boost::mpi::request>;


public: //Ctor

    TaskCommunicator( int _nBuffers=10 )  
    {
        buffer_.init(_nBuffers);
    }

    //No copy or Assign 
	TaskCommunicator(const TaskCommunicator&) = delete;
	TaskCommunicator& operator=(const TaskCommunicator&) & = delete;
    ~TaskCommunicator()=default;

public:

    /** * @brief Insert task into buffer queue */
    task_ptr_t insert(const id_type& _taskId ,
                      task_data_t* _data, 
                      int _rank_other
                      ) noexcept
    {
        ++nActive_tasks_;
        auto task_ptr = std::make_shared<task_t>(_taskId);
        task_ptr->attach_data(_data);
        task_ptr->rank_other()=_rank_other;
        buffer_queue_.push_back( task_ptr );

        return task_ptr;
    }

    /** * @brief Post an answer of this task */
    template<class TaskPtr, class DataPtr>
    task_ptr_t post_answer(TaskPtr _task_ptr, DataPtr* _data) noexcept
    {
        return insert(_task_ptr->id(), _data, _task_ptr->rank_other());
    }

    /** * @brief Check if all tasks are done and nothing is in the queue */
    bool done() const noexcept
    {
        return tasks_.size()==0  && 
               buffer_queue_.size()==0 && 
               unconfirmed_tasks_.size()==0;
    }

    /** * @brief Check if all tasks are done and nothing is in the queue */
    task_ptr_t post_task(task_data_t* _dat, int _rank_other, bool use_tag=false, int _tag=100)
    {
        if(use_tag)
        {
            auto task_ptr=insert(_tag, _dat, _rank_other);
            return task_ptr;
        }
        auto tag= tag_gen().get<task_t::tag()>(tag_rank(_rank_other));
        auto task_ptr=insert(tag, _dat, _rank_other);
        tag= tag_gen().generate<task_t::tag()>(tag_rank(_rank_other));
        return task_ptr;
    }

    /** * @brief Start communication (send or receive for this task)*/
    task_vector_t start_communication()
    {
        task_vector_t res;
        std::size_t mCount=0;
        std::size_t size = buffer_queue_.size();
        while(buffer_.is_free() && !buffer_queue_.empty())
        {
            auto task =buffer_queue_.front();

            //If message does not exisit (for recv), check other posted messages
            if( !message_exists(task))
            {
                buffer_queue_.push_back(task);
                buffer_queue_.pop_front();
            }
            else
            {
                auto ptr = buffer_.get_free_buffer();
                task->attach_buffer( ptr );
                to_buffer(task);
                sendRecv(task);

                tasks_.push_back(task);
                res.push_back(task);
                buffer_queue_.pop_front();
            }
            if(mCount==size) break;
            ++mCount;
        } 
        return res;
    }

    /** * @brief Finish communication (send or receive for this task)
     *           Task will also be completed at the same time
     * */
    task_vector_t finish_communication()
    {
        task_vector_t finished_;
        for(auto it=tasks_.begin();it!=tasks_.end();)
        {
            auto& t=*it;
            if( auto status_opt = t->test() )
            {
                finished_.push_back(t);
                from_buffer(t);
                
                t->deattach_buffer();
                insert_unconfirmed_tasks(t);
                it=tasks_.erase(it);
                --nActive_tasks_;
            }
            else { ++it; }
        }
        check_unconfirmed_tasks();
        return finished_;
    }
    
    const auto& get_buffer_queue() const noexcept {return buffer_queue_;}
    auto& get_buffer_queue() noexcept {return buffer_queue_;}

    void clear()
    {
        tasks_.clear();
        unconfirmed_tasks_.clear();
        buffer_queue_.clear();
    }

protected:
    int tag_rank(int _rank_other) const noexcept 
    {
        return this->derived().tag_rank_impl(_rank_other);
    }
    void sendRecv(task_ptr_t _t) const noexcept 
    {
        this->derived().sendRecv_impl(_t);
    }
    void to_buffer(task_ptr_t _t) const noexcept 
    {
        this->derived().to_buffer_impl(_t); 
    }
    void from_buffer(task_ptr_t _t) const noexcept 
    {
         this->derived().from_buffer_impl(_t); 
    }

    bool message_exists(task_ptr_t _t) const noexcept
    {
        return this->derived().message_exists_impl(_t);
    }



    void insert_unconfirmed_tasks(task_ptr_t& t) noexcept
    {
        if(t->requires_confirmation())
        {
            auto confirmed=t->confirmed();
            if(!confirmed)
            {
                unconfirmed_tasks_.push_back(t);
            }
        }
    }
    void check_unconfirmed_tasks() noexcept
    {
        for(auto it=unconfirmed_tasks_.begin(); it!=unconfirmed_tasks_.end();)
        {
            if((*it)->confirmed())
            {
                it=unconfirmed_tasks_.erase(it);
            }
            else { ++it; }
        }
    }



protected:

    boost::mpi::communicator comm_; ///< Mpi communicator.
    int nActive_tasks_=0;           ///< Number of active tasks.

    task_vector_t tasks_;             ///< Taks that currently are being send.
    buffer_container_t buffer_;       ///< Data buffer to be send for each task.
    buffer_queue_t buffer_queue_;     ///< Queue of tasks to fill the send buffer
    task_vector_t unconfirmed_tasks_; ///< Unconfirmed tasks

};

template<class TaskType> 
struct SendTaskCommunicator  
:public TaskCommunicator<TaskType, SendTaskCommunicator<TaskType>>
{

public: 
    using super_type = TaskCommunicator<TaskType, SendTaskCommunicator<TaskType>>;
    using task_ptr_t = typename super_type::task_ptr_t;

public: //Ctors
    using super_type::TaskCommunicator;

public: //Memebers:

    int tag_rank_impl(int rank_other) const noexcept { return this->comm_.rank(); }
    void sendRecv_impl(task_ptr_t _t) const noexcept {_t->isend(this->comm_);}
    void to_buffer_impl(task_ptr_t _t) const noexcept 
    {
        _t->assign_data2buffer();
    }
    void from_buffer_impl(task_ptr_t _t) const noexcept { } 
    bool message_exists_impl( task_ptr_t _t  ) const noexcept{ return true; }
private:
    
};

template<class TaskType> 
class  RecvTaskCommunicator  
:public TaskCommunicator<TaskType, RecvTaskCommunicator<TaskType>>
{
public: 
    using super_type = TaskCommunicator<TaskType, RecvTaskCommunicator<TaskType>>;
    using task_ptr_t = typename super_type::task_ptr_t;

public: //Ctors
    using super_type::TaskCommunicator;

public: //members
    int tag_rank_impl(int _rank_other) const noexcept {return _rank_other;  }
    void sendRecv_impl(task_ptr_t _t) const noexcept {_t->irecv(this->comm_);}
    void to_buffer_impl(task_ptr_t _t) const noexcept { } 
    void from_buffer_impl(task_ptr_t _t) const noexcept 
    {
        _t->assign_buffer2data();
    }
    bool message_exists_impl( task_ptr_t _t  ) const noexcept{ 
        if(this->comm_.iprobe(_t->rank_other(), _t->id()))
            return true;
        else return false;

    }
};



}  //namespace sr_mpi

#endif 
