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

#ifndef INCLUDED_TAKS_BASE__HPP
#define INCLUDED_TAKS_BASE__HPP

#include "tags.hpp"
#include "task_buffer.hpp"

#include <iblgf/utilities/crtp.hpp>

namespace iblgf
{
namespace sr_mpi
{
template<class TaskType>
struct InfluenceFieldBuffer : crtp::Crtp<TaskType, InfluenceFieldBuffer> //callback function executed on send. addition on recv
{
    template<class Buffer>
    void attach_buffer(Buffer* _b) noexcept
    {
        task_.comm_buffer_ = _b;
        task_.comm_buffer_->attach();
    }

    //send: Compute influence field and ship
    void assign_data2buffer() noexcept { sendCallback_(send_buffer()); }

    //recv:  Add influence field contributuin
    void assign_buffer2data() noexcept
    {
        std::transform(task_.data().begin(), task_.data().end(),
            task_.comm_buffer_->data().begin(), task_.data().begin(),
            std::plus<typename TaskType::data_type::value_type>());
    }
    void deattach_buffer() noexcept
    {
        task_.comm_buffer_->detach();
        task_.comm_buffer_ = nullptr;
    }
    auto& send_buffer() noexcept { return task_.comm_buffer_->data(); }
    auto& recv_buffer() noexcept { return task_.comm_buffer_->data(); }

    template<class Function>
    void register_sendCallback(Function& _f) noexcept
    {
        sendCallback_ = _f;
    }

  private:
    TaskType&                                 task_ = this->derived();
    std::function<void(std::vector<double>&)> sendCallback_;
};

template<class TaskType>
struct CopyAssign : crtp::Crtp<TaskType, CopyAssign> //simple copy on send and recv
{
    template<class Buffer>
    void attach_buffer(Buffer* _b) noexcept
    {
        task_.comm_buffer_ = _b;
        task_.comm_buffer_->attach();
    }
    void assign_data2buffer() noexcept
    {
        task_.comm_buffer_->data() = task_.data();
    }
    void assign_buffer2data() noexcept
    {
        task_.data() = task_.comm_buffer_->data();
    }
    void deattach_buffer() noexcept
    {
        task_.comm_buffer_->detach();
        task_.comm_buffer_ = nullptr;
    }
    auto& send_buffer() noexcept { return task_.comm_buffer_->data(); }
    auto& recv_buffer() noexcept { return task_.comm_buffer_->data(); }

  private:
    TaskType& task_ = this->derived();
};

template<class TaskType>
struct AddAssignRecv : CopyAssign<TaskType> //adds received data to local data on recv
{
    template<class Buffer>
    void attach_buffer(Buffer* _b) noexcept
    {
        task_.comm_buffer_ = _b;
        task_.comm_buffer_->attach();
    }
    //send
    void assign_data2buffer() noexcept
    {
        //inplace send ...
        //task_.comm_buffer_->data()=task_.data();
    }

    void assign_buffer2data() noexcept
    {
        std::transform(task_.data().begin(), task_.data().end(),
            task_.comm_buffer_->data().begin(), task_.data().begin(),
            std::plus<typename TaskType::data_type::value_type>());
    }
    void deattach_buffer() noexcept
    {
        task_.comm_buffer_->detach();
        task_.comm_buffer_ = nullptr;
    }
    auto& send_buffer() noexcept { return *task_.data_; }
    auto& recv_buffer() noexcept { return task_.comm_buffer_->data(); }

  private:
    TaskType& task_ = this->derived();
};

template<class TaskType>
struct OrAssignRecv : CopyAssign<TaskType>
{
    template<class Buffer>
    void attach_buffer(Buffer* _b) noexcept
    {
        task_.comm_buffer_ = _b;
        task_.comm_buffer_->attach();
    }
    void assign_data2buffer() noexcept
    {
        //task_.comm_buffer_->data()=task_.data();
    }
    void assign_buffer2data() noexcept
    {
        task_.data() = task_.data() || task_.comm_buffer_->data();
    }
    void deattach_buffer() noexcept
    {
        task_.comm_buffer_->detach();
        task_.comm_buffer_ = nullptr;
    }
    auto& send_buffer() noexcept { return *task_.data_; }
    auto& recv_buffer() noexcept { return task_.comm_buffer_->data(); }

  private:
    TaskType& task_ = this->derived();
};

template<class TaskType>
struct Inplace : crtp::Crtp<TaskType, Inplace>
{
    template<class Buffer>
    void attach_buffer(Buffer* _b) noexcept
    {
    }
    void assign_data2buffer() noexcept {}
    void assign_buffer2data() noexcept {}
    void deattach_buffer() noexcept {}

    auto& send_buffer() noexcept { return *task_.data_; }
    auto& recv_buffer() noexcept { return *task_.data_; }

  private:
    TaskType& task_ = this->derived();
};

template<class BufferType,
    template<class> class BufferPolicy, //Assign Mixin
    class ID = int>
class Task_base : public BufferPolicy<Task_base<BufferType, BufferPolicy, ID>>
{
  public:
    using id_type = ID;
    using task_buffer_t = BufferType;
    using data_type = typename BufferType::data_type;

    using buffer_policy_t =
        BufferPolicy<Task_base<BufferType, BufferPolicy, ID>>;

  public: //Friends:
    friend buffer_policy_t;

  public: //Ctor:
    Task_base(id_type _id)
    : id_(_id)
    {
    }

    Task_base() = default;

  public: //access:
    const id_type& id() const noexcept { return id_; }
    id_type&       id() noexcept { return id_; }

    const int& rank_other() const noexcept { return rank_other_; }
    int&       rank_other() noexcept { return rank_other_; }

    const auto& data() const noexcept { return *data_; }
    auto&       data() noexcept { return *data_; }

    const auto& comm_buffer() const noexcept { return *comm_buffer_; }
    auto&       comm_buffer() noexcept { return *comm_buffer_; }

    const auto& comm_data() const noexcept { return comm_buffer_->data(); }
    auto&       comm_data() noexcept { return *comm_buffer_->data(); }

    const auto& reuest() const noexcept { return request_; }
    auto&       request() noexcept { return request_; }

  public: //memebers:
    auto test() noexcept { return request_.test(); }
    bool confirmed()
    {
        if (confirmed_) return true;
        if (!request_confirmation_)
        {
            confirmed_ = true;
            return confirmed_;
        }
        else
        {
            if (auto opt = confirmation_request_.test()) { confirmed_ = true; }
        }
        return confirmed_;
    }
    const bool& requires_confirmation() const noexcept
    {
        return request_confirmation_;
    }
    bool& requires_confirmation() noexcept { return request_confirmation_; }

    void attach_data(data_type* _s) noexcept { data_ = _s; }

    void isend(const boost::mpi::communicator& _comm)
    {
        this->request_ =
            _comm.isend(this->rank_other_, this->id_, (this->send_buffer()));
        if (this->request_confirmation_)
        {
            this->confirmation_request_ =
                _comm.irecv(this->rank_other_, tags::confirmation);
        }
    }
    void irecv(const boost::mpi::communicator& _comm)
    {
        this->request_ =
            _comm.irecv(this->rank_other_, this->id_, (this->recv_buffer()));
        if (this->request_confirmation_)
        {
            this->confirmation_request_ =
                _comm.isend(this->rank_other_, tags::confirmation);
        }
    }
    void wait_confirmation()
    {
        if (confirmed_) return;
        confirmation_request_.wait();
    }

  protected:
    int                 rank_other_ = -1;
    id_type             id_ = 0;
    data_type*          data_ = nullptr;
    task_buffer_t*      comm_buffer_ = nullptr;
    boost::mpi::request request_;

    //Confirmation mechanism
    boost::mpi::request confirmation_request_;
    bool                request_confirmation_ = true;

    bool confirmed_ = false;
};

template<int Tag, class T,
    template<class> class BufferPolicy = CopyAssign, //Assign Mixin,
    class MetaDataType = void, class ID = int>
class Task : public Task_base<TaskBuffer<Tag, T, ID>, BufferPolicy, ID>
{
  public:
    using super_type = Task_base<TaskBuffer<Tag, T, ID>, BufferPolicy, ID>;
    using super_type::Task_base;
    using id_type = ID;
    using buffer_type = TaskBuffer<Tag, T, ID>;
    using buffer_container_type = typename buffer_type::container_t;
    using data_type = T;

  public: //Ctors
    Task()=default;
    Task(id_type _id) :super_type(_id){}

    using inplace_task_type = Task<Tag, T, Inplace, MetaDataType, ID>;

    static constexpr int tag() { return Tag; }

    const MetaDataType*& meta() const { return meta_; }
    MetaDataType*&       meta() { return meta_; }

    const MetaDataType*& octant() const { return meta_; }
    MetaDataType*&       octant() { return meta_; }

  private:
    MetaDataType* meta_ = nullptr;
};

} // namespace sr_mpi
} // namespace iblgf
#endif
