#ifndef INCLUDED_TAKS_BUFFER__HPP
#define INCLUDED_TAKS_BUFFER__HPP

namespace sr_mpi
{

template<class Buffer>
class TaskBufferContainer;

template<int Tag, class T, class ID=int>
class TaskBuffer
{

public: //Ctors:
    using container_t = TaskBufferContainer<TaskBuffer>;
    using data_type = T;

    friend container_t;

public: //Ctors:
    TaskBuffer()=default;
    TaskBuffer( container_t* _c )
    :container_(_c)
    {}

public: //memebers:

    void detach() noexcept
    {
        is_free_=true;
        container_->nActive_buffers_-=1;
    }

    void attach() noexcept
    {
        is_free_=false;
        container_->nActive_buffers_+=1;
    }

    T& data()noexcept {return buffer_; }
    const T& data()const noexcept {return buffer_; }

    T* ptr()noexcept{return &buffer_;}

    const bool& is_free()const noexcept{return is_free_;}
    bool& is_free()noexcept{return is_free_;}


private: //memebers:
    T buffer_;
    container_t* container_=nullptr;
    bool is_free_=true;
};

template<class Buffer>
class TaskBufferContainer
{

public:
    using buffer_data_t =  typename Buffer::data_type;
    friend Buffer;

public:
    TaskBufferContainer()=default;
    TaskBufferContainer(int _nBuffers)
    :buffers_(_nBuffers)
    {
        init(_nBuffers);
    }

    void init(int _nBuffers)
    {
        buffers_.resize(_nBuffers);
        for(auto & b: buffers_)
        {
            b.container_=this;
        }
    }

    Buffer* get_free_buffer() noexcept
    {
        if(!is_free())return nullptr;
        for(std::size_t i=0;i<buffers_.size();++i)
        {
            if(buffers_[i].is_free())
            {
                return &buffers_[i];
            }
        }
        return nullptr;
    }

    bool is_free()const noexcept
    {
        return nActive_buffers_ < static_cast<int>(buffers_.size());
    }

private:
    std::vector<Buffer> buffers_;
    int nActive_buffers_=0;
};




}


#endif
