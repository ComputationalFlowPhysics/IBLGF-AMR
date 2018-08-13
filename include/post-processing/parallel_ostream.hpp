#ifndef INCLUDED_PARALLEL_OSTREAM_HPP
#define INCLUDED_PARALLEL_OSTREAM_HPP

#include <boost/mpi.hpp>
#include <boost/mpi/communicator.hpp>
#include <ostream>

namespace parallel_ostream
{

class ParallelOstream
{
public:
    ParallelOstream (std::ostream& _ostream=std::cout,
                     const bool _active = [](){boost::mpi::communicator world; return world.rank()==0; }() )
    :ostream_(_ostream),
    active_(_active)
    {
    }
    
    bool active() const noexcept{return active_; }
    void active(const bool _a) noexcept{active_=_a;}

    std::ostream& ostream() const noexcept
    {
        return ostream_;
    }

    template <typename T>
    const ParallelOstream& operator << (const T &t) const noexcept
    {
        if (active_)
            ostream_ << t;
        return *this;
    }

    const ParallelOstream& operator<< ( std::ostream& (*p) (std::ostream &)) const noexcept
    {
        if (active_)
            ostream_ << p;
        return *this;
    }
    template <typename T>
    const ParallelOstream& operator << (T &t) const noexcept
    {
        if (active_)
            ostream_ << t;
        return *this;
    }




private:
    std::ostream  &ostream_;
    bool active_;
};

   
} //namespace
#endif 
