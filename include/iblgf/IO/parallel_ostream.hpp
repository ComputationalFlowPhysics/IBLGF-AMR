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

#ifndef INCLUDED_PARALLEL_OSTREAM_HPP
#define INCLUDED_PARALLEL_OSTREAM_HPP

#include <ostream>
#include <boost/mpi.hpp>
#include <boost/mpi/communicator.hpp>

namespace iblgf
{
namespace parallel_ostream
{
class ParallelOstream
{
  public:
    explicit ParallelOstream(
        const bool _active, std::ostream& _ostream = std::cout)
    : ostream_(_ostream)
    , active_(_active)
    {
    }

    explicit ParallelOstream(int _rank = 0, std::ostream& _ostream = std::cout)
    : ostream_(_ostream)
    , active_(is_active_rank(_rank))
    {
    }

    explicit ParallelOstream(
        const std::string& _filename, int _rank, std::ofstream& _ofstream)
    : ostream_(_ofstream)
    , active_(is_active_rank(_rank))
    {
        open(_filename, _ofstream);
    }

    void open(const std::string& _filename, std::ofstream& _ofs)
    {
        if (active_) _ofs.open(_filename);
    }
    void close(std::ofstream _ofs)
    {
        if (active_) _ofs.close();
    }

    bool is_active_rank(int _r)
    {
        boost::mpi::communicator world;
        return world.rank() == _r;
    }

    bool active() const noexcept { return active_; }
    void active(const bool _a) noexcept { active_ = _a; }

    std::ostream& ostream() const noexcept { return ostream_; }

    template<typename T>
    const ParallelOstream& operator<<(const T& t) const noexcept
    {
        if (active_) ostream_ << t;
        return *this;
    }

    const ParallelOstream& operator<<(std::ostream& (*p)(std::ostream&)) const
        noexcept
    {
        if (active_) ostream_ << p;
        return *this;
    }
    template<typename T>
    const ParallelOstream& operator<<(T& t) const noexcept
    {
        if (active_) ostream_ << t;
        return *this;
    }

  private:
    std::ostream& ostream_;
    bool          active_;
};

} // namespace parallel_ostream
} // namespace iblgf
#endif
