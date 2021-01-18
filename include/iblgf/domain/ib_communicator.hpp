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

#ifndef IMMERSED_BOUNDARY_COMMUNICATOR_HPP
#define IMMERSED_BOUNDARY_COMMUNICATOR_HPP

#include <vector>
#include <cassert>
#include <functional>
#include <iblgf/global.hpp>
#include <iblgf/domain/octree/octant.hpp>

namespace iblgf
{
namespace ib
{
/** @brief Non-blocking MPI communication for immersed boundary points.
 *
 *  @detail: This class operates in two modes, where once can send either
 *  locally_owned points and assign the force values to the ghost points.
 *  Or one can send ghost-point and add their value to the locally owned points.
 *  Ghost points are defines as points which have the locally owned blocks
 *  in their influence list.
 */
template<class IBType>
class ib_communicator
{
  public: // member types
    using ib_type = IBType;
    using real_coordinate_type = typename ib_type::real_coordinate_type;
    using coordinate_type = typename ib_type::coordinate_type;

  public: // Ctors
    ib_communicator(const ib_communicator& other) = default;
    ib_communicator(ib_communicator&& other) = default;
    ib_communicator& operator=(const ib_communicator& other) & = default;
    ib_communicator& operator=(ib_communicator&& other) & = default;
    ~ib_communicator() = default;

    ib_communicator(ib_type* _ib)
    : ib_(_ib)
    , locally_owned_indices_(comm_.size())
    , ghost_indices_(comm_.size())
    , locally_owned_data_(comm_.size())
    , ghost_data_(comm_.size())
    , send_reqs_(comm_.size())
    , recv_reqs_(comm_.size())
    {
    }

  public: // init functions
    /**@brief Compute communication indices. Locally owned inidices, are
     * indices of locally owned forces, which will be send the ranks with
     * block that are influenced by that force.
     * Ghost indices, are indices, which are not locally owned but influenced
     * by my block
     * */
    void compute_indices() noexcept
    {
        const auto my_rank = comm_.rank();
        this->clear();
        for (std::size_t i = 0; i < ib_->size(); ++i)
        {
            if (ib_->rank(i) == my_rank)
            {
                std::set<int> unique_inflRanks;
                for (auto& infl_block : ib_->influence_list(i))
                {
                    if (infl_block->rank() != my_rank)
                    { unique_inflRanks.insert(infl_block->rank()); }
                }
                for (auto& ur : unique_inflRanks)
                { locally_owned_indices_[ur].push_back(i); }
            }
            else
            {
                for (auto& infl_block : ib_->influence_list(i))
                {
                    if (infl_block->rank() == my_rank)
                    {
                        ghost_indices_[ib_->rank(i)].push_back(i);
                        break;
                    }
                }
            }
        }
    }

    /**@brief Clear all data and resize vectors* */
    void clear()
    {
        locally_owned_indices_.clear();
        ghost_indices_.clear();
        locally_owned_data_.clear();
        ghost_data_.clear();
        send_reqs_.clear();
        recv_reqs_.clear();

        locally_owned_indices_.resize(comm_.size());
        ghost_indices_.resize(comm_.size());
        locally_owned_data_.resize(comm_.size());
        ghost_data_.resize(comm_.size());
        send_reqs_.resize(comm_.size());
        recv_reqs_.resize(comm_.size());
    }

    /**@brief Pack all messages, start send/recvs and store requests.
     * */
    template <class VecType>
    void start_communication(bool send_locally_owned, VecType& f)
    {
        pack_messages(send_locally_owned, f);

        auto& send_indices =
            send_locally_owned ? locally_owned_indices_ : ghost_indices_;
        auto& recv_indices =
            send_locally_owned ? ghost_indices_ : locally_owned_indices_;
        auto& send_data =
            send_locally_owned ? locally_owned_data_ : ghost_data_;
        auto& recv_data =
            send_locally_owned ? ghost_data_ : locally_owned_data_;

        //MPI tag is the sending rank
        const auto my_rank = comm_.rank();
        for (int rank_other = 0; rank_other < comm_.size(); ++rank_other)
        {
            if (!send_indices[rank_other].empty())
            {
                //dest=rank_other
                send_reqs_[rank_other] =
                    comm_.isend(rank_other, my_rank, send_data[rank_other]);
            }
            if (!recv_indices[rank_other].empty())
            {
                //source=rank_other
                recv_reqs_[rank_other] =
                    comm_.irecv(rank_other, rank_other, recv_data[rank_other]);
            }
        }
    }

    /**@brief Finish communication, i.e. wait for all request and
     * unpack the mesaages
     * */
    template <class VecType>
    void finish_communication(bool send_locally_owned, VecType& f)
    {
        boost::mpi::wait_all(send_reqs_.begin() + 1, send_reqs_.end());
        boost::mpi::wait_all(recv_reqs_.begin() + 1, recv_reqs_.end());
        this->unpack_messages(send_locally_owned, f);
    }

    /**@brief Blocking communication call of immersed boundary points.
     * Use start/finish individually for non-blocking communication
     * */
    template <class VecType>
    void communicate(bool send_locally_owned, VecType& f)
    {
        start_communication(send_locally_owned, f);
        finish_communication(send_locally_owned, f);
    }

  private: //Private member: Packing/Unpacking
    /**@brief Pack all messages to the send buffer.
     * */
    template <class VecType>
    void pack_messages(bool send_locally_owned, VecType& f) noexcept
    {
        //send
        auto& indices =
            send_locally_owned ? locally_owned_indices_ : ghost_indices_;
        auto& data = send_locally_owned ? locally_owned_data_ : ghost_data_;

        for (std::size_t rank = 0; rank < indices.size(); ++rank)
        {
            data[rank].resize(indices[rank].size());
            for (std::size_t i = 0; i < indices[rank].size(); ++i)
            { data[rank][i] = f[indices[rank][i]]; }
        }
    }

    /**@brief Unpack all received messages. If send_locally_owned=true,
     * this will assign the values to the ghost forces, else it will be
     * added to the forces.
     * */
    template <class VecType>
    void unpack_messages(bool send_locally_owned, VecType& f )
    {
        auto& indices =
            send_locally_owned ? ghost_indices_ : locally_owned_indices_;
        auto& data = send_locally_owned ? ghost_data_ : locally_owned_data_;

        //Unpack received data
        //If sending locally owned, this will assign the data, else when
        //sending ghost data, we will add the received contribution
        for (std::size_t rank = 0; rank < indices.size(); ++rank)
        {
            assert(indices[rank].size() == data[rank].size());
            for (std::size_t i = 0; i < indices[rank].size(); ++i)
            {
                f[indices[rank][i]] =
                    send_locally_owned
                        ? data[rank][i]
                        : f[indices[rank][i]] + data[rank][i];
            }
        }
    }

  public:
    ib_type* ib_;

    boost::mpi::communicator comm_;

    std::vector<std::vector<std::size_t>> locally_owned_indices_;
    std::vector<std::vector<std::size_t>> ghost_indices_;

    std::vector<std::vector<real_coordinate_type>> locally_owned_data_;
    std::vector<std::vector<real_coordinate_type>> ghost_data_;

    std::vector<boost::mpi::request> send_reqs_;
    std::vector<boost::mpi::request> recv_reqs_;
};

} // namespace ib
} // namespace iblgf

#endif

