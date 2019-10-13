#ifndef DOMAIN_INCLUDED_COMPUTE_TASK_HPP
#define DOMAIN_INCLUDED_COMPUTE_TASK_HPP

#include <vector>
#include <stdexcept>

#include <boost/mpi/communicator.hpp>

#include <global.hpp>

namespace domain
{

/** @brief ProcessType Server 
 *  Master/Server process.
 *  Stores the full tree structure without the data.
 *  Responsible for load balancing and listens to 
 *  the client/worker processes.
 */
template<class OctantType>
class ComputeTask
{

public:
    using octant_t  = OctantType;
public:

    ComputeTask(const ComputeTask&  other) = default;
    ComputeTask(      ComputeTask&& other) = default;
    ComputeTask& operator=(const ComputeTask&  other) & = default;
    ComputeTask& operator=(      ComputeTask&& other) & = default;
    ~ComputeTask() = default;

    ComputeTask() = default;

    ComputeTask(octant_t* _o, int _rank, float_type _load)
    : octant_(_o), rank_(_rank),load_(_load)
    {
    }


    octant_t* octant()noexcept {return octant_;}
    const auto& key()const noexcept {return octant_->key();}
    auto& key()noexcept {return octant_->key();}
    const int& rank()const noexcept {return rank_;}
    int& rank()noexcept {return rank_;}
    const float_type& load()const noexcept {return load_;}
    float_type& load()noexcept {return load_;}


private:
    octant_t* octant_=nullptr;
    int rank_=-1;
    float_type load_=0;

};

}

#endif
