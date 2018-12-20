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
template<class Key>
class ComputeTask
{

public:
    using key_t  = Key;
public:

    ComputeTask(const ComputeTask&  other) = default;
    ComputeTask(      ComputeTask&& other) = default;
    ComputeTask& operator=(const ComputeTask&  other) & = default;
    ComputeTask& operator=(      ComputeTask&& other) & = default;
    ~ComputeTask() = default;

    ComputeTask() = default;

    ComputeTask(key_t _k, int _rank, float_type _load)
    : key_(_k), rank_(_rank),load_(_load)
    {
    }

    const Key& key()const noexcept {return key_;}
    Key& key()noexcept {return key_;}
    const int& rank()const noexcept {return rank_;}
    int& rank()noexcept {return rank_;}
    const float_type& load()const noexcept {return load_;}
    float_type& load()noexcept {return load_;}


private:
    Key key_;
    int rank_;
    float_type load_;

private:

    friend class boost::serialization::access;
                  
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & key_;
        ar & rank_;
        ar & load_;
    }
};

}

#endif
