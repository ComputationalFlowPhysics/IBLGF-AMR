#ifndef OCTREE_INCLUDED_NODE_HPP
#define OCTREE_INCLUDED_NODE_HPP

#include <vector>
#include <memory>
#include <cmath>
#include <set>
#include <string>
#include <map>

// IBLGF-specific
#include <global.hpp>
#include <domain/octree/key.hpp>

namespace octree
{

enum class node_flag : int
{
    octant,
    hanging,
    boundary_node
};


template<int Dim, class DataType>
class Tree;

template<int Dim, class DataType>
class Octant_base
{

public:

    using key_type             = Key<Dim>;
    using coordinate_type      = typename key_type::coordinate_type;
    using real_coordinate_type = types::coordinate_type<float_type,Dim>;
    using key_index_type       = typename key_type::value_type;

    using tree_type = Tree<Dim, DataType>;

    static constexpr int num_children(){ return pow(2,Dim); };
    static constexpr int fmm_max_idx(){ return 30; };

    using rank_type = std::array<int, fmm_max_idx()>;
    static constexpr rank_type rank_default()
    {
        rank_type r;
        std::fill(r.begin(),r.end(),-1);

        return r;
    };


public:

	Octant_base()                                      = delete;
	Octant_base(const Octant_base& other)              = default;
	Octant_base(Octant_base&& other)                   = default;
	Octant_base& operator=(const Octant_base& other) & = default;
	Octant_base& operator=(Octant_base&& other)      & = default;
	~Octant_base()                                     = default;



    Octant_base(const coordinate_type& _x, int _level)
    :key_(key_type(_x,_level))
     {}

    Octant_base(const key_type& _k)
    :key_(_k)
    { }

public:

    /** @brief Get octant key*/
    const key_type& key() const noexcept{return key_;}


    /** @brief Octant level relative to root of tree*/
    auto level() const noexcept{return key_.level();}

    /** @brief Octant level relative to root of tree*/
    auto tree_level() const noexcept{return key_.level();}

    /** @brief Get octant coordinate (integer) based on tree structure */
    coordinate_type tree_coordinate() const noexcept
    {
        return key_.coordinate();
    }

    friend std::ostream& operator<<(std::ostream& os, const Octant_base& n)
    {
        os<<n.key_;
        return os;
    }

    const auto& rank_list()const noexcept{return ranks_;}
    auto& rank_list() noexcept{return ranks_;}


    auto rank_list_unique() noexcept
    {
        std::set<int> unique_ranks;
        for (auto r:ranks_)
        {
            if (r>0)
                unique_ranks.emplace(r);
        }

        return unique_ranks;
    }

    const int& rank(int rank_idx=0)const noexcept
    {
        // rand_idx==0 means not using fmm ranks
        return ranks_[rank_idx];
    }

    int& rank(int rank_idx=0) noexcept{return ranks_[rank_idx];}

private: //Serialization

    friend class boost::serialization::access;


    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & key_;
        ar & rank_;
    }

protected:
    key_type key_;
    rank_type ranks_=rank_default();

};



} //namespace octree
#endif
