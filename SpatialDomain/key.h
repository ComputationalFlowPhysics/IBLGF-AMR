#ifndef INCLUDED_KEY_HPP
#define INCLUDED_KEY_HPP


#include <array>
#include <iomanip>
#include "vector.h"
#include "bitmasks.h"

namespace octree
{

template<int Dim>
class Key;
template<int Dim>
bool operator< (const Key<Dim>& l, const Key<Dim>& r);
template<int Dim>
bool operator<=(const Key<Dim>& l, const Key<Dim>& r);
template<int Dim>
bool operator> (const Key<Dim>& l, const Key<Dim>& r);
template<int Dim>
bool operator>=(const Key<Dim>& l, const Key<Dim>& r);
template<int Dim>
bool operator==(const Key<Dim>& l, const Key<Dim>& r);

template<int Dim>
class Key{};


template <>
class Key<3>
{

public:

    static constexpr int Dim=3;
    using id_t = std::uint_fast64_t;
    using coordinate_t = math::vector<int,Dim>;
    using bitmask_t = Bitmasks<Dim>;
    static constexpr unsigned int max_level = bitmask_t::max_level;

	Key() = default;
	Key(id_t _id) : id_(_id) {}
	Key(const Key& other) = default;
	Key(Key&& other) = default;
	Key& operator=(const Key& other) & = default;
	Key& operator=(Key&& other) & = default;
	~Key() = default;

	Key(const coordinate_t& x, int level)
	: Key(x.x(),x.y(),x.z(),level) {}
	
	Key(id_t x, id_t y, id_t z, int level)
	: id_(0)
	{
		for (int i=level; i>0; --i)
		{
			id_ |= ((x & bitmask_t::lo_mask) << ((max_level-i)*3+7));
			id_ |= ((y & bitmask_t::lo_mask) << ((max_level-i)*3+8));
			id_ |= ((z & bitmask_t::lo_mask) << ((max_level-i)*3+9));
			x >>= 1;
			y >>= 1;
			z >>= 1;
		}
		id_ |= (static_cast<id_t>(level) << 2);
	}

    coordinate_t coordinate( ) const noexcept
    {
		const int level = this->level();
		id_t x(0), y(0), z(0);
		id_t k = id_ >> (7 + 3*(max_level-level));
		for (int i=0; i<level; ++i)
		{
			x |= ((k & bitmask_t::lo_mask) << i);
			k >>= 1;
			y |= ((k & bitmask_t::lo_mask) << i);
			k >>= 1;
			z |= ((k & bitmask_t::lo_mask) << i);
			k >>= 1;
		}
    	return coordinate_t{{{static_cast<int>(x),
                             static_cast<int>(y),
                             static_cast<int>(z)}}};
    }

   int level() const noexcept
    {
        return static_cast<int>((bitmask_t::level_mask & id_) >> 2);
    }

    id_t parent_key() const noexcept
    {
		const auto level = this->level();
        return (bitmask_t::coord_mask_arr[level-1] & id_) | ((level - 1) << 2);
    }

	Key child_key(int i) const
	{
		const int level = ((bitmask_t::level_mask & id_) >> 2);
		return {(bitmask_t::coord_mask_arr[level] & id_) | 
                (static_cast<id_t>(i) << ((max_level-(level+1))*3+7)) | 
                ((level+1) << 2)};
	}

	Key next_key() const
	{
		const int level = ((bitmask_t::level_mask & id_) >> 2);
		if (id_ >= (bitmask_t::max_arr[level]-3)) return bitmask_t::min_arr[level];
		return {((((bitmask_t::coord_mask_arr[level] & id_) >> 
                  ((max_level-level)*3+7)) + 1) << 
                  ((max_level-level)*3+7)) | 
                    bitmask_t::min_arr[level]};
	}


	
	Key prev_key() const
	{
		const int level = ((bitmask_t::level_mask & id_) >> 2);
		if (id_ <= (bitmask_t::min_arr[level]+3)) return bitmask_t::max_arr[level]-3;
		return {((((bitmask_t::coord_mask_arr[level] & id_) >> 
                  ((max_level-level)*3+7)) - 1) << 
                  ((max_level-level)*3+7)) | 
                  bitmask_t::min_arr[level]};
	}

	bool is_branch_of(const Key& k) const
	{
		return (k>=*this) && (k<next_key());
	}
	

    id_t id() const noexcept { return id_; }


    void print_binary()
    {
        auto size=sizeof(id_t)*CHAR_BIT-1;
        for (decltype(size) i = 0; i< size+1; i++)
        {
            std::cout<<std::setw(3)<<std::left<<size-i;
        }
        std::cout<<std::endl;
        for (decltype(size) i = 0; i< size+1; i++)
        {
            std::cout<<std::setw(3)<<std::left<<"|";
        }
        std::cout<<std::endl;
        for (int i = size; i >= 0; i--)
        {
            std::cout<<std::setw(3)<<std::left 
            << ((id_ >> i) & 1);
        }
        std::cout<<std::endl;

    }


private:
    id_t id_;

};


template<int Dim>
bool operator< (const Key<Dim>& l, const Key<Dim>& r) { return l.id()< r.id();}
template<int Dim>
bool operator<=(const Key<Dim>& l, const Key<Dim>& r) { return l.id()<=r.id();}
template<int Dim>
bool operator> (const Key<Dim>& l, const Key<Dim>& r) { return l.id()> r.id();}
template<int Dim>
bool operator>=(const Key<Dim>& l, const Key<Dim>& r) { return l.id()>=r.id();}
template<int Dim>
bool operator==(const Key<Dim>& l, const Key<Dim>& r) { return l.id()==r.id();}



}//namespace octree

#endif 
