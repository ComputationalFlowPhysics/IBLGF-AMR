#ifndef INCLUDED_TAG_GENERATOR_HPP
#define INCLUDED_TAG_GENERATOR_HPP

#include<vector>

#include "tags.hpp"

/** @brief TagGenerator
 *  Generate mpi tags using a periodic grid based upon baseTags of type
 *  enum, as well as the message number for that tag. Allows to keep
 *  track of multiple messages per tag and processor.
 *
 *  Requirement: Enum starts at zero and has nTags as
 *               last element
 */
namespace sr_mpi
{

template<class Enum, class Tag=int>
class TagGenerator
{

public:
    using enum_utype  =  typename std::underlying_type<Enum>::type;
    using tag_type =Tag;
    static_assert(std::is_same<tag_type, enum_utype>::value,"TagGenerator: Tag is not int-type");

public:
   static constexpr int nMessages_=1000;

public: // ctors

	TagGenerator(const TagGenerator&) = delete;
	TagGenerator(TagGenerator&&) = delete;
	TagGenerator& operator=(const TagGenerator&) & = delete;
	TagGenerator& operator=(TagGenerator&&) & = delete;
    ~TagGenerator()=default;


private:   
    TagGenerator(int _rank=boost::mpi::communicator().rank(),
                 int _nProcs=boost::mpi::communicator().size() )
    :rank_(_rank),
    nTags_(to_integral(Enum::nTags)), 
    nProcs_( _nProcs ), 
    message_count_(_nProcs,std::vector<int>(to_integral(Enum::nTags),0))
    {
    }
    
	friend TagGenerator<tags::type>& tag_gen(); 

public: //members

    template<int E>
    tag_type get() const noexcept
    {
        return get(std::integral_constant<tag_type, E>::value);
    }
    template<int E>
    tag_type get(int _rank) const noexcept
    {
        return get(std::integral_constant<tag_type, E>::value, _rank);
    }

    template<class T>
    tag_type get(T _base_tag, int _rank  ) const noexcept
    {
        return index(_base_tag, _rank);
    }

    template<int E>
    tag_type generate(int _rank) noexcept
    {
        return generate(std::integral_constant<tag_type, E>::value, _rank);
    }

    template<class T>
    tag_type generate(T _base_tag, int _rank  ) noexcept
    {
        const auto idx= index(_base_tag, _rank);
        message_count_[_rank][_base_tag]+=1;
        return idx;
    }

private: //member functions

   template<class T>
   constexpr tag_type to_integral( const T&  _t) const noexcept
   {
       return static_cast<tag_type>(_t);
   }

   template<class T>
   inline tag_type index(T _base_tag, 
                         tag_type _message_nr, 
                         int _rank) const noexcept
   {
       return to_integral(_base_tag) + nTags_*(_message_nr%nMessages_);
   }
   template<class T>
   tag_type index(T _base_tag, int _rank)const noexcept
   {
        return index(_base_tag,message_count_[_rank][_base_tag], _rank);
   }
   template<class T>
   tag_type index(T _base_tag)const noexcept
   {
        return index(_base_tag,message_count_[rank_][_base_tag], rank_);
   }

private: //data memebers

   const int rank_;                              ///< Default rank! or specified
   const int nTags_;                             ///< Total number of Tags.
   int nProcs_;                                  ///< Total number of Tags.
   std::vector<std::vector<int>> message_count_; ///< number of messages per tag
};

TagGenerator<tags::type>& tag_gen()
{
    static TagGenerator<tags::type> _tags;
    return _tags;
}

}

#endif 
