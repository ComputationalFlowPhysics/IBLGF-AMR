#ifndef INCLUDED_TAG_GENERATOR_HPP
#define INCLUDED_TAG_GENERATOR_HPP

#include<vector>

#include <mpi/tags.hpp>


/** @brief TagGenerator
 *  Generate mpi tags using a 3d periodic grid based upon baseTags of type
 *  enum, the message number for that tag as well as the rank.  Allows to keep
 *  track of multiple messages per tag.
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
    static_assert(std::is_same<tag_type, enum_utype>::value);

public:
   static constexpr int nMessages_=1000;

public: // ctors

	TagGenerator(const TagGenerator&) = delete;
	TagGenerator(TagGenerator&&) = delete;
	TagGenerator& operator=(const TagGenerator&) & = delete;
	TagGenerator& operator=(TagGenerator&&) & = delete;
    ~TagGenerator()=default;


private:   
    TagGenerator(int _rank=boost::mpi::communicator().rank(),int _nProcs=boost::mpi::communicator().size() )
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


    inline tag_type base_tag( tag_type _tag  ) const noexcept
    {
        return  _tag / (nMessages_*nProcs_);
    }

    inline tag_type message_nr(tag_type _tag) const noexcept
    {
        return  (_tag-base_tag(_tag)*nTags_) / nMessages_;
    }

    inline tag_type rank(tag_type _tag) const noexcept
    {
        const auto base_tag = base_tag(_tag);
        const tag_type nr =  (_tag-base_tag*nTags_)/ nMessages_;
        return  _tag-base_tag*nTags_ - nr*nMessages_;
    }

    const int& nProcessors() const noexcept {return nProcs_;}
    int& nProcessors() noexcept {return nProcs_;}

public:  //Some output functions
    struct TagInfo
    {
        tag_type rank;
        tag_type message_nr;
        tag_type base_tag;
        tag_type index;

        friend std::ostream& operator<<(std::ostream& os, const TagInfo t)
        {
            os<<"index: "<<t.index<<" "
              <<"rank :"<<t.rank<<" "
              <<"message_nr "<<t.message_nr<<" "
              <<"base Tag "<<t.base_tag<<" ";
              return os;
        }
    };

    TagInfo info( tag_type _tag )
    {
        TagInfo i;
        i.base_tag = _tag / (nProcs_*nMessages_);
        i.message_nr =  (_tag-i.base_tag*nProcs_*nMessages_ )/ nProcs_;
        i.rank =  _tag-i.base_tag*nProcs_*nMessages_ - i.message_nr*nProcs_;
        i.index =  _tag;
        return i;
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
       //x=_rank,       Nx=nProcs_, Nx_Stride=1;
       //y=_message_nr, Ny= nMessages_, Ny_stride=Nx
       //z=base_tag,    Nz=nTags_, Nz_stride=Nx*Ny
       const auto base_tag=to_integral(_base_tag);
       return _rank+(_message_nr%nMessages_)*nProcs_ + 
                        base_tag*nProcs_*nMessages_;
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
