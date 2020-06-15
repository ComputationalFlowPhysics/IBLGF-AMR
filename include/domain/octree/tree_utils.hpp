#ifndef INCLUDED_UTILS_HPP
#define INCLUDED_UTILS_HPP

#include <iostream>
#include <queue>
#include <stack>
#include <iomanip>
#include <domain/octree/key.hpp>
#include <utilities/crtp.hpp>

namespace octree{
template<class MapType>
class MapKeyIterator : public MapType::iterator
{

public:
    using typename MapType::mapped_type;
    using typename MapType::key_type;
    using iterator_t =typename MapType::iterator;

public:

    MapKeyIterator()
    :iterator_t(){};

    MapKeyIterator(iterator_t it_ )
    :iterator_t(it_){};

public:
    key_type* operator->() noexcept
    {
        return (key_type* const )&( iterator_t::operator -> ( )->first );
    }
    const key_type& operator*(){return iterator_t::operator*().first; }
};

template<class MapType>
class MapValueIterator : public MapType::iterator
{
public:
    using mapped_type =typename MapType::mapped_type;
    using key_type =typename MapType::key_type;
    using iterator_t =typename MapType::iterator;

public:
    MapValueIterator()
    : iterator_t(){}

    MapValueIterator( const iterator_t& _it )
    : iterator_t(_it){}

    mapped_type* operator->() noexcept
    {
        return (mapped_type* const )&( iterator_t::operator->()->second );
    }
    const mapped_type& operator*() { return iterator_t::operator*().second; }
};


template<class MapType>
class MapValuePtrIterator : public MapType::iterator
{
public:
    using mapped_type =typename MapType::mapped_type;
    using key_type =typename MapType::key_type;
    using iterator_t =typename MapType::iterator;

public:
    MapValuePtrIterator()
    : iterator_t(){}

    MapValuePtrIterator( const iterator_t& _it )
    : iterator_t(_it){}

    mapped_type operator->() const noexcept
    {
        return iterator_t::operator->()->second ;
    }
    mapped_type operator*() { return iterator_t::operator*().second; }
    mapped_type ptr() { return iterator_t::operator*().second; }
};


namespace tuple_utils
{

    template <typename Tuple, typename F, std::size_t ...Indices>
    void for_each_impl(Tuple&& tuple, F&& f, std::index_sequence<Indices...>)
    {
        using swallow = int[];
        (void)swallow{1, (f(std::get<Indices>(std::forward<Tuple>(tuple))), void(), int{})...  };
    }

    template <typename Tuple, typename F>
    void for_each(Tuple&& tuple, F&& f)
    {
        constexpr std::size_t N = std::tuple_size<std::remove_reference_t<Tuple>>::value;
        for_each_impl(std::forward<Tuple>(tuple), std::forward<F>(f),
                std::make_index_sequence<N>{});
    }
}



namespace detail
{

template<class T, class Derived>
class IteratorBase : public crtp::Crtps<Derived, IteratorBase<T,Derived>>
{

public: //member types
    using difference_type=std::ptrdiff_t;
    using size_type=std::size_t;
    using value_type=T;
    using pointer=T*;
    using const_pointer=const T*;
    using reference=T&;
    using const_reference=const T&;
    using iterator_category=std::forward_iterator_tag;

public: //Ctors:

    IteratorBase(const IteratorBase&) = default;
    IteratorBase(IteratorBase&&) = default;
	IteratorBase& operator=(const IteratorBase&) & = default;
	IteratorBase& operator=(IteratorBase&&) & = default;
    ~IteratorBase()=default;

    IteratorBase() =default;
    IteratorBase(pointer _ptr) : current_(_ptr), end_(false){}

public: //Acess
    auto& operator*() noexcept { return *(this->current_); }
    const auto& operator*() const noexcept { return *(this->current_); }
    auto operator->() noexcept { return current_; }
    auto operator->() const noexcept { return current_; }

    void swap(IteratorBase& other) noexcept
    {
        std::swap(current_,other.current_);
    }

    //Incremet/Decrement
    IteratorBase& operator++() noexcept //precrement
    {
        return this->derived()->operator ++();
    }

    IteratorBase operator++(int) noexcept //postcrement
    {
        const IteratorBase tmp(*this);
        ++(*this);
        return tmp;
    }

    IteratorBase& operator+=(size_type n)
    {
        for(size_type i=0;i<n;++i){++(*this);}
        return *this;
    }

    friend bool operator==(const IteratorBase& lhs, const IteratorBase& rhs)noexcept
    {
        return (lhs.end_ ? (lhs.end_==rhs.end_) :
                (rhs.end_ ? (false) : (lhs.current_==rhs.current_)));
        return lhs.current_==rhs.current_;
    }
    friend bool operator!=(const IteratorBase& lhs, const IteratorBase& rhs)noexcept
    {
        return ! operator==(lhs,rhs);
    }

    pointer ptr()const noexcept{ return current_; }

protected:
    pointer current_=nullptr;
    bool end_ = true;
};


template<class Node,int Dim=3>
struct IteratorBfs : public IteratorBase<Node, IteratorBfs<Node,Dim>>
{

public:

    using iterator_base_type = IteratorBase<Node,IteratorBfs<Node,Dim>>;
    using node_type = Node;

public: //member types

    using difference_type=typename iterator_base_type::difference_type;
    using size_type=typename iterator_base_type::size_type;
    using value_type=typename iterator_base_type::value_type;
    using pointer=typename iterator_base_type::pointer;
    using const_pointer=typename iterator_base_type::const_pointer;
    using reference=typename iterator_base_type::reference;
    using const_reference=typename iterator_base_type::const_reference;
    using iterator_category=typename iterator_base_type::iterator_category;

public: //ctors

    using iterator_base_type::IteratorBase;

    IteratorBfs(const IteratorBfs&) = default;
    IteratorBfs(IteratorBfs&&) = default;
	IteratorBfs& operator=(const IteratorBfs&) & = default;
    ~IteratorBfs()=default;
    IteratorBfs()=default;


    IteratorBfs& operator=(IteratorBfs&& other)
    {
        this->current_ = other->current_;
        this->queue_ = std::move(other->queue_);
    }

    IteratorBfs(node_type* _ptr) noexcept
    :   iterator_base_type(_ptr)
    {
        queue_.push(_ptr);
    }


public: //operator overloads


    IteratorBfs& operator++() noexcept
    {
        if(queue_.empty()) {
            this->end_=true;
            return *this;
        }
        this->current_=queue_.front();
        queue_.pop();
        //for(int i =0; i<static_cast<int>(node_type::nChildren); ++i)
        for(int i =0; i<node_type::num_children(); ++i)
        {
            if(this->current_->child(i))
                queue_.push(this->current_->child(i));
        }
        if (this->current_->key().level() == 0){
            this->operator++();
        }
        return *this;
    }


    friend void swap(IteratorBfs& l, IteratorBfs& r) noexcept
    {
        static_cast<iterator_base_type>(l).swap(r);
        l.queue_.swap(r.queue_);
    }

private:

    std::queue<node_type*> queue_;
};

template<class Node,int Dim=3>
struct IteratorDfs : public IteratorBase<Node, IteratorDfs<Node,Dim>>
{

public:

    using iterator_base_type = IteratorBase<Node,IteratorDfs<Node,Dim>>;
    using node_type = Node;

public: //member types

    using difference_type=typename iterator_base_type::difference_type;
    using size_type=typename iterator_base_type::size_type;
    using value_type=typename iterator_base_type::value_type;
    using pointer=typename iterator_base_type::pointer;
    using const_pointer=typename iterator_base_type::const_pointer;
    using reference=typename iterator_base_type::reference;
    using const_reference=typename iterator_base_type::const_reference;
    using iterator_category=typename iterator_base_type::iterator_category;

public: //ctors

    using iterator_base_type::IteratorBase;

    IteratorDfs(const IteratorDfs&) = default;
	IteratorDfs& operator=(const IteratorDfs&) & = default;
    ~IteratorDfs()=default;
    IteratorDfs()=default;

    IteratorDfs& operator=(IteratorDfs&& other)
    {
        this->current_ = other->current_;
        this->stack_ = std::move(other->stack_);
    }
    IteratorDfs(node_type* _ptr) noexcept
    :iterator_base_type(_ptr) { stack_.push(_ptr); }

public: //operator overloads

    IteratorDfs& operator++() noexcept
    {
        if(stack_.empty()) {
            this->end_=true;
            return *this;
        }
        this->current_=stack_.top();
        stack_.pop();
        for(int i =node_type::num_children()-1; i>=0; --i)
        {
            if(this->current_->child(i))
                stack_.push(this->current_->child(i));
        }
        if (this->current_->key().level() == 0){
            this->operator++();
        }
        return *this;
    }

    friend void swap(IteratorDfs& l, IteratorDfs& r) noexcept
    {
        static_cast<iterator_base_type>(l).swap(r);
        l.stack_.swap(r.stack_);
    }

private:
    std::stack<node_type*> stack_;
};


/** @brief Iterator node with fullfilled condition (lambda function)*/
template<class Iterator>
struct ConditionalIterator : public Iterator
{
public:
    using node_type=typename Iterator::node_type;
    using super_type=Iterator;
public: //Ctors
    using Iterator::Iterator; //inherite base class ctors
    //using Iterator::operator=;

    using condition_t =std::function<bool(ConditionalIterator&)>;

public: //Ctors
    ConditionalIterator(node_type* _ptr, condition_t _c ) noexcept
    :super_type(_ptr), cond_(_c)
    {
        if(!cond_(*this))
             this->operator++();
    }

public: //Operators
    ConditionalIterator& operator++() noexcept
    {
        if(this->_end) return *this;
        super_type::operator++();
        if(this->_end)return *this;
        if(!cond_(*this))
        {
             this->operator++();
        }
        return *this;
    }

private:
    condition_t cond_;
};





} //namespace detail

} //namespace octree

#endif //INCLUDED_UTILS_HPP
