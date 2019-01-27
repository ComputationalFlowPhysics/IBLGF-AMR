#ifndef INCLUDED_UTILS_HPP
#define INCLUDED_UTILS_HPP

#include <iostream>
#include <queue>
#include <iomanip>
#include <domain/octree/key.hpp>

namespace octree{

template<typename T>
constexpr T pow(const T& base, const int exp)
{
    return  exp == 0 ? 1 : base*pow(base,exp-1);
}

template<int Dim, int D=Dim-1>
struct rcIterator
{

    template<class BlockType, class Function>
    static void apply(const BlockType& _b,
                      const Function& f)
    {
        const auto base=_b.base();
        const auto extent=_b.extent();
        auto p=base;
        rcIterator<Dim, D>::apply_impl(p,f,base, extent );
    }
    template<class ArrayType, class Function>
    static void apply(const ArrayType& _base,
                      const ArrayType& _extent,
                      const Function& f)
    {
        auto p=_base;
        rcIterator<Dim, D>::apply_impl(p,f,_base, _extent );
    }
    template<class ArrayType, class Function>
    static void apply_impl(ArrayType& _p, 
                           const Function& f,
                           const ArrayType& _base, 
                           const ArrayType& _extent
                       )
    {
        for(std::size_t k=0; k<static_cast<std::size_t>(_extent[D]);++k)
        {
            _p[D]=_base[D]+k;
            rcIterator<Dim, D-1>::apply_impl(_p,f, _base, _extent);
        }
    }
};

template<int Dim>
struct rcIterator<Dim,0>
{
    template<class ArrayType, class Function>
    static void apply_impl(ArrayType& _p, 
                           const Function& f,
                           const ArrayType& _base, 
                           const ArrayType& _extent )
    {
        for(std::size_t k=0; k<static_cast<std::size_t>(_extent[0]);++k)
        {
            _p[0]=_base[0]+k;
            f(_p);
        }
    }
};

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

template<class Node>
struct iterator_base
{
    using node_type = Node;

    iterator_base() = delete;
    iterator_base(const iterator_base&) noexcept = default;
    iterator_base(iterator_base&&) noexcept = default;
    iterator_base(node_type* _ptr) noexcept : ptr_(_ptr) {}

    friend bool operator==(const iterator_base& l, const iterator_base& r)
    {
        return (l._end ? (l._end==r._end) : (r._end ? (false) : (l.ptr_==r.ptr_)));
    }
    friend bool operator!=(const iterator_base& l, const iterator_base& r)
    {
        return !operator==(l,r);
    }
    
    node_type* ptr_;
    bool _end = false;

    void swap(iterator_base& other) noexcept
    {
        const auto tmp = ptr_;
        ptr_ = other.ptr_;
        other.ptr_ = tmp;
        std::swap(_end, other._end);
    }

    node_type* ptr()const noexcept {return ptr_;}

};

template<class Node,int Dim=3>
struct iterator_depth_first : public iterator_base<Node>
{
public:
    using key_type = Key<Dim>;
    using iterator_base_type = iterator_base<Node>;
    using node_type = typename iterator_base_type::node_type;
    iterator_depth_first(const iterator_depth_first&) noexcept = default;
    iterator_depth_first(iterator_depth_first&&) noexcept = default;

    iterator_depth_first& operator=(iterator_depth_first other) noexcept
    {
        swap(*this,other);
        return *this;
    }

    iterator_depth_first& operator=(iterator_depth_first&& other) noexcept
    {
        this->ptr_ = other->ptr_;
        this->_child_number = std::move(other._child_number);
    } 
    
public:

    iterator_depth_first(node_type* _ptr) noexcept
    :   iterator_base_type(_ptr) 
    {
        _child_number.fill(0);
    }

    iterator_depth_first() noexcept
    :   iterator_base_type(nullptr)
    {
        _child_number.fill(0);
        this->_end = true;
    }


public:

    node_type& operator*() noexcept { return *(this->ptr_); }
    const node_type& operator*() const noexcept { return *(this->ptr_); }
    node_type* operator->() noexcept { return this->ptr_; }
    const node_type* operator->() const noexcept { return this->ptr_; }

    iterator_depth_first& operator++() noexcept
    {
        const auto current_level = this->ptr_->key().level();
        if (current_level == key_type::max_level())
        {
            this->ptr_ = get_parent();
            return this->operator++();
        }
        auto& next_child = _child_number[current_level];
        while (next_child < 8 && 
               this->ptr_->child(next_child) == nullptr) 
        {
            ++next_child;
        }
        if (next_child >= 8|| current_level==level_max)
        {
            if (current_level == 0)
            {
                this->_end = true;
                return *this;
            }
            else
            {
                next_child = 0;
                this->ptr_ = get_parent();
                return this->operator++();
            }
        }
        else
        {
            _child_number[current_level+1] = 0;
            this->ptr_ = this->ptr_->child(next_child); 
            ++next_child;
            if(current_level<level_min-1)this->operator++();
            return *this;
        }
    }

    friend void swap(iterator_depth_first& l, iterator_depth_first& r) noexcept
    {
        static_cast<iterator_base_type>(l).swap(r);
        l._child_number.swap(r._child_number);
    }

    const int& max_level()const noexcept{return level_max;}
    int& max_level(){return level_max;}
    const int& min_level()const noexcept{return level_min;}
    int& min_level(){return level_min;}

private:

    node_type* get_parent() const noexcept
    {
        return this->ptr_->parent();    
    }
    std::array<short,key_type::max_level()> _child_number;
    int level_max=key_type::max_level();
    int level_min=0;
};

template<class Node,int Dim=3>
struct iterator_breadth_first : public iterator_base<Node>
{
public:
    using key_type = Key<Dim>;
    using iterator_base_type = iterator_base<Node>;
    using node_type = typename iterator_base_type::node_type;
    iterator_breadth_first(const iterator_breadth_first&) = default;
    iterator_breadth_first(iterator_breadth_first&&)  = default;

    iterator_breadth_first& operator=(iterator_breadth_first other) 
    {
        swap(*this,other);
        return *this;
    }

    iterator_breadth_first& operator=(iterator_breadth_first&& other) 
    {
        this->ptr_ = other->ptr_;
        this->queue_ = std::move(other->queue_);
    } 
    
public:

    iterator_breadth_first(node_type* _ptr) noexcept
    :   iterator_base_type(_ptr) 
    {
        queue_.push(_ptr);
    }

    iterator_breadth_first() noexcept
    :   iterator_base_type(nullptr)
    {
        this->_end = true;
    }

public:

    node_type& operator*() noexcept { return *(this->ptr_); }
    const node_type& operator*() const noexcept { return *(this->ptr_); }
    node_type* operator->() noexcept { return this->ptr_; }
    const node_type* operator->() const noexcept { return this->ptr_; }

    iterator_breadth_first& operator++() noexcept
    {
        if(queue_.empty()) {
            this->_end=true;
            return *this;
        }
        this->ptr_=queue_.front();
        queue_.pop();
        for(int i =0; i<8; ++i)
        {
            if(this->ptr_->child(i))
                queue_.push(this->ptr_->child(i));
        }
        if (this->ptr_->key().level() == 0){
            this->operator++();
        } 
        return *this;
    }

  
    friend void swap(iterator_breadth_first& l, iterator_breadth_first& r) noexcept
    {
        static_cast<iterator_base_type>(l).swap(r);
        l.queue_.swap(r.queue_);
    }

private:

    std::queue<node_type*> queue_;
};





} //namespace detail

} //namespace octree

#endif //INCLUDED_UTILS_HPP
