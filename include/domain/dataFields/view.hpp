#ifndef INCLUDED_LGF_DOMAIN_VIEW_HPP
#define INCLUDED_LGF_DOMAIN_VIEW_HPP

#include <vector>
#include <iostream>

// IBLGF-specific
#include <types.hpp>
#include <domain/dataFields/datafield_utils.hpp>
#include <domain/dataFields/blockDescriptor.hpp>
#include <utilities/rcIterator.hpp>


namespace domain
{

//Exent generators
template<class View, int Dim>
class ViewIterator
{
public: 
    using view_t =  View;
    using coordinate_t = typename View::base_t;
    using element_t = typename view_t::element_t;

    struct dummy_t {};

public: 
    ViewIterator()=delete;
    ~ViewIterator() =default;

    ViewIterator(View* _view)
    :view_(_view),coordinate_(_view->base())
    { 
    }


public: 

    element_t& operator*() noexcept { return view_->field()->get(coordinate_); }
    const element_t& operator*()const  noexcept { return view_->field()->get(coordinate_); }
    element_t* operator->() noexcept { return view_->field()->get_ptr(coordinate_); }
    const element_t* operator->()const  noexcept { return (view_->field()->get_ptr(coordinate_)); }
    bool operator!=(dummy_t) const noexcept { return !_end; }

    inline auto& operator++() noexcept
    {
        for (int i = 0;i < Dim;++i)
        {
            if (coordinate_[i]-view_->base()[i] < view_->extent()[i] - 1)
            {
                coordinate_[i]+=view_->stride()[i];
                for (int j = 0;j < i;++j)
                {
                    coordinate_[j] = view_->base()[j];
                }
                return *this;
            }
        }
        _end = true;
        return *this;
    }


private:
    view_t* view_;
    coordinate_t coordinate_;
    bool _end = false;

};



template<class Field, int Dim=3>
class View : public BlockDescriptor<int, Dim>
{

public: //member types
    using block_type = BlockDescriptor<int ,Dim>;
    using super_type = block_type;
    using extent_t = typename block_type::extent_t;
    using base_t = typename block_type::base_t;
    using element_t = typename Field::data_type;

    using iterator_t = ViewIterator<View, Dim>;

   
public: //Ctors:

    View()=delete;
    ~View() =default;
    View(const View& rhs)=default;
	View& operator=(const View&) & = default ;

    View(View&& rhs)=default;
	View& operator=(View&&) & = default;

    View(Field* _f, const block_type _view, extent_t _stride=extent_t(1))
    :super_type(_view),field_(_f), stride_(_stride)
    {
    }

    template<class Function>
    void iterate(Function _f) noexcept
    {
        rcIterator<Dim>::iterate(this->base(), this->extent(),stride_,
               [this, &_f](const auto& _p)
               {
                   _f(field_->get(_p));
               });
    }

public: //member functions

    iterator_t begin(){return iterator_t(this); }
    auto end(){ return typename iterator_t::dummy_t{}; }

    auto& stride()noexcept {return stride_;}
    const auto& stride()const noexcept {return stride_;}

    Field* field()noexcept { return field_; }


    //TODO: void assign(view rhs)

public: //protected memeber:
    Field* field_=nullptr;
    extent_t stride_=extent_t(1);
};



} //namespace domain

#endif 
