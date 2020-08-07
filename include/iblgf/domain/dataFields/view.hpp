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

#ifndef INCLUDED_LGF_DOMAIN_VIEW_HPP
#define INCLUDED_LGF_DOMAIN_VIEW_HPP

#include <vector>
#include <iostream>

// IBLGF-specific
#include <iblgf/types.hpp>
#include <iblgf/domain/dataFields/blockDescriptor.hpp>
#include <iblgf/utilities/block_iterator.hpp>
#include <iblgf/utilities/tuple_utilities.hpp>

namespace iblgf
{
namespace domain
{
//Exent generators
template<class View, int Dim>
class ViewIterator
{
  public:
    using view_t = View;
    using coordinate_t = typename View::coordinate_type;
    using element_t = typename view_t::element_t;

    struct dummy_t
    {
    };

  public:
    ViewIterator() = delete;
    ~ViewIterator() = default;

    ViewIterator(View* _view)
    : view_(_view)
    , coordinate_(_view->base())
    {
    }

  public:
    element_t& operator*() noexcept { return view_->field()->get(coordinate_); }
    const element_t& operator*() const noexcept
    {
        return view_->field()->get(coordinate_);
    }
    element_t* operator->() noexcept
    {
        return view_->field()->get_ptr(coordinate_);
    }
    const element_t* operator->() const noexcept
    {
        return (view_->field()->get_ptr(coordinate_));
    }
    bool operator!=(dummy_t) const noexcept { return !_end; }

    inline auto& operator++() noexcept
    {
        for (int i = 0; i < Dim; ++i)
        {
            if (coordinate_[i] - view_->base()[i] < view_->extent()[i] - 1)
            {
                coordinate_[i] += view_->stride()[i];
                for (int j = 0; j < i; ++j)
                { coordinate_[j] = view_->base()[j]; }
                return *this;
            }
        }
        _end = true;
        return *this;
    }
    const auto& coordinate() const noexcept { return coordinate_; }

  private:
    view_t*      view_;
    coordinate_t coordinate_;
    bool         _end = false;
};

template<class Field, int Dim = 3>
class View : public BlockDescriptor<int, Dim>
{
  public: //member types
    using block_type = BlockDescriptor<int, Dim>;
    using super_type = block_type;
    using coordinate_type = typename block_type::coordinate_type;
    using element_t = typename Field::data_type;

    using iterator_t = ViewIterator<View, Dim>;

  public: //Ctors:
    View() = default;
    ~View() = default;
    View(const View& rhs) = default;
    View& operator=(const View&) & = default;

    View(View&& rhs) = default;
    View& operator=(View&&) & = default;

    View(Field* _f, const block_type _view,
        coordinate_type _stride = coordinate_type(1))
    : super_type(_view)
    , field_(_f)
    , stride_(_stride)
    {
        //TODO: Sanity checks on the blockd_type that is passed.
        //Needs to be contained within the fields block
    }

    template<class Function>
    void iterate(Function _f) noexcept
    {
        BlockIterator<Dim>::iterate(this->base(), this->extent(), stride_,
            [this, &_f](const auto& _p) { _f(field_->get(_p)); });
    }

  public: //member functions
    iterator_t begin() { return iterator_t(this); }
    auto       end() { return typename iterator_t::dummy_t{}; }

    auto&       stride() noexcept { return stride_; }
    const auto& stride() const noexcept { return stride_; }

    const Field*& field() const noexcept { return field_; }
    Field*&       field() noexcept { return field_; }

    template<class Container>
    View& operator=(Container& _c) noexcept
    {
        auto it_c = _c.begin();
        this->iterate([&it_c](auto& n) {
            n = *it_c;
            ++it_c;
        });
        return *this;
    }
    template<class Container>
    void assign_toBuffer(Container& _c) noexcept
    {
        auto it_c = _c.begin();
        this->iterate([&it_c](auto& n) {
            *it_c = n;
            ++it_c;
        });
    }
    template<class Container>
    void assign_toView(Container& _c) noexcept
    {
        *this = _c;
    }

  public: //unary arithmetic operator overloads
    /** @brief Scalar assign operator */
    View& operator=(const element_t& element) noexcept
    {
        for (auto& e : *this) { e = element; }
        return *this;
    }

    /** @{
     * @brief element wise add operator */
    View& operator+=(View& other) noexcept
    {
        assert((this->size() == other.size()) && (stride() == other.stride()));
        auto it_other = other.begin();
        this->iterate([&it_other](auto& n) {
            n += *it_other;
            ++it_other;
        });
        return *this;
    }
    View& operator+=(const element_t& element) noexcept
    {
        for (auto& e : *this) { e += element; }
        return *this;
    }
    /** @} */

    /** @{
     * @brief element wise subtract operator */
    View& operator-=(View& other) noexcept
    {
        assert((this->size() == other.size()) && (stride() == other.stride()));
        auto it_other = other.begin();
        this->iterate([&it_other](auto& n) {
            n -= *it_other;
            ++it_other;
        });
        return *this;
    }
    View& operator-=(const element_t& element) noexcept
    {
        for (auto& e : *this) { e -= element; }
        return *this;
    }
    /** @} */

    /** @{
     * @brief element wise multiply operator */
    View& operator*=(View& other) noexcept
    {
        assert((this->size() == other.size()) && (stride() == other.stride()));
        auto it_other = other.begin();
        this->iterate([&it_other](auto& n) {
            n *= *it_other;
            ++it_other;
        });
        return *this;
    }
    View& operator*=(const element_t& element) noexcept
    {
        for (auto& e : *this) { e *= element; }
        return *this;
    }
    /** @} */

    /** @{
     * @brief element wise divide operator */
    View& operator/=(View& other) noexcept
    {
        assert((this->size() == other.size()) && (stride() == other.stride()));
        auto it_other = other.begin();
        this->iterate([&it_other](auto& n) {
            n /= *it_other;
            ++it_other;
        });
        return *this;
    }
    View& operator/=(const element_t& element) noexcept
    {
        for (auto& e : *this) { e /= element; }
        return *this;
    }
    /** @} */

  public: //protected memeber:
    Field*          field_ = nullptr;
    coordinate_type stride_ = coordinate_type(1);
};

} //namespace domain
} // namespace iblgf

#endif
