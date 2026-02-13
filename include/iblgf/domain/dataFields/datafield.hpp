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

#ifndef INCLUDED_LGF_DOMAIN_DATAFIELD_HPP
#define INCLUDED_LGF_DOMAIN_DATAFIELD_HPP

#include <vector>
#include <tuple>
#include <iostream>
#include <algorithm>
#include <optional>
// IBLGF-specific
#include <iblgf/types.hpp>
#include <iblgf/linalg/linalg.hpp>
#include <iblgf/utilities/tuple_utilities.hpp>
#include <iblgf/domain/dataFields/blockDescriptor.hpp>
#include <iblgf/domain/dataFields/view.hpp>

#include <boost/preprocessor/tuple/size.hpp>
#include <boost/preprocessor/tuple/elem.hpp>
#include <boost/preprocessor/repetition/enum.hpp>
#include <boost/preprocessor/tuple/enum.hpp>
#include <boost/preprocessor/repetition/repeat.hpp>
#include <boost/preprocessor/tuple/pop_front.hpp>

namespace iblgf
{
namespace domain
{
enum class MeshObject : int
{
    cell,
    face,
    edge,
    vertex
};

class Datafield_trait
{
};

template<class DataType, std::size_t Dim>
class DataField : public BlockDescriptor<int, Dim>
{
  public: //member types
    using size_type = types::size_type;
    using data_type = DataType;
    static constexpr std::size_t dimension() { return Dim; };

    template<typename T>
    using vector_type = types::vector_type<T, Dim>;
    using buffer_d_t = vector_type<int>;
    using super_type = BlockDescriptor<int, Dim>;

    using block_type = BlockDescriptor<int, Dim>;
    using coordinate_t = typename block_type::coordinate_type;
    using view_type = View<DataField, Dim>;

  public: //Ctors:
    DataField() = default;
    ~DataField() = default;

    //TODO: Remove that cube_ alias to avoid all that
    DataField(const DataField& rhs)
    : data_(rhs.data_)
    , lowBuffer_(rhs.lowBuffer_)
    , highBuffer_(rhs.highBuffer_)
    , real_block_(rhs.real_block_)
    /*, cube_(std::make_unique<linalg::Cube_t>(&data_[0], real_block_.extent()[0],
          real_block_.extent()[1], real_block_.extent()[2]))*/
    {
	if (Dim == 2) { 
	    cube_ = std::make_unique<linalg::Cube_t>(&data_[0], real_block_.extent()[0],real_block_.extent()[1]);
	} else {
	    cube_ = std::make_unique<linalg::Cube_t>(&data_[0], real_block_.extent()[0],real_block_.extent()[1], real_block_.extent()[2]);
	}
    }

    DataField& operator=(const DataField& _other)
    {
        if (this == &_other) return *this;
        data_=_other.data_;
        lowBuffer_ = _other.lowBuffer_;
        highBuffer_ = _other.highBuffer_;
        real_block_ = _other.real_block_;
	if (Dim == 2) { 
	    cube_ = std::make_unique<linalg::Cube_t>(&data_[0], real_block_.extent()[0],real_block_.extent()[1]);
	} else {
	    cube_ = std::make_unique<linalg::Cube_t>(&data_[0], real_block_.extent()[0],real_block_.extent()[1], real_block_.extent()[2]);
	}
        return *this;
    }

    DataField(DataField&& rhs)
    : data_(std::move(rhs.data_))
    , lowBuffer_(std::move(rhs.lowBuffer_))
    , highBuffer_(std::move(rhs.highBuffer_))
    , real_block_(std::move(rhs.real_block_))
    /*, cube_(std::make_unique<linalg::Cube_t>(&data_[0], real_block_.extent()[0],
          real_block_.extent()[1], real_block_.extent()[2]))*/
    {	
	if (Dim == 2) { 
	    cube_ = std::make_unique<linalg::Cube_t>(&data_[0], real_block_.extent()[0],real_block_.extent()[1]);
	} else {
	    cube_ = std::make_unique<linalg::Cube_t>(&data_[0], real_block_.extent()[0],real_block_.extent()[1], real_block_.extent()[2]);
	}
    }
    DataField& operator=(DataField&& _other)
    {
        data_ = std::move(_other.data_);
        lowBuffer_ = std::move(_other.lowBuffer_);
        highBuffer_ = std::move(_other.highBuffer_);
        real_block_ = std::move(_other.real_block_);
	if (Dim == 2) { 
	    cube_ = std::make_unique<linalg::Cube_t>(&data_[0], real_block_.extent()[0],real_block_.extent()[1]);
	} else {
	    cube_ = std::make_unique<linalg::Cube_t>(&data_[0], real_block_.extent()[0],real_block_.extent()[1], real_block_.extent()[2]);
	}
        return *this;
    }

  public: //member functions
    /** @brief Initialize the datafield and grow according to buffer */
    void initialize(block_type _b, buffer_d_t _lb = buffer_d_t(0),
        buffer_d_t _hb = buffer_d_t(0), bool _allocate = true,
        bool _default = false, data_type _dval = data_type()) noexcept
    {
        lowBuffer_ = _lb;
        highBuffer_ = _hb;

        this->real_block_.base(_b.base() - lowBuffer_);
        this->real_block_.extent(_b.extent() + lowBuffer_ + highBuffer_);
        this->real_block_.level() = _b.level();

        this->base(_b.base());
        this->extent(_b.extent());
        this->level() = _b.level();
        if (_allocate) data_.resize(real_block_.size());
        if (_default) { std::fill(data_.begin(), data_.end(), _dval); }

        const auto ext = real_block_.extent();
        /*cube_ = std::make_unique<linalg::Cube_t>(
            (types::float_type*)&data_[0], ext[0], ext[1], ext[2]);*/
	if (Dim == 2) {
	    cube_ = std::make_unique<linalg::Cube_t>(
            (types::float_type*)&data_[0], ext[0], ext[1]);
	} else {
	    cube_ = std::make_unique<linalg::Cube_t>(
            (types::float_type*)&data_[0], ext[0], ext[1], ext[2]);
	}
    }

    auto begin() noexcept { return data_.begin(); }
    auto end() noexcept { return data_.end(); }

  public: //Access
    auto&       operator[](size_type i) noexcept { return data_[i]; }
    const auto& operator[](size_type i) const noexcept { return data_[i]; }

    auto& data() { return data_; }
    auto  data_ptr() { return &data_; }

    auto& linalg_data() { return cube_->data_; }
    auto& linalg() { return cube_; }

    inline data_type* get_ptr(const coordinate_t& _c) noexcept
    {
        return &data_[real_block_.index(_c)];
    }
    inline const data_type* get_ptr(const coordinate_t& _c) const noexcept
    {
        return &data_[real_block_.index(_c)];
    }
    inline data_type& get(const coordinate_t& _c) noexcept
    {
        return data_[real_block_.index(_c)];
    }
    inline const data_type& get(const coordinate_t& _c) const noexcept
    {
        return data_[real_block_.index(_c)];
    }

    inline const data_type& get_real_local(
        const coordinate_t& _c) const noexcept
    {
        return data_[real_block_.index_zeroBase(_c)];
    }
    inline data_type& get_real_local(const coordinate_t& _c) noexcept
    {
        return data_[real_block_.index_zeroBase(_c)];
    }

    //IJK access
    /*inline const data_type& get_real_local(
        int _i, int _j, int _k) const noexcept
    {
        return data_[real_block_.index_zeroBase(_i, _j, _k)];
    }
    inline data_type& get_real_local(int _i, int _j, int _k) noexcept
    {
        return data_[real_block_.index_zeroBase(_i, _j, _k)];
    }*/
    inline const data_type& get_real_local(
        int _i, int _j) const noexcept
    {
        return data_[real_block_.index_zeroBase(_i, _j)];
    }
    inline data_type& get_real_local(int _i, int _j) noexcept
    {
        return data_[real_block_.index_zeroBase(_i, _j)];
    }

    inline const data_type& get_real_local(
        int _i, int _j, int _k) const noexcept
    {
        return data_[real_block_.index_zeroBase(_i, _j, _k)];
    }
    inline data_type& get_real_local(int _i, int _j, int _k) noexcept
    {
        return data_[real_block_.index_zeroBase(_i, _j, _k)];
    }

  public: //unary arithmetic operator overloads
    /** @brief Scalar assign operator */
    DataField& operator=(const data_type& element) noexcept
    {
        std::fill(data_.begin(), data_.end(), element);
        return *this;
    }

    /** @{
     * @brief element wise add operator */
    DataField& operator+=(const DataField& other) noexcept
    {
        for (unsigned int i = 0; i < data_.size(); ++i) data_[i] += other[i];
        return *this;
    }
    DataField& operator+=(const data_type& element) noexcept
    {
        for (unsigned int i = 0; i < data_.size(); ++i) data_[i] += element;
        return *this;
    }
    /** @} */

    /** @{
     * @brief element wise subtract operator */
    DataField& operator-=(const DataField& other) noexcept
    {
        for (unsigned int i = 0; i < data_.size(); ++i) data_[i] -= other[i];
        return *this;
    }
    DataField& operator-=(const data_type& element) noexcept
    {
        for (unsigned int i = 0; i < data_.size(); ++i) data_[i] -= element;
        return *this;
    }
    /** @} */

    /** @{
     * @brief element wise multiply operator */
    DataField& operator*=(const DataField& other) noexcept
    {
        for (unsigned int i = 0; i < data_.size(); ++i) data_[i] *= other[i];
        return *this;
    }
    DataField& operator*=(const data_type& element) noexcept
    {
        for (unsigned int i = 0; i < data_.size(); ++i) data_[i] *= element;
        return *this;
    }
    /** @} */

    /** @{
     * @brief element wise divide operator */
    DataField& operator/=(const DataField& other) noexcept
    {
        for (unsigned int i = 0; i < data_.size(); ++i) data_[i] /= other[i];
        return *this;
    }
    DataField& operator/=(const data_type& element) noexcept
    {
        for (unsigned int i = 0; i < data_.size(); ++i) data_[i] /= element;
        return *this;
    }
    /** @} */

  public: //binary arithmetic operator overloads
    /** @{
     * @brief element wise add operator */
    friend auto operator+(DataField lhs, const DataField& rhs) noexcept
    {
        return lhs += rhs;
    }
    friend auto operator+(DataField lhs, const data_type& rhs) noexcept
    {
        return lhs += rhs;
    }
    friend auto operator+(const data_type& lhs, DataField rhs) noexcept
    {
        return rhs += lhs;
    }
    /** @} */

    /** @{
     * @brief element wise subtract operator */
    friend auto operator-(DataField lhs, const DataField& rhs) noexcept
    {
        return lhs -= rhs;
    }
    friend auto operator-(DataField lhs, const data_type& rhs) noexcept
    {
        return lhs -= rhs;
    }
    friend auto operator-(const data_type& lhs, DataField rhs) noexcept
    {
        return rhs -= lhs;
    }
    /** @} */

    /** @{
     * @brief element wise multiply operator */
    friend auto operator*(DataField lhs, const DataField& rhs) noexcept
    {
        return lhs *= rhs;
    }
    friend auto operator*(DataField lhs, const data_type& rhs) noexcept
    {
        return lhs *= rhs;
    }
    friend auto operator*(const data_type& lhs, DataField rhs) noexcept
    {
        return rhs *= lhs;
    }
    /** @} */

    /** @{
   * @brief element wise divide operator */
    friend auto operator/(DataField lhs, const DataField& rhs) noexcept
    {
        return lhs /= rhs;
    }
    friend auto operator/(DataField lhs, const data_type& rhs) noexcept
    {
        return lhs /= rhs;
    }
    friend auto operator/(const data_type& lhs, DataField rhs) noexcept
    {
        for (unsigned int i = 0; i < rhs.size(); ++i) rhs[i] = lhs / rhs[i];
    }
    /** @} */

  public: //Misc members:
    template<class BlockType, class OverlapType>
    bool buffer_overlap(
        const BlockType& other, OverlapType& overlap, int level) const noexcept
    {
        return real_block_.overlap(other, overlap, level);
    }

    template<class BlockType>
    auto send_view(const BlockType& other) noexcept
    {
        auto overlap = real_block_;
        auto has_overlap = this->overlap(other.real_block_, overlap);
        return has_overlap ? std::optional<view_type>(view(overlap))
                           : std::nullopt;
    }
    template<class BlockType>
    auto recv_view(const BlockType& other) noexcept
    {
        auto overlap = real_block_;
        auto has_overlap = other.overlap(this->real_block_, overlap);
        return has_overlap ? std::optional<view_type>(view(overlap))
                           : std::nullopt;
    }

    buffer_d_t&       lbuffer() noexcept { return lowBuffer_; }
    const buffer_d_t& lbuffer() const noexcept { return lowBuffer_; }
    buffer_d_t&       hbuffer() noexcept { return highBuffer_; }
    const buffer_d_t& hbuffer() const noexcept { return highBuffer_; }

    const block_type& real_block() const noexcept { return real_block_; }
    block_type&       real_block() noexcept { return real_block_; }

    auto size() const noexcept { return data_.size(); }

    /** @brief Get a (sub-)view of the datafield */
    auto view(
        const block_type& _b, coordinate_t _stride = coordinate_t(1)) noexcept
    {
        return view_type(this, _b, _stride);
    }
    auto domain_view() noexcept { return view(*this); }
    auto real_domain_view() noexcept { return view(real_block_); }

  protected:                                //protected memebers:
    std::vector<data_type> data_;           ///< actual data
    buffer_d_t lowBuffer_ = buffer_d_t(0);  ///< Buffer in negative direction
    buffer_d_t highBuffer_ = buffer_d_t(0); ///< Buffer in positive direction
    block_type real_block_; ///< Block descriptorinlcuding buffer

    /** @brief Linear algebra wrapper for the actual data */
    std::unique_ptr<linalg::Cube_t> cube_;
};

/****************************************************************************/

template<class Traits>
class Field;

template<class Tag, class DataType, std::size_t NFields, std::size_t lBuff,
    std::size_t hBuff, MeshObject MeshType, std::size_t Dim, bool _output>
struct field_traits
{
    using tag_type = Tag;
    using data_type = DataType;
    using field_type = Field<field_traits>;
    using data_field_t = DataField<data_type, Dim>;

    static auto                  name() noexcept { return tag_type::c_str(); }
    static constexpr bool        output() { return _output; }
    static constexpr tag_type    tag() { return tag_type{}; }
    static constexpr std::size_t nFields() { return NFields; }
    static constexpr MeshObject  mesh_type() { return MeshType; }
    static constexpr std::size_t lowBuffer() { return lBuff; }
    static constexpr std::size_t highBuffer() { return hBuff; }
};

template<class Traits>
class Field : public Traits
{
  public:
    using traits = Traits;
    using tag_type = typename Traits::tag_type;
    using data_type = typename traits::data_type;
    using data_field_t = typename traits::data_field_t;
    using view_type = typename data_field_t::view_type;
    using block_type = typename data_field_t::block_type;

  public:
    auto&       operator[](std::size_t i) noexcept { return fields_[i]; }
    const auto& operator[](std::size_t i) const noexcept { return fields_[i]; }

    void initialize(block_type _b, bool _allocate = true, bool _default = false,
        data_type _dval = data_type())
    {
        for (std::size_t i = 0; i < this->nFields(); ++i)
        {
            fields_[i].initialize(_b, this->lowBuffer(), this->highBuffer(),
                _allocate, _default, _dval);
        }
    }

  private:
    std::array<data_field_t, traits::nFields()> fields_;
};

#define STRINGIFY(X) #X

#define make_field_type_impl(                                                  \
    Dim, Name, DataType, NFields, lBuff, hBuff, MeshObjectType, output)        \
    struct Name##_tag_helper                                                   \
    {                                                                          \
        static constexpr tuple_tag_h Name##_tag{STRINGIFY(Name)};              \
    };                                                                         \
    using Name##_traits_type =                                                 \
        field_traits<                                                          \
            IBLGF_TAG_TYPE(Name##_tag_helper::Name##_tag),                     \
            DataType,                                                          \
            NFields, lBuff, hBuff, MeshObject::MeshObjectType, Dim, output>;   \
    using Name##_type = Field<Name##_traits_type>;                             \
    static constexpr Name##_traits_type Name{};

#ifdef IBLGF_COMPILE_CUDA
#define IBLGF_TAG_TYPE(tag) tag_type_ptr<&tag>
#else
#define IBLGF_TAG_TYPE(tag) tag_type<tag>
#endif

#define make_field_type_impl_default(                                          \
    Dim, key, DataType, NFields, lBuffer, hBuffer, MeshObjectType)             \
    make_field_type_impl(                                                      \
        Dim, key, DataType, NFields, lBuffer, hBuffer, MeshObjectType, true)

#define GET_FIELD_MACRO(_1, _2, _3, _4, _5, _6, _7, _8, NAME, ...) NAME
#define make_field_type(...)                                                   \
    GET_FIELD_MACRO(                                                           \
        __VA_ARGS__, make_field_type_impl, make_field_type_impl_default)       \
    (__VA_ARGS__)

#define COMMA ,

#define FIELD_N(Dim, FIELD_TUPLE)                                              \
    make_field_type(Dim,                                                       \
        BOOST_PP_TUPLE_ENUM(BOOST_PP_TUPLE_SIZE(FIELD_TUPLE), FIELD_TUPLE))

#define FIELD_DECL(z, n, FIELD_TUPLES)                                         \
    FIELD_N(BOOST_PP_TUPLE_ELEM(0, FIELD_TUPLES),                              \
        BOOST_PP_TUPLE_ELEM(n, BOOST_PP_TUPLE_ELEM(1, FIELD_TUPLES)))

#define FIELD_NAME_N(z, n, FIELD_TUPLES)                                       \
    COMMA                                                                      \
    BOOST_PP_CAT(                                                              \
        BOOST_PP_TUPLE_ELEM(0, BOOST_PP_TUPLE_ELEM(n, FIELD_TUPLES)), _type)

#define MAKE_TUPLE_ALIAS(FIELD_TUPLES)                                         \
    using fields_tuple_t = std::tuple<BOOST_PP_CAT(                            \
        BOOST_PP_TUPLE_ELEM(0, BOOST_PP_TUPLE_ELEM(0, FIELD_TUPLES)), _type)   \
            BOOST_PP_REPEAT(                                                   \
                BOOST_PP_TUPLE_SIZE(BOOST_PP_TUPLE_POP_FRONT(FIELD_TUPLES)),   \
                FIELD_NAME_N, BOOST_PP_TUPLE_POP_FRONT(FIELD_TUPLES))>;

#define REGISTER_FIELDS(Dim, FIELD_TUPLES)                                     \
    BOOST_PP_REPEAT(                                                           \
        BOOST_PP_TUPLE_SIZE(FIELD_TUPLES), FIELD_DECL, (Dim, FIELD_TUPLES))    \
    MAKE_TUPLE_ALIAS(FIELD_TUPLES)

} //namespace domain
} // namespace iblgf

#endif
