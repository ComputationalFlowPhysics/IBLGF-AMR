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
#include <iostream>
#include <algorithm>

// IBLGF-specific
#include <iblgf/types.hpp>
#include <iblgf/linalg/linalg.hpp>
#include <iblgf/domain/dataFields/datafield_utils.hpp>
#include <iblgf/domain/dataFields/blockDescriptor.hpp>
#include <iblgf/domain/dataFields/array_ref.hpp>
#include <iblgf/domain/dataFields/view.hpp>

#include <boost/preprocessor/tuple/size.hpp>
#include <boost/preprocessor/tuple/elem.hpp>
#include <boost/preprocessor/repetition/enum.hpp>
#include <boost/preprocessor/tuple/enum.hpp>
#include <boost/preprocessor/repetition/repeat.hpp>
#include <boost/preprocessor/tuple/pop_front.hpp>
#include <tuple>

namespace domain
{
enum class MeshObject : int
{
    cell,
    face,
    edge,
    vertex
};

template<class DataType, std::size_t Dim>
class DataField : public BlockDescriptor<int, Dim>
{
  public: //member types
    using size_type = types::size_type;
    using data_type = DataType;

    template<typename T>
    using vector_type = types::vector_type<T, Dim>;
    using buffer_d_t = vector_type<int>;
    using super_type = BlockDescriptor<int, Dim>;

    using block_type = BlockDescriptor<int, Dim>;
    using coordinate_t = typename block_type::base_t;
    using view_type = View<DataField, Dim>;

  public: //Ctors:
    DataField() = default;

    ~DataField() = default;
    DataField(const DataField& rhs) = delete;
    DataField& operator=(const DataField&) & = delete;

    DataField(DataField&& rhs) = default;
    DataField& operator=(DataField&&) & = default;

    DataField(const buffer_d_t& _lBuffer, const buffer_d_t& _hBuffer)
    : lowBuffer_(_lBuffer)
    , highBuffer_(_hBuffer)
    {
    }
    DataField(const int _lBuffer, const int _hBuffer)
    : lowBuffer_(_lBuffer)
    , highBuffer_(_hBuffer)
    {
    }

  public: //member functions
    /** @brief Initialize the datafield and grow according to buffer
     *
     *  @param[in] _b Blockdescriptor
     */
    void initialize(block_type _b, bool _allocate = true, bool _default = false,
        DataType _dval = DataType())
    {
        this->real_block_.base(_b.base() - lowBuffer_);
        this->real_block_.extent(_b.extent() + lowBuffer_ + highBuffer_);
        this->real_block_.level() = _b.level();

        this->base(_b.base());
        this->extent(_b.extent());
        this->level() = _b.level();
        if (_allocate) data_.resize(real_block_.nPoints());
        if (_default) { std::fill(data_.begin(), data_.end(), _dval); }

        auto ext = real_block_.extent();
        cube_ = std::make_unique<linalg::Cube_t>(
            (types::float_type*)&data_[0], ext[0], ext[1], ext[2]);
    }

    auto&       operator[](size_type i) noexcept { return data_[i]; }
    const auto& operator[](size_type i) const noexcept { return data_[i]; }

    auto begin() const noexcept { return data_.begin(); }
    auto end() const noexcept { return data_.end(); }

    auto& data() { return data_; }
    auto  data_ptr() { return &data_; }

    auto& linalg_data() { return cube_->data_; }
    auto& linalg() { return cube_; }

    auto size() const noexcept { return data_.size(); }

    inline DataType* get_ptr(const coordinate_t& _c) noexcept
    {
        return &data_[real_block_.index(_c)];
    }
    inline const DataType* get_ptr(const coordinate_t& _c) const noexcept
    {
        return &data_[real_block_.index(_c)];
    }
    inline DataType& get(const coordinate_t& _c) noexcept
    {
        return data_[real_block_.index(_c)];
    }
    inline const DataType& get(const coordinate_t& _c) const noexcept
    {
        return data_[real_block_.index(_c)];
    }

    //inline const DataType&
    //get_local(const coordinate_t& _c) const noexcept
    //{
    //    return data_[ real_block_.index_zeroBase(_c+lowBuffer_)];
    //}
    //inline DataType&
    //get_local(const coordinate_t& _c) noexcept
    //{
    //    return data_[ real_block_.index_zeroBase(_c+lowBuffer_)];
    //}
    inline const DataType& get_real_local(const coordinate_t& _c) const noexcept
    {
        return data_[real_block_.index_zeroBase(_c)];
    }
    inline DataType& get_real_local(const coordinate_t& _c) noexcept
    {
        return data_[real_block_.index_zeroBase(_c)];
    }

    //IJK access
    inline const DataType& get(int _i, int _j, int _k) const noexcept
    {
        return data_[real_block_.index(_i, _j, _k)];
    }
    inline DataType& get(int _i, int _j, int _k) noexcept
    {
        return data_[real_block_.index(_i, _j, _k)];
    }
    inline const DataType& get_real_local(int _i, int _j, int _k) const noexcept
    {
        return data_[real_block_.index_zeroBase(_i, _j, _k)];
    }
    inline DataType& get_real_local(int _i, int _j, int _k) noexcept
    {
        return data_[real_block_.index_zeroBase(_i, _j, _k)];
    }

    //inline const DataType&
    //get_local(int _i, int _j, int _k) const noexcept
    //{
    //    return data_[real_block_.index_zeroBase(_i+lowBuffer_[0],
    //                                            _j+lowBuffer_[1],
    //                                            _k+lowBuffer_[2])];
    //}
    //inline DataType& get_local(int _i, int _j, int _k) noexcept
    //{
    //    return data_[real_block_.index_zeroBase(_i+lowBuffer_[0],
    //                                            _j+lowBuffer_[1],
    //                                            _k+lowBuffer_[2])];
    //}
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

    /** @brief Get a (sub-)view of the datafield */
    auto view(
        const block_type& _b, coordinate_t _stride = coordinate_t(1)) noexcept
    {
        return view_type(this, _b, _stride);
    }
    auto domain_view() noexcept { return view(*this); }
    auto real_domain_view() noexcept { return view(real_block_); }

  protected:                     //protected memeber:
    std::vector<DataType> data_; ///< actual data
    std::unique_ptr<linalg::Cube_t>
               cube_; ///< Linear algebra wrapper for the actual data >
    buffer_d_t lowBuffer_ = buffer_d_t(0);  ///< Buffer in negative direction
    buffer_d_t highBuffer_ = buffer_d_t(0); ///< Buffer in positive direction
    block_type real_block_; ///< Block descriptorinlcuding buffer
};

#define STRINGIFY(X) #X

#define make_field_type_impl(                                                  \
    Dim, key, DataType, NFields, lBuffer, hBuffer, MeshObjectType, _output)    \
    class key                                                                  \
    {                                                                          \
      public:                                                                  \
        using data_field_t = DataField<DataType, Dim>;                         \
        using view_type = typename data_field_t::view_type;                    \
        static constexpr const char* name_ = STRINGIFY(key);                   \
        static constexpr MeshObject  mesh_type = MeshObject::MeshObjectType;   \
        static constexpr std::size_t nFields = NFields;                        \
        static constexpr bool        output = _output;                         \
        key()                                                                  \
        {                                                                      \
            for (std::size_t i = 0; i < nFields; ++i)                          \
            { fields_[i] = data_field_t(lBuffer, hBuffer); }                   \
        }                                                                      \
        static auto name() noexcept { return key::name_; }                     \
        auto&       operator[](size_type i) noexcept { return fields_[i]; }    \
        const auto& operator[](size_type i) const noexcept                     \
        {                                                                      \
            return fields_[i];                                                 \
        }                                                                      \
        std::array<data_field_t, nFields> fields_;                             \
    };

//For 7 parameters with defaulted output
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
    BOOST_PP_TUPLE_ELEM(0, BOOST_PP_TUPLE_ELEM(n, FIELD_TUPLES))

#define MAKE_TUPLE_ALIAS(FIELD_TUPLES)                                         \
    using fields_tuple_t = std::tuple<BOOST_PP_TUPLE_ELEM(                     \
        0, BOOST_PP_TUPLE_ELEM(0, FIELD_TUPLES))                               \
            BOOST_PP_REPEAT(                                                   \
                BOOST_PP_TUPLE_SIZE(BOOST_PP_TUPLE_POP_FRONT(FIELD_TUPLES)),   \
                FIELD_NAME_N, BOOST_PP_TUPLE_POP_FRONT(FIELD_TUPLES))>;

#define MAKE_PARAM_CLASS(NAME, FIELD_TUPLES)                                   \
    struct NAME                                                                \
    {                                                                          \
        MAKE_TUPLE_ALIAS(FIELD_TUPLES)                                         \
    };

#define REGISTER_FIELDS(Dim, FIELD_TUPLES)                                     \
    BOOST_PP_REPEAT(                                                           \
        BOOST_PP_TUPLE_SIZE(FIELD_TUPLES), FIELD_DECL, (Dim, FIELD_TUPLES))    \
    MAKE_TUPLE_ALIAS(FIELD_TUPLES)

} //namespace domain

#endif
