#ifndef INCLUDED_LGF_DOMAIN_DATAFIELD_HPP
#define INCLUDED_LGF_DOMAIN_DATAFIELD_HPP

#include <vector>
#include <iostream>

// IBLGF-specific
#include <types.hpp>
#include <linalg/linalg.hpp>
#include <domain/dataFields/datafield_utils.hpp>
#include <domain/dataFields/blockDescriptor.hpp>
#include <domain/dataFields/array_ref.hpp>
#include <domain/dataFields/view.hpp>


namespace domain
{

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
    static constexpr int dimension(){return Dim;}

    using block_type = BlockDescriptor<int,Dim>;
    using coordinate_t = typename block_type::base_t;
    using view_type =View<DataField, Dim>;

public: //Ctors:

    DataField()=default;

    ~DataField() =default;
    DataField(const DataField& rhs)=delete;
	DataField& operator=(const DataField&) & = delete ;

    DataField(DataField&& rhs)=default;
	DataField& operator=(DataField&&) & = default;

    //DataField(size_type _size, DataType _initValue=0)
    //{
    //    data_.resize(_size);
    //    std::fill(data_.begin(), data_.end(),_initValue);
    //}

    DataField(const buffer_d_t& _lBuffer, const buffer_d_t& _hBuffer)
    : lowBuffer_(_lBuffer), highBuffer_(_hBuffer)
    {}
    DataField(const int _lBuffer, const int _hBuffer)
    : lowBuffer_(_lBuffer), highBuffer_(_hBuffer)
    {}


public: //member functions

    /** @brief Initialize the datafield and grow according to buffer
     *
     *  @param[in] _b Blockdescriptor
     */
    void initialize(block_type _b)
    {
        this->real_block_.base(_b.base()-lowBuffer_);
        this->real_block_.extent(_b.extent()+lowBuffer_+highBuffer_);
        this->real_block_.level()=_b.level();

        this->base(_b.base());
        this->extent(_b.extent());
        this->level()= _b.level();
        data_.resize(real_block_.nPoints());

        auto ext = real_block_.extent();
        cube_ = std::unique_ptr<linalg::Cube_t> (new linalg::Cube_t( (types::float_type*) &data_[0], ext[0], ext[1], ext[2]));
    }

    auto& operator[](size_type i ) noexcept {return data_[i];}
    const auto& operator[](size_type i )const noexcept {return data_[i];}

    auto begin()const noexcept {return data_.begin();}
    auto end()const noexcept{return data_.end();}

    auto& data(){return data_;}
    auto& linalg_data(){return cube_->data_;}

    auto size()const noexcept{return data_.size();}

    //Get ijk-data
    inline DataType* get_ptr(const coordinate_t& _c) noexcept
    {
        return &data_[real_block_.globalCoordinate_to_index(_c[0],
                                                           _c[1],
                                                           _c[2])];
    }
    inline const DataType* get_ptr(const coordinate_t& _c) const noexcept
    {
        return &data_[real_block_.globalCoordinate_to_index(_c[0],
                                                           _c[1],
                                                           _c[2])];
    }
    inline DataType& get(const coordinate_t& _c) noexcept
    {
        return data_[real_block_.globalCoordinate_to_index(_c[0],
                                                           _c[1],
                                                           _c[2])];
    }
    inline const DataType& get(const coordinate_t& _c) const noexcept
    {
        return data_[real_block_.globalCoordinate_to_index(_c[0],
                                                           _c[1],
                                                           _c[2])];
    }
    inline const DataType& get(int _i, int _j, int _k) const noexcept
    {
        return data_[real_block_.globalCoordinate_to_index(_i,
                                                           _j,
                                                           _k)];
    }
    inline DataType& get(int _i, int _j, int _k) noexcept
    {
        return data_[real_block_.globalCoordinate_to_index(_i,
                                                           _j,
                                                           _k)];
    }

    inline const DataType&
    get_real_local(int _i, int _j, int _k) const noexcept
    {
        return data_[real_block_.localCoordinate_to_index(_i,_j,_k)];
    }
    inline DataType& get_real_local(int _i, int _j, int _k) noexcept
    {
        return data_[real_block_.localCoordinate_to_index(_i,_j,_k)];
    }

    inline const DataType&
    get_local(int _i, int _j, int _k) const noexcept
    {
        return data_[
            real_block_.localCoordinate_to_index(_i+lowBuffer_[0],
                                                 _j+lowBuffer_[1],
                                                 _k+lowBuffer_[2])
        ];
    }

    inline DataType& get_local(int _i, int _j, int _k) noexcept
    {
        return data_[
            real_block_.localCoordinate_to_index(_i+lowBuffer_[0],
                                                 _j+lowBuffer_[1],
                                                 _k+lowBuffer_[2])
        ];
    }

    template<class BlockType, class OverlapType>
    bool buffer_overlap(const BlockType& other,
                 OverlapType& overlap, int level ) const noexcept
    {
        return real_block_.overlap(other, overlap, level);
    }


    buffer_d_t& lbuffer()noexcept{return lowBuffer_;}
    const buffer_d_t& lbuffer()const noexcept{return lowBuffer_;}
    buffer_d_t& hbuffer()noexcept{return highBuffer_;}
    const buffer_d_t& hbuffer()const noexcept{return highBuffer_;}

    const block_type& real_block()const noexcept{return real_block_;}
    block_type& real_block()noexcept{return real_block_;}

    /** @brief Get a (sub-)view of the datafield
     */
    auto view(const block_type& _b,
              coordinate_t _stride=coordinate_t(1)) noexcept
    {
        return view_type(this,_b, _stride);
    }
    auto domain_view() noexcept
    {
        return view(*this);
    }
    auto real_domain_view()noexcept
    {
        return view(real_block_);
    }


protected: //protected memeber:

    std::vector<DataType> data_;          ///< actual data
    std::unique_ptr<linalg::Cube_t> cube_;///< Linear algebra wrapper for the actual data >
    buffer_d_t lowBuffer_ =buffer_d_t(0); ///< Buffer in negative direction
    buffer_d_t highBuffer_=buffer_d_t(0); ///< Buffer in positive direction
    block_type real_block_;               ///< Block descriptorinlcuding buffer

};


#define STRINGIFY(X) #X

#define make_field_type_nb(key, DataType)                                   \
template<std::size_t _Dim>                                                  \
class key : public DataField<DataType,_Dim>                                 \
{                                                                           \
    public:                                                                 \
    using data_field_t=DataField<DataType,_Dim>;                            \
    static constexpr const char* name_= STRINGIFY(key);                     \
    key (): data_field_t()                                                  \
    {                                                                       \
    }                                                                       \
  static auto  name()noexcept{return key::name_;}                           \
                                                                            \
};                                                                          \




#define make_field_type_b(key, DataType, lBuffer, hBuffer)                  \
template<std::size_t _Dim>                                                  \
class key : public DataField<DataType,_Dim>                                 \
{                                                                           \
    public:                                                                 \
    using data_field_t=DataField<DataType,_Dim>;                            \
    static constexpr const char* name_= STRINGIFY(key);                     \
    key (): data_field_t(lBuffer, hBuffer)                                  \
    {                                                                       \
    }                                                                       \
  static auto  name()noexcept{return key::name_;}                           \
                                                                            \
};                                                                          \


//Dummy marco for three params
#define FOO3(DataType, lBuffer, hBuffer) bla

#define GET_FIELD_MACRO(_1,_2,_3,_4,NAME,...) NAME
#define make_field_type(...)                                                \
GET_FIELD_MACRO(__VA_ARGS__,                                                \
                make_field_type_b,                                          \
                FOO3,                                                       \
                make_field_type_nb)                                         \
                (__VA_ARGS__)                                               \


} //namespace domain

#endif
