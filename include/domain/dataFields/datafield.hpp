#ifndef INCLUDED_LGF_DOMAIN_DATAFIELD_HPP
#define INCLUDED_LGF_DOMAIN_DATAFIELD_HPP

#include <vector>
#include <iostream>

// IBLGF-specific
#include <types.hpp>
#include <domain/dataFields/datafield_utils.hpp>
#include <domain/dataFields/blockDescriptor.hpp>


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

public: //Ctors:

    DataField()=default;

    ~DataField() =default;
    DataField(const DataField& rhs)=delete;
	DataField& operator=(const DataField&) & = delete ;

    DataField(DataField&& rhs)=default;
	DataField& operator=(DataField&&) & = default;

    DataField(size_type _size, DataType _initValue=0)
    {
        data_.resize(_size);
        std::fill(data_.begin(), data_.end(),_initValue);
    }

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
        real_block_.base(_b.base()-lowBuffer_);
        real_block_.extent(_b.extent()+lowBuffer_+highBuffer_);
        real_block_.level()=_b.level();

        this->base(_b.base());
        this->extent(_b.extent());
        this->level()= _b.level();
        auto size=real_block_.size();
        data_.resize(size);
    }

    DataType& operator[](size_type i ) noexcept {return data_[i];}
    const DataType& operator[](size_type i )const noexcept {return data_[i];}

    auto begin()const noexcept {return data_.begin();}
    auto end()const noexcept{return data_.end();}

    auto& data(){return data_;}

    auto size()const noexcept{return data_.size();}

    //Get ijk-data
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
    
  
    buffer_d_t lbuffer()const noexcept{return lowBuffer_;}
    buffer_d_t hbuffer()const noexcept{return highBuffer_;}

    const block_type& real_block()const noexcept{return real_block_;}
    block_type& real_block()noexcept{return real_block_;}




protected: //protected memeber:

    std::vector<DataType> data_;          ///< actual data
    block_type real_block_;               ///< Block descriptorinlcuding buffer
    buffer_d_t lowBuffer_ =buffer_d_t(0); ///< Buffer in negative direction
    buffer_d_t highBuffer_=buffer_d_t(0); ///< Buffer in positive direction
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
