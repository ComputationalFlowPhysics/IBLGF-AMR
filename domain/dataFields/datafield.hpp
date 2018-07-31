#ifndef INCLUDED_LGF_DOMAIN_DATAFIELD_HPP
#define INCLUDED_LGF_DOMAIN_DATAFIELD_HPP

#include <vector>
#include <iostream>

// IBLGF-specific
#include <types.hpp>
#include <domain/dataFields/datafield_utils.hpp>


namespace domain
{

template<class DataType, std::size_t Dim>
class DataField 
{

public: //member types

    using size_type = types::size_type;
    using data_type = DataType;

    template<typename T>
    using vector_type = types::vector_type<T, Dim>;
    static constexpr int dimension(){return Dim;}

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


public: //member functions

    void resize(size_type _size)    
    {
        data_.resize(_size);
    }

    DataType& operator[](size_type i ) noexcept {return data_[i];}
    const DataType& operator[](size_type i )const noexcept {return data_[i];}

    auto begin()const noexcept {return data_.begin();}
    auto end()const noexcept{return data_.end();}

    auto& data(){return data_;}

    auto size()const noexcept{return data_.size();}

protected: //protected memeber:

    std::vector<DataType> data_; 
};


#define STRINGIFY(X) #X

#define make_field_type(key, DataType)                                      \
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
                                                                            \
                                                                            \


} //namespace domain

#endif 
