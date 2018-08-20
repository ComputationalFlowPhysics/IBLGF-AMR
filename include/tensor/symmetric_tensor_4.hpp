#ifndef MATH_INCLUDED_TENSOR_SYMMETRIC_TENSOR_4_HPP
#define MATH_INCLUDED_TENSOR_SYMMETRIC_TENSOR_4_HPP

#include <tensor/tensor_base.hpp>

namespace math {

// symmetric rank-4 tensor
// data stored in lower part
// column major format (fortran)
template<typename T, std::size_t N>
class symmetric_tensor<T,4,N> : public detail::symmetric_tensor_access<symmetric_tensor<T,4,N>,T,4,N>
{
public: // static member functions

	static constexpr std::size_t size() { return detail::s4(N,N); }
	
public: // member types

	using data_type = std::array<T,size()>;
	using value_type = typename data_type::value_type;
	using size_type = typename data_type::size_type;
	using difference_type = typename data_type::difference_type;
	using reference = typename data_type::reference;
	using const_reference = typename data_type::const_reference;
	using pointer = typename data_type::pointer;
	using const_pointer = typename data_type::const_pointer;
	using iterator = typename data_type::iterator;
	using const_iterator = typename data_type::const_iterator;
	using reverse_iterator = typename data_type::reverse_iterator;
	using const_reverse_iterator = typename data_type::const_reverse_iterator;

public: // ctors

	symmetric_tensor() = default;
	symmetric_tensor(const T& element) noexcept { data.fill(element); }
	symmetric_tensor(const symmetric_tensor&) = default;
	symmetric_tensor(symmetric_tensor&&) = default;
	template<typename T2>
	symmetric_tensor(const symmetric_tensor<T2,4,N>& other) { for(unsigned int i=0; i<size(); ++i) data[i] = static_cast<T>(other.data[i]); }
	symmetric_tensor(const std::array<T,size()>& _data) noexcept : data(_data) {}
	symmetric_tensor(std::array<T,size()>&& _data) noexcept : data(std::move(_data)) {}
	
	symmetric_tensor& operator=(const symmetric_tensor&) & = default;
	symmetric_tensor& operator=(symmetric_tensor&&) & = default;
	symmetric_tensor& operator=(const T& element) & { data.fill(element); }
	
public: // member functions

	reference operator[](size_type pos) { return data[pos]; }
	const_reference operator[](size_type pos) const { return data[pos]; }
	
	reference operator()(size_type i, size_type j, size_type k, size_type l) 
	{
		if (i < j) std::swap(i,j);
		if (k < l) std::swap(k,l);
		if (j < k) std::swap(j,k);
		if (i < j) std::swap(i,j);
		if (k < l) std::swap(k,l);
		if (j < k) std::swap(j,k);
		return data[detail::s4(l,N)+detail::s3(k-l,N-l)+detail::s2(j-k,N-k)+i-j];
	}
	
	const_reference operator()(size_type i, size_type j, size_type k, size_type l) const
	{
		if (i < j) std::swap(i,j);
		if (k < l) std::swap(k,l);
		if (j < k) std::swap(j,k);
		if (i < j) std::swap(i,j);
		if (k < l) std::swap(k,l);
		if (j < k) std::swap(j,k);
		return data[detail::s4(l,N)+detail::s3(k-l,N-l)+detail::s2(j-k,N-k)+i-j];
	}
	
	reference front() { return data.front(); }
	const_reference front() const { return data.front(); }
	reference back() { return data.back(); }
	const_reference back() const { return data.back(); }

	friend std::ostream& operator<<(std::ostream& os, const symmetric_tensor& t)
	{
		os << "(";
		for (unsigned int l=0; l<N; ++l)
		{
			os << "(";
			for (unsigned int k=0; k<N; ++k)
			{
				os << "(";
				for (unsigned int j=0; j<N; ++j)
				{
					os << "(";
					for (unsigned int i=0; i<N; ++i)
					{
						os << t(i,j,k,l);
						if (i<N-1) os << ", ";
					}
					os << ")";
					if (j<N-1) os << ", ";
				}
				os << ")";
				if (k<N-1) os << ", ";
			}
			os << ")";
			if (l<N-1) os << ", ";
		}
		os << ")";
		return os;
	}
public: // iterators

	iterator begin() { return data.begin(); }
	const_iterator begin() const { return data.begin(); }
	const_iterator cbegin() const { return data.cbegin(); }
	iterator end() { return data.end(); }
	const_iterator end() const { return data.end(); }
	const_iterator cend() const { return data.cend(); }
	reverse_iterator rbegin() { return data.rbegin(); }
	const_reverse_iterator rbegin() const { return data.rbegin(); }
	const_reverse_iterator crbegin() const { return data.crbegin(); }
	reverse_iterator rend() { return data.rend(); }
	const_reverse_iterator rend() const { return data.rend(); }
	const_reverse_iterator crend() const { return data.crend(); }
	
public: // arithmetic member functions

	symmetric_tensor& operator+=(const symmetric_tensor& other) { for (unsigned int i=0; i<size(); ++i) data[i]+=other[i]; return *this; }
	symmetric_tensor& operator+=(const T& element) { for (unsigned int i=0; i<size(); ++i) data[i]+=element; return *this; }
	symmetric_tensor& operator-=(const symmetric_tensor& other) { for (unsigned int i=0; i<size(); ++i) data[i]-=other[i]; return *this; }
	symmetric_tensor& operator-=(const T& element) { for (unsigned int i=0; i<size(); ++i) data[i]-=element; return *this; }
	symmetric_tensor& operator*=(const symmetric_tensor& other) { for (unsigned int i=0; i<size(); ++i) data[i]*=other[i]; return *this; }
	symmetric_tensor& operator*=(const T& element) { for (unsigned int i=0; i<size(); ++i) data[i]*=element; return *this; }
	symmetric_tensor& operator/=(const symmetric_tensor& other) { for (unsigned int i=0; i<size(); ++i) data[i]/=other[i]; return *this; }
	symmetric_tensor& operator/=(const T& element) { for (unsigned int i=0; i<size(); ++i) data[i]/=element; return *this; }

public: // members

	data_type data;

private: // serialization

	friend class boost::serialization::access;

	template<class Archive>
	void serialize(Archive & ar, const unsigned int version) const
	{
		ar & data;
	}
};

template< class T, std::size_t N >
bool operator==( const math::symmetric_tensor<T,4,N>& lhs, const math::symmetric_tensor<T,4,N>& rhs ) { return lhs.data == rhs.data; }

template< class T, std::size_t N >
bool operator!=( const math::symmetric_tensor<T,4,N>& lhs, const math::symmetric_tensor<T,4,N>& rhs ) { return lhs.data != rhs.data; }

template< class T, std::size_t N >
bool operator<( const math::symmetric_tensor<T,4,N>& lhs, const math::symmetric_tensor<T,4,N>& rhs ) { return lhs.data < rhs.data; }

template< class T, std::size_t N >
bool operator<=( const math::symmetric_tensor<T,4,N>& lhs, const math::symmetric_tensor<T,4,N>& rhs ) { return lhs.data <= rhs.data; }

template< class T, std::size_t N >
bool operator>( const math::symmetric_tensor<T,4,N>& lhs, const math::symmetric_tensor<T,4,N>& rhs ) { return lhs.data > rhs.data; }

template< class T, std::size_t N >
bool operator>=( const math::symmetric_tensor<T,4,N>& lhs, const math::symmetric_tensor<T,4,N>& rhs ) { return lhs.data >= rhs.data; }

template<typename T, std::size_t N>
math::symmetric_tensor<T,4,N> operator-(math::symmetric_tensor<T,4,N> t) 
{
	for (std::size_t i=0; i<math::symmetric_tensor<T,4,N>::size(); ++i) t.data[i] = -t.data[i]; 
	return std::move(t);
}

template<typename U, typename V, std::size_t N>
typename std::common_type<U,V>::type contract(const math::symmetric_tensor<U,4,N>& lhs, const math::symmetric_tensor<V,4,N>& rhs) noexcept
{
	typename std::common_type<U,V>::type res(lhs(0,0,0,0)*rhs(0,0,0,0));
	for (std::size_t l=0; l<N; ++l)
		for (std::size_t k=0; k<N; ++k)
			for (std::size_t j=0; j<N; ++j)
				for (std::size_t i=(l>0?0:1); i<N; ++i)
					res += lhs(i,j,k,l)*rhs(i,j,k,l);
	return res;
}

template<typename U, typename V>
typename std::common_type<U,V>::type contract(const math::symmetric_tensor<U,4,2>& lhs, const math::symmetric_tensor<V,4,2>& rhs) noexcept
{
	return lhs.xxxx()*rhs.xxxx() + lhs.yyyy()*rhs.yyyy() + 4*lhs.xxxy()*rhs.xxxy() + 4*lhs.xyyy()*rhs.xyyy() + 6*lhs.xxyy()*rhs.xxyy();
}

template<typename U, typename V>
typename std::common_type<U,V>::type contract(const math::symmetric_tensor<U,4,3>& lhs, const math::symmetric_tensor<V,4,3>& rhs) noexcept
{
	return lhs.xxxx()*rhs.xxxx() + lhs.yyyy()*rhs.yyyy() + lhs.zzzz()*rhs.zzzz()
	+  4*lhs.xxxy()*rhs.xxxy() +  4*lhs.xxxz()*rhs.xxxz()
	+  4*lhs.xyyy()*rhs.xyyy() +  4*lhs.yyyz()*rhs.yyyz()
	+  4*lhs.xzzz()*rhs.xzzz() +  4*lhs.yzzz()*rhs.yzzz()
	+  6*lhs.xxyy()*rhs.xxyy() +  6*lhs.xxzz()*rhs.xxzz() +  6*lhs.yyzz()*rhs.yyzz()
	+ 12*lhs.xxyz()*rhs.xxyz() + 12*lhs.xyyz()*rhs.xyyz() + 12*lhs.xyzz()*rhs.xyzz();
}


template<typename T, std::size_t N>
T trace(const math::symmetric_tensor<T,4,N>& a) noexcept
{
    T res(a(0,0));
    for (std::size_t i=1; i<N; ++i) res += a(i,i);
    return res;
}

template<typename T>
T trace(const math::symmetric_tensor<T,4,2>& a) noexcept
{
    return a.xxxx()+a.yyyy();
}

template<typename T>
T trace(const math::symmetric_tensor<T,4,3>& a) noexcept
{
    return a.xxxx()+a.yyyy()+a.zzzz();
}

template<typename T, std::size_t N>
math::symmetric_tensor<T,4,N> transpose(math::symmetric_tensor<T,4,N> other) noexcept { return std::move(other); }


template<typename U, typename V, std::size_t N>
math::symmetric_tensor<typename std::common_type<U,V>::type,4,N> operator+(const math::symmetric_tensor<U,4,N>& lhs, const math::symmetric_tensor<V,4,N>& rhs)
{ 
	math::symmetric_tensor<typename std::common_type<U,V>::type,4,N> res(lhs);
	for (std::size_t i=0; i<math::symmetric_tensor<U,4,N>::size(); ++i) res[i] += rhs[i]; 
	return std::move(res);
}

template<typename U, typename V, std::size_t N>
math::symmetric_tensor<typename std::common_type<U,V>::type,4,N> operator+(const math::symmetric_tensor<U,4,N>& lhs, const V& rhs) 
{
	math::symmetric_tensor<typename std::common_type<U,V>::type,4,N> res(lhs);
	for (std::size_t i=0; i<math::symmetric_tensor<U,4,N>::size(); ++i) res[i] += rhs; 
	return std::move(res);
}

template<typename U, typename V, std::size_t N>
math::symmetric_tensor<typename std::common_type<U,V>::type,4,N> operator+(const U& lhs, const math::symmetric_tensor<V,4,N>& rhs) 
{
	math::symmetric_tensor<typename std::common_type<U,V>::type,4,N> res(rhs);
	for (std::size_t i=0; i<math::symmetric_tensor<U,4,N>::size(); ++i) res[i] += lhs; 
	return std::move(res);
}

template<typename U, typename V, std::size_t N>
math::symmetric_tensor<typename std::common_type<U,V>::type,4,N> operator-(const math::symmetric_tensor<U,4,N>& lhs, const math::symmetric_tensor<V,4,N>& rhs)
{ 
	math::symmetric_tensor<typename std::common_type<U,V>::type,4,N> res(lhs);
	for (std::size_t i=0; i<math::symmetric_tensor<U,4,N>::size(); ++i) res[i] -= rhs[i]; 
	return std::move(res);
}

template<typename U, typename V, std::size_t N>
math::symmetric_tensor<typename std::common_type<U,V>::type,4,N> operator-(const math::symmetric_tensor<U,4,N>& lhs, const V& rhs) 
{
	math::symmetric_tensor<typename std::common_type<U,V>::type,4,N> res(lhs);
	for (std::size_t i=0; i<math::symmetric_tensor<U,4,N>::size(); ++i) res[i] -= rhs; 
	return std::move(res);
}

template<typename U, typename V, std::size_t N>
math::symmetric_tensor<typename std::common_type<U,V>::type,4,N> operator-(const U& lhs, const math::symmetric_tensor<V,4,N>& rhs) 
{
	math::symmetric_tensor<typename std::common_type<U,V>::type,4,N> res(-rhs);
	for (std::size_t i=0; i<math::symmetric_tensor<U,4,N>::size(); ++i) res[i] += lhs; 
	return std::move(res);
}

template<typename U, typename V, std::size_t N>
math::symmetric_tensor<typename std::common_type<U,V>::type,4,N> operator*(const math::symmetric_tensor<U,4,N>& lhs, const math::symmetric_tensor<V,4,N>& rhs)
{ 
	math::symmetric_tensor<typename std::common_type<U,V>::type,4,N> res(lhs);
	for (std::size_t i=0; i<math::symmetric_tensor<U,4,N>::size(); ++i) res[i] *= rhs[i]; 
	return std::move(res);
}

template<typename U, typename V, std::size_t N>
math::symmetric_tensor<typename std::common_type<U,V>::type,4,N> operator*(const math::symmetric_tensor<U,4,N>& lhs, const V& rhs) 
{
	math::symmetric_tensor<typename std::common_type<U,V>::type,4,N> res(lhs);
	for (std::size_t i=0; i<math::symmetric_tensor<U,4,N>::size(); ++i) res[i] *= rhs; 
	return std::move(res);
}

template<typename U, typename V, std::size_t N>
math::symmetric_tensor<typename std::common_type<U,V>::type,4,N> operator*(const U& lhs, const math::symmetric_tensor<V,4,N>& rhs) 
{
	math::symmetric_tensor<typename std::common_type<U,V>::type,4,N> res(rhs);
	for (std::size_t i=0; i<math::symmetric_tensor<U,4,N>::size(); ++i) res[i] *= lhs; 
	return std::move(res);
}

template<typename U, typename V, std::size_t N>
math::symmetric_tensor<typename std::common_type<U,V>::type,4,N> operator/(const math::symmetric_tensor<U,4,N>& lhs, const math::symmetric_tensor<V,4,N>& rhs)
{ 
	math::symmetric_tensor<typename std::common_type<U,V>::type,4,N> res(lhs);
	for (std::size_t i=0; i<math::symmetric_tensor<U,4,N>::size(); ++i) res[i] /= rhs[i]; 
	return std::move(res);
}

template<typename U, typename V, std::size_t N>
math::symmetric_tensor<typename std::common_type<U,V>::type,4,N> operator/(const math::symmetric_tensor<U,4,N>& lhs, const V& rhs) 
{
	math::symmetric_tensor<typename std::common_type<U,V>::type,4,N> res(lhs);
	for (std::size_t i=0; i<math::symmetric_tensor<U,4,N>::size(); ++i) res[i] /= rhs; 
	return std::move(res);
}

template<typename U, typename V, std::size_t N>
math::symmetric_tensor<typename std::common_type<U,V>::type,4,N> operator/(const U& lhs, const math::symmetric_tensor<V,4,N>& rhs) 
{
	math::symmetric_tensor<typename std::common_type<U,V>::type,4,N> res(rhs);
	for (std::size_t i=0; i<math::symmetric_tensor<U,4,N>::size(); ++i) res[i] = lhs / res[i]; 
	return std::move(res);
}

} // namespace lb

#endif // LB_INCLUDED_TENSOR_SYMMETRIC_TENSOR_4_HPP
