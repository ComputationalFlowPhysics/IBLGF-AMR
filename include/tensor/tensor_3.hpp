#ifndef MATH_INCLUDED_TENSOR_TENSOR_3_HPP
#define MATH_INCLUDED_TENSOR_TENSOR_3_HPP

#include <tensor/symmetric_tensor_3.hpp>

namespace math {

// rank-3 tensor
// column major format (fortran)
template<typename T, std::size_t Nx, std::size_t Ny, std::size_t Nz>
class tensor<T,Nx,Ny,Nz>
{
public: // static member functions

	static constexpr std::size_t size() { return Nx*Ny*Nz; }	

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

	tensor() = default;
	tensor(const T& element) noexcept { data.fill(element); }
	tensor(const tensor&) = default;
	tensor(tensor&&) = default;
	template<typename T2>
	tensor(const tensor<T2,Nx,Ny,Nz>& other) { for (unsigned int i=0; i<size(); ++i) data[i] = static_cast<T>(other.data[i]); }
	tensor(const std::array<T,size()>& _data) noexcept : data(_data) {}
	tensor(std::array<T,size()>&& _data) noexcept : data(std::move(_data)) {}
	
	tensor& operator=(const tensor&) & = default;
	tensor& operator=(tensor&&) & = default;
	tensor& operator=(const T& element) & { data.fill(element); }
	
public: // member functions

	reference operator[](size_type pos) { return data[pos]; }
	const_reference operator[](size_type pos) const { return data[pos]; }
	
	reference operator()(size_type i, size_type j, size_type k) 
	{
		return data[k*Nx*Ny+j*Nx+i];
	}
	
	const_reference operator()(size_type i, size_type j, size_type k) const
	{
		return data[k*Nx*Ny+j*Nx+i];
	}
	
	reference front() { return data.front(); }
	const_reference front() const { return data.front(); }
	reference back() { return data.back(); }
	const_reference back() const { return data.back(); }
	
	friend std::ostream& operator<<(std::ostream& os, const tensor& t)
	{
		os << "(";
		for (unsigned int k=0; k<Nz; ++k)
		{
			os << "(";
			for (unsigned int j=0; j<Ny; ++j)
			{
				os << "(";
				for (unsigned int i=0; i<Nx; ++i)
				{
					os << t(i,j,k);
					if (i<Nx-1) os << ", ";
				}
				os << ")";
				if (j<Ny-1) os << ", ";
			}
			os << ")";
			if (k<Nz-1) os << ", ";
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

	tensor& operator+=(const tensor& other) { for (unsigned int i=0; i<size(); ++i) data[i]+=other[i]; return *this; }
	tensor& operator+=(const T& element) { for (unsigned int i=0; i<size(); ++i) data[i]+=element; return *this; }
	tensor& operator-=(const tensor& other) { for (unsigned int i=0; i<size(); ++i) data[i]-=other[i]; return *this; }
	tensor& operator-=(const T& element) { for (unsigned int i=0; i<size(); ++i) data[i]-=element; return *this; }
	tensor& operator*=(const tensor& other) { for (unsigned int i=0; i<size(); ++i) data[i]*=other[i]; return *this; }
	tensor& operator*=(const T& element) { for (unsigned int i=0; i<size(); ++i) data[i]*=element; return *this; }
	tensor& operator/=(const tensor& other) { for (unsigned int i=0; i<size(); ++i) data[i]/=other[i]; return *this; }
	tensor& operator/=(const T& element) { for (unsigned int i=0; i<size(); ++i) data[i]/=element; return *this; }

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


// quadratic rank-3 tensor
template<typename T, std::size_t N>
class tensor<T,N,N,N> : public detail::tensor_access<tensor<T,N,N,N>,T,N,N,N>
{
public: // static member functions

	static constexpr std::size_t size() { return N*N*N; }
	
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

	tensor() = default;
	tensor(const T& element) noexcept { data.fill(element); }
	tensor(const tensor&) = default;
	tensor(tensor&&) = default;
	template<typename T2>
	tensor(const tensor<T2,N,N,N>& other) { for (unsigned int i=0; i<size(); ++i) data[i] = static_cast<T>(other.data[i]); }
	tensor(const std::array<T,size()>& _data) noexcept : data(_data) {}
	tensor(std::array<T,size()>&& _data) noexcept : data(std::move(_data)) {}
	tensor(const symmetric_tensor<T,3,N>& other)
	{
		for (std::size_t k=0; k<N; ++k)
			for (std::size_t j=0; j<N; ++j)
				for (std::size_t i=0; i<N; ++i)
					(*this)(i,j,k) = other(i,j,k);
	}
	template<typename T2>
	tensor(const symmetric_tensor<T2,3,N>& other)
	{
		for (std::size_t k=0; k<N; ++k)
			for (std::size_t j=0; j<N; ++j)
				for (std::size_t i=0; i<N; ++i)
					(*this)(i,j,k) = static_cast<T>(other(i,j,k));
	}
	
	tensor& operator=(const tensor&) & = default;
	tensor& operator=(tensor&&) & = default;
	tensor& operator=(const T& element) & { data.fill(element); }
	
public: // member functions

	reference operator[](size_type pos) { return data[pos]; }
	const_reference operator[](size_type pos) const { return data[pos]; }
	
	reference operator()(size_type i, size_type j, size_type k) 
	{
		return data[k*N*N+j*N+i];
	}
	
	const_reference operator()(size_type i, size_type j, size_type k) const
	{
		return data[k*N*N+j*N+i];
	}
	
	reference front() { return data.front(); }
	const_reference front() const { return data.front(); }
	reference back() { return data.back(); }
	const_reference back() const { return data.back(); }
	
	friend std::ostream& operator<<(std::ostream& os, const tensor& t)
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
					os << t(i,j,k);
					if (i<N-1) os << ", ";
				}
				os << ")";
				if (j<N-1) os << ", ";
			}
			os << ")";
			if (k<N-1) os << ", ";
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

	tensor& operator+=(const tensor& other) { for (unsigned int i=0; i<size(); ++i) data[i]+=other[i]; return *this; }
	tensor& operator+=(const T& element) { for (unsigned int i=0; i<size(); ++i) data[i]+=element; return *this; }
	tensor& operator-=(const tensor& other) { for (unsigned int i=0; i<size(); ++i) data[i]-=other[i]; return *this; }
	tensor& operator-=(const T& element) { for (unsigned int i=0; i<size(); ++i) data[i]-=element; return *this; }
	tensor& operator*=(const tensor& other) { for (unsigned int i=0; i<size(); ++i) data[i]*=other[i]; return *this; }
	tensor& operator*=(const T& element) { for (unsigned int i=0; i<size(); ++i) data[i]*=element; return *this; }
	tensor& operator/=(const tensor& other) { for (unsigned int i=0; i<size(); ++i) data[i]/=other[i]; return *this; }
	tensor& operator/=(const T& element) { for (unsigned int i=0; i<size(); ++i) data[i]/=element; return *this; }

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

template< class T, std::size_t Nx, std::size_t Ny, std::size_t Nz >
bool operator==( const math::tensor<T,Nx,Ny,Nz>& lhs, const math::tensor<T,Nx,Ny,Nz>& rhs ) { return lhs.data == rhs.data; }

template< class T, std::size_t Nx, std::size_t Ny, std::size_t Nz >
bool operator!=( const math::tensor<T,Nx,Ny,Nz>& lhs, const math::tensor<T,Nx,Ny,Nz>& rhs ) { return lhs.data != rhs.data; }

template< class T, std::size_t Nx, std::size_t Ny, std::size_t Nz >
bool operator<( const math::tensor<T,Nx,Ny,Nz>& lhs, const math::tensor<T,Nx,Ny,Nz>& rhs ) { return lhs.data < rhs.data; }

template< class T, std::size_t Nx, std::size_t Ny, std::size_t Nz >
bool operator<=( const math::tensor<T,Nx,Ny,Nz>& lhs, const math::tensor<T,Nx,Ny,Nz>& rhs ) { return lhs.data <= rhs.data; }

template< class T, std::size_t Nx, std::size_t Ny, std::size_t Nz >
bool operator>( const math::tensor<T,Nx,Ny,Nz>& lhs, const math::tensor<T,Nx,Ny,Nz>& rhs ) { return lhs.data > rhs.data; }

template< class T, std::size_t Nx, std::size_t Ny, std::size_t Nz >
bool operator>=( const math::tensor<T,Nx,Ny,Nz>& lhs, const math::tensor<T,Nx,Ny,Nz>& rhs ) { return lhs.data >= rhs.data; }

template< class T, std::size_t Nx, std::size_t Ny, std::size_t Nz >
math::tensor<T,Nx,Ny,Nz> operator-(math::tensor<T,Nx,Ny,Nz> t) 
{
	for (std::size_t i=0; i<math::tensor<T,Nx,Ny,Nz>::size(); ++i) t.data[i] = -t.data[i]; 
	return std::move(t);
}

template<typename U, typename V, std::size_t Nx, std::size_t Ny, std::size_t Nz>
typename std::common_type<U,V>::type contract(const math::tensor<U,Nx,Ny,Nz>& lhs, const math::tensor<V,Nx,Ny,Nz>& rhs) noexcept
{
	typename std::common_type<U,V>::type res(lhs(0,0,0)*rhs(0,0,0));
	for (std::size_t k=0; k<Nz; ++k)
		for (std::size_t j=0; j<Ny; ++j)
			for (std::size_t i=(k>0?0:1); i<Nx; ++i)
				res += lhs(i,j,k)*rhs(i,j,k);
	return res;
}

template<typename U, typename V, std::size_t N>
typename std::common_type<U,V>::type contract(const math::tensor<U,N,N,N>& lhs, const math::symmetric_tensor<V,3,N>& rhs) noexcept
{
	typename std::common_type<U,V>::type res(lhs(0,0,0)*rhs(0,0,0));
	for (std::size_t k=0; k<N; ++k)
		for (std::size_t j=0; j<N; ++j)
			for (std::size_t i=(k>0?0:1); i<N; ++i)
				res += lhs(i,j,k)*rhs(i,j,k);
	return res;
}

template<typename U, typename V, std::size_t N>
typename std::common_type<U,V>::type contract(const math::symmetric_tensor<U,3,N>& lhs, const math::tensor<V,N,N,N>& rhs) noexcept
{
	typename std::common_type<U,V>::type res(lhs(0,0,0)*rhs(0,0,0));
	for (std::size_t k=0; k<N; ++k)
		for (std::size_t j=0; j<N; ++j)
			for (std::size_t i=(k>0?0:1); i<N; ++i)
				res += lhs(i,j,k)*rhs(i,j,k);
	return res;
}



template<typename T, std::size_t Nx, std::size_t Ny, std::size_t Nz>
T trace(const math::tensor<T,Nx,Ny,Nz>& a) noexcept
{
    T res(a(0,0,0));
    for (std::size_t i=1; i<std::min(std::min(Nx,Ny),Nz); ++i) res += a(i,i,i);
    return res;
}

template<typename T>
T trace(const math::tensor<T,2,2,2>& a) noexcept
{
    return a.xxx()+a.yyy();
}

template<typename T>
T trace(const math::tensor<T,3,3,3>& a) noexcept
{
    return a.xxx()+a.yyy()+a.zzz();
}


template<typename U, typename V, std::size_t Nx, std::size_t Ny, std::size_t Nz>
math::tensor<typename std::common_type<U,V>::type,Nx,Ny,Nz> operator+(const math::tensor<U,Nx,Ny,Nz>& lhs, const math::tensor<V,Nx,Ny,Nz>& rhs)
{ 
	math::tensor<typename std::common_type<U,V>::type,Nx,Ny,Nz> res(lhs);
	for (std::size_t i=0; i<math::tensor<U,Nx,Ny,Nz>::size(); ++i) res[i] += rhs[i]; 
	return std::move(res);
}

template<typename U, typename V, std::size_t Nx, std::size_t Ny, std::size_t Nz>
math::tensor<typename std::common_type<U,V>::type,Nx,Ny,Nz> operator+(const math::tensor<U,Nx,Ny,Nz>& lhs, const V& rhs) 
{
	math::tensor<typename std::common_type<U,V>::type,Nx,Ny,Nz> res(lhs);
	for (std::size_t i=0; i<math::tensor<U,Nx,Ny,Nz>::size(); ++i) res[i] += rhs; 
	return std::move(res);
}

template<typename U, typename V, std::size_t Nx, std::size_t Ny, std::size_t Nz>
math::tensor<typename std::common_type<U,V>::type,Nx,Ny,Nz> operator+(const U& lhs, const math::tensor<V,Nx,Ny,Nz>& rhs) 
{
	math::tensor<typename std::common_type<U,V>::type,Nx,Ny,Nz> res(rhs);
	for (std::size_t i=0; i<math::tensor<U,Nx,Ny,Nz>::size(); ++i) res[i] += lhs; 
	return std::move(res);
}

template<typename U, typename V, std::size_t Nx, std::size_t Ny, std::size_t Nz>
math::tensor<typename std::common_type<U,V>::type,Nx,Ny,Nz> operator-(const math::tensor<U,Nx,Ny,Nz>& lhs, const math::tensor<V,Nx,Ny,Nz>& rhs)
{ 
	math::tensor<typename std::common_type<U,V>::type,Nx,Ny,Nz> res(lhs);
	for (std::size_t i=0; i<math::tensor<U,Nx,Ny,Nz>::size(); ++i) res[i] -= rhs[i]; 
	return std::move(res);
}

template<typename U, typename V, std::size_t Nx, std::size_t Ny, std::size_t Nz>
math::tensor<typename std::common_type<U,V>::type,Nx,Ny,Nz> operator-(const math::tensor<U,Nx,Ny,Nz>& lhs, const V& rhs) 
{
	math::tensor<typename std::common_type<U,V>::type,Nx,Ny,Nz> res(lhs);
	for (std::size_t i=0; i<math::tensor<U,Nx,Ny,Nz>::size(); ++i) res[i] -= rhs; 
	return std::move(res);
}

template<typename U, typename V, std::size_t Nx, std::size_t Ny, std::size_t Nz>
math::tensor<typename std::common_type<U,V>::type,Nx,Ny,Nz> operator-(const U& lhs, const math::tensor<V,Nx,Ny,Nz>& rhs) 
{
	math::tensor<typename std::common_type<U,V>::type,Nx,Ny,Nz> res(-rhs);
	for (std::size_t i=0; i<math::tensor<U,Nx,Ny,Nz>::size(); ++i) res[i] += lhs; 
	return std::move(res);
}

template<typename U, typename V, std::size_t Nx, std::size_t Ny, std::size_t Nz>
math::tensor<typename std::common_type<U,V>::type,Nx,Ny,Nz> operator*(const math::tensor<U,Nx,Ny,Nz>& lhs, const math::tensor<V,Nx,Ny,Nz>& rhs)
{ 
	math::tensor<typename std::common_type<U,V>::type,Nx,Ny,Nz> res(lhs);
	for (std::size_t i=0; i<math::tensor<U,Nx,Ny,Nz>::size(); ++i) res[i] *= rhs[i]; 
	return std::move(res);
}

template<typename U, typename V, std::size_t Nx, std::size_t Ny, std::size_t Nz>
math::tensor<typename std::common_type<U,V>::type,Nx,Ny,Nz> operator*(const math::tensor<U,Nx,Ny,Nz>& lhs, const V& rhs) 
{
	math::tensor<typename std::common_type<U,V>::type,Nx,Ny,Nz> res(lhs);
	for (std::size_t i=0; i<math::tensor<U,Nx,Ny,Nz>::size(); ++i) res[i] *= rhs; 
	return std::move(res);
}

template<typename U, typename V, std::size_t Nx, std::size_t Ny, std::size_t Nz>
math::tensor<typename std::common_type<U,V>::type,Nx,Ny,Nz> operator*(const U& lhs, const math::tensor<V,Nx,Ny,Nz>& rhs) 
{
	math::tensor<typename std::common_type<U,V>::type,Nx,Ny,Nz> res(rhs);
	for (std::size_t i=0; i<math::tensor<U,Nx,Ny,Nz>::size(); ++i) res[i] *= lhs; 
	return std::move(res);
}

template<typename U, typename V, std::size_t Nx, std::size_t Ny, std::size_t Nz>
math::tensor<typename std::common_type<U,V>::type,Nx,Ny,Nz> operator/(const math::tensor<U,Nx,Ny,Nz>& lhs, const math::tensor<V,Nx,Ny,Nz>& rhs)
{ 
	math::tensor<typename std::common_type<U,V>::type,Nx,Ny,Nz> res(lhs);
	for (std::size_t i=0; i<math::tensor<U,Nx,Ny,Nz>::size(); ++i) res[i] /= rhs[i]; 
	return std::move(res);
}

template<typename U, typename V, std::size_t Nx, std::size_t Ny, std::size_t Nz>
math::tensor<typename std::common_type<U,V>::type,Nx,Ny,Nz> operator/(const math::tensor<U,Nx,Ny,Nz>& lhs, const V& rhs) 
{
	math::tensor<typename std::common_type<U,V>::type,Nx,Ny,Nz> res(lhs);
	for (std::size_t i=0; i<math::tensor<U,Nx,Ny,Nz>::size(); ++i) res[i] /= rhs; 
	return std::move(res);
}

template<typename U, typename V, std::size_t Nx, std::size_t Ny, std::size_t Nz>
math::tensor<typename std::common_type<U,V>::type,Nx,Ny,Nz> operator/(const U& lhs, const math::tensor<V,Nx,Ny,Nz>& rhs) 
{
	math::tensor<typename std::common_type<U,V>::type,Nx,Ny,Nz> res(rhs);
	for (std::size_t i=0; i<math::tensor<U,Nx,Ny,Nz>::size(); ++i) res[i] = lhs / res[i]; 
	return std::move(res);
}


template<typename U, typename V, std::size_t N>
math::tensor<typename std::common_type<U,V>::type,N,N,N> operator+(const math::tensor<U,N,N,N>& lhs, const math::symmetric_tensor<V,3,N>& rhs)
{ 
	math::tensor<typename std::common_type<U,V>::type,N,N,N> res(lhs);
	for (std::size_t k=0; k<N; ++k)
		for (std::size_t j=0; j<N; ++j)
			for (std::size_t i=0; i<N; ++i)
				res(i,j,k) += rhs(i,j,k);
	return std::move(res);
}

template<typename U, typename V, std::size_t N>
math::tensor<typename std::common_type<U,V>::type,N,N,N> operator+(const math::symmetric_tensor<U,3,N>& lhs, const math::tensor<V,N,N,N>& rhs)
{ 
	math::tensor<typename std::common_type<U,V>::type,N,N,N> res(rhs);
	for (std::size_t k=0; k<N; ++k)
		for (std::size_t j=0; j<N; ++j)
			for (std::size_t i=0; i<N; ++i)
				res(i,j,k) += lhs(i,j,k);
	return std::move(res);
}

template<typename U, typename V, std::size_t N>
math::tensor<typename std::common_type<U,V>::type,N,N,N> operator-(const math::tensor<U,N,N,N>& lhs, const math::symmetric_tensor<V,3,N>& rhs)
{ 
	math::tensor<typename std::common_type<U,V>::type,N,N,N> res(lhs);
	for (std::size_t k=0; k<N; ++k)
		for (std::size_t j=0; j<N; ++j)
			for (std::size_t i=0; i<N; ++i)
				res(i,j,k) -= rhs(i,j,k);
	return std::move(res);
}

template<typename U, typename V, std::size_t N>
math::tensor<typename std::common_type<U,V>::type,N,N,N> operator-(const math::symmetric_tensor<U,3,N>& lhs, const math::tensor<V,N,N,N>& rhs)
{ 
	math::tensor<typename std::common_type<U,V>::type,N,N,N> res(-rhs);
	for (std::size_t k=0; k<N; ++k)
		for (std::size_t j=0; j<N; ++j)
			for (std::size_t i=0; i<N; ++i)
				res(i,j,k) += lhs(i,j,k);
	return std::move(res);
}

template<typename U, typename V, std::size_t N>
math::tensor<typename std::common_type<U,V>::type,N,N,N> operator*(const math::tensor<U,N,N,N>& lhs, const math::symmetric_tensor<V,3,N>& rhs)
{ 
	math::tensor<typename std::common_type<U,V>::type,N,N,N> res(lhs);
	for (std::size_t k=0; k<N; ++k)
		for (std::size_t j=0; j<N; ++j)
			for (std::size_t i=0; i<N; ++i)
				res(i,j,k) *= rhs(i,j,k);
	return std::move(res);
}

template<typename U, typename V, std::size_t N>
math::tensor<typename std::common_type<U,V>::type,N,N,N> operator*(const math::symmetric_tensor<U,3,N>& lhs, const math::tensor<V,N,N,N>& rhs)
{ 
	math::tensor<typename std::common_type<U,V>::type,N,N,N> res(rhs);
	for (std::size_t k=0; k<N; ++k)
		for (std::size_t j=0; j<N; ++j)
			for (std::size_t i=0; i<N; ++i)
				res(i,j,k) *= lhs(i,j,k);
	return std::move(res);
}

template<typename U, typename V, std::size_t N>
math::tensor<typename std::common_type<U,V>::type,N,N,N> operator/(const math::tensor<U,N,N,N>& lhs, const math::symmetric_tensor<V,3,N>& rhs)
{ 
	math::tensor<typename std::common_type<U,V>::type,N,N,N> res(lhs);
	for (std::size_t k=0; k<N; ++k)
		for (std::size_t j=0; j<N; ++j)
			for (std::size_t i=0; i<N; ++i)
				res(i,j,k) /= rhs(i,j,k);
	return std::move(res);
}

template<typename U, typename V, std::size_t N>
math::tensor<typename std::common_type<U,V>::type,N,N,N> operator/(const math::symmetric_tensor<U,3,N>& lhs, const math::tensor<V,N,N,N>& rhs)
{ 
	math::tensor<typename std::common_type<U,V>::type,N,N,N> res;
	for (std::size_t k=0; k<N; ++k)
		for (std::size_t j=0; j<N; ++j)
			for (std::size_t i=0; i<N; ++i)
				res(i,j) = lhs(i,j,k)/rhs(i,j,k);
	return std::move(res);
}

} // namespace math

#endif // #define MATH_INCLUDED_TENSOR_TENSOR_3_HPP
