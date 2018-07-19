#ifndef MATH_INCLUDED_TENSOR_TENSOR_2_HPP
#define MATH_INCLUDED_TENSOR_TENSOR_2_HPP

#include <tensor/symmetric_tensor_2.hpp>

namespace math {

// rank-2 tensor
// column major format (fortran)
template<typename T, std::size_t Nx, std::size_t Ny>
class tensor<T,Nx,Ny>
{
public: // static member functions

	static constexpr std::size_t size() { return Nx*Ny; }	

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
	tensor(const tensor<T2,Nx,Ny>& other) { for (unsigned int i=0; i<size(); ++i) data[i] = static_cast<T>(other.data[i]); }
	tensor(const std::array<T,size()>& _data) noexcept : data(_data) {}
	tensor(std::array<T,size()>&& _data) noexcept : data(std::move(_data)) {}
	
	tensor& operator=(const tensor&) & = default;
	tensor& operator=(tensor&&) & = default;
	tensor& operator=(const T& element) & { data.fill(element); }
	
public: // member functions

	reference operator[](size_type pos) { return data[pos]; }
	const_reference operator[](size_type pos) const { return data[pos]; }
	
	reference operator()(size_type i, size_type j) 
	{
		return data[j*Nx+i];
	}
	
	const_reference operator()(size_type i, size_type j) const
	{
		return data[j*Nx+i];
	}
	
	reference front() { return data.front(); }
	const_reference front() const { return data.front(); }
	reference back() { return data.back(); }
	const_reference back() const { return data.back(); }
	
	vector<T,Ny> get_row(size_type i) const
	{
		vector<T,Ny> res;
		for (size_type j=0; j<Ny; ++j) res[j] = (*this)(i,j);
		return std::move(res);
	}
	
	vector<T,Nx> get_col(size_type j) const
	{
		vector<T,Nx> res;
		for (size_type i=0; i<Nx; ++i) res[i] = (*this)(i,j);
		return std::move(res);
	}
	
	void set_row(size_type i, const vector<T,Ny>& row)
	{
		for (size_type j=0; j<Ny; ++j) (*this)(i,j) = row[j];
	}
	
	void set_col(size_type j, const vector<T,Nx>& col) const
	{
		for (size_type i=0; i<Nx; ++i) (*this)(i,j) = col[i];
	}
	
	friend std::ostream& operator<<(std::ostream& os, const tensor& t)
	{
		os << "(";
		for (unsigned int j=0; j<Ny; ++j)
		{
			os << "(";
			for (unsigned int i=0; i<Nx; ++i)
			{
				os << t(i,j);
				if (i<Nx-1) os << ", ";
			}
			os << ")";
			if (j<Ny-1) os << ", ";
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



// quadratic rank-2 tensor
template<typename T, std::size_t N>
class tensor<T,N,N> : public detail::tensor_access<tensor<T,N,N>,T,N,N>
{
public: // static member functions

	static constexpr std::size_t size() { return N*N; }
	
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
	tensor(const tensor<T2,N,N>& other) { for (unsigned int i=0; i<size(); ++i) data[i] = static_cast<T>(other.data[i]); }
	tensor(const std::array<T,size()>& _data) noexcept : data(_data) {}
	tensor(std::array<T,size()>&& _data) noexcept : data(std::move(_data)) {}
	tensor(const symmetric_tensor<T,2,N>& other)
	{
		for (std::size_t j=0; j<N; ++j)
			for (std::size_t i=0; i<N; ++i)
				(*this)(i,j) = other(i,j);
	}
	template<typename T2>
	tensor(const symmetric_tensor<T2,2,N>& other)
	{
		for (std::size_t j=0; j<N; ++j)
			for (std::size_t i=0; i<N; ++i)
				(*this)(i,j) = static_cast<T>(other(i,j));
	}
	
	tensor& operator=(const tensor&) & = default;
	tensor& operator=(tensor&&) & = default;
	tensor& operator=(const T& element) & { data.fill(element); }
	
public: // member functions

	reference operator[](size_type pos) { return data[pos]; }
	const_reference operator[](size_type pos) const { return data[pos]; }
	
	reference operator()(size_type i, size_type j) 
	{
		return data[j*N+i];
	}
	
	const_reference operator()(size_type i, size_type j) const
	{
		return data[j*N+i];
	}
	
	reference front() { return data.front(); }
	const_reference front() const { return data.front(); }
	reference back() { return data.back(); }
	const_reference back() const { return data.back(); }
	
	vector<T,N> get_row(size_type i) const
	{
		vector<T,N> res;
		for (size_type j=0; j<N; ++j) res[j] = (*this)(i,j);
		return std::move(res);
	}
	
	vector<T,N> get_col(size_type j) const
	{
		vector<T,N> res;
		for (size_type i=0; i<N; ++i) res[i] = (*this)(i,j);
		return std::move(res);
	}
	
	void set_row(size_type i, const vector<T,N>& row)
	{
		for (size_type j=0; j<N; ++j) (*this)(i,j) = row[j];
	}
	
	void set_col(size_type j, const vector<T,N>& col) const
	{
		for (size_type i=0; i<N; ++i) (*this)(i,j) = col[i];
	}
	
	friend std::ostream& operator<<(std::ostream& os, const tensor& t)
	{
		os << "(";
		for (unsigned int j=0; j<N; ++j)
		{
			os << "(";
			for (unsigned int i=0; i<N; ++i)
			{
				os << t(i,j);
				if (i<N-1) os << ", ";
			}
			os << ")";
			if (j<N-1) os << ", ";
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

template< class T, std::size_t Nx, std::size_t Ny >
bool operator==( const math::tensor<T,Nx,Ny>& lhs, const math::tensor<T,Nx,Ny>& rhs ) { return lhs.data == rhs.data; }

template< class T, std::size_t Nx, std::size_t Ny >
bool operator!=( const math::tensor<T,Nx,Ny>& lhs, const math::tensor<T,Nx,Ny>& rhs ) { return lhs.data != rhs.data; }

template< class T, std::size_t Nx, std::size_t Ny >
bool operator<( const math::tensor<T,Nx,Ny>& lhs, const math::tensor<T,Nx,Ny>& rhs ) { return lhs.data < rhs.data; }

template< class T, std::size_t Nx, std::size_t Ny >
bool operator<=( const math::tensor<T,Nx,Ny>& lhs, const math::tensor<T,Nx,Ny>& rhs ) { return lhs.data <= rhs.data; }

template< class T, std::size_t Nx, std::size_t Ny >
bool operator>( const math::tensor<T,Nx,Ny>& lhs, const math::tensor<T,Nx,Ny>& rhs ) { return lhs.data > rhs.data; }

template< class T, std::size_t Nx, std::size_t Ny >
bool operator>=( const math::tensor<T,Nx,Ny>& lhs, const math::tensor<T,Nx,Ny>& rhs ) { return lhs.data >= rhs.data; }

template< class T, std::size_t Nx, std::size_t Ny >
math::tensor<T,Nx,Ny> operator-(math::tensor<T,Nx,Ny> t) 
{
	for (std::size_t i=0; i<math::tensor<T,Nx,Ny>::size(); ++i) t.data[i] = -t.data[i]; 
	return std::move(t);
}

template<typename U, typename V, std::size_t Nx, std::size_t Ny>
typename std::common_type<U,V>::type contract(const math::tensor<U,Nx,Ny>& lhs, const math::tensor<V,Nx,Ny>& rhs) noexcept
{
	typename std::common_type<U,V>::type res(lhs(0,0)*rhs(0,0));
	for (std::size_t j=0; j<Ny; ++j)
		for (std::size_t i=(j>0?0:1); i<Nx; ++i)
			res += lhs(i,j)*rhs(i,j);
	return res;
}

template<typename U, typename V, std::size_t N>
typename std::common_type<U,V>::type contract(const math::tensor<U,N,N>& lhs, const math::symmetric_tensor<V,2,N>& rhs) noexcept
{
	typename std::common_type<U,V>::type res(lhs(0,0)*rhs(0,0));
	for (std::size_t j=0; j<N; ++j)
		for (std::size_t i=(j>0?0:1); i<N; ++i)
			res += lhs(i,j)*rhs(i,j);
	return res;
}

template<typename U, typename V, std::size_t N>
typename std::common_type<U,V>::type contract(const math::symmetric_tensor<U,2,N>& lhs, const math::tensor<V,N,N>& rhs) noexcept
{
	typename std::common_type<U,V>::type res(lhs(0,0)*rhs(0,0));
	for (std::size_t j=0; j<N; ++j)
		for (std::size_t i=(j>0?0:1); i<N; ++i)
			res += lhs(i,j)*rhs(i,j);
	return res;
}



template<typename T, std::size_t Nx, std::size_t Ny>
T trace(const math::tensor<T,Nx,Ny>& a) noexcept
{
    T res(a(0,0));
    for (std::size_t i=1; i<std::min(Nx,Ny); ++i) res += a(i,i);
    return res;
}

template<typename T>
T trace(const math::tensor<T,2,2>& a) noexcept
{
    return a.xx()+a.yy();
}

template<typename T>
T trace(const math::tensor<T,3,3>& a) noexcept
{
    return a.xx()+a.yy()+a.zz();
}

template<typename T, std::size_t Nx, std::size_t Ny>
math::tensor<T,Ny,Nx> transpose(const math::tensor<T,Nx,Ny>& other) 
{
	math::tensor<T,Ny,Nx> res;
	for (std::size_t j=0; j<Ny; ++j)
		for (std::size_t i=0; i<Nx; ++i)
			res(j,i) = other(i,j);
	return std::move(res); 
}


template<typename U, typename V, std::size_t Nx, std::size_t Ny>
math::tensor<typename std::common_type<U,V>::type,Nx,Ny> operator+(const math::tensor<U,Nx,Ny>& lhs, const math::tensor<V,Nx,Ny>& rhs)
{ 
	math::tensor<typename std::common_type<U,V>::type,Nx,Ny> res(lhs);
	for (std::size_t i=0; i<math::tensor<U,Nx,Ny>::size(); ++i) res[i] += rhs[i]; 
	return std::move(res);
}

template<typename U, typename V, std::size_t Nx, std::size_t Ny>
math::tensor<typename std::common_type<U,V>::type,Nx,Ny> operator+(const math::tensor<U,Nx,Ny>& lhs, const V& rhs) 
{
	math::tensor<typename std::common_type<U,V>::type,Nx,Ny> res(lhs);
	for (std::size_t i=0; i<math::tensor<U,Nx,Ny>::size(); ++i) res[i] += rhs; 
	return std::move(res);
}

template<typename U, typename V, std::size_t Nx, std::size_t Ny>
math::tensor<typename std::common_type<U,V>::type,Nx,Ny> operator+(const U& lhs, const math::tensor<V,Nx,Ny>& rhs) 
{
	math::tensor<typename std::common_type<U,V>::type,Nx,Ny> res(rhs);
	for (std::size_t i=0; i<math::tensor<U,Nx,Ny>::size(); ++i) res[i] += lhs; 
	return std::move(res);
}

template<typename U, typename V, std::size_t Nx, std::size_t Ny>
math::tensor<typename std::common_type<U,V>::type,Nx,Ny> operator-(const math::tensor<U,Nx,Ny>& lhs, const math::tensor<V,Nx,Ny>& rhs)
{ 
	math::tensor<typename std::common_type<U,V>::type,Nx,Ny> res(lhs);
	for (std::size_t i=0; i<math::tensor<U,Nx,Ny>::size(); ++i) res[i] -= rhs[i]; 
	return std::move(res);
}

template<typename U, typename V, std::size_t Nx, std::size_t Ny>
math::tensor<typename std::common_type<U,V>::type,Nx,Ny> operator-(const math::tensor<U,Nx,Ny>& lhs, const V& rhs) 
{
	math::tensor<typename std::common_type<U,V>::type,Nx,Ny> res(lhs);
	for (std::size_t i=0; i<math::tensor<U,Nx,Ny>::size(); ++i) res[i] -= rhs; 
	return std::move(res);
}

template<typename U, typename V, std::size_t Nx, std::size_t Ny>
math::tensor<typename std::common_type<U,V>::type,Nx,Ny> operator-(const U& lhs, const math::tensor<V,Nx,Ny>& rhs) 
{
	math::tensor<typename std::common_type<U,V>::type,Nx,Ny> res(-rhs);
	for (std::size_t i=0; i<math::tensor<U,Nx,Ny>::size(); ++i) res[i] += lhs; 
	return std::move(res);
}

template<typename U, typename V, std::size_t Nx, std::size_t Ny>
math::tensor<typename std::common_type<U,V>::type,Nx,Ny> operator*(const math::tensor<U,Nx,Ny>& lhs, const math::tensor<V,Nx,Ny>& rhs)
{ 
	math::tensor<typename std::common_type<U,V>::type,Nx,Ny> res(lhs);
	for (std::size_t i=0; i<math::tensor<U,Nx,Ny>::size(); ++i) res[i] *= rhs[i]; 
	return std::move(res);
}

template<typename U, typename V, std::size_t Nx, std::size_t Ny>
math::tensor<typename std::common_type<U,V>::type,Nx,Ny> operator*(const math::tensor<U,Nx,Ny>& lhs, const V& rhs) 
{
	math::tensor<typename std::common_type<U,V>::type,Nx,Ny> res(lhs);
	for (std::size_t i=0; i<math::tensor<U,Nx,Ny>::size(); ++i) res[i] *= rhs; 
	return std::move(res);
}

template<typename U, typename V, std::size_t Nx, std::size_t Ny>
math::tensor<typename std::common_type<U,V>::type,Nx,Ny> operator*(const U& lhs, const math::tensor<V,Nx,Ny>& rhs) 
{
	math::tensor<typename std::common_type<U,V>::type,Nx,Ny> res(rhs);
	for (std::size_t i=0; i<math::tensor<U,Nx,Ny>::size(); ++i) res[i] *= lhs; 
	return std::move(res);
}

template<typename U, typename V, std::size_t Nx, std::size_t Ny>
math::tensor<typename std::common_type<U,V>::type,Nx,Ny> operator/(const math::tensor<U,Nx,Ny>& lhs, const math::tensor<V,Nx,Ny>& rhs)
{ 
	math::tensor<typename std::common_type<U,V>::type,Nx,Ny> res(lhs);
	for (std::size_t i=0; i<math::tensor<U,Nx,Ny>::size(); ++i) res[i] /= rhs[i]; 
	return std::move(res);
}

template<typename U, typename V, std::size_t Nx, std::size_t Ny>
math::tensor<typename std::common_type<U,V>::type,Nx,Ny> operator/(const math::tensor<U,Nx,Ny>& lhs, const V& rhs) 
{
	math::tensor<typename std::common_type<U,V>::type,Nx,Ny> res(lhs);
	for (std::size_t i=0; i<math::tensor<U,Nx,Ny>::size(); ++i) res[i] /= rhs; 
	return std::move(res);
}

template<typename U, typename V, std::size_t Nx, std::size_t Ny>
math::tensor<typename std::common_type<U,V>::type,Nx,Ny> operator/(const U& lhs, const math::tensor<V,Nx,Ny>& rhs) 
{
	math::tensor<typename std::common_type<U,V>::type,Nx,Ny> res(rhs);
	for (std::size_t i=0; i<math::tensor<U,Nx,Ny>::size(); ++i) res[i] = lhs / res[i]; 
	return std::move(res);
}


template<typename U, typename V, std::size_t N>
math::tensor<typename std::common_type<U,V>::type,N,N> operator+(const math::tensor<U,N,N>& lhs, const math::symmetric_tensor<V,2,N>& rhs)
{ 
	math::tensor<typename std::common_type<U,V>::type,N,N> res(lhs);
	for (std::size_t j=0; j<N; ++j)
		for (std::size_t i=0; i<N; ++i)
			res(i,j) += rhs(i,j);
	return std::move(res);
}

template<typename U, typename V, std::size_t N>
math::tensor<typename std::common_type<U,V>::type,N,N> operator+(const math::symmetric_tensor<U,2,N>& lhs, const math::tensor<V,N,N>& rhs)
{ 
	math::tensor<typename std::common_type<U,V>::type,N,N> res(rhs);
	for (std::size_t j=0; j<N; ++j)
		for (std::size_t i=0; i<N; ++i)
			res(i,j) += lhs(i,j);
	return std::move(res);
}

template<typename U, typename V, std::size_t N>
math::tensor<typename std::common_type<U,V>::type,N,N> operator-(const math::tensor<U,N,N>& lhs, const math::symmetric_tensor<V,2,N>& rhs)
{ 
	math::tensor<typename std::common_type<U,V>::type,N,N> res(lhs);
	for (std::size_t j=0; j<N; ++j)
		for (std::size_t i=0; i<N; ++i)
			res(i,j) -= rhs(i,j);
	return std::move(res);
}

template<typename U, typename V, std::size_t N>
math::tensor<typename std::common_type<U,V>::type,N,N> operator-(const math::symmetric_tensor<U,2,N>& lhs, const math::tensor<V,N,N>& rhs)
{ 
	math::tensor<typename std::common_type<U,V>::type,N,N> res(-rhs);
	for (std::size_t j=0; j<N; ++j)
		for (std::size_t i=0; i<N; ++i)
			res(i,j) += lhs(i,j);
	return std::move(res);
}

template<typename U, typename V, std::size_t N>
math::tensor<typename std::common_type<U,V>::type,N,N> operator*(const math::tensor<U,N,N>& lhs, const math::symmetric_tensor<V,2,N>& rhs)
{ 
	math::tensor<typename std::common_type<U,V>::type,N,N> res(lhs);
	for (std::size_t j=0; j<N; ++j)
		for (std::size_t i=0; i<N; ++i)
			res(i,j) *= rhs(i,j);
	return std::move(res);
}

template<typename U, typename V, std::size_t N>
math::tensor<typename std::common_type<U,V>::type,N,N> operator*(const math::symmetric_tensor<U,2,N>& lhs, const math::tensor<V,N,N>& rhs)
{ 
	math::tensor<typename std::common_type<U,V>::type,N,N> res(rhs);
	for (std::size_t j=0; j<N; ++j)
		for (std::size_t i=0; i<N; ++i)
			res(i,j) *= lhs(i,j);
	return std::move(res);
}

template<typename U, typename V, std::size_t N>
math::tensor<typename std::common_type<U,V>::type,N,N> operator/(const math::tensor<U,N,N>& lhs, const math::symmetric_tensor<V,2,N>& rhs)
{ 
	math::tensor<typename std::common_type<U,V>::type,N,N> res(lhs);
	for (std::size_t j=0; j<N; ++j)
		for (std::size_t i=0; i<N; ++i)
			res(i,j) /= rhs(i,j);
	return std::move(res);
}

template<typename U, typename V, std::size_t N>
math::tensor<typename std::common_type<U,V>::type,N,N> operator/(const math::symmetric_tensor<U,2,N>& lhs, const math::tensor<V,N,N>& rhs)
{ 
	math::tensor<typename std::common_type<U,V>::type,N,N> res;
	for (std::size_t j=0; j<N; ++j)
		for (std::size_t i=0; i<N; ++i)
			res(i,j) = lhs(i,j)/rhs(i,j);
	return std::move(res);
}





template<typename U, typename V, std::size_t N, std::size_t M>
math::vector<typename std::common_type<U,V>::type,M> dot(const math::tensor<U,M,N>& lhs, const math::vector<V,N>& rhs)
{ 
	math::vector<typename std::common_type<U,V>::type,M> res(0.0);
	for (std::size_t i=0; i<M; ++i)
		for (std::size_t j=0; j<N; ++j)
			res(i) += lhs(i,j)*rhs(j);
	return std::move(res);
}

template<typename U, typename V, std::size_t N>
math::vector<typename std::common_type<U,V>::type,N> dot(const math::symmetric_tensor<U,2,N>& lhs, const math::vector<V,N>& rhs)
{ 
	math::vector<typename std::common_type<U,V>::type,N> res(0.0);
	for (std::size_t i=0; i<N; ++i)
		for (std::size_t j=0; j<N; ++j)
			res(i) += lhs(i,j)*rhs(j);
	return std::move(res);
}

template<typename U, std::size_t N>
symmetric_tensor<U,2,N> AtpA(const tensor<U,N,N>& A)
{
	const auto tmp = transpose(A) + A;
	symmetric_tensor<U,2,N> res;
	for (std::size_t i=0; i<N; ++i)
		for (std::size_t j=i; j<N; ++j)
			res(i,j) += tmp(i,j);
	return std::move(res);
}

/*
template<typename U, typename V, std::size_t N>
math::tensor<typename std::common_type<U,V>::type,N,N> operator*(const math::tensor<U,N,N>& lhs, const math::tensor<U,N,N>& rhs)
{ 
	math::tensor<typename std::common_type<U,V>::type,N,N> res(lhs);
	for (std::size_t j=0; j<N; ++j)
		for (std::size_t i=0; i<N; ++i)
			res(i,j) *= rhs(i,j);
	return std::move(res);
}*/


} // namespace math

#endif // #define MATH_INCLUDED_TENSOR_TENSOR_2_HPP
