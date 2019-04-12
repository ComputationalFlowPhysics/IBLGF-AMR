#ifndef MATH_INCLUDED_TENSOR_TENSOR_BASE_HPP
#define MATH_INCLUDED_TENSOR_TENSOR_BASE_HPP

#include <tensor/vector.hpp>

namespace math {

template<typename T, std::size_t... Dimension>
class tensor {};

template<typename T, std::size_t Rank, std::size_t N>
class symmetric_tensor {};



// rank-1 tensor
template<typename T, std::size_t N>
using tensor_1 = vector<T,N>;


// rank-2 tensors
template<typename T, std::size_t Nx, std::size_t Ny>
using tensor_2 = tensor<T,Nx,Ny>;

template<typename T, std::size_t N>
using quadratic_tensor_2 = tensor<T,N,N>;

template<typename T, std::size_t N>
using symmetric_tensor_2 = symmetric_tensor<T,2,N>;


// rank-3 tensors
template<typename T, std::size_t Nx, std::size_t Ny, std::size_t Nz>
using tensor_3 = tensor<T,Nx,Ny,Nz>;

template<typename T, std::size_t N>
using quadratic_tensor_3 = tensor<T,N,N,N>;

template<typename T, std::size_t N>
using symmetric_tensor_3 = symmetric_tensor<T,3,N>;


// rank-4 tensors
template<typename T, std::size_t Nx, std::size_t Ny, std::size_t Nz, std::size_t Nw>
using tensor_4 = tensor<T,Nx,Ny,Nz,Nw>;

template<typename T, std::size_t N>
using quadratic_tensor_4 = tensor<T,N,N,N,N>;

template<typename T, std::size_t N>
using symmetric_tensor_4 = symmetric_tensor<T,4,N>;




namespace detail {

	constexpr std::size_t s2(std::size_t j, std::size_t n) { return (j*(1+2*n) - j*j)/2; }
	constexpr std::size_t s3(std::size_t k, std::size_t n) { return ((k*k*k + k*(2+6*n+3*n*n)) - k*k*(3+3*n) )/6; }
	constexpr std::size_t s4(std::size_t w, std::size_t n) { return (w*w*w*(6+4*n) + w*(6+22*n+18*n*n+4*n*n*n) - w*w*w*w - w*w*(11+18*n+6*n*n))/24; }
	
	
	template<typename Derived, typename T, std::size_t... Dimension>
	struct tensor_access {};

	template<typename Derived, typename T, std::size_t Rank, std::size_t N>
	struct symmetric_tensor_access {};
	
	// Rank = 2, N = 1
	template<typename Derived, typename T>
	struct tensor_access<Derived,T,1,1>
	{
		using reference = typename std::array<T,1>::reference;
		using const_reference=typename std::array<T,1>::const_reference;
		reference xx() { return static_cast<Derived*>(this)->data[0]; }
		const_reference xx() const { return static_cast<const Derived*>(this)->data[0]; }
	};

	// Rank = 2, N = 2
	template<typename Derived, typename T>
	struct tensor_access<Derived,T,2,2>
	{
		using reference = typename std::array<T,1>::reference;
		using const_reference=typename std::array<T,1>::const_reference;
		reference xx() { return static_cast<Derived*>(this)->data[0]; }
		const_reference xx() const { return static_cast<const Derived*>(this)->data[0]; }
		reference yx() { return static_cast<Derived*>(this)->data[1]; }
		const_reference yx() const { return static_cast<const Derived*>(this)->data[1]; }
		reference xy() { return static_cast<Derived*>(this)->data[2]; }
		const_reference xy() const { return static_cast<const Derived*>(this)->data[2]; }
		reference yy() { return static_cast<Derived*>(this)->data[3]; }
		const_reference yy() const { return static_cast<const Derived*>(this)->data[3]; }
	};

	// Rank = 2, N = 3
	template<typename Derived, typename T>
	struct tensor_access<Derived,T,3,3>
	{
		using reference = typename std::array<T,1>::reference;
		using const_reference=typename std::array<T,1>::const_reference;
		reference xx() { return static_cast<Derived*>(this)->data[0]; }
		const_reference xx() const { return static_cast<const Derived*>(this)->data[0]; }
		reference yx() { return static_cast<Derived*>(this)->data[1]; }
		const_reference yx() const { return static_cast<const Derived*>(this)->data[1]; }
		reference zx() { return static_cast<Derived*>(this)->data[2]; }
		const_reference zx() const { return static_cast<const Derived*>(this)->data[2]; }
		reference xy() { return static_cast<Derived*>(this)->data[3]; }
		const_reference xy() const { return static_cast<const Derived*>(this)->data[3]; }
		reference yy() { return static_cast<Derived*>(this)->data[4]; }
		const_reference yy() const { return static_cast<const Derived*>(this)->data[4]; }
		reference zy() { return static_cast<Derived*>(this)->data[5]; }
		const_reference zy() const { return static_cast<const Derived*>(this)->data[5]; }
		reference xz() { return static_cast<Derived*>(this)->data[6]; }
		const_reference xz() const { return static_cast<const Derived*>(this)->data[6]; }
		reference yz() { return static_cast<Derived*>(this)->data[7]; }
		const_reference yz() const { return static_cast<const Derived*>(this)->data[7]; }
		reference zz() { return static_cast<Derived*>(this)->data[8]; }
		const_reference zz() const { return static_cast<const Derived*>(this)->data[8]; }
	};

	// Rank = 2, N = 4
	template<typename Derived, typename T>
	struct tensor_access<Derived,T,4,4>
	{
		using reference = typename std::array<T,1>::reference;
		using const_reference=typename std::array<T,1>::const_reference;
		reference xx() { return static_cast<Derived*>(this)->data[0]; }
		const_reference xx() const { return static_cast<const Derived*>(this)->data[0]; }
		reference yx() { return static_cast<Derived*>(this)->data[1]; }
		const_reference yx() const { return static_cast<const Derived*>(this)->data[1]; }
		reference zx() { return static_cast<Derived*>(this)->data[2]; }
		const_reference zx() const { return static_cast<const Derived*>(this)->data[2]; }
		reference wx() { return static_cast<Derived*>(this)->data[3]; }
		const_reference wx() const { return static_cast<const Derived*>(this)->data[3]; }
		reference xy() { return static_cast<Derived*>(this)->data[4]; }
		const_reference xy() const { return static_cast<const Derived*>(this)->data[4]; }
		reference yy() { return static_cast<Derived*>(this)->data[5]; }
		const_reference yy() const { return static_cast<const Derived*>(this)->data[5]; }
		reference zy() { return static_cast<Derived*>(this)->data[6]; }
		const_reference zy() const { return static_cast<const Derived*>(this)->data[6]; }
		reference wy() { return static_cast<Derived*>(this)->data[7]; }
		const_reference wy() const { return static_cast<const Derived*>(this)->data[7]; }
		reference xz() { return static_cast<Derived*>(this)->data[8]; }
		const_reference xz() const { return static_cast<const Derived*>(this)->data[8]; }
		reference yz() { return static_cast<Derived*>(this)->data[9]; }
		const_reference yz() const { return static_cast<const Derived*>(this)->data[9]; }
		reference zz() { return static_cast<Derived*>(this)->data[10]; }
		const_reference zz() const { return static_cast<const Derived*>(this)->data[10]; }
		reference wz() { return static_cast<Derived*>(this)->data[11]; }
		const_reference wz() const { return static_cast<const Derived*>(this)->data[11]; }
		reference xw() { return static_cast<Derived*>(this)->data[12]; }
		const_reference xw() const { return static_cast<const Derived*>(this)->data[12]; }
		reference yw() { return static_cast<Derived*>(this)->data[13]; }
		const_reference yw() const { return static_cast<const Derived*>(this)->data[13]; }
		reference zw() { return static_cast<Derived*>(this)->data[14]; }
		const_reference zw() const { return static_cast<const Derived*>(this)->data[14]; }
		reference ww() { return static_cast<Derived*>(this)->data[15]; }
		const_reference ww() const { return static_cast<const Derived*>(this)->data[15]; }
	};


	// Rank = 3, N = 1
	template<typename Derived, typename T>
	struct tensor_access<Derived,T,1,1,1>
	{
		using reference = typename std::array<T,1>::reference;
		using const_reference=typename std::array<T,1>::const_reference;
		reference xxx() { return static_cast<Derived*>(this)->data[0]; }
		const_reference xxx() const { return static_cast<const Derived*>(this)->data[0]; }
	};

	// Rank = 3, N = 2
	template<typename Derived, typename T>
	struct tensor_access<Derived,T,2,2,2>
	{
		using reference = typename std::array<T,1>::reference;
		using const_reference=typename std::array<T,1>::const_reference;
		reference xxx() { return static_cast<Derived*>(this)->data[0]; }
		const_reference xxx() const { return static_cast<const Derived*>(this)->data[0]; }
		reference yxx() { return static_cast<Derived*>(this)->data[1]; }
		const_reference yxx() const { return static_cast<const Derived*>(this)->data[1]; }
		reference xyx() { return static_cast<Derived*>(this)->data[2]; }
		const_reference xyx() const { return static_cast<const Derived*>(this)->data[2]; }
		reference yyx() { return static_cast<Derived*>(this)->data[3]; }
		const_reference yyx() const { return static_cast<const Derived*>(this)->data[3]; }
		reference xxy() { return static_cast<Derived*>(this)->data[4]; }
		const_reference xxy() const { return static_cast<const Derived*>(this)->data[4]; }
		reference yxy() { return static_cast<Derived*>(this)->data[5]; }
		const_reference yxy() const { return static_cast<const Derived*>(this)->data[5]; }
		reference xyy() { return static_cast<Derived*>(this)->data[6]; }
		const_reference xyy() const { return static_cast<const Derived*>(this)->data[6]; }
		reference yyy() { return static_cast<Derived*>(this)->data[7]; }
		const_reference yyy() const { return static_cast<const Derived*>(this)->data[7]; }
	};

	// Rank = 3, N = 3
	template<typename Derived, typename T>
	struct tensor_access<Derived,T,3,3,3>
	{
		using reference = typename std::array<T,1>::reference;
		using const_reference=typename std::array<T,1>::const_reference;
		reference xxx() { return static_cast<Derived*>(this)->data[0]; }
		const_reference xxx() const { return static_cast<const Derived*>(this)->data[0]; }
		reference yxx() { return static_cast<Derived*>(this)->data[1]; }
		const_reference yxx() const { return static_cast<const Derived*>(this)->data[1]; }
		reference zxx() { return static_cast<Derived*>(this)->data[2]; }
		const_reference zxx() const { return static_cast<const Derived*>(this)->data[2]; }
		reference xyx() { return static_cast<Derived*>(this)->data[3]; }
		const_reference xyx() const { return static_cast<const Derived*>(this)->data[3]; }
		reference yyx() { return static_cast<Derived*>(this)->data[4]; }
		const_reference yyx() const { return static_cast<const Derived*>(this)->data[4]; }
		reference zyx() { return static_cast<Derived*>(this)->data[5]; }
		const_reference zyx() const { return static_cast<const Derived*>(this)->data[5]; }
		reference xzx() { return static_cast<Derived*>(this)->data[6]; }
		const_reference xzx() const { return static_cast<const Derived*>(this)->data[6]; }
		reference yzx() { return static_cast<Derived*>(this)->data[7]; }
		const_reference yzx() const { return static_cast<const Derived*>(this)->data[7]; }
		reference zzx() { return static_cast<Derived*>(this)->data[8]; }
		const_reference zzx() const { return static_cast<const Derived*>(this)->data[8]; }
		reference xxy() { return static_cast<Derived*>(this)->data[9]; }
		const_reference xxy() const { return static_cast<const Derived*>(this)->data[9]; }
		reference yxy() { return static_cast<Derived*>(this)->data[10]; }
		const_reference yxy() const { return static_cast<const Derived*>(this)->data[10]; }
		reference zxy() { return static_cast<Derived*>(this)->data[11]; }
		const_reference zxy() const { return static_cast<const Derived*>(this)->data[11]; }
		reference xyy() { return static_cast<Derived*>(this)->data[12]; }
		const_reference xyy() const { return static_cast<const Derived*>(this)->data[12]; }
		reference yyy() { return static_cast<Derived*>(this)->data[13]; }
		const_reference yyy() const { return static_cast<const Derived*>(this)->data[13]; }
		reference zyy() { return static_cast<Derived*>(this)->data[14]; }
		const_reference zyy() const { return static_cast<const Derived*>(this)->data[14]; }
		reference xzy() { return static_cast<Derived*>(this)->data[15]; }
		const_reference xzy() const { return static_cast<const Derived*>(this)->data[15]; }
		reference yzy() { return static_cast<Derived*>(this)->data[16]; }
		const_reference yzy() const { return static_cast<const Derived*>(this)->data[16]; }
		reference zzy() { return static_cast<Derived*>(this)->data[17]; }
		const_reference zzy() const { return static_cast<const Derived*>(this)->data[17]; }
		reference xxz() { return static_cast<Derived*>(this)->data[18]; }
		const_reference xxz() const { return static_cast<const Derived*>(this)->data[18]; }
		reference yxz() { return static_cast<Derived*>(this)->data[19]; }
		const_reference yxz() const { return static_cast<const Derived*>(this)->data[19]; }
		reference zxz() { return static_cast<Derived*>(this)->data[20]; }
		const_reference zxz() const { return static_cast<const Derived*>(this)->data[20]; }
		reference xyz() { return static_cast<Derived*>(this)->data[21]; }
		const_reference xyz() const { return static_cast<const Derived*>(this)->data[21]; }
		reference yyz() { return static_cast<Derived*>(this)->data[22]; }
		const_reference yyz() const { return static_cast<const Derived*>(this)->data[22]; }
		reference zyz() { return static_cast<Derived*>(this)->data[23]; }
		const_reference zyz() const { return static_cast<const Derived*>(this)->data[23]; }
		reference xzz() { return static_cast<Derived*>(this)->data[24]; }
		const_reference xzz() const { return static_cast<const Derived*>(this)->data[24]; }
		reference yzz() { return static_cast<Derived*>(this)->data[25]; }
		const_reference yzz() const { return static_cast<const Derived*>(this)->data[25]; }
		reference zzz() { return static_cast<Derived*>(this)->data[26]; }
		const_reference zzz() const { return static_cast<const Derived*>(this)->data[26]; }
	};

	// Rank = 3, N = 4
	template<typename Derived, typename T>
	struct tensor_access<Derived,T,4,4,4>
	{
		using reference = typename std::array<T,1>::reference;
		using const_reference=typename std::array<T,1>::const_reference;
		reference xxx() { return static_cast<Derived*>(this)->data[0]; }
		const_reference xxx() const { return static_cast<const Derived*>(this)->data[0]; }
		reference yxx() { return static_cast<Derived*>(this)->data[1]; }
		const_reference yxx() const { return static_cast<const Derived*>(this)->data[1]; }
		reference zxx() { return static_cast<Derived*>(this)->data[2]; }
		const_reference zxx() const { return static_cast<const Derived*>(this)->data[2]; }
		reference wxx() { return static_cast<Derived*>(this)->data[3]; }
		const_reference wxx() const { return static_cast<const Derived*>(this)->data[3]; }
		reference xyx() { return static_cast<Derived*>(this)->data[4]; }
		const_reference xyx() const { return static_cast<const Derived*>(this)->data[4]; }
		reference yyx() { return static_cast<Derived*>(this)->data[5]; }
		const_reference yyx() const { return static_cast<const Derived*>(this)->data[5]; }
		reference zyx() { return static_cast<Derived*>(this)->data[6]; }
		const_reference zyx() const { return static_cast<const Derived*>(this)->data[6]; }
		reference wyx() { return static_cast<Derived*>(this)->data[7]; }
		const_reference wyx() const { return static_cast<const Derived*>(this)->data[7]; }
		reference xzx() { return static_cast<Derived*>(this)->data[8]; }
		const_reference xzx() const { return static_cast<const Derived*>(this)->data[8]; }
		reference yzx() { return static_cast<Derived*>(this)->data[9]; }
		const_reference yzx() const { return static_cast<const Derived*>(this)->data[9]; }
		reference zzx() { return static_cast<Derived*>(this)->data[10]; }
		const_reference zzx() const { return static_cast<const Derived*>(this)->data[10]; }
		reference wzx() { return static_cast<Derived*>(this)->data[11]; }
		const_reference wzx() const { return static_cast<const Derived*>(this)->data[11]; }
		reference xwx() { return static_cast<Derived*>(this)->data[12]; }
		const_reference xwx() const { return static_cast<const Derived*>(this)->data[12]; }
		reference ywx() { return static_cast<Derived*>(this)->data[13]; }
		const_reference ywx() const { return static_cast<const Derived*>(this)->data[13]; }
		reference zwx() { return static_cast<Derived*>(this)->data[14]; }
		const_reference zwx() const { return static_cast<const Derived*>(this)->data[14]; }
		reference wwx() { return static_cast<Derived*>(this)->data[15]; }
		const_reference wwx() const { return static_cast<const Derived*>(this)->data[15]; }
		reference xxy() { return static_cast<Derived*>(this)->data[16]; }
		const_reference xxy() const { return static_cast<const Derived*>(this)->data[16]; }
		reference yxy() { return static_cast<Derived*>(this)->data[17]; }
		const_reference yxy() const { return static_cast<const Derived*>(this)->data[17]; }
		reference zxy() { return static_cast<Derived*>(this)->data[18]; }
		const_reference zxy() const { return static_cast<const Derived*>(this)->data[18]; }
		reference wxy() { return static_cast<Derived*>(this)->data[19]; }
		const_reference wxy() const { return static_cast<const Derived*>(this)->data[19]; }
		reference xyy() { return static_cast<Derived*>(this)->data[20]; }
		const_reference xyy() const { return static_cast<const Derived*>(this)->data[20]; }
		reference yyy() { return static_cast<Derived*>(this)->data[21]; }
		const_reference yyy() const { return static_cast<const Derived*>(this)->data[21]; }
		reference zyy() { return static_cast<Derived*>(this)->data[22]; }
		const_reference zyy() const { return static_cast<const Derived*>(this)->data[22]; }
		reference wyy() { return static_cast<Derived*>(this)->data[23]; }
		const_reference wyy() const { return static_cast<const Derived*>(this)->data[23]; }
		reference xzy() { return static_cast<Derived*>(this)->data[24]; }
		const_reference xzy() const { return static_cast<const Derived*>(this)->data[24]; }
		reference yzy() { return static_cast<Derived*>(this)->data[25]; }
		const_reference yzy() const { return static_cast<const Derived*>(this)->data[25]; }
		reference zzy() { return static_cast<Derived*>(this)->data[26]; }
		const_reference zzy() const { return static_cast<const Derived*>(this)->data[26]; }
		reference wzy() { return static_cast<Derived*>(this)->data[27]; }
		const_reference wzy() const { return static_cast<const Derived*>(this)->data[27]; }
		reference xwy() { return static_cast<Derived*>(this)->data[28]; }
		const_reference xwy() const { return static_cast<const Derived*>(this)->data[28]; }
		reference ywy() { return static_cast<Derived*>(this)->data[29]; }
		const_reference ywy() const { return static_cast<const Derived*>(this)->data[29]; }
		reference zwy() { return static_cast<Derived*>(this)->data[30]; }
		const_reference zwy() const { return static_cast<const Derived*>(this)->data[30]; }
		reference wwy() { return static_cast<Derived*>(this)->data[31]; }
		const_reference wwy() const { return static_cast<const Derived*>(this)->data[31]; }
		reference xxz() { return static_cast<Derived*>(this)->data[32]; }
		const_reference xxz() const { return static_cast<const Derived*>(this)->data[32]; }
		reference yxz() { return static_cast<Derived*>(this)->data[33]; }
		const_reference yxz() const { return static_cast<const Derived*>(this)->data[33]; }
		reference zxz() { return static_cast<Derived*>(this)->data[34]; }
		const_reference zxz() const { return static_cast<const Derived*>(this)->data[34]; }
		reference wxz() { return static_cast<Derived*>(this)->data[35]; }
		const_reference wxz() const { return static_cast<const Derived*>(this)->data[35]; }
		reference xyz() { return static_cast<Derived*>(this)->data[36]; }
		const_reference xyz() const { return static_cast<const Derived*>(this)->data[36]; }
		reference yyz() { return static_cast<Derived*>(this)->data[37]; }
		const_reference yyz() const { return static_cast<const Derived*>(this)->data[37]; }
		reference zyz() { return static_cast<Derived*>(this)->data[38]; }
		const_reference zyz() const { return static_cast<const Derived*>(this)->data[38]; }
		reference wyz() { return static_cast<Derived*>(this)->data[39]; }
		const_reference wyz() const { return static_cast<const Derived*>(this)->data[39]; }
		reference xzz() { return static_cast<Derived*>(this)->data[40]; }
		const_reference xzz() const { return static_cast<const Derived*>(this)->data[40]; }
		reference yzz() { return static_cast<Derived*>(this)->data[41]; }
		const_reference yzz() const { return static_cast<const Derived*>(this)->data[41]; }
		reference zzz() { return static_cast<Derived*>(this)->data[42]; }
		const_reference zzz() const { return static_cast<const Derived*>(this)->data[42]; }
		reference wzz() { return static_cast<Derived*>(this)->data[43]; }
		const_reference wzz() const { return static_cast<const Derived*>(this)->data[43]; }
		reference xwz() { return static_cast<Derived*>(this)->data[44]; }
		const_reference xwz() const { return static_cast<const Derived*>(this)->data[44]; }
		reference ywz() { return static_cast<Derived*>(this)->data[45]; }
		const_reference ywz() const { return static_cast<const Derived*>(this)->data[45]; }
		reference zwz() { return static_cast<Derived*>(this)->data[46]; }
		const_reference zwz() const { return static_cast<const Derived*>(this)->data[46]; }
		reference wwz() { return static_cast<Derived*>(this)->data[47]; }
		const_reference wwz() const { return static_cast<const Derived*>(this)->data[47]; }
		reference xxw() { return static_cast<Derived*>(this)->data[48]; }
		const_reference xxw() const { return static_cast<const Derived*>(this)->data[48]; }
		reference yxw() { return static_cast<Derived*>(this)->data[49]; }
		const_reference yxw() const { return static_cast<const Derived*>(this)->data[49]; }
		reference zxw() { return static_cast<Derived*>(this)->data[50]; }
		const_reference zxw() const { return static_cast<const Derived*>(this)->data[50]; }
		reference wxw() { return static_cast<Derived*>(this)->data[51]; }
		const_reference wxw() const { return static_cast<const Derived*>(this)->data[51]; }
		reference xyw() { return static_cast<Derived*>(this)->data[52]; }
		const_reference xyw() const { return static_cast<const Derived*>(this)->data[52]; }
		reference yyw() { return static_cast<Derived*>(this)->data[53]; }
		const_reference yyw() const { return static_cast<const Derived*>(this)->data[53]; }
		reference zyw() { return static_cast<Derived*>(this)->data[54]; }
		const_reference zyw() const { return static_cast<const Derived*>(this)->data[54]; }
		reference wyw() { return static_cast<Derived*>(this)->data[55]; }
		const_reference wyw() const { return static_cast<const Derived*>(this)->data[55]; }
		reference xzw() { return static_cast<Derived*>(this)->data[56]; }
		const_reference xzw() const { return static_cast<const Derived*>(this)->data[56]; }
		reference yzw() { return static_cast<Derived*>(this)->data[57]; }
		const_reference yzw() const { return static_cast<const Derived*>(this)->data[57]; }
		reference zzw() { return static_cast<Derived*>(this)->data[58]; }
		const_reference zzw() const { return static_cast<const Derived*>(this)->data[58]; }
		reference wzw() { return static_cast<Derived*>(this)->data[59]; }
		const_reference wzw() const { return static_cast<const Derived*>(this)->data[59]; }
		reference xww() { return static_cast<Derived*>(this)->data[60]; }
		const_reference xww() const { return static_cast<const Derived*>(this)->data[60]; }
		reference yww() { return static_cast<Derived*>(this)->data[61]; }
		const_reference yww() const { return static_cast<const Derived*>(this)->data[61]; }
		reference zww() { return static_cast<Derived*>(this)->data[62]; }
		const_reference zww() const { return static_cast<const Derived*>(this)->data[62]; }
		reference www() { return static_cast<Derived*>(this)->data[63]; }
		const_reference www() const { return static_cast<const Derived*>(this)->data[63]; }
	};

	// Rank = 4, N = 1
	template<typename Derived, typename T>
	struct tensor_access<Derived,T,1,1,1,1>
	{
		using reference = typename std::array<T,1>::reference;
		using const_reference=typename std::array<T,1>::const_reference;
		reference xxxx() { return static_cast<Derived*>(this)->data[0]; }
		const_reference xxxx() const { return static_cast<const Derived*>(this)->data[0]; }
	};

	// Rank = 4, N = 2
	template<typename Derived, typename T>
	struct tensor_access<Derived,T,2,2,2,2>
	{
		using reference = typename std::array<T,1>::reference;
		using const_reference=typename std::array<T,1>::const_reference;
		reference xxxx() { return static_cast<Derived*>(this)->data[0]; }
		const_reference xxxx() const { return static_cast<const Derived*>(this)->data[0]; }
		reference yxxx() { return static_cast<Derived*>(this)->data[1]; }
		const_reference yxxx() const { return static_cast<const Derived*>(this)->data[1]; }
		reference xyxx() { return static_cast<Derived*>(this)->data[2]; }
		const_reference xyxx() const { return static_cast<const Derived*>(this)->data[2]; }
		reference yyxx() { return static_cast<Derived*>(this)->data[3]; }
		const_reference yyxx() const { return static_cast<const Derived*>(this)->data[3]; }
		reference xxyx() { return static_cast<Derived*>(this)->data[4]; }
		const_reference xxyx() const { return static_cast<const Derived*>(this)->data[4]; }
		reference yxyx() { return static_cast<Derived*>(this)->data[5]; }
		const_reference yxyx() const { return static_cast<const Derived*>(this)->data[5]; }
		reference xyyx() { return static_cast<Derived*>(this)->data[6]; }
		const_reference xyyx() const { return static_cast<const Derived*>(this)->data[6]; }
		reference yyyx() { return static_cast<Derived*>(this)->data[7]; }
		const_reference yyyx() const { return static_cast<const Derived*>(this)->data[7]; }
		reference xxxy() { return static_cast<Derived*>(this)->data[8]; }
		const_reference xxxy() const { return static_cast<const Derived*>(this)->data[8]; }
		reference yxxy() { return static_cast<Derived*>(this)->data[9]; }
		const_reference yxxy() const { return static_cast<const Derived*>(this)->data[9]; }
		reference xyxy() { return static_cast<Derived*>(this)->data[10]; }
		const_reference xyxy() const { return static_cast<const Derived*>(this)->data[10]; }
		reference yyxy() { return static_cast<Derived*>(this)->data[11]; }
		const_reference yyxy() const { return static_cast<const Derived*>(this)->data[11]; }
		reference xxyy() { return static_cast<Derived*>(this)->data[12]; }
		const_reference xxyy() const { return static_cast<const Derived*>(this)->data[12]; }
		reference yxyy() { return static_cast<Derived*>(this)->data[13]; }
		const_reference yxyy() const { return static_cast<const Derived*>(this)->data[13]; }
		reference xyyy() { return static_cast<Derived*>(this)->data[14]; }
		const_reference xyyy() const { return static_cast<const Derived*>(this)->data[14]; }
		reference yyyy() { return static_cast<Derived*>(this)->data[15]; }
		const_reference yyyy() const { return static_cast<const Derived*>(this)->data[15]; }
	};

	// Rank = 4, N = 3
	template<typename Derived, typename T>
	struct tensor_access<Derived,T,3,3,3,3>
	{
		using reference = typename std::array<T,1>::reference;
		using const_reference=typename std::array<T,1>::const_reference;
		reference xxxx() { return static_cast<Derived*>(this)->data[0]; }
		const_reference xxxx() const { return static_cast<const Derived*>(this)->data[0]; }
		reference yxxx() { return static_cast<Derived*>(this)->data[1]; }
		const_reference yxxx() const { return static_cast<const Derived*>(this)->data[1]; }
		reference zxxx() { return static_cast<Derived*>(this)->data[2]; }
		const_reference zxxx() const { return static_cast<const Derived*>(this)->data[2]; }
		reference xyxx() { return static_cast<Derived*>(this)->data[3]; }
		const_reference xyxx() const { return static_cast<const Derived*>(this)->data[3]; }
		reference yyxx() { return static_cast<Derived*>(this)->data[4]; }
		const_reference yyxx() const { return static_cast<const Derived*>(this)->data[4]; }
		reference zyxx() { return static_cast<Derived*>(this)->data[5]; }
		const_reference zyxx() const { return static_cast<const Derived*>(this)->data[5]; }
		reference xzxx() { return static_cast<Derived*>(this)->data[6]; }
		const_reference xzxx() const { return static_cast<const Derived*>(this)->data[6]; }
		reference yzxx() { return static_cast<Derived*>(this)->data[7]; }
		const_reference yzxx() const { return static_cast<const Derived*>(this)->data[7]; }
		reference zzxx() { return static_cast<Derived*>(this)->data[8]; }
		const_reference zzxx() const { return static_cast<const Derived*>(this)->data[8]; }
		reference xxyx() { return static_cast<Derived*>(this)->data[9]; }
		const_reference xxyx() const { return static_cast<const Derived*>(this)->data[9]; }
		reference yxyx() { return static_cast<Derived*>(this)->data[10]; }
		const_reference yxyx() const { return static_cast<const Derived*>(this)->data[10]; }
		reference zxyx() { return static_cast<Derived*>(this)->data[11]; }
		const_reference zxyx() const { return static_cast<const Derived*>(this)->data[11]; }
		reference xyyx() { return static_cast<Derived*>(this)->data[12]; }
		const_reference xyyx() const { return static_cast<const Derived*>(this)->data[12]; }
		reference yyyx() { return static_cast<Derived*>(this)->data[13]; }
		const_reference yyyx() const { return static_cast<const Derived*>(this)->data[13]; }
		reference zyyx() { return static_cast<Derived*>(this)->data[14]; }
		const_reference zyyx() const { return static_cast<const Derived*>(this)->data[14]; }
		reference xzyx() { return static_cast<Derived*>(this)->data[15]; }
		const_reference xzyx() const { return static_cast<const Derived*>(this)->data[15]; }
		reference yzyx() { return static_cast<Derived*>(this)->data[16]; }
		const_reference yzyx() const { return static_cast<const Derived*>(this)->data[16]; }
		reference zzyx() { return static_cast<Derived*>(this)->data[17]; }
		const_reference zzyx() const { return static_cast<const Derived*>(this)->data[17]; }
		reference xxzx() { return static_cast<Derived*>(this)->data[18]; }
		const_reference xxzx() const { return static_cast<const Derived*>(this)->data[18]; }
		reference yxzx() { return static_cast<Derived*>(this)->data[19]; }
		const_reference yxzx() const { return static_cast<const Derived*>(this)->data[19]; }
		reference zxzx() { return static_cast<Derived*>(this)->data[20]; }
		const_reference zxzx() const { return static_cast<const Derived*>(this)->data[20]; }
		reference xyzx() { return static_cast<Derived*>(this)->data[21]; }
		const_reference xyzx() const { return static_cast<const Derived*>(this)->data[21]; }
		reference yyzx() { return static_cast<Derived*>(this)->data[22]; }
		const_reference yyzx() const { return static_cast<const Derived*>(this)->data[22]; }
		reference zyzx() { return static_cast<Derived*>(this)->data[23]; }
		const_reference zyzx() const { return static_cast<const Derived*>(this)->data[23]; }
		reference xzzx() { return static_cast<Derived*>(this)->data[24]; }
		const_reference xzzx() const { return static_cast<const Derived*>(this)->data[24]; }
		reference yzzx() { return static_cast<Derived*>(this)->data[25]; }
		const_reference yzzx() const { return static_cast<const Derived*>(this)->data[25]; }
		reference zzzx() { return static_cast<Derived*>(this)->data[26]; }
		const_reference zzzx() const { return static_cast<const Derived*>(this)->data[26]; }
		reference xxxy() { return static_cast<Derived*>(this)->data[27]; }
		const_reference xxxy() const { return static_cast<const Derived*>(this)->data[27]; }
		reference yxxy() { return static_cast<Derived*>(this)->data[28]; }
		const_reference yxxy() const { return static_cast<const Derived*>(this)->data[28]; }
		reference zxxy() { return static_cast<Derived*>(this)->data[29]; }
		const_reference zxxy() const { return static_cast<const Derived*>(this)->data[29]; }
		reference xyxy() { return static_cast<Derived*>(this)->data[30]; }
		const_reference xyxy() const { return static_cast<const Derived*>(this)->data[30]; }
		reference yyxy() { return static_cast<Derived*>(this)->data[31]; }
		const_reference yyxy() const { return static_cast<const Derived*>(this)->data[31]; }
		reference zyxy() { return static_cast<Derived*>(this)->data[32]; }
		const_reference zyxy() const { return static_cast<const Derived*>(this)->data[32]; }
		reference xzxy() { return static_cast<Derived*>(this)->data[33]; }
		const_reference xzxy() const { return static_cast<const Derived*>(this)->data[33]; }
		reference yzxy() { return static_cast<Derived*>(this)->data[34]; }
		const_reference yzxy() const { return static_cast<const Derived*>(this)->data[34]; }
		reference zzxy() { return static_cast<Derived*>(this)->data[35]; }
		const_reference zzxy() const { return static_cast<const Derived*>(this)->data[35]; }
		reference xxyy() { return static_cast<Derived*>(this)->data[36]; }
		const_reference xxyy() const { return static_cast<const Derived*>(this)->data[36]; }
		reference yxyy() { return static_cast<Derived*>(this)->data[37]; }
		const_reference yxyy() const { return static_cast<const Derived*>(this)->data[37]; }
		reference zxyy() { return static_cast<Derived*>(this)->data[38]; }
		const_reference zxyy() const { return static_cast<const Derived*>(this)->data[38]; }
		reference xyyy() { return static_cast<Derived*>(this)->data[39]; }
		const_reference xyyy() const { return static_cast<const Derived*>(this)->data[39]; }
		reference yyyy() { return static_cast<Derived*>(this)->data[40]; }
		const_reference yyyy() const { return static_cast<const Derived*>(this)->data[40]; }
		reference zyyy() { return static_cast<Derived*>(this)->data[41]; }
		const_reference zyyy() const { return static_cast<const Derived*>(this)->data[41]; }
		reference xzyy() { return static_cast<Derived*>(this)->data[42]; }
		const_reference xzyy() const { return static_cast<const Derived*>(this)->data[42]; }
		reference yzyy() { return static_cast<Derived*>(this)->data[43]; }
		const_reference yzyy() const { return static_cast<const Derived*>(this)->data[43]; }
		reference zzyy() { return static_cast<Derived*>(this)->data[44]; }
		const_reference zzyy() const { return static_cast<const Derived*>(this)->data[44]; }
		reference xxzy() { return static_cast<Derived*>(this)->data[45]; }
		const_reference xxzy() const { return static_cast<const Derived*>(this)->data[45]; }
		reference yxzy() { return static_cast<Derived*>(this)->data[46]; }
		const_reference yxzy() const { return static_cast<const Derived*>(this)->data[46]; }
		reference zxzy() { return static_cast<Derived*>(this)->data[47]; }
		const_reference zxzy() const { return static_cast<const Derived*>(this)->data[47]; }
		reference xyzy() { return static_cast<Derived*>(this)->data[48]; }
		const_reference xyzy() const { return static_cast<const Derived*>(this)->data[48]; }
		reference yyzy() { return static_cast<Derived*>(this)->data[49]; }
		const_reference yyzy() const { return static_cast<const Derived*>(this)->data[49]; }
		reference zyzy() { return static_cast<Derived*>(this)->data[50]; }
		const_reference zyzy() const { return static_cast<const Derived*>(this)->data[50]; }
		reference xzzy() { return static_cast<Derived*>(this)->data[51]; }
		const_reference xzzy() const { return static_cast<const Derived*>(this)->data[51]; }
		reference yzzy() { return static_cast<Derived*>(this)->data[52]; }
		const_reference yzzy() const { return static_cast<const Derived*>(this)->data[52]; }
		reference zzzy() { return static_cast<Derived*>(this)->data[53]; }
		const_reference zzzy() const { return static_cast<const Derived*>(this)->data[53]; }
		reference xxxz() { return static_cast<Derived*>(this)->data[54]; }
		const_reference xxxz() const { return static_cast<const Derived*>(this)->data[54]; }
		reference yxxz() { return static_cast<Derived*>(this)->data[55]; }
		const_reference yxxz() const { return static_cast<const Derived*>(this)->data[55]; }
		reference zxxz() { return static_cast<Derived*>(this)->data[56]; }
		const_reference zxxz() const { return static_cast<const Derived*>(this)->data[56]; }
		reference xyxz() { return static_cast<Derived*>(this)->data[57]; }
		const_reference xyxz() const { return static_cast<const Derived*>(this)->data[57]; }
		reference yyxz() { return static_cast<Derived*>(this)->data[58]; }
		const_reference yyxz() const { return static_cast<const Derived*>(this)->data[58]; }
		reference zyxz() { return static_cast<Derived*>(this)->data[59]; }
		const_reference zyxz() const { return static_cast<const Derived*>(this)->data[59]; }
		reference xzxz() { return static_cast<Derived*>(this)->data[60]; }
		const_reference xzxz() const { return static_cast<const Derived*>(this)->data[60]; }
		reference yzxz() { return static_cast<Derived*>(this)->data[61]; }
		const_reference yzxz() const { return static_cast<const Derived*>(this)->data[61]; }
		reference zzxz() { return static_cast<Derived*>(this)->data[62]; }
		const_reference zzxz() const { return static_cast<const Derived*>(this)->data[62]; }
		reference xxyz() { return static_cast<Derived*>(this)->data[63]; }
		const_reference xxyz() const { return static_cast<const Derived*>(this)->data[63]; }
		reference yxyz() { return static_cast<Derived*>(this)->data[64]; }
		const_reference yxyz() const { return static_cast<const Derived*>(this)->data[64]; }
		reference zxyz() { return static_cast<Derived*>(this)->data[65]; }
		const_reference zxyz() const { return static_cast<const Derived*>(this)->data[65]; }
		reference xyyz() { return static_cast<Derived*>(this)->data[66]; }
		const_reference xyyz() const { return static_cast<const Derived*>(this)->data[66]; }
		reference yyyz() { return static_cast<Derived*>(this)->data[67]; }
		const_reference yyyz() const { return static_cast<const Derived*>(this)->data[67]; }
		reference zyyz() { return static_cast<Derived*>(this)->data[68]; }
		const_reference zyyz() const { return static_cast<const Derived*>(this)->data[68]; }
		reference xzyz() { return static_cast<Derived*>(this)->data[69]; }
		const_reference xzyz() const { return static_cast<const Derived*>(this)->data[69]; }
		reference yzyz() { return static_cast<Derived*>(this)->data[70]; }
		const_reference yzyz() const { return static_cast<const Derived*>(this)->data[70]; }
		reference zzyz() { return static_cast<Derived*>(this)->data[71]; }
		const_reference zzyz() const { return static_cast<const Derived*>(this)->data[71]; }
		reference xxzz() { return static_cast<Derived*>(this)->data[72]; }
		const_reference xxzz() const { return static_cast<const Derived*>(this)->data[72]; }
		reference yxzz() { return static_cast<Derived*>(this)->data[73]; }
		const_reference yxzz() const { return static_cast<const Derived*>(this)->data[73]; }
		reference zxzz() { return static_cast<Derived*>(this)->data[74]; }
		const_reference zxzz() const { return static_cast<const Derived*>(this)->data[74]; }
		reference xyzz() { return static_cast<Derived*>(this)->data[75]; }
		const_reference xyzz() const { return static_cast<const Derived*>(this)->data[75]; }
		reference yyzz() { return static_cast<Derived*>(this)->data[76]; }
		const_reference yyzz() const { return static_cast<const Derived*>(this)->data[76]; }
		reference zyzz() { return static_cast<Derived*>(this)->data[77]; }
		const_reference zyzz() const { return static_cast<const Derived*>(this)->data[77]; }
		reference xzzz() { return static_cast<Derived*>(this)->data[78]; }
		const_reference xzzz() const { return static_cast<const Derived*>(this)->data[78]; }
		reference yzzz() { return static_cast<Derived*>(this)->data[79]; }
		const_reference yzzz() const { return static_cast<const Derived*>(this)->data[79]; }
		reference zzzz() { return static_cast<Derived*>(this)->data[80]; }
		const_reference zzzz() const { return static_cast<const Derived*>(this)->data[80]; }
	};

	// Rank = 4, N = 4
	template<typename Derived, typename T>
	struct tensor_access<Derived,T,4,4,4,4>
	{
		using reference = typename std::array<T,1>::reference;
		using const_reference=typename std::array<T,1>::const_reference;
		reference xxxx() { return static_cast<Derived*>(this)->data[0]; }
		const_reference xxxx() const { return static_cast<const Derived*>(this)->data[0]; }
		reference yxxx() { return static_cast<Derived*>(this)->data[1]; }
		const_reference yxxx() const { return static_cast<const Derived*>(this)->data[1]; }
		reference zxxx() { return static_cast<Derived*>(this)->data[2]; }
		const_reference zxxx() const { return static_cast<const Derived*>(this)->data[2]; }
		reference wxxx() { return static_cast<Derived*>(this)->data[3]; }
		const_reference wxxx() const { return static_cast<const Derived*>(this)->data[3]; }
		reference xyxx() { return static_cast<Derived*>(this)->data[4]; }
		const_reference xyxx() const { return static_cast<const Derived*>(this)->data[4]; }
		reference yyxx() { return static_cast<Derived*>(this)->data[5]; }
		const_reference yyxx() const { return static_cast<const Derived*>(this)->data[5]; }
		reference zyxx() { return static_cast<Derived*>(this)->data[6]; }
		const_reference zyxx() const { return static_cast<const Derived*>(this)->data[6]; }
		reference wyxx() { return static_cast<Derived*>(this)->data[7]; }
		const_reference wyxx() const { return static_cast<const Derived*>(this)->data[7]; }
		reference xzxx() { return static_cast<Derived*>(this)->data[8]; }
		const_reference xzxx() const { return static_cast<const Derived*>(this)->data[8]; }
		reference yzxx() { return static_cast<Derived*>(this)->data[9]; }
		const_reference yzxx() const { return static_cast<const Derived*>(this)->data[9]; }
		reference zzxx() { return static_cast<Derived*>(this)->data[10]; }
		const_reference zzxx() const { return static_cast<const Derived*>(this)->data[10]; }
		reference wzxx() { return static_cast<Derived*>(this)->data[11]; }
		const_reference wzxx() const { return static_cast<const Derived*>(this)->data[11]; }
		reference xwxx() { return static_cast<Derived*>(this)->data[12]; }
		const_reference xwxx() const { return static_cast<const Derived*>(this)->data[12]; }
		reference ywxx() { return static_cast<Derived*>(this)->data[13]; }
		const_reference ywxx() const { return static_cast<const Derived*>(this)->data[13]; }
		reference zwxx() { return static_cast<Derived*>(this)->data[14]; }
		const_reference zwxx() const { return static_cast<const Derived*>(this)->data[14]; }
		reference wwxx() { return static_cast<Derived*>(this)->data[15]; }
		const_reference wwxx() const { return static_cast<const Derived*>(this)->data[15]; }
		reference xxyx() { return static_cast<Derived*>(this)->data[16]; }
		const_reference xxyx() const { return static_cast<const Derived*>(this)->data[16]; }
		reference yxyx() { return static_cast<Derived*>(this)->data[17]; }
		const_reference yxyx() const { return static_cast<const Derived*>(this)->data[17]; }
		reference zxyx() { return static_cast<Derived*>(this)->data[18]; }
		const_reference zxyx() const { return static_cast<const Derived*>(this)->data[18]; }
		reference wxyx() { return static_cast<Derived*>(this)->data[19]; }
		const_reference wxyx() const { return static_cast<const Derived*>(this)->data[19]; }
		reference xyyx() { return static_cast<Derived*>(this)->data[20]; }
		const_reference xyyx() const { return static_cast<const Derived*>(this)->data[20]; }
		reference yyyx() { return static_cast<Derived*>(this)->data[21]; }
		const_reference yyyx() const { return static_cast<const Derived*>(this)->data[21]; }
		reference zyyx() { return static_cast<Derived*>(this)->data[22]; }
		const_reference zyyx() const { return static_cast<const Derived*>(this)->data[22]; }
		reference wyyx() { return static_cast<Derived*>(this)->data[23]; }
		const_reference wyyx() const { return static_cast<const Derived*>(this)->data[23]; }
		reference xzyx() { return static_cast<Derived*>(this)->data[24]; }
		const_reference xzyx() const { return static_cast<const Derived*>(this)->data[24]; }
		reference yzyx() { return static_cast<Derived*>(this)->data[25]; }
		const_reference yzyx() const { return static_cast<const Derived*>(this)->data[25]; }
		reference zzyx() { return static_cast<Derived*>(this)->data[26]; }
		const_reference zzyx() const { return static_cast<const Derived*>(this)->data[26]; }
		reference wzyx() { return static_cast<Derived*>(this)->data[27]; }
		const_reference wzyx() const { return static_cast<const Derived*>(this)->data[27]; }
		reference xwyx() { return static_cast<Derived*>(this)->data[28]; }
		const_reference xwyx() const { return static_cast<const Derived*>(this)->data[28]; }
		reference ywyx() { return static_cast<Derived*>(this)->data[29]; }
		const_reference ywyx() const { return static_cast<const Derived*>(this)->data[29]; }
		reference zwyx() { return static_cast<Derived*>(this)->data[30]; }
		const_reference zwyx() const { return static_cast<const Derived*>(this)->data[30]; }
		reference wwyx() { return static_cast<Derived*>(this)->data[31]; }
		const_reference wwyx() const { return static_cast<const Derived*>(this)->data[31]; }
		reference xxzx() { return static_cast<Derived*>(this)->data[32]; }
		const_reference xxzx() const { return static_cast<const Derived*>(this)->data[32]; }
		reference yxzx() { return static_cast<Derived*>(this)->data[33]; }
		const_reference yxzx() const { return static_cast<const Derived*>(this)->data[33]; }
		reference zxzx() { return static_cast<Derived*>(this)->data[34]; }
		const_reference zxzx() const { return static_cast<const Derived*>(this)->data[34]; }
		reference wxzx() { return static_cast<Derived*>(this)->data[35]; }
		const_reference wxzx() const { return static_cast<const Derived*>(this)->data[35]; }
		reference xyzx() { return static_cast<Derived*>(this)->data[36]; }
		const_reference xyzx() const { return static_cast<const Derived*>(this)->data[36]; }
		reference yyzx() { return static_cast<Derived*>(this)->data[37]; }
		const_reference yyzx() const { return static_cast<const Derived*>(this)->data[37]; }
		reference zyzx() { return static_cast<Derived*>(this)->data[38]; }
		const_reference zyzx() const { return static_cast<const Derived*>(this)->data[38]; }
		reference wyzx() { return static_cast<Derived*>(this)->data[39]; }
		const_reference wyzx() const { return static_cast<const Derived*>(this)->data[39]; }
		reference xzzx() { return static_cast<Derived*>(this)->data[40]; }
		const_reference xzzx() const { return static_cast<const Derived*>(this)->data[40]; }
		reference yzzx() { return static_cast<Derived*>(this)->data[41]; }
		const_reference yzzx() const { return static_cast<const Derived*>(this)->data[41]; }
		reference zzzx() { return static_cast<Derived*>(this)->data[42]; }
		const_reference zzzx() const { return static_cast<const Derived*>(this)->data[42]; }
		reference wzzx() { return static_cast<Derived*>(this)->data[43]; }
		const_reference wzzx() const { return static_cast<const Derived*>(this)->data[43]; }
		reference xwzx() { return static_cast<Derived*>(this)->data[44]; }
		const_reference xwzx() const { return static_cast<const Derived*>(this)->data[44]; }
		reference ywzx() { return static_cast<Derived*>(this)->data[45]; }
		const_reference ywzx() const { return static_cast<const Derived*>(this)->data[45]; }
		reference zwzx() { return static_cast<Derived*>(this)->data[46]; }
		const_reference zwzx() const { return static_cast<const Derived*>(this)->data[46]; }
		reference wwzx() { return static_cast<Derived*>(this)->data[47]; }
		const_reference wwzx() const { return static_cast<const Derived*>(this)->data[47]; }
		reference xxwx() { return static_cast<Derived*>(this)->data[48]; }
		const_reference xxwx() const { return static_cast<const Derived*>(this)->data[48]; }
		reference yxwx() { return static_cast<Derived*>(this)->data[49]; }
		const_reference yxwx() const { return static_cast<const Derived*>(this)->data[49]; }
		reference zxwx() { return static_cast<Derived*>(this)->data[50]; }
		const_reference zxwx() const { return static_cast<const Derived*>(this)->data[50]; }
		reference wxwx() { return static_cast<Derived*>(this)->data[51]; }
		const_reference wxwx() const { return static_cast<const Derived*>(this)->data[51]; }
		reference xywx() { return static_cast<Derived*>(this)->data[52]; }
		const_reference xywx() const { return static_cast<const Derived*>(this)->data[52]; }
		reference yywx() { return static_cast<Derived*>(this)->data[53]; }
		const_reference yywx() const { return static_cast<const Derived*>(this)->data[53]; }
		reference zywx() { return static_cast<Derived*>(this)->data[54]; }
		const_reference zywx() const { return static_cast<const Derived*>(this)->data[54]; }
		reference wywx() { return static_cast<Derived*>(this)->data[55]; }
		const_reference wywx() const { return static_cast<const Derived*>(this)->data[55]; }
		reference xzwx() { return static_cast<Derived*>(this)->data[56]; }
		const_reference xzwx() const { return static_cast<const Derived*>(this)->data[56]; }
		reference yzwx() { return static_cast<Derived*>(this)->data[57]; }
		const_reference yzwx() const { return static_cast<const Derived*>(this)->data[57]; }
		reference zzwx() { return static_cast<Derived*>(this)->data[58]; }
		const_reference zzwx() const { return static_cast<const Derived*>(this)->data[58]; }
		reference wzwx() { return static_cast<Derived*>(this)->data[59]; }
		const_reference wzwx() const { return static_cast<const Derived*>(this)->data[59]; }
		reference xwwx() { return static_cast<Derived*>(this)->data[60]; }
		const_reference xwwx() const { return static_cast<const Derived*>(this)->data[60]; }
		reference ywwx() { return static_cast<Derived*>(this)->data[61]; }
		const_reference ywwx() const { return static_cast<const Derived*>(this)->data[61]; }
		reference zwwx() { return static_cast<Derived*>(this)->data[62]; }
		const_reference zwwx() const { return static_cast<const Derived*>(this)->data[62]; }
		reference wwwx() { return static_cast<Derived*>(this)->data[63]; }
		const_reference wwwx() const { return static_cast<const Derived*>(this)->data[63]; }
		reference xxxy() { return static_cast<Derived*>(this)->data[64]; }
		const_reference xxxy() const { return static_cast<const Derived*>(this)->data[64]; }
		reference yxxy() { return static_cast<Derived*>(this)->data[65]; }
		const_reference yxxy() const { return static_cast<const Derived*>(this)->data[65]; }
		reference zxxy() { return static_cast<Derived*>(this)->data[66]; }
		const_reference zxxy() const { return static_cast<const Derived*>(this)->data[66]; }
		reference wxxy() { return static_cast<Derived*>(this)->data[67]; }
		const_reference wxxy() const { return static_cast<const Derived*>(this)->data[67]; }
		reference xyxy() { return static_cast<Derived*>(this)->data[68]; }
		const_reference xyxy() const { return static_cast<const Derived*>(this)->data[68]; }
		reference yyxy() { return static_cast<Derived*>(this)->data[69]; }
		const_reference yyxy() const { return static_cast<const Derived*>(this)->data[69]; }
		reference zyxy() { return static_cast<Derived*>(this)->data[70]; }
		const_reference zyxy() const { return static_cast<const Derived*>(this)->data[70]; }
		reference wyxy() { return static_cast<Derived*>(this)->data[71]; }
		const_reference wyxy() const { return static_cast<const Derived*>(this)->data[71]; }
		reference xzxy() { return static_cast<Derived*>(this)->data[72]; }
		const_reference xzxy() const { return static_cast<const Derived*>(this)->data[72]; }
		reference yzxy() { return static_cast<Derived*>(this)->data[73]; }
		const_reference yzxy() const { return static_cast<const Derived*>(this)->data[73]; }
		reference zzxy() { return static_cast<Derived*>(this)->data[74]; }
		const_reference zzxy() const { return static_cast<const Derived*>(this)->data[74]; }
		reference wzxy() { return static_cast<Derived*>(this)->data[75]; }
		const_reference wzxy() const { return static_cast<const Derived*>(this)->data[75]; }
		reference xwxy() { return static_cast<Derived*>(this)->data[76]; }
		const_reference xwxy() const { return static_cast<const Derived*>(this)->data[76]; }
		reference ywxy() { return static_cast<Derived*>(this)->data[77]; }
		const_reference ywxy() const { return static_cast<const Derived*>(this)->data[77]; }
		reference zwxy() { return static_cast<Derived*>(this)->data[78]; }
		const_reference zwxy() const { return static_cast<const Derived*>(this)->data[78]; }
		reference wwxy() { return static_cast<Derived*>(this)->data[79]; }
		const_reference wwxy() const { return static_cast<const Derived*>(this)->data[79]; }
		reference xxyy() { return static_cast<Derived*>(this)->data[80]; }
		const_reference xxyy() const { return static_cast<const Derived*>(this)->data[80]; }
		reference yxyy() { return static_cast<Derived*>(this)->data[81]; }
		const_reference yxyy() const { return static_cast<const Derived*>(this)->data[81]; }
		reference zxyy() { return static_cast<Derived*>(this)->data[82]; }
		const_reference zxyy() const { return static_cast<const Derived*>(this)->data[82]; }
		reference wxyy() { return static_cast<Derived*>(this)->data[83]; }
		const_reference wxyy() const { return static_cast<const Derived*>(this)->data[83]; }
		reference xyyy() { return static_cast<Derived*>(this)->data[84]; }
		const_reference xyyy() const { return static_cast<const Derived*>(this)->data[84]; }
		reference yyyy() { return static_cast<Derived*>(this)->data[85]; }
		const_reference yyyy() const { return static_cast<const Derived*>(this)->data[85]; }
		reference zyyy() { return static_cast<Derived*>(this)->data[86]; }
		const_reference zyyy() const { return static_cast<const Derived*>(this)->data[86]; }
		reference wyyy() { return static_cast<Derived*>(this)->data[87]; }
		const_reference wyyy() const { return static_cast<const Derived*>(this)->data[87]; }
		reference xzyy() { return static_cast<Derived*>(this)->data[88]; }
		const_reference xzyy() const { return static_cast<const Derived*>(this)->data[88]; }
		reference yzyy() { return static_cast<Derived*>(this)->data[89]; }
		const_reference yzyy() const { return static_cast<const Derived*>(this)->data[89]; }
		reference zzyy() { return static_cast<Derived*>(this)->data[90]; }
		const_reference zzyy() const { return static_cast<const Derived*>(this)->data[90]; }
		reference wzyy() { return static_cast<Derived*>(this)->data[91]; }
		const_reference wzyy() const { return static_cast<const Derived*>(this)->data[91]; }
		reference xwyy() { return static_cast<Derived*>(this)->data[92]; }
		const_reference xwyy() const { return static_cast<const Derived*>(this)->data[92]; }
		reference ywyy() { return static_cast<Derived*>(this)->data[93]; }
		const_reference ywyy() const { return static_cast<const Derived*>(this)->data[93]; }
		reference zwyy() { return static_cast<Derived*>(this)->data[94]; }
		const_reference zwyy() const { return static_cast<const Derived*>(this)->data[94]; }
		reference wwyy() { return static_cast<Derived*>(this)->data[95]; }
		const_reference wwyy() const { return static_cast<const Derived*>(this)->data[95]; }
		reference xxzy() { return static_cast<Derived*>(this)->data[96]; }
		const_reference xxzy() const { return static_cast<const Derived*>(this)->data[96]; }
		reference yxzy() { return static_cast<Derived*>(this)->data[97]; }
		const_reference yxzy() const { return static_cast<const Derived*>(this)->data[97]; }
		reference zxzy() { return static_cast<Derived*>(this)->data[98]; }
		const_reference zxzy() const { return static_cast<const Derived*>(this)->data[98]; }
		reference wxzy() { return static_cast<Derived*>(this)->data[99]; }
		const_reference wxzy() const { return static_cast<const Derived*>(this)->data[99]; }
		reference xyzy() { return static_cast<Derived*>(this)->data[100]; }
		const_reference xyzy() const { return static_cast<const Derived*>(this)->data[100]; }
		reference yyzy() { return static_cast<Derived*>(this)->data[101]; }
		const_reference yyzy() const { return static_cast<const Derived*>(this)->data[101]; }
		reference zyzy() { return static_cast<Derived*>(this)->data[102]; }
		const_reference zyzy() const { return static_cast<const Derived*>(this)->data[102]; }
		reference wyzy() { return static_cast<Derived*>(this)->data[103]; }
		const_reference wyzy() const { return static_cast<const Derived*>(this)->data[103]; }
		reference xzzy() { return static_cast<Derived*>(this)->data[104]; }
		const_reference xzzy() const { return static_cast<const Derived*>(this)->data[104]; }
		reference yzzy() { return static_cast<Derived*>(this)->data[105]; }
		const_reference yzzy() const { return static_cast<const Derived*>(this)->data[105]; }
		reference zzzy() { return static_cast<Derived*>(this)->data[106]; }
		const_reference zzzy() const { return static_cast<const Derived*>(this)->data[106]; }
		reference wzzy() { return static_cast<Derived*>(this)->data[107]; }
		const_reference wzzy() const { return static_cast<const Derived*>(this)->data[107]; }
		reference xwzy() { return static_cast<Derived*>(this)->data[108]; }
		const_reference xwzy() const { return static_cast<const Derived*>(this)->data[108]; }
		reference ywzy() { return static_cast<Derived*>(this)->data[109]; }
		const_reference ywzy() const { return static_cast<const Derived*>(this)->data[109]; }
		reference zwzy() { return static_cast<Derived*>(this)->data[110]; }
		const_reference zwzy() const { return static_cast<const Derived*>(this)->data[110]; }
		reference wwzy() { return static_cast<Derived*>(this)->data[111]; }
		const_reference wwzy() const { return static_cast<const Derived*>(this)->data[111]; }
		reference xxwy() { return static_cast<Derived*>(this)->data[112]; }
		const_reference xxwy() const { return static_cast<const Derived*>(this)->data[112]; }
		reference yxwy() { return static_cast<Derived*>(this)->data[113]; }
		const_reference yxwy() const { return static_cast<const Derived*>(this)->data[113]; }
		reference zxwy() { return static_cast<Derived*>(this)->data[114]; }
		const_reference zxwy() const { return static_cast<const Derived*>(this)->data[114]; }
		reference wxwy() { return static_cast<Derived*>(this)->data[115]; }
		const_reference wxwy() const { return static_cast<const Derived*>(this)->data[115]; }
		reference xywy() { return static_cast<Derived*>(this)->data[116]; }
		const_reference xywy() const { return static_cast<const Derived*>(this)->data[116]; }
		reference yywy() { return static_cast<Derived*>(this)->data[117]; }
		const_reference yywy() const { return static_cast<const Derived*>(this)->data[117]; }
		reference zywy() { return static_cast<Derived*>(this)->data[118]; }
		const_reference zywy() const { return static_cast<const Derived*>(this)->data[118]; }
		reference wywy() { return static_cast<Derived*>(this)->data[119]; }
		const_reference wywy() const { return static_cast<const Derived*>(this)->data[119]; }
		reference xzwy() { return static_cast<Derived*>(this)->data[120]; }
		const_reference xzwy() const { return static_cast<const Derived*>(this)->data[120]; }
		reference yzwy() { return static_cast<Derived*>(this)->data[121]; }
		const_reference yzwy() const { return static_cast<const Derived*>(this)->data[121]; }
		reference zzwy() { return static_cast<Derived*>(this)->data[122]; }
		const_reference zzwy() const { return static_cast<const Derived*>(this)->data[122]; }
		reference wzwy() { return static_cast<Derived*>(this)->data[123]; }
		const_reference wzwy() const { return static_cast<const Derived*>(this)->data[123]; }
		reference xwwy() { return static_cast<Derived*>(this)->data[124]; }
		const_reference xwwy() const { return static_cast<const Derived*>(this)->data[124]; }
		reference ywwy() { return static_cast<Derived*>(this)->data[125]; }
		const_reference ywwy() const { return static_cast<const Derived*>(this)->data[125]; }
		reference zwwy() { return static_cast<Derived*>(this)->data[126]; }
		const_reference zwwy() const { return static_cast<const Derived*>(this)->data[126]; }
		reference wwwy() { return static_cast<Derived*>(this)->data[127]; }
		const_reference wwwy() const { return static_cast<const Derived*>(this)->data[127]; }
		reference xxxz() { return static_cast<Derived*>(this)->data[128]; }
		const_reference xxxz() const { return static_cast<const Derived*>(this)->data[128]; }
		reference yxxz() { return static_cast<Derived*>(this)->data[129]; }
		const_reference yxxz() const { return static_cast<const Derived*>(this)->data[129]; }
		reference zxxz() { return static_cast<Derived*>(this)->data[130]; }
		const_reference zxxz() const { return static_cast<const Derived*>(this)->data[130]; }
		reference wxxz() { return static_cast<Derived*>(this)->data[131]; }
		const_reference wxxz() const { return static_cast<const Derived*>(this)->data[131]; }
		reference xyxz() { return static_cast<Derived*>(this)->data[132]; }
		const_reference xyxz() const { return static_cast<const Derived*>(this)->data[132]; }
		reference yyxz() { return static_cast<Derived*>(this)->data[133]; }
		const_reference yyxz() const { return static_cast<const Derived*>(this)->data[133]; }
		reference zyxz() { return static_cast<Derived*>(this)->data[134]; }
		const_reference zyxz() const { return static_cast<const Derived*>(this)->data[134]; }
		reference wyxz() { return static_cast<Derived*>(this)->data[135]; }
		const_reference wyxz() const { return static_cast<const Derived*>(this)->data[135]; }
		reference xzxz() { return static_cast<Derived*>(this)->data[136]; }
		const_reference xzxz() const { return static_cast<const Derived*>(this)->data[136]; }
		reference yzxz() { return static_cast<Derived*>(this)->data[137]; }
		const_reference yzxz() const { return static_cast<const Derived*>(this)->data[137]; }
		reference zzxz() { return static_cast<Derived*>(this)->data[138]; }
		const_reference zzxz() const { return static_cast<const Derived*>(this)->data[138]; }
		reference wzxz() { return static_cast<Derived*>(this)->data[139]; }
		const_reference wzxz() const { return static_cast<const Derived*>(this)->data[139]; }
		reference xwxz() { return static_cast<Derived*>(this)->data[140]; }
		const_reference xwxz() const { return static_cast<const Derived*>(this)->data[140]; }
		reference ywxz() { return static_cast<Derived*>(this)->data[141]; }
		const_reference ywxz() const { return static_cast<const Derived*>(this)->data[141]; }
		reference zwxz() { return static_cast<Derived*>(this)->data[142]; }
		const_reference zwxz() const { return static_cast<const Derived*>(this)->data[142]; }
		reference wwxz() { return static_cast<Derived*>(this)->data[143]; }
		const_reference wwxz() const { return static_cast<const Derived*>(this)->data[143]; }
		reference xxyz() { return static_cast<Derived*>(this)->data[144]; }
		const_reference xxyz() const { return static_cast<const Derived*>(this)->data[144]; }
		reference yxyz() { return static_cast<Derived*>(this)->data[145]; }
		const_reference yxyz() const { return static_cast<const Derived*>(this)->data[145]; }
		reference zxyz() { return static_cast<Derived*>(this)->data[146]; }
		const_reference zxyz() const { return static_cast<const Derived*>(this)->data[146]; }
		reference wxyz() { return static_cast<Derived*>(this)->data[147]; }
		const_reference wxyz() const { return static_cast<const Derived*>(this)->data[147]; }
		reference xyyz() { return static_cast<Derived*>(this)->data[148]; }
		const_reference xyyz() const { return static_cast<const Derived*>(this)->data[148]; }
		reference yyyz() { return static_cast<Derived*>(this)->data[149]; }
		const_reference yyyz() const { return static_cast<const Derived*>(this)->data[149]; }
		reference zyyz() { return static_cast<Derived*>(this)->data[150]; }
		const_reference zyyz() const { return static_cast<const Derived*>(this)->data[150]; }
		reference wyyz() { return static_cast<Derived*>(this)->data[151]; }
		const_reference wyyz() const { return static_cast<const Derived*>(this)->data[151]; }
		reference xzyz() { return static_cast<Derived*>(this)->data[152]; }
		const_reference xzyz() const { return static_cast<const Derived*>(this)->data[152]; }
		reference yzyz() { return static_cast<Derived*>(this)->data[153]; }
		const_reference yzyz() const { return static_cast<const Derived*>(this)->data[153]; }
		reference zzyz() { return static_cast<Derived*>(this)->data[154]; }
		const_reference zzyz() const { return static_cast<const Derived*>(this)->data[154]; }
		reference wzyz() { return static_cast<Derived*>(this)->data[155]; }
		const_reference wzyz() const { return static_cast<const Derived*>(this)->data[155]; }
		reference xwyz() { return static_cast<Derived*>(this)->data[156]; }
		const_reference xwyz() const { return static_cast<const Derived*>(this)->data[156]; }
		reference ywyz() { return static_cast<Derived*>(this)->data[157]; }
		const_reference ywyz() const { return static_cast<const Derived*>(this)->data[157]; }
		reference zwyz() { return static_cast<Derived*>(this)->data[158]; }
		const_reference zwyz() const { return static_cast<const Derived*>(this)->data[158]; }
		reference wwyz() { return static_cast<Derived*>(this)->data[159]; }
		const_reference wwyz() const { return static_cast<const Derived*>(this)->data[159]; }
		reference xxzz() { return static_cast<Derived*>(this)->data[160]; }
		const_reference xxzz() const { return static_cast<const Derived*>(this)->data[160]; }
		reference yxzz() { return static_cast<Derived*>(this)->data[161]; }
		const_reference yxzz() const { return static_cast<const Derived*>(this)->data[161]; }
		reference zxzz() { return static_cast<Derived*>(this)->data[162]; }
		const_reference zxzz() const { return static_cast<const Derived*>(this)->data[162]; }
		reference wxzz() { return static_cast<Derived*>(this)->data[163]; }
		const_reference wxzz() const { return static_cast<const Derived*>(this)->data[163]; }
		reference xyzz() { return static_cast<Derived*>(this)->data[164]; }
		const_reference xyzz() const { return static_cast<const Derived*>(this)->data[164]; }
		reference yyzz() { return static_cast<Derived*>(this)->data[165]; }
		const_reference yyzz() const { return static_cast<const Derived*>(this)->data[165]; }
		reference zyzz() { return static_cast<Derived*>(this)->data[166]; }
		const_reference zyzz() const { return static_cast<const Derived*>(this)->data[166]; }
		reference wyzz() { return static_cast<Derived*>(this)->data[167]; }
		const_reference wyzz() const { return static_cast<const Derived*>(this)->data[167]; }
		reference xzzz() { return static_cast<Derived*>(this)->data[168]; }
		const_reference xzzz() const { return static_cast<const Derived*>(this)->data[168]; }
		reference yzzz() { return static_cast<Derived*>(this)->data[169]; }
		const_reference yzzz() const { return static_cast<const Derived*>(this)->data[169]; }
		reference zzzz() { return static_cast<Derived*>(this)->data[170]; }
		const_reference zzzz() const { return static_cast<const Derived*>(this)->data[170]; }
		reference wzzz() { return static_cast<Derived*>(this)->data[171]; }
		const_reference wzzz() const { return static_cast<const Derived*>(this)->data[171]; }
		reference xwzz() { return static_cast<Derived*>(this)->data[172]; }
		const_reference xwzz() const { return static_cast<const Derived*>(this)->data[172]; }
		reference ywzz() { return static_cast<Derived*>(this)->data[173]; }
		const_reference ywzz() const { return static_cast<const Derived*>(this)->data[173]; }
		reference zwzz() { return static_cast<Derived*>(this)->data[174]; }
		const_reference zwzz() const { return static_cast<const Derived*>(this)->data[174]; }
		reference wwzz() { return static_cast<Derived*>(this)->data[175]; }
		const_reference wwzz() const { return static_cast<const Derived*>(this)->data[175]; }
		reference xxwz() { return static_cast<Derived*>(this)->data[176]; }
		const_reference xxwz() const { return static_cast<const Derived*>(this)->data[176]; }
		reference yxwz() { return static_cast<Derived*>(this)->data[177]; }
		const_reference yxwz() const { return static_cast<const Derived*>(this)->data[177]; }
		reference zxwz() { return static_cast<Derived*>(this)->data[178]; }
		const_reference zxwz() const { return static_cast<const Derived*>(this)->data[178]; }
		reference wxwz() { return static_cast<Derived*>(this)->data[179]; }
		const_reference wxwz() const { return static_cast<const Derived*>(this)->data[179]; }
		reference xywz() { return static_cast<Derived*>(this)->data[180]; }
		const_reference xywz() const { return static_cast<const Derived*>(this)->data[180]; }
		reference yywz() { return static_cast<Derived*>(this)->data[181]; }
		const_reference yywz() const { return static_cast<const Derived*>(this)->data[181]; }
		reference zywz() { return static_cast<Derived*>(this)->data[182]; }
		const_reference zywz() const { return static_cast<const Derived*>(this)->data[182]; }
		reference wywz() { return static_cast<Derived*>(this)->data[183]; }
		const_reference wywz() const { return static_cast<const Derived*>(this)->data[183]; }
		reference xzwz() { return static_cast<Derived*>(this)->data[184]; }
		const_reference xzwz() const { return static_cast<const Derived*>(this)->data[184]; }
		reference yzwz() { return static_cast<Derived*>(this)->data[185]; }
		const_reference yzwz() const { return static_cast<const Derived*>(this)->data[185]; }
		reference zzwz() { return static_cast<Derived*>(this)->data[186]; }
		const_reference zzwz() const { return static_cast<const Derived*>(this)->data[186]; }
		reference wzwz() { return static_cast<Derived*>(this)->data[187]; }
		const_reference wzwz() const { return static_cast<const Derived*>(this)->data[187]; }
		reference xwwz() { return static_cast<Derived*>(this)->data[188]; }
		const_reference xwwz() const { return static_cast<const Derived*>(this)->data[188]; }
		reference ywwz() { return static_cast<Derived*>(this)->data[189]; }
		const_reference ywwz() const { return static_cast<const Derived*>(this)->data[189]; }
		reference zwwz() { return static_cast<Derived*>(this)->data[190]; }
		const_reference zwwz() const { return static_cast<const Derived*>(this)->data[190]; }
		reference wwwz() { return static_cast<Derived*>(this)->data[191]; }
		const_reference wwwz() const { return static_cast<const Derived*>(this)->data[191]; }
		reference xxxw() { return static_cast<Derived*>(this)->data[192]; }
		const_reference xxxw() const { return static_cast<const Derived*>(this)->data[192]; }
		reference yxxw() { return static_cast<Derived*>(this)->data[193]; }
		const_reference yxxw() const { return static_cast<const Derived*>(this)->data[193]; }
		reference zxxw() { return static_cast<Derived*>(this)->data[194]; }
		const_reference zxxw() const { return static_cast<const Derived*>(this)->data[194]; }
		reference wxxw() { return static_cast<Derived*>(this)->data[195]; }
		const_reference wxxw() const { return static_cast<const Derived*>(this)->data[195]; }
		reference xyxw() { return static_cast<Derived*>(this)->data[196]; }
		const_reference xyxw() const { return static_cast<const Derived*>(this)->data[196]; }
		reference yyxw() { return static_cast<Derived*>(this)->data[197]; }
		const_reference yyxw() const { return static_cast<const Derived*>(this)->data[197]; }
		reference zyxw() { return static_cast<Derived*>(this)->data[198]; }
		const_reference zyxw() const { return static_cast<const Derived*>(this)->data[198]; }
		reference wyxw() { return static_cast<Derived*>(this)->data[199]; }
		const_reference wyxw() const { return static_cast<const Derived*>(this)->data[199]; }
		reference xzxw() { return static_cast<Derived*>(this)->data[200]; }
		const_reference xzxw() const { return static_cast<const Derived*>(this)->data[200]; }
		reference yzxw() { return static_cast<Derived*>(this)->data[201]; }
		const_reference yzxw() const { return static_cast<const Derived*>(this)->data[201]; }
		reference zzxw() { return static_cast<Derived*>(this)->data[202]; }
		const_reference zzxw() const { return static_cast<const Derived*>(this)->data[202]; }
		reference wzxw() { return static_cast<Derived*>(this)->data[203]; }
		const_reference wzxw() const { return static_cast<const Derived*>(this)->data[203]; }
		reference xwxw() { return static_cast<Derived*>(this)->data[204]; }
		const_reference xwxw() const { return static_cast<const Derived*>(this)->data[204]; }
		reference ywxw() { return static_cast<Derived*>(this)->data[205]; }
		const_reference ywxw() const { return static_cast<const Derived*>(this)->data[205]; }
		reference zwxw() { return static_cast<Derived*>(this)->data[206]; }
		const_reference zwxw() const { return static_cast<const Derived*>(this)->data[206]; }
		reference wwxw() { return static_cast<Derived*>(this)->data[207]; }
		const_reference wwxw() const { return static_cast<const Derived*>(this)->data[207]; }
		reference xxyw() { return static_cast<Derived*>(this)->data[208]; }
		const_reference xxyw() const { return static_cast<const Derived*>(this)->data[208]; }
		reference yxyw() { return static_cast<Derived*>(this)->data[209]; }
		const_reference yxyw() const { return static_cast<const Derived*>(this)->data[209]; }
		reference zxyw() { return static_cast<Derived*>(this)->data[210]; }
		const_reference zxyw() const { return static_cast<const Derived*>(this)->data[210]; }
		reference wxyw() { return static_cast<Derived*>(this)->data[211]; }
		const_reference wxyw() const { return static_cast<const Derived*>(this)->data[211]; }
		reference xyyw() { return static_cast<Derived*>(this)->data[212]; }
		const_reference xyyw() const { return static_cast<const Derived*>(this)->data[212]; }
		reference yyyw() { return static_cast<Derived*>(this)->data[213]; }
		const_reference yyyw() const { return static_cast<const Derived*>(this)->data[213]; }
		reference zyyw() { return static_cast<Derived*>(this)->data[214]; }
		const_reference zyyw() const { return static_cast<const Derived*>(this)->data[214]; }
		reference wyyw() { return static_cast<Derived*>(this)->data[215]; }
		const_reference wyyw() const { return static_cast<const Derived*>(this)->data[215]; }
		reference xzyw() { return static_cast<Derived*>(this)->data[216]; }
		const_reference xzyw() const { return static_cast<const Derived*>(this)->data[216]; }
		reference yzyw() { return static_cast<Derived*>(this)->data[217]; }
		const_reference yzyw() const { return static_cast<const Derived*>(this)->data[217]; }
		reference zzyw() { return static_cast<Derived*>(this)->data[218]; }
		const_reference zzyw() const { return static_cast<const Derived*>(this)->data[218]; }
		reference wzyw() { return static_cast<Derived*>(this)->data[219]; }
		const_reference wzyw() const { return static_cast<const Derived*>(this)->data[219]; }
		reference xwyw() { return static_cast<Derived*>(this)->data[220]; }
		const_reference xwyw() const { return static_cast<const Derived*>(this)->data[220]; }
		reference ywyw() { return static_cast<Derived*>(this)->data[221]; }
		const_reference ywyw() const { return static_cast<const Derived*>(this)->data[221]; }
		reference zwyw() { return static_cast<Derived*>(this)->data[222]; }
		const_reference zwyw() const { return static_cast<const Derived*>(this)->data[222]; }
		reference wwyw() { return static_cast<Derived*>(this)->data[223]; }
		const_reference wwyw() const { return static_cast<const Derived*>(this)->data[223]; }
		reference xxzw() { return static_cast<Derived*>(this)->data[224]; }
		const_reference xxzw() const { return static_cast<const Derived*>(this)->data[224]; }
		reference yxzw() { return static_cast<Derived*>(this)->data[225]; }
		const_reference yxzw() const { return static_cast<const Derived*>(this)->data[225]; }
		reference zxzw() { return static_cast<Derived*>(this)->data[226]; }
		const_reference zxzw() const { return static_cast<const Derived*>(this)->data[226]; }
		reference wxzw() { return static_cast<Derived*>(this)->data[227]; }
		const_reference wxzw() const { return static_cast<const Derived*>(this)->data[227]; }
		reference xyzw() { return static_cast<Derived*>(this)->data[228]; }
		const_reference xyzw() const { return static_cast<const Derived*>(this)->data[228]; }
		reference yyzw() { return static_cast<Derived*>(this)->data[229]; }
		const_reference yyzw() const { return static_cast<const Derived*>(this)->data[229]; }
		reference zyzw() { return static_cast<Derived*>(this)->data[230]; }
		const_reference zyzw() const { return static_cast<const Derived*>(this)->data[230]; }
		reference wyzw() { return static_cast<Derived*>(this)->data[231]; }
		const_reference wyzw() const { return static_cast<const Derived*>(this)->data[231]; }
		reference xzzw() { return static_cast<Derived*>(this)->data[232]; }
		const_reference xzzw() const { return static_cast<const Derived*>(this)->data[232]; }
		reference yzzw() { return static_cast<Derived*>(this)->data[233]; }
		const_reference yzzw() const { return static_cast<const Derived*>(this)->data[233]; }
		reference zzzw() { return static_cast<Derived*>(this)->data[234]; }
		const_reference zzzw() const { return static_cast<const Derived*>(this)->data[234]; }
		reference wzzw() { return static_cast<Derived*>(this)->data[235]; }
		const_reference wzzw() const { return static_cast<const Derived*>(this)->data[235]; }
		reference xwzw() { return static_cast<Derived*>(this)->data[236]; }
		const_reference xwzw() const { return static_cast<const Derived*>(this)->data[236]; }
		reference ywzw() { return static_cast<Derived*>(this)->data[237]; }
		const_reference ywzw() const { return static_cast<const Derived*>(this)->data[237]; }
		reference zwzw() { return static_cast<Derived*>(this)->data[238]; }
		const_reference zwzw() const { return static_cast<const Derived*>(this)->data[238]; }
		reference wwzw() { return static_cast<Derived*>(this)->data[239]; }
		const_reference wwzw() const { return static_cast<const Derived*>(this)->data[239]; }
		reference xxww() { return static_cast<Derived*>(this)->data[240]; }
		const_reference xxww() const { return static_cast<const Derived*>(this)->data[240]; }
		reference yxww() { return static_cast<Derived*>(this)->data[241]; }
		const_reference yxww() const { return static_cast<const Derived*>(this)->data[241]; }
		reference zxww() { return static_cast<Derived*>(this)->data[242]; }
		const_reference zxww() const { return static_cast<const Derived*>(this)->data[242]; }
		reference wxww() { return static_cast<Derived*>(this)->data[243]; }
		const_reference wxww() const { return static_cast<const Derived*>(this)->data[243]; }
		reference xyww() { return static_cast<Derived*>(this)->data[244]; }
		const_reference xyww() const { return static_cast<const Derived*>(this)->data[244]; }
		reference yyww() { return static_cast<Derived*>(this)->data[245]; }
		const_reference yyww() const { return static_cast<const Derived*>(this)->data[245]; }
		reference zyww() { return static_cast<Derived*>(this)->data[246]; }
		const_reference zyww() const { return static_cast<const Derived*>(this)->data[246]; }
		reference wyww() { return static_cast<Derived*>(this)->data[247]; }
		const_reference wyww() const { return static_cast<const Derived*>(this)->data[247]; }
		reference xzww() { return static_cast<Derived*>(this)->data[248]; }
		const_reference xzww() const { return static_cast<const Derived*>(this)->data[248]; }
		reference yzww() { return static_cast<Derived*>(this)->data[249]; }
		const_reference yzww() const { return static_cast<const Derived*>(this)->data[249]; }
		reference zzww() { return static_cast<Derived*>(this)->data[250]; }
		const_reference zzww() const { return static_cast<const Derived*>(this)->data[250]; }
		reference wzww() { return static_cast<Derived*>(this)->data[251]; }
		const_reference wzww() const { return static_cast<const Derived*>(this)->data[251]; }
		reference xwww() { return static_cast<Derived*>(this)->data[252]; }
		const_reference xwww() const { return static_cast<const Derived*>(this)->data[252]; }
		reference ywww() { return static_cast<Derived*>(this)->data[253]; }
		const_reference ywww() const { return static_cast<const Derived*>(this)->data[253]; }
		reference zwww() { return static_cast<Derived*>(this)->data[254]; }
		const_reference zwww() const { return static_cast<const Derived*>(this)->data[254]; }
		reference wwww() { return static_cast<Derived*>(this)->data[255]; }
		const_reference wwww() const { return static_cast<const Derived*>(this)->data[255]; }
	};




	// Rank = 2, N = 1
	template<typename Derived, typename T>
	struct symmetric_tensor_access<Derived,T,2,1>
	{
		using reference = typename std::array<T,1>::reference;
		using const_reference=typename std::array<T,1>::const_reference;
		reference xx() { return static_cast<Derived*>(this)->data[0]; }
		const_reference xx() const { return static_cast<const Derived*>(this)->data[0]; }
	};

	// Rank = 2, N = 2
	template<typename Derived, typename T>
	struct symmetric_tensor_access<Derived,T,2,2>
	{
		using reference = typename std::array<T,1>::reference;
		using const_reference=typename std::array<T,1>::const_reference;
		reference xx() { return static_cast<Derived*>(this)->data[0]; }
		const_reference xx() const { return static_cast<const Derived*>(this)->data[0]; }
		reference yx() { return static_cast<Derived*>(this)->data[1]; }
		const_reference yx() const { return static_cast<const Derived*>(this)->data[1]; }
		reference xy() { return static_cast<Derived*>(this)->data[1]; }
		const_reference xy() const { return static_cast<const Derived*>(this)->data[1]; }
		reference yy() { return static_cast<Derived*>(this)->data[2]; }
		const_reference yy() const { return static_cast<const Derived*>(this)->data[2]; }
	};

	// Rank = 2, N = 3
	template<typename Derived, typename T>
	struct symmetric_tensor_access<Derived,T,2,3>
	{
		using reference = typename std::array<T,1>::reference;
		using const_reference=typename std::array<T,1>::const_reference;
		reference xx() { return static_cast<Derived*>(this)->data[0]; }
		const_reference xx() const { return static_cast<const Derived*>(this)->data[0]; }
		reference yx() { return static_cast<Derived*>(this)->data[1]; }
		const_reference yx() const { return static_cast<const Derived*>(this)->data[1]; }
		reference xy() { return static_cast<Derived*>(this)->data[1]; }
		const_reference xy() const { return static_cast<const Derived*>(this)->data[1]; }
		reference zx() { return static_cast<Derived*>(this)->data[2]; }
		const_reference zx() const { return static_cast<const Derived*>(this)->data[2]; }
		reference xz() { return static_cast<Derived*>(this)->data[2]; }
		const_reference xz() const { return static_cast<const Derived*>(this)->data[2]; }
		reference yy() { return static_cast<Derived*>(this)->data[3]; }
		const_reference yy() const { return static_cast<const Derived*>(this)->data[3]; }
		reference zy() { return static_cast<Derived*>(this)->data[4]; }
		const_reference zy() const { return static_cast<const Derived*>(this)->data[4]; }
		reference yz() { return static_cast<Derived*>(this)->data[4]; }
		const_reference yz() const { return static_cast<const Derived*>(this)->data[4]; }
		reference zz() { return static_cast<Derived*>(this)->data[5]; }
		const_reference zz() const { return static_cast<const Derived*>(this)->data[5]; }
	};

	// Rank = 2, N = 4
	template<typename Derived, typename T>
	struct symmetric_tensor_access<Derived,T,2,4>
	{
		using reference = typename std::array<T,1>::reference;
		using const_reference=typename std::array<T,1>::const_reference;
		reference xx() { return static_cast<Derived*>(this)->data[0]; }
		const_reference xx() const { return static_cast<const Derived*>(this)->data[0]; }
		reference yx() { return static_cast<Derived*>(this)->data[1]; }
		const_reference yx() const { return static_cast<const Derived*>(this)->data[1]; }
		reference xy() { return static_cast<Derived*>(this)->data[1]; }
		const_reference xy() const { return static_cast<const Derived*>(this)->data[1]; }
		reference zx() { return static_cast<Derived*>(this)->data[2]; }
		const_reference zx() const { return static_cast<const Derived*>(this)->data[2]; }
		reference xz() { return static_cast<Derived*>(this)->data[2]; }
		const_reference xz() const { return static_cast<const Derived*>(this)->data[2]; }
		reference wx() { return static_cast<Derived*>(this)->data[3]; }
		const_reference wx() const { return static_cast<const Derived*>(this)->data[3]; }
		reference xw() { return static_cast<Derived*>(this)->data[3]; }
		const_reference xw() const { return static_cast<const Derived*>(this)->data[3]; }
		reference yy() { return static_cast<Derived*>(this)->data[4]; }
		const_reference yy() const { return static_cast<const Derived*>(this)->data[4]; }
		reference zy() { return static_cast<Derived*>(this)->data[5]; }
		const_reference zy() const { return static_cast<const Derived*>(this)->data[5]; }
		reference yz() { return static_cast<Derived*>(this)->data[5]; }
		const_reference yz() const { return static_cast<const Derived*>(this)->data[5]; }
		reference wy() { return static_cast<Derived*>(this)->data[6]; }
		const_reference wy() const { return static_cast<const Derived*>(this)->data[6]; }
		reference yw() { return static_cast<Derived*>(this)->data[6]; }
		const_reference yw() const { return static_cast<const Derived*>(this)->data[6]; }
		reference zz() { return static_cast<Derived*>(this)->data[7]; }
		const_reference zz() const { return static_cast<const Derived*>(this)->data[7]; }
		reference wz() { return static_cast<Derived*>(this)->data[8]; }
		const_reference wz() const { return static_cast<const Derived*>(this)->data[8]; }
		reference zw() { return static_cast<Derived*>(this)->data[8]; }
		const_reference zw() const { return static_cast<const Derived*>(this)->data[8]; }
		reference ww() { return static_cast<Derived*>(this)->data[9]; }
		const_reference ww() const { return static_cast<const Derived*>(this)->data[9]; }
	};
	
	// Rank = 3, N = 1
	template<typename Derived, typename T>
	struct symmetric_tensor_access<Derived,T,3,1>
	{
		using reference = typename std::array<T,1>::reference;
		using const_reference=typename std::array<T,1>::const_reference;
		reference xxx() { return static_cast<Derived*>(this)->data[0]; }
		const_reference xxx() const { return static_cast<const Derived*>(this)->data[0]; }
	};

	// Rank = 3, N = 2
	template<typename Derived, typename T>
	struct symmetric_tensor_access<Derived,T,3,2>
	{
		using reference = typename std::array<T,1>::reference;
		using const_reference=typename std::array<T,1>::const_reference;
		reference xxx() { return static_cast<Derived*>(this)->data[0]; }
		const_reference xxx() const { return static_cast<const Derived*>(this)->data[0]; }
		reference yxx() { return static_cast<Derived*>(this)->data[1]; }
		const_reference yxx() const { return static_cast<const Derived*>(this)->data[1]; }
		reference xyx() { return static_cast<Derived*>(this)->data[1]; }
		const_reference xyx() const { return static_cast<const Derived*>(this)->data[1]; }
		reference xxy() { return static_cast<Derived*>(this)->data[1]; }
		const_reference xxy() const { return static_cast<const Derived*>(this)->data[1]; }
		reference yyx() { return static_cast<Derived*>(this)->data[2]; }
		const_reference yyx() const { return static_cast<const Derived*>(this)->data[2]; }
		reference yxy() { return static_cast<Derived*>(this)->data[2]; }
		const_reference yxy() const { return static_cast<const Derived*>(this)->data[2]; }
		reference xyy() { return static_cast<Derived*>(this)->data[2]; }
		const_reference xyy() const { return static_cast<const Derived*>(this)->data[2]; }
		reference yyy() { return static_cast<Derived*>(this)->data[3]; }
		const_reference yyy() const { return static_cast<const Derived*>(this)->data[3]; }
	};

	// Rank = 3, N = 3
	template<typename Derived, typename T>
	struct symmetric_tensor_access<Derived,T,3,3>
	{
		using reference = typename std::array<T,1>::reference;
		using const_reference=typename std::array<T,1>::const_reference;
		reference xxx() { return static_cast<Derived*>(this)->data[0]; }
		const_reference xxx() const { return static_cast<const Derived*>(this)->data[0]; }
		reference yxx() { return static_cast<Derived*>(this)->data[1]; }
		const_reference yxx() const { return static_cast<const Derived*>(this)->data[1]; }
		reference xyx() { return static_cast<Derived*>(this)->data[1]; }
		const_reference xyx() const { return static_cast<const Derived*>(this)->data[1]; }
		reference xxy() { return static_cast<Derived*>(this)->data[1]; }
		const_reference xxy() const { return static_cast<const Derived*>(this)->data[1]; }
		reference zxx() { return static_cast<Derived*>(this)->data[2]; }
		const_reference zxx() const { return static_cast<const Derived*>(this)->data[2]; }
		reference xzx() { return static_cast<Derived*>(this)->data[2]; }
		const_reference xzx() const { return static_cast<const Derived*>(this)->data[2]; }
		reference xxz() { return static_cast<Derived*>(this)->data[2]; }
		const_reference xxz() const { return static_cast<const Derived*>(this)->data[2]; }
		reference yyx() { return static_cast<Derived*>(this)->data[3]; }
		const_reference yyx() const { return static_cast<const Derived*>(this)->data[3]; }
		reference yxy() { return static_cast<Derived*>(this)->data[3]; }
		const_reference yxy() const { return static_cast<const Derived*>(this)->data[3]; }
		reference xyy() { return static_cast<Derived*>(this)->data[3]; }
		const_reference xyy() const { return static_cast<const Derived*>(this)->data[3]; }
		reference zyx() { return static_cast<Derived*>(this)->data[4]; }
		const_reference zyx() const { return static_cast<const Derived*>(this)->data[4]; }
		reference zxy() { return static_cast<Derived*>(this)->data[4]; }
		const_reference zxy() const { return static_cast<const Derived*>(this)->data[4]; }
		reference yzx() { return static_cast<Derived*>(this)->data[4]; }
		const_reference yzx() const { return static_cast<const Derived*>(this)->data[4]; }
		reference yxz() { return static_cast<Derived*>(this)->data[4]; }
		const_reference yxz() const { return static_cast<const Derived*>(this)->data[4]; }
		reference xzy() { return static_cast<Derived*>(this)->data[4]; }
		const_reference xzy() const { return static_cast<const Derived*>(this)->data[4]; }
		reference xyz() { return static_cast<Derived*>(this)->data[4]; }
		const_reference xyz() const { return static_cast<const Derived*>(this)->data[4]; }
		reference zzx() { return static_cast<Derived*>(this)->data[5]; }
		const_reference zzx() const { return static_cast<const Derived*>(this)->data[5]; }
		reference zxz() { return static_cast<Derived*>(this)->data[5]; }
		const_reference zxz() const { return static_cast<const Derived*>(this)->data[5]; }
		reference xzz() { return static_cast<Derived*>(this)->data[5]; }
		const_reference xzz() const { return static_cast<const Derived*>(this)->data[5]; }
		reference yyy() { return static_cast<Derived*>(this)->data[6]; }
		const_reference yyy() const { return static_cast<const Derived*>(this)->data[6]; }
		reference zyy() { return static_cast<Derived*>(this)->data[7]; }
		const_reference zyy() const { return static_cast<const Derived*>(this)->data[7]; }
		reference yzy() { return static_cast<Derived*>(this)->data[7]; }
		const_reference yzy() const { return static_cast<const Derived*>(this)->data[7]; }
		reference yyz() { return static_cast<Derived*>(this)->data[7]; }
		const_reference yyz() const { return static_cast<const Derived*>(this)->data[7]; }
		reference zzy() { return static_cast<Derived*>(this)->data[8]; }
		const_reference zzy() const { return static_cast<const Derived*>(this)->data[8]; }
		reference zyz() { return static_cast<Derived*>(this)->data[8]; }
		const_reference zyz() const { return static_cast<const Derived*>(this)->data[8]; }
		reference yzz() { return static_cast<Derived*>(this)->data[8]; }
		const_reference yzz() const { return static_cast<const Derived*>(this)->data[8]; }
		reference zzz() { return static_cast<Derived*>(this)->data[9]; }
		const_reference zzz() const { return static_cast<const Derived*>(this)->data[9]; }
	};

	// Rank = 3, N = 4
	template<typename Derived, typename T>
	struct symmetric_tensor_access<Derived,T,3,4>
	{
		using reference = typename std::array<T,1>::reference;
		using const_reference=typename std::array<T,1>::const_reference;
		reference xxx() { return static_cast<Derived*>(this)->data[0]; }
		const_reference xxx() const { return static_cast<const Derived*>(this)->data[0]; }
		reference yxx() { return static_cast<Derived*>(this)->data[1]; }
		const_reference yxx() const { return static_cast<const Derived*>(this)->data[1]; }
		reference xyx() { return static_cast<Derived*>(this)->data[1]; }
		const_reference xyx() const { return static_cast<const Derived*>(this)->data[1]; }
		reference xxy() { return static_cast<Derived*>(this)->data[1]; }
		const_reference xxy() const { return static_cast<const Derived*>(this)->data[1]; }
		reference zxx() { return static_cast<Derived*>(this)->data[2]; }
		const_reference zxx() const { return static_cast<const Derived*>(this)->data[2]; }
		reference xzx() { return static_cast<Derived*>(this)->data[2]; }
		const_reference xzx() const { return static_cast<const Derived*>(this)->data[2]; }
		reference xxz() { return static_cast<Derived*>(this)->data[2]; }
		const_reference xxz() const { return static_cast<const Derived*>(this)->data[2]; }
		reference wxx() { return static_cast<Derived*>(this)->data[3]; }
		const_reference wxx() const { return static_cast<const Derived*>(this)->data[3]; }
		reference xwx() { return static_cast<Derived*>(this)->data[3]; }
		const_reference xwx() const { return static_cast<const Derived*>(this)->data[3]; }
		reference xxw() { return static_cast<Derived*>(this)->data[3]; }
		const_reference xxw() const { return static_cast<const Derived*>(this)->data[3]; }
		reference yyx() { return static_cast<Derived*>(this)->data[4]; }
		const_reference yyx() const { return static_cast<const Derived*>(this)->data[4]; }
		reference yxy() { return static_cast<Derived*>(this)->data[4]; }
		const_reference yxy() const { return static_cast<const Derived*>(this)->data[4]; }
		reference xyy() { return static_cast<Derived*>(this)->data[4]; }
		const_reference xyy() const { return static_cast<const Derived*>(this)->data[4]; }
		reference zyx() { return static_cast<Derived*>(this)->data[5]; }
		const_reference zyx() const { return static_cast<const Derived*>(this)->data[5]; }
		reference zxy() { return static_cast<Derived*>(this)->data[5]; }
		const_reference zxy() const { return static_cast<const Derived*>(this)->data[5]; }
		reference yzx() { return static_cast<Derived*>(this)->data[5]; }
		const_reference yzx() const { return static_cast<const Derived*>(this)->data[5]; }
		reference yxz() { return static_cast<Derived*>(this)->data[5]; }
		const_reference yxz() const { return static_cast<const Derived*>(this)->data[5]; }
		reference xzy() { return static_cast<Derived*>(this)->data[5]; }
		const_reference xzy() const { return static_cast<const Derived*>(this)->data[5]; }
		reference xyz() { return static_cast<Derived*>(this)->data[5]; }
		const_reference xyz() const { return static_cast<const Derived*>(this)->data[5]; }
		reference wyx() { return static_cast<Derived*>(this)->data[6]; }
		const_reference wyx() const { return static_cast<const Derived*>(this)->data[6]; }
		reference wxy() { return static_cast<Derived*>(this)->data[6]; }
		const_reference wxy() const { return static_cast<const Derived*>(this)->data[6]; }
		reference ywx() { return static_cast<Derived*>(this)->data[6]; }
		const_reference ywx() const { return static_cast<const Derived*>(this)->data[6]; }
		reference yxw() { return static_cast<Derived*>(this)->data[6]; }
		const_reference yxw() const { return static_cast<const Derived*>(this)->data[6]; }
		reference xwy() { return static_cast<Derived*>(this)->data[6]; }
		const_reference xwy() const { return static_cast<const Derived*>(this)->data[6]; }
		reference xyw() { return static_cast<Derived*>(this)->data[6]; }
		const_reference xyw() const { return static_cast<const Derived*>(this)->data[6]; }
		reference zzx() { return static_cast<Derived*>(this)->data[7]; }
		const_reference zzx() const { return static_cast<const Derived*>(this)->data[7]; }
		reference zxz() { return static_cast<Derived*>(this)->data[7]; }
		const_reference zxz() const { return static_cast<const Derived*>(this)->data[7]; }
		reference xzz() { return static_cast<Derived*>(this)->data[7]; }
		const_reference xzz() const { return static_cast<const Derived*>(this)->data[7]; }
		reference wzx() { return static_cast<Derived*>(this)->data[8]; }
		const_reference wzx() const { return static_cast<const Derived*>(this)->data[8]; }
		reference wxz() { return static_cast<Derived*>(this)->data[8]; }
		const_reference wxz() const { return static_cast<const Derived*>(this)->data[8]; }
		reference zwx() { return static_cast<Derived*>(this)->data[8]; }
		const_reference zwx() const { return static_cast<const Derived*>(this)->data[8]; }
		reference zxw() { return static_cast<Derived*>(this)->data[8]; }
		const_reference zxw() const { return static_cast<const Derived*>(this)->data[8]; }
		reference xwz() { return static_cast<Derived*>(this)->data[8]; }
		const_reference xwz() const { return static_cast<const Derived*>(this)->data[8]; }
		reference xzw() { return static_cast<Derived*>(this)->data[8]; }
		const_reference xzw() const { return static_cast<const Derived*>(this)->data[8]; }
		reference wwx() { return static_cast<Derived*>(this)->data[9]; }
		const_reference wwx() const { return static_cast<const Derived*>(this)->data[9]; }
		reference wxw() { return static_cast<Derived*>(this)->data[9]; }
		const_reference wxw() const { return static_cast<const Derived*>(this)->data[9]; }
		reference xww() { return static_cast<Derived*>(this)->data[9]; }
		const_reference xww() const { return static_cast<const Derived*>(this)->data[9]; }
		reference yyy() { return static_cast<Derived*>(this)->data[10]; }
		const_reference yyy() const { return static_cast<const Derived*>(this)->data[10]; }
		reference zyy() { return static_cast<Derived*>(this)->data[11]; }
		const_reference zyy() const { return static_cast<const Derived*>(this)->data[11]; }
		reference yzy() { return static_cast<Derived*>(this)->data[11]; }
		const_reference yzy() const { return static_cast<const Derived*>(this)->data[11]; }
		reference yyz() { return static_cast<Derived*>(this)->data[11]; }
		const_reference yyz() const { return static_cast<const Derived*>(this)->data[11]; }
		reference wyy() { return static_cast<Derived*>(this)->data[12]; }
		const_reference wyy() const { return static_cast<const Derived*>(this)->data[12]; }
		reference ywy() { return static_cast<Derived*>(this)->data[12]; }
		const_reference ywy() const { return static_cast<const Derived*>(this)->data[12]; }
		reference yyw() { return static_cast<Derived*>(this)->data[12]; }
		const_reference yyw() const { return static_cast<const Derived*>(this)->data[12]; }
		reference zzy() { return static_cast<Derived*>(this)->data[13]; }
		const_reference zzy() const { return static_cast<const Derived*>(this)->data[13]; }
		reference zyz() { return static_cast<Derived*>(this)->data[13]; }
		const_reference zyz() const { return static_cast<const Derived*>(this)->data[13]; }
		reference yzz() { return static_cast<Derived*>(this)->data[13]; }
		const_reference yzz() const { return static_cast<const Derived*>(this)->data[13]; }
		reference wzy() { return static_cast<Derived*>(this)->data[14]; }
		const_reference wzy() const { return static_cast<const Derived*>(this)->data[14]; }
		reference wyz() { return static_cast<Derived*>(this)->data[14]; }
		const_reference wyz() const { return static_cast<const Derived*>(this)->data[14]; }
		reference zwy() { return static_cast<Derived*>(this)->data[14]; }
		const_reference zwy() const { return static_cast<const Derived*>(this)->data[14]; }
		reference zyw() { return static_cast<Derived*>(this)->data[14]; }
		const_reference zyw() const { return static_cast<const Derived*>(this)->data[14]; }
		reference ywz() { return static_cast<Derived*>(this)->data[14]; }
		const_reference ywz() const { return static_cast<const Derived*>(this)->data[14]; }
		reference yzw() { return static_cast<Derived*>(this)->data[14]; }
		const_reference yzw() const { return static_cast<const Derived*>(this)->data[14]; }
		reference wwy() { return static_cast<Derived*>(this)->data[15]; }
		const_reference wwy() const { return static_cast<const Derived*>(this)->data[15]; }
		reference wyw() { return static_cast<Derived*>(this)->data[15]; }
		const_reference wyw() const { return static_cast<const Derived*>(this)->data[15]; }
		reference yww() { return static_cast<Derived*>(this)->data[15]; }
		const_reference yww() const { return static_cast<const Derived*>(this)->data[15]; }
		reference zzz() { return static_cast<Derived*>(this)->data[16]; }
		const_reference zzz() const { return static_cast<const Derived*>(this)->data[16]; }
		reference wzz() { return static_cast<Derived*>(this)->data[17]; }
		const_reference wzz() const { return static_cast<const Derived*>(this)->data[17]; }
		reference zwz() { return static_cast<Derived*>(this)->data[17]; }
		const_reference zwz() const { return static_cast<const Derived*>(this)->data[17]; }
		reference zzw() { return static_cast<Derived*>(this)->data[17]; }
		const_reference zzw() const { return static_cast<const Derived*>(this)->data[17]; }
		reference wwz() { return static_cast<Derived*>(this)->data[18]; }
		const_reference wwz() const { return static_cast<const Derived*>(this)->data[18]; }
		reference wzw() { return static_cast<Derived*>(this)->data[18]; }
		const_reference wzw() const { return static_cast<const Derived*>(this)->data[18]; }
		reference zww() { return static_cast<Derived*>(this)->data[18]; }
		const_reference zww() const { return static_cast<const Derived*>(this)->data[18]; }
		reference www() { return static_cast<Derived*>(this)->data[19]; }
		const_reference www() const { return static_cast<const Derived*>(this)->data[19]; }
	};

	// Rank = 4, N = 1
	template<typename Derived, typename T>
	struct symmetric_tensor_access<Derived,T,4,1>
	{
		using reference = typename std::array<T,1>::reference;
		using const_reference=typename std::array<T,1>::const_reference;
		reference xxxx() { return static_cast<Derived*>(this)->data[0]; }
		const_reference xxxx() const { return static_cast<const Derived*>(this)->data[0]; }
	};

	// Rank = 4, N = 2
	template<typename Derived, typename T>
	struct symmetric_tensor_access<Derived,T,4,2>
	{
		using reference = typename std::array<T,1>::reference;
		using const_reference=typename std::array<T,1>::const_reference;
		reference xxxx() { return static_cast<Derived*>(this)->data[0]; }
		const_reference xxxx() const { return static_cast<const Derived*>(this)->data[0]; }
		reference yxxx() { return static_cast<Derived*>(this)->data[1]; }
		const_reference yxxx() const { return static_cast<const Derived*>(this)->data[1]; }
		reference xyxx() { return static_cast<Derived*>(this)->data[1]; }
		const_reference xyxx() const { return static_cast<const Derived*>(this)->data[1]; }
		reference xxyx() { return static_cast<Derived*>(this)->data[1]; }
		const_reference xxyx() const { return static_cast<const Derived*>(this)->data[1]; }
		reference xxxy() { return static_cast<Derived*>(this)->data[1]; }
		const_reference xxxy() const { return static_cast<const Derived*>(this)->data[1]; }
		reference yyxx() { return static_cast<Derived*>(this)->data[2]; }
		const_reference yyxx() const { return static_cast<const Derived*>(this)->data[2]; }
		reference yxyx() { return static_cast<Derived*>(this)->data[2]; }
		const_reference yxyx() const { return static_cast<const Derived*>(this)->data[2]; }
		reference yxxy() { return static_cast<Derived*>(this)->data[2]; }
		const_reference yxxy() const { return static_cast<const Derived*>(this)->data[2]; }
		reference xyyx() { return static_cast<Derived*>(this)->data[2]; }
		const_reference xyyx() const { return static_cast<const Derived*>(this)->data[2]; }
		reference xyxy() { return static_cast<Derived*>(this)->data[2]; }
		const_reference xyxy() const { return static_cast<const Derived*>(this)->data[2]; }
		reference xxyy() { return static_cast<Derived*>(this)->data[2]; }
		const_reference xxyy() const { return static_cast<const Derived*>(this)->data[2]; }
		reference yyyx() { return static_cast<Derived*>(this)->data[3]; }
		const_reference yyyx() const { return static_cast<const Derived*>(this)->data[3]; }
		reference yyxy() { return static_cast<Derived*>(this)->data[3]; }
		const_reference yyxy() const { return static_cast<const Derived*>(this)->data[3]; }
		reference yxyy() { return static_cast<Derived*>(this)->data[3]; }
		const_reference yxyy() const { return static_cast<const Derived*>(this)->data[3]; }
		reference xyyy() { return static_cast<Derived*>(this)->data[3]; }
		const_reference xyyy() const { return static_cast<const Derived*>(this)->data[3]; }
		reference yyyy() { return static_cast<Derived*>(this)->data[4]; }
		const_reference yyyy() const { return static_cast<const Derived*>(this)->data[4]; }
	};

	// Rank = 4, N = 3
	template<typename Derived, typename T>
	struct symmetric_tensor_access<Derived,T,4,3>
	{
		using reference = typename std::array<T,1>::reference;
		using const_reference=typename std::array<T,1>::const_reference;
		reference xxxx() { return static_cast<Derived*>(this)->data[0]; }
		const_reference xxxx() const { return static_cast<const Derived*>(this)->data[0]; }
		reference yxxx() { return static_cast<Derived*>(this)->data[1]; }
		const_reference yxxx() const { return static_cast<const Derived*>(this)->data[1]; }
		reference xyxx() { return static_cast<Derived*>(this)->data[1]; }
		const_reference xyxx() const { return static_cast<const Derived*>(this)->data[1]; }
		reference xxyx() { return static_cast<Derived*>(this)->data[1]; }
		const_reference xxyx() const { return static_cast<const Derived*>(this)->data[1]; }
		reference xxxy() { return static_cast<Derived*>(this)->data[1]; }
		const_reference xxxy() const { return static_cast<const Derived*>(this)->data[1]; }
		reference zxxx() { return static_cast<Derived*>(this)->data[2]; }
		const_reference zxxx() const { return static_cast<const Derived*>(this)->data[2]; }
		reference xzxx() { return static_cast<Derived*>(this)->data[2]; }
		const_reference xzxx() const { return static_cast<const Derived*>(this)->data[2]; }
		reference xxzx() { return static_cast<Derived*>(this)->data[2]; }
		const_reference xxzx() const { return static_cast<const Derived*>(this)->data[2]; }
		reference xxxz() { return static_cast<Derived*>(this)->data[2]; }
		const_reference xxxz() const { return static_cast<const Derived*>(this)->data[2]; }
		reference yyxx() { return static_cast<Derived*>(this)->data[3]; }
		const_reference yyxx() const { return static_cast<const Derived*>(this)->data[3]; }
		reference yxyx() { return static_cast<Derived*>(this)->data[3]; }
		const_reference yxyx() const { return static_cast<const Derived*>(this)->data[3]; }
		reference yxxy() { return static_cast<Derived*>(this)->data[3]; }
		const_reference yxxy() const { return static_cast<const Derived*>(this)->data[3]; }
		reference xyyx() { return static_cast<Derived*>(this)->data[3]; }
		const_reference xyyx() const { return static_cast<const Derived*>(this)->data[3]; }
		reference xyxy() { return static_cast<Derived*>(this)->data[3]; }
		const_reference xyxy() const { return static_cast<const Derived*>(this)->data[3]; }
		reference xxyy() { return static_cast<Derived*>(this)->data[3]; }
		const_reference xxyy() const { return static_cast<const Derived*>(this)->data[3]; }
		reference zyxx() { return static_cast<Derived*>(this)->data[4]; }
		const_reference zyxx() const { return static_cast<const Derived*>(this)->data[4]; }
		reference zxyx() { return static_cast<Derived*>(this)->data[4]; }
		const_reference zxyx() const { return static_cast<const Derived*>(this)->data[4]; }
		reference zxxy() { return static_cast<Derived*>(this)->data[4]; }
		const_reference zxxy() const { return static_cast<const Derived*>(this)->data[4]; }
		reference yzxx() { return static_cast<Derived*>(this)->data[4]; }
		const_reference yzxx() const { return static_cast<const Derived*>(this)->data[4]; }
		reference yxzx() { return static_cast<Derived*>(this)->data[4]; }
		const_reference yxzx() const { return static_cast<const Derived*>(this)->data[4]; }
		reference yxxz() { return static_cast<Derived*>(this)->data[4]; }
		const_reference yxxz() const { return static_cast<const Derived*>(this)->data[4]; }
		reference xzyx() { return static_cast<Derived*>(this)->data[4]; }
		const_reference xzyx() const { return static_cast<const Derived*>(this)->data[4]; }
		reference xzxy() { return static_cast<Derived*>(this)->data[4]; }
		const_reference xzxy() const { return static_cast<const Derived*>(this)->data[4]; }
		reference xyzx() { return static_cast<Derived*>(this)->data[4]; }
		const_reference xyzx() const { return static_cast<const Derived*>(this)->data[4]; }
		reference xyxz() { return static_cast<Derived*>(this)->data[4]; }
		const_reference xyxz() const { return static_cast<const Derived*>(this)->data[4]; }
		reference xxzy() { return static_cast<Derived*>(this)->data[4]; }
		const_reference xxzy() const { return static_cast<const Derived*>(this)->data[4]; }
		reference xxyz() { return static_cast<Derived*>(this)->data[4]; }
		const_reference xxyz() const { return static_cast<const Derived*>(this)->data[4]; }
		reference zzxx() { return static_cast<Derived*>(this)->data[5]; }
		const_reference zzxx() const { return static_cast<const Derived*>(this)->data[5]; }
		reference zxzx() { return static_cast<Derived*>(this)->data[5]; }
		const_reference zxzx() const { return static_cast<const Derived*>(this)->data[5]; }
		reference zxxz() { return static_cast<Derived*>(this)->data[5]; }
		const_reference zxxz() const { return static_cast<const Derived*>(this)->data[5]; }
		reference xzzx() { return static_cast<Derived*>(this)->data[5]; }
		const_reference xzzx() const { return static_cast<const Derived*>(this)->data[5]; }
		reference xzxz() { return static_cast<Derived*>(this)->data[5]; }
		const_reference xzxz() const { return static_cast<const Derived*>(this)->data[5]; }
		reference xxzz() { return static_cast<Derived*>(this)->data[5]; }
		const_reference xxzz() const { return static_cast<const Derived*>(this)->data[5]; }
		reference yyyx() { return static_cast<Derived*>(this)->data[6]; }
		const_reference yyyx() const { return static_cast<const Derived*>(this)->data[6]; }
		reference yyxy() { return static_cast<Derived*>(this)->data[6]; }
		const_reference yyxy() const { return static_cast<const Derived*>(this)->data[6]; }
		reference yxyy() { return static_cast<Derived*>(this)->data[6]; }
		const_reference yxyy() const { return static_cast<const Derived*>(this)->data[6]; }
		reference xyyy() { return static_cast<Derived*>(this)->data[6]; }
		const_reference xyyy() const { return static_cast<const Derived*>(this)->data[6]; }
		reference zyyx() { return static_cast<Derived*>(this)->data[7]; }
		const_reference zyyx() const { return static_cast<const Derived*>(this)->data[7]; }
		reference zyxy() { return static_cast<Derived*>(this)->data[7]; }
		const_reference zyxy() const { return static_cast<const Derived*>(this)->data[7]; }
		reference zxyy() { return static_cast<Derived*>(this)->data[7]; }
		const_reference zxyy() const { return static_cast<const Derived*>(this)->data[7]; }
		reference yzyx() { return static_cast<Derived*>(this)->data[7]; }
		const_reference yzyx() const { return static_cast<const Derived*>(this)->data[7]; }
		reference yzxy() { return static_cast<Derived*>(this)->data[7]; }
		const_reference yzxy() const { return static_cast<const Derived*>(this)->data[7]; }
		reference yyzx() { return static_cast<Derived*>(this)->data[7]; }
		const_reference yyzx() const { return static_cast<const Derived*>(this)->data[7]; }
		reference yyxz() { return static_cast<Derived*>(this)->data[7]; }
		const_reference yyxz() const { return static_cast<const Derived*>(this)->data[7]; }
		reference yxzy() { return static_cast<Derived*>(this)->data[7]; }
		const_reference yxzy() const { return static_cast<const Derived*>(this)->data[7]; }
		reference yxyz() { return static_cast<Derived*>(this)->data[7]; }
		const_reference yxyz() const { return static_cast<const Derived*>(this)->data[7]; }
		reference xzyy() { return static_cast<Derived*>(this)->data[7]; }
		const_reference xzyy() const { return static_cast<const Derived*>(this)->data[7]; }
		reference xyzy() { return static_cast<Derived*>(this)->data[7]; }
		const_reference xyzy() const { return static_cast<const Derived*>(this)->data[7]; }
		reference xyyz() { return static_cast<Derived*>(this)->data[7]; }
		const_reference xyyz() const { return static_cast<const Derived*>(this)->data[7]; }
		reference zzyx() { return static_cast<Derived*>(this)->data[8]; }
		const_reference zzyx() const { return static_cast<const Derived*>(this)->data[8]; }
		reference zzxy() { return static_cast<Derived*>(this)->data[8]; }
		const_reference zzxy() const { return static_cast<const Derived*>(this)->data[8]; }
		reference zyzx() { return static_cast<Derived*>(this)->data[8]; }
		const_reference zyzx() const { return static_cast<const Derived*>(this)->data[8]; }
		reference zyxz() { return static_cast<Derived*>(this)->data[8]; }
		const_reference zyxz() const { return static_cast<const Derived*>(this)->data[8]; }
		reference zxzy() { return static_cast<Derived*>(this)->data[8]; }
		const_reference zxzy() const { return static_cast<const Derived*>(this)->data[8]; }
		reference zxyz() { return static_cast<Derived*>(this)->data[8]; }
		const_reference zxyz() const { return static_cast<const Derived*>(this)->data[8]; }
		reference yzzx() { return static_cast<Derived*>(this)->data[8]; }
		const_reference yzzx() const { return static_cast<const Derived*>(this)->data[8]; }
		reference yzxz() { return static_cast<Derived*>(this)->data[8]; }
		const_reference yzxz() const { return static_cast<const Derived*>(this)->data[8]; }
		reference yxzz() { return static_cast<Derived*>(this)->data[8]; }
		const_reference yxzz() const { return static_cast<const Derived*>(this)->data[8]; }
		reference xzzy() { return static_cast<Derived*>(this)->data[8]; }
		const_reference xzzy() const { return static_cast<const Derived*>(this)->data[8]; }
		reference xzyz() { return static_cast<Derived*>(this)->data[8]; }
		const_reference xzyz() const { return static_cast<const Derived*>(this)->data[8]; }
		reference xyzz() { return static_cast<Derived*>(this)->data[8]; }
		const_reference xyzz() const { return static_cast<const Derived*>(this)->data[8]; }
		reference zzzx() { return static_cast<Derived*>(this)->data[9]; }
		const_reference zzzx() const { return static_cast<const Derived*>(this)->data[9]; }
		reference zzxz() { return static_cast<Derived*>(this)->data[9]; }
		const_reference zzxz() const { return static_cast<const Derived*>(this)->data[9]; }
		reference zxzz() { return static_cast<Derived*>(this)->data[9]; }
		const_reference zxzz() const { return static_cast<const Derived*>(this)->data[9]; }
		reference xzzz() { return static_cast<Derived*>(this)->data[9]; }
		const_reference xzzz() const { return static_cast<const Derived*>(this)->data[9]; }
		reference yyyy() { return static_cast<Derived*>(this)->data[10]; }
		const_reference yyyy() const { return static_cast<const Derived*>(this)->data[10]; }
		reference zyyy() { return static_cast<Derived*>(this)->data[11]; }
		const_reference zyyy() const { return static_cast<const Derived*>(this)->data[11]; }
		reference yzyy() { return static_cast<Derived*>(this)->data[11]; }
		const_reference yzyy() const { return static_cast<const Derived*>(this)->data[11]; }
		reference yyzy() { return static_cast<Derived*>(this)->data[11]; }
		const_reference yyzy() const { return static_cast<const Derived*>(this)->data[11]; }
		reference yyyz() { return static_cast<Derived*>(this)->data[11]; }
		const_reference yyyz() const { return static_cast<const Derived*>(this)->data[11]; }
		reference zzyy() { return static_cast<Derived*>(this)->data[12]; }
		const_reference zzyy() const { return static_cast<const Derived*>(this)->data[12]; }
		reference zyzy() { return static_cast<Derived*>(this)->data[12]; }
		const_reference zyzy() const { return static_cast<const Derived*>(this)->data[12]; }
		reference zyyz() { return static_cast<Derived*>(this)->data[12]; }
		const_reference zyyz() const { return static_cast<const Derived*>(this)->data[12]; }
		reference yzzy() { return static_cast<Derived*>(this)->data[12]; }
		const_reference yzzy() const { return static_cast<const Derived*>(this)->data[12]; }
		reference yzyz() { return static_cast<Derived*>(this)->data[12]; }
		const_reference yzyz() const { return static_cast<const Derived*>(this)->data[12]; }
		reference yyzz() { return static_cast<Derived*>(this)->data[12]; }
		const_reference yyzz() const { return static_cast<const Derived*>(this)->data[12]; }
		reference zzzy() { return static_cast<Derived*>(this)->data[13]; }
		const_reference zzzy() const { return static_cast<const Derived*>(this)->data[13]; }
		reference zzyz() { return static_cast<Derived*>(this)->data[13]; }
		const_reference zzyz() const { return static_cast<const Derived*>(this)->data[13]; }
		reference zyzz() { return static_cast<Derived*>(this)->data[13]; }
		const_reference zyzz() const { return static_cast<const Derived*>(this)->data[13]; }
		reference yzzz() { return static_cast<Derived*>(this)->data[13]; }
		const_reference yzzz() const { return static_cast<const Derived*>(this)->data[13]; }
		reference zzzz() { return static_cast<Derived*>(this)->data[14]; }
		const_reference zzzz() const { return static_cast<const Derived*>(this)->data[14]; }
	};

	// Rank = 4, N = 4
	template<typename Derived, typename T>
	struct symmetric_tensor_access<Derived,T,4,4>
	{
		using reference = typename std::array<T,1>::reference;
		using const_reference=typename std::array<T,1>::const_reference;
		reference xxxx() { return static_cast<Derived*>(this)->data[0]; }
		const_reference xxxx() const { return static_cast<const Derived*>(this)->data[0]; }
		reference yxxx() { return static_cast<Derived*>(this)->data[1]; }
		const_reference yxxx() const { return static_cast<const Derived*>(this)->data[1]; }
		reference xyxx() { return static_cast<Derived*>(this)->data[1]; }
		const_reference xyxx() const { return static_cast<const Derived*>(this)->data[1]; }
		reference xxyx() { return static_cast<Derived*>(this)->data[1]; }
		const_reference xxyx() const { return static_cast<const Derived*>(this)->data[1]; }
		reference xxxy() { return static_cast<Derived*>(this)->data[1]; }
		const_reference xxxy() const { return static_cast<const Derived*>(this)->data[1]; }
		reference zxxx() { return static_cast<Derived*>(this)->data[2]; }
		const_reference zxxx() const { return static_cast<const Derived*>(this)->data[2]; }
		reference xzxx() { return static_cast<Derived*>(this)->data[2]; }
		const_reference xzxx() const { return static_cast<const Derived*>(this)->data[2]; }
		reference xxzx() { return static_cast<Derived*>(this)->data[2]; }
		const_reference xxzx() const { return static_cast<const Derived*>(this)->data[2]; }
		reference xxxz() { return static_cast<Derived*>(this)->data[2]; }
		const_reference xxxz() const { return static_cast<const Derived*>(this)->data[2]; }
		reference wxxx() { return static_cast<Derived*>(this)->data[3]; }
		const_reference wxxx() const { return static_cast<const Derived*>(this)->data[3]; }
		reference xwxx() { return static_cast<Derived*>(this)->data[3]; }
		const_reference xwxx() const { return static_cast<const Derived*>(this)->data[3]; }
		reference xxwx() { return static_cast<Derived*>(this)->data[3]; }
		const_reference xxwx() const { return static_cast<const Derived*>(this)->data[3]; }
		reference xxxw() { return static_cast<Derived*>(this)->data[3]; }
		const_reference xxxw() const { return static_cast<const Derived*>(this)->data[3]; }
		reference yyxx() { return static_cast<Derived*>(this)->data[4]; }
		const_reference yyxx() const { return static_cast<const Derived*>(this)->data[4]; }
		reference yxyx() { return static_cast<Derived*>(this)->data[4]; }
		const_reference yxyx() const { return static_cast<const Derived*>(this)->data[4]; }
		reference yxxy() { return static_cast<Derived*>(this)->data[4]; }
		const_reference yxxy() const { return static_cast<const Derived*>(this)->data[4]; }
		reference xyyx() { return static_cast<Derived*>(this)->data[4]; }
		const_reference xyyx() const { return static_cast<const Derived*>(this)->data[4]; }
		reference xyxy() { return static_cast<Derived*>(this)->data[4]; }
		const_reference xyxy() const { return static_cast<const Derived*>(this)->data[4]; }
		reference xxyy() { return static_cast<Derived*>(this)->data[4]; }
		const_reference xxyy() const { return static_cast<const Derived*>(this)->data[4]; }
		reference zyxx() { return static_cast<Derived*>(this)->data[5]; }
		const_reference zyxx() const { return static_cast<const Derived*>(this)->data[5]; }
		reference zxyx() { return static_cast<Derived*>(this)->data[5]; }
		const_reference zxyx() const { return static_cast<const Derived*>(this)->data[5]; }
		reference zxxy() { return static_cast<Derived*>(this)->data[5]; }
		const_reference zxxy() const { return static_cast<const Derived*>(this)->data[5]; }
		reference yzxx() { return static_cast<Derived*>(this)->data[5]; }
		const_reference yzxx() const { return static_cast<const Derived*>(this)->data[5]; }
		reference yxzx() { return static_cast<Derived*>(this)->data[5]; }
		const_reference yxzx() const { return static_cast<const Derived*>(this)->data[5]; }
		reference yxxz() { return static_cast<Derived*>(this)->data[5]; }
		const_reference yxxz() const { return static_cast<const Derived*>(this)->data[5]; }
		reference xzyx() { return static_cast<Derived*>(this)->data[5]; }
		const_reference xzyx() const { return static_cast<const Derived*>(this)->data[5]; }
		reference xzxy() { return static_cast<Derived*>(this)->data[5]; }
		const_reference xzxy() const { return static_cast<const Derived*>(this)->data[5]; }
		reference xyzx() { return static_cast<Derived*>(this)->data[5]; }
		const_reference xyzx() const { return static_cast<const Derived*>(this)->data[5]; }
		reference xyxz() { return static_cast<Derived*>(this)->data[5]; }
		const_reference xyxz() const { return static_cast<const Derived*>(this)->data[5]; }
		reference xxzy() { return static_cast<Derived*>(this)->data[5]; }
		const_reference xxzy() const { return static_cast<const Derived*>(this)->data[5]; }
		reference xxyz() { return static_cast<Derived*>(this)->data[5]; }
		const_reference xxyz() const { return static_cast<const Derived*>(this)->data[5]; }
		reference wyxx() { return static_cast<Derived*>(this)->data[6]; }
		const_reference wyxx() const { return static_cast<const Derived*>(this)->data[6]; }
		reference wxyx() { return static_cast<Derived*>(this)->data[6]; }
		const_reference wxyx() const { return static_cast<const Derived*>(this)->data[6]; }
		reference wxxy() { return static_cast<Derived*>(this)->data[6]; }
		const_reference wxxy() const { return static_cast<const Derived*>(this)->data[6]; }
		reference ywxx() { return static_cast<Derived*>(this)->data[6]; }
		const_reference ywxx() const { return static_cast<const Derived*>(this)->data[6]; }
		reference yxwx() { return static_cast<Derived*>(this)->data[6]; }
		const_reference yxwx() const { return static_cast<const Derived*>(this)->data[6]; }
		reference yxxw() { return static_cast<Derived*>(this)->data[6]; }
		const_reference yxxw() const { return static_cast<const Derived*>(this)->data[6]; }
		reference xwyx() { return static_cast<Derived*>(this)->data[6]; }
		const_reference xwyx() const { return static_cast<const Derived*>(this)->data[6]; }
		reference xwxy() { return static_cast<Derived*>(this)->data[6]; }
		const_reference xwxy() const { return static_cast<const Derived*>(this)->data[6]; }
		reference xywx() { return static_cast<Derived*>(this)->data[6]; }
		const_reference xywx() const { return static_cast<const Derived*>(this)->data[6]; }
		reference xyxw() { return static_cast<Derived*>(this)->data[6]; }
		const_reference xyxw() const { return static_cast<const Derived*>(this)->data[6]; }
		reference xxwy() { return static_cast<Derived*>(this)->data[6]; }
		const_reference xxwy() const { return static_cast<const Derived*>(this)->data[6]; }
		reference xxyw() { return static_cast<Derived*>(this)->data[6]; }
		const_reference xxyw() const { return static_cast<const Derived*>(this)->data[6]; }
		reference zzxx() { return static_cast<Derived*>(this)->data[7]; }
		const_reference zzxx() const { return static_cast<const Derived*>(this)->data[7]; }
		reference zxzx() { return static_cast<Derived*>(this)->data[7]; }
		const_reference zxzx() const { return static_cast<const Derived*>(this)->data[7]; }
		reference zxxz() { return static_cast<Derived*>(this)->data[7]; }
		const_reference zxxz() const { return static_cast<const Derived*>(this)->data[7]; }
		reference xzzx() { return static_cast<Derived*>(this)->data[7]; }
		const_reference xzzx() const { return static_cast<const Derived*>(this)->data[7]; }
		reference xzxz() { return static_cast<Derived*>(this)->data[7]; }
		const_reference xzxz() const { return static_cast<const Derived*>(this)->data[7]; }
		reference xxzz() { return static_cast<Derived*>(this)->data[7]; }
		const_reference xxzz() const { return static_cast<const Derived*>(this)->data[7]; }
		reference wzxx() { return static_cast<Derived*>(this)->data[8]; }
		const_reference wzxx() const { return static_cast<const Derived*>(this)->data[8]; }
		reference wxzx() { return static_cast<Derived*>(this)->data[8]; }
		const_reference wxzx() const { return static_cast<const Derived*>(this)->data[8]; }
		reference wxxz() { return static_cast<Derived*>(this)->data[8]; }
		const_reference wxxz() const { return static_cast<const Derived*>(this)->data[8]; }
		reference zwxx() { return static_cast<Derived*>(this)->data[8]; }
		const_reference zwxx() const { return static_cast<const Derived*>(this)->data[8]; }
		reference zxwx() { return static_cast<Derived*>(this)->data[8]; }
		const_reference zxwx() const { return static_cast<const Derived*>(this)->data[8]; }
		reference zxxw() { return static_cast<Derived*>(this)->data[8]; }
		const_reference zxxw() const { return static_cast<const Derived*>(this)->data[8]; }
		reference xwzx() { return static_cast<Derived*>(this)->data[8]; }
		const_reference xwzx() const { return static_cast<const Derived*>(this)->data[8]; }
		reference xwxz() { return static_cast<Derived*>(this)->data[8]; }
		const_reference xwxz() const { return static_cast<const Derived*>(this)->data[8]; }
		reference xzwx() { return static_cast<Derived*>(this)->data[8]; }
		const_reference xzwx() const { return static_cast<const Derived*>(this)->data[8]; }
		reference xzxw() { return static_cast<Derived*>(this)->data[8]; }
		const_reference xzxw() const { return static_cast<const Derived*>(this)->data[8]; }
		reference xxwz() { return static_cast<Derived*>(this)->data[8]; }
		const_reference xxwz() const { return static_cast<const Derived*>(this)->data[8]; }
		reference xxzw() { return static_cast<Derived*>(this)->data[8]; }
		const_reference xxzw() const { return static_cast<const Derived*>(this)->data[8]; }
		reference wwxx() { return static_cast<Derived*>(this)->data[9]; }
		const_reference wwxx() const { return static_cast<const Derived*>(this)->data[9]; }
		reference wxwx() { return static_cast<Derived*>(this)->data[9]; }
		const_reference wxwx() const { return static_cast<const Derived*>(this)->data[9]; }
		reference wxxw() { return static_cast<Derived*>(this)->data[9]; }
		const_reference wxxw() const { return static_cast<const Derived*>(this)->data[9]; }
		reference xwwx() { return static_cast<Derived*>(this)->data[9]; }
		const_reference xwwx() const { return static_cast<const Derived*>(this)->data[9]; }
		reference xwxw() { return static_cast<Derived*>(this)->data[9]; }
		const_reference xwxw() const { return static_cast<const Derived*>(this)->data[9]; }
		reference xxww() { return static_cast<Derived*>(this)->data[9]; }
		const_reference xxww() const { return static_cast<const Derived*>(this)->data[9]; }
		reference yyyx() { return static_cast<Derived*>(this)->data[10]; }
		const_reference yyyx() const { return static_cast<const Derived*>(this)->data[10]; }
		reference yyxy() { return static_cast<Derived*>(this)->data[10]; }
		const_reference yyxy() const { return static_cast<const Derived*>(this)->data[10]; }
		reference yxyy() { return static_cast<Derived*>(this)->data[10]; }
		const_reference yxyy() const { return static_cast<const Derived*>(this)->data[10]; }
		reference xyyy() { return static_cast<Derived*>(this)->data[10]; }
		const_reference xyyy() const { return static_cast<const Derived*>(this)->data[10]; }
		reference zyyx() { return static_cast<Derived*>(this)->data[11]; }
		const_reference zyyx() const { return static_cast<const Derived*>(this)->data[11]; }
		reference zyxy() { return static_cast<Derived*>(this)->data[11]; }
		const_reference zyxy() const { return static_cast<const Derived*>(this)->data[11]; }
		reference zxyy() { return static_cast<Derived*>(this)->data[11]; }
		const_reference zxyy() const { return static_cast<const Derived*>(this)->data[11]; }
		reference yzyx() { return static_cast<Derived*>(this)->data[11]; }
		const_reference yzyx() const { return static_cast<const Derived*>(this)->data[11]; }
		reference yzxy() { return static_cast<Derived*>(this)->data[11]; }
		const_reference yzxy() const { return static_cast<const Derived*>(this)->data[11]; }
		reference yyzx() { return static_cast<Derived*>(this)->data[11]; }
		const_reference yyzx() const { return static_cast<const Derived*>(this)->data[11]; }
		reference yyxz() { return static_cast<Derived*>(this)->data[11]; }
		const_reference yyxz() const { return static_cast<const Derived*>(this)->data[11]; }
		reference yxzy() { return static_cast<Derived*>(this)->data[11]; }
		const_reference yxzy() const { return static_cast<const Derived*>(this)->data[11]; }
		reference yxyz() { return static_cast<Derived*>(this)->data[11]; }
		const_reference yxyz() const { return static_cast<const Derived*>(this)->data[11]; }
		reference xzyy() { return static_cast<Derived*>(this)->data[11]; }
		const_reference xzyy() const { return static_cast<const Derived*>(this)->data[11]; }
		reference xyzy() { return static_cast<Derived*>(this)->data[11]; }
		const_reference xyzy() const { return static_cast<const Derived*>(this)->data[11]; }
		reference xyyz() { return static_cast<Derived*>(this)->data[11]; }
		const_reference xyyz() const { return static_cast<const Derived*>(this)->data[11]; }
		reference wyyx() { return static_cast<Derived*>(this)->data[12]; }
		const_reference wyyx() const { return static_cast<const Derived*>(this)->data[12]; }
		reference wyxy() { return static_cast<Derived*>(this)->data[12]; }
		const_reference wyxy() const { return static_cast<const Derived*>(this)->data[12]; }
		reference wxyy() { return static_cast<Derived*>(this)->data[12]; }
		const_reference wxyy() const { return static_cast<const Derived*>(this)->data[12]; }
		reference ywyx() { return static_cast<Derived*>(this)->data[12]; }
		const_reference ywyx() const { return static_cast<const Derived*>(this)->data[12]; }
		reference ywxy() { return static_cast<Derived*>(this)->data[12]; }
		const_reference ywxy() const { return static_cast<const Derived*>(this)->data[12]; }
		reference yywx() { return static_cast<Derived*>(this)->data[12]; }
		const_reference yywx() const { return static_cast<const Derived*>(this)->data[12]; }
		reference yyxw() { return static_cast<Derived*>(this)->data[12]; }
		const_reference yyxw() const { return static_cast<const Derived*>(this)->data[12]; }
		reference yxwy() { return static_cast<Derived*>(this)->data[12]; }
		const_reference yxwy() const { return static_cast<const Derived*>(this)->data[12]; }
		reference yxyw() { return static_cast<Derived*>(this)->data[12]; }
		const_reference yxyw() const { return static_cast<const Derived*>(this)->data[12]; }
		reference xwyy() { return static_cast<Derived*>(this)->data[12]; }
		const_reference xwyy() const { return static_cast<const Derived*>(this)->data[12]; }
		reference xywy() { return static_cast<Derived*>(this)->data[12]; }
		const_reference xywy() const { return static_cast<const Derived*>(this)->data[12]; }
		reference xyyw() { return static_cast<Derived*>(this)->data[12]; }
		const_reference xyyw() const { return static_cast<const Derived*>(this)->data[12]; }
		reference zzyx() { return static_cast<Derived*>(this)->data[13]; }
		const_reference zzyx() const { return static_cast<const Derived*>(this)->data[13]; }
		reference zzxy() { return static_cast<Derived*>(this)->data[13]; }
		const_reference zzxy() const { return static_cast<const Derived*>(this)->data[13]; }
		reference zyzx() { return static_cast<Derived*>(this)->data[13]; }
		const_reference zyzx() const { return static_cast<const Derived*>(this)->data[13]; }
		reference zyxz() { return static_cast<Derived*>(this)->data[13]; }
		const_reference zyxz() const { return static_cast<const Derived*>(this)->data[13]; }
		reference zxzy() { return static_cast<Derived*>(this)->data[13]; }
		const_reference zxzy() const { return static_cast<const Derived*>(this)->data[13]; }
		reference zxyz() { return static_cast<Derived*>(this)->data[13]; }
		const_reference zxyz() const { return static_cast<const Derived*>(this)->data[13]; }
		reference yzzx() { return static_cast<Derived*>(this)->data[13]; }
		const_reference yzzx() const { return static_cast<const Derived*>(this)->data[13]; }
		reference yzxz() { return static_cast<Derived*>(this)->data[13]; }
		const_reference yzxz() const { return static_cast<const Derived*>(this)->data[13]; }
		reference yxzz() { return static_cast<Derived*>(this)->data[13]; }
		const_reference yxzz() const { return static_cast<const Derived*>(this)->data[13]; }
		reference xzzy() { return static_cast<Derived*>(this)->data[13]; }
		const_reference xzzy() const { return static_cast<const Derived*>(this)->data[13]; }
		reference xzyz() { return static_cast<Derived*>(this)->data[13]; }
		const_reference xzyz() const { return static_cast<const Derived*>(this)->data[13]; }
		reference xyzz() { return static_cast<Derived*>(this)->data[13]; }
		const_reference xyzz() const { return static_cast<const Derived*>(this)->data[13]; }
		reference wzyx() { return static_cast<Derived*>(this)->data[14]; }
		const_reference wzyx() const { return static_cast<const Derived*>(this)->data[14]; }
		reference wzxy() { return static_cast<Derived*>(this)->data[14]; }
		const_reference wzxy() const { return static_cast<const Derived*>(this)->data[14]; }
		reference wyzx() { return static_cast<Derived*>(this)->data[14]; }
		const_reference wyzx() const { return static_cast<const Derived*>(this)->data[14]; }
		reference wyxz() { return static_cast<Derived*>(this)->data[14]; }
		const_reference wyxz() const { return static_cast<const Derived*>(this)->data[14]; }
		reference wxzy() { return static_cast<Derived*>(this)->data[14]; }
		const_reference wxzy() const { return static_cast<const Derived*>(this)->data[14]; }
		reference wxyz() { return static_cast<Derived*>(this)->data[14]; }
		const_reference wxyz() const { return static_cast<const Derived*>(this)->data[14]; }
		reference zwyx() { return static_cast<Derived*>(this)->data[14]; }
		const_reference zwyx() const { return static_cast<const Derived*>(this)->data[14]; }
		reference zwxy() { return static_cast<Derived*>(this)->data[14]; }
		const_reference zwxy() const { return static_cast<const Derived*>(this)->data[14]; }
		reference zywx() { return static_cast<Derived*>(this)->data[14]; }
		const_reference zywx() const { return static_cast<const Derived*>(this)->data[14]; }
		reference zyxw() { return static_cast<Derived*>(this)->data[14]; }
		const_reference zyxw() const { return static_cast<const Derived*>(this)->data[14]; }
		reference zxwy() { return static_cast<Derived*>(this)->data[14]; }
		const_reference zxwy() const { return static_cast<const Derived*>(this)->data[14]; }
		reference zxyw() { return static_cast<Derived*>(this)->data[14]; }
		const_reference zxyw() const { return static_cast<const Derived*>(this)->data[14]; }
		reference ywzx() { return static_cast<Derived*>(this)->data[14]; }
		const_reference ywzx() const { return static_cast<const Derived*>(this)->data[14]; }
		reference ywxz() { return static_cast<Derived*>(this)->data[14]; }
		const_reference ywxz() const { return static_cast<const Derived*>(this)->data[14]; }
		reference yzwx() { return static_cast<Derived*>(this)->data[14]; }
		const_reference yzwx() const { return static_cast<const Derived*>(this)->data[14]; }
		reference yzxw() { return static_cast<Derived*>(this)->data[14]; }
		const_reference yzxw() const { return static_cast<const Derived*>(this)->data[14]; }
		reference yxwz() { return static_cast<Derived*>(this)->data[14]; }
		const_reference yxwz() const { return static_cast<const Derived*>(this)->data[14]; }
		reference yxzw() { return static_cast<Derived*>(this)->data[14]; }
		const_reference yxzw() const { return static_cast<const Derived*>(this)->data[14]; }
		reference xwzy() { return static_cast<Derived*>(this)->data[14]; }
		const_reference xwzy() const { return static_cast<const Derived*>(this)->data[14]; }
		reference xwyz() { return static_cast<Derived*>(this)->data[14]; }
		const_reference xwyz() const { return static_cast<const Derived*>(this)->data[14]; }
		reference xzwy() { return static_cast<Derived*>(this)->data[14]; }
		const_reference xzwy() const { return static_cast<const Derived*>(this)->data[14]; }
		reference xzyw() { return static_cast<Derived*>(this)->data[14]; }
		const_reference xzyw() const { return static_cast<const Derived*>(this)->data[14]; }
		reference xywz() { return static_cast<Derived*>(this)->data[14]; }
		const_reference xywz() const { return static_cast<const Derived*>(this)->data[14]; }
		reference xyzw() { return static_cast<Derived*>(this)->data[14]; }
		const_reference xyzw() const { return static_cast<const Derived*>(this)->data[14]; }
		reference wwyx() { return static_cast<Derived*>(this)->data[15]; }
		const_reference wwyx() const { return static_cast<const Derived*>(this)->data[15]; }
		reference wwxy() { return static_cast<Derived*>(this)->data[15]; }
		const_reference wwxy() const { return static_cast<const Derived*>(this)->data[15]; }
		reference wywx() { return static_cast<Derived*>(this)->data[15]; }
		const_reference wywx() const { return static_cast<const Derived*>(this)->data[15]; }
		reference wyxw() { return static_cast<Derived*>(this)->data[15]; }
		const_reference wyxw() const { return static_cast<const Derived*>(this)->data[15]; }
		reference wxwy() { return static_cast<Derived*>(this)->data[15]; }
		const_reference wxwy() const { return static_cast<const Derived*>(this)->data[15]; }
		reference wxyw() { return static_cast<Derived*>(this)->data[15]; }
		const_reference wxyw() const { return static_cast<const Derived*>(this)->data[15]; }
		reference ywwx() { return static_cast<Derived*>(this)->data[15]; }
		const_reference ywwx() const { return static_cast<const Derived*>(this)->data[15]; }
		reference ywxw() { return static_cast<Derived*>(this)->data[15]; }
		const_reference ywxw() const { return static_cast<const Derived*>(this)->data[15]; }
		reference yxww() { return static_cast<Derived*>(this)->data[15]; }
		const_reference yxww() const { return static_cast<const Derived*>(this)->data[15]; }
		reference xwwy() { return static_cast<Derived*>(this)->data[15]; }
		const_reference xwwy() const { return static_cast<const Derived*>(this)->data[15]; }
		reference xwyw() { return static_cast<Derived*>(this)->data[15]; }
		const_reference xwyw() const { return static_cast<const Derived*>(this)->data[15]; }
		reference xyww() { return static_cast<Derived*>(this)->data[15]; }
		const_reference xyww() const { return static_cast<const Derived*>(this)->data[15]; }
		reference zzzx() { return static_cast<Derived*>(this)->data[16]; }
		const_reference zzzx() const { return static_cast<const Derived*>(this)->data[16]; }
		reference zzxz() { return static_cast<Derived*>(this)->data[16]; }
		const_reference zzxz() const { return static_cast<const Derived*>(this)->data[16]; }
		reference zxzz() { return static_cast<Derived*>(this)->data[16]; }
		const_reference zxzz() const { return static_cast<const Derived*>(this)->data[16]; }
		reference xzzz() { return static_cast<Derived*>(this)->data[16]; }
		const_reference xzzz() const { return static_cast<const Derived*>(this)->data[16]; }
		reference wzzx() { return static_cast<Derived*>(this)->data[17]; }
		const_reference wzzx() const { return static_cast<const Derived*>(this)->data[17]; }
		reference wzxz() { return static_cast<Derived*>(this)->data[17]; }
		const_reference wzxz() const { return static_cast<const Derived*>(this)->data[17]; }
		reference wxzz() { return static_cast<Derived*>(this)->data[17]; }
		const_reference wxzz() const { return static_cast<const Derived*>(this)->data[17]; }
		reference zwzx() { return static_cast<Derived*>(this)->data[17]; }
		const_reference zwzx() const { return static_cast<const Derived*>(this)->data[17]; }
		reference zwxz() { return static_cast<Derived*>(this)->data[17]; }
		const_reference zwxz() const { return static_cast<const Derived*>(this)->data[17]; }
		reference zzwx() { return static_cast<Derived*>(this)->data[17]; }
		const_reference zzwx() const { return static_cast<const Derived*>(this)->data[17]; }
		reference zzxw() { return static_cast<Derived*>(this)->data[17]; }
		const_reference zzxw() const { return static_cast<const Derived*>(this)->data[17]; }
		reference zxwz() { return static_cast<Derived*>(this)->data[17]; }
		const_reference zxwz() const { return static_cast<const Derived*>(this)->data[17]; }
		reference zxzw() { return static_cast<Derived*>(this)->data[17]; }
		const_reference zxzw() const { return static_cast<const Derived*>(this)->data[17]; }
		reference xwzz() { return static_cast<Derived*>(this)->data[17]; }
		const_reference xwzz() const { return static_cast<const Derived*>(this)->data[17]; }
		reference xzwz() { return static_cast<Derived*>(this)->data[17]; }
		const_reference xzwz() const { return static_cast<const Derived*>(this)->data[17]; }
		reference xzzw() { return static_cast<Derived*>(this)->data[17]; }
		const_reference xzzw() const { return static_cast<const Derived*>(this)->data[17]; }
		reference wwzx() { return static_cast<Derived*>(this)->data[18]; }
		const_reference wwzx() const { return static_cast<const Derived*>(this)->data[18]; }
		reference wwxz() { return static_cast<Derived*>(this)->data[18]; }
		const_reference wwxz() const { return static_cast<const Derived*>(this)->data[18]; }
		reference wzwx() { return static_cast<Derived*>(this)->data[18]; }
		const_reference wzwx() const { return static_cast<const Derived*>(this)->data[18]; }
		reference wzxw() { return static_cast<Derived*>(this)->data[18]; }
		const_reference wzxw() const { return static_cast<const Derived*>(this)->data[18]; }
		reference wxwz() { return static_cast<Derived*>(this)->data[18]; }
		const_reference wxwz() const { return static_cast<const Derived*>(this)->data[18]; }
		reference wxzw() { return static_cast<Derived*>(this)->data[18]; }
		const_reference wxzw() const { return static_cast<const Derived*>(this)->data[18]; }
		reference zwwx() { return static_cast<Derived*>(this)->data[18]; }
		const_reference zwwx() const { return static_cast<const Derived*>(this)->data[18]; }
		reference zwxw() { return static_cast<Derived*>(this)->data[18]; }
		const_reference zwxw() const { return static_cast<const Derived*>(this)->data[18]; }
		reference zxww() { return static_cast<Derived*>(this)->data[18]; }
		const_reference zxww() const { return static_cast<const Derived*>(this)->data[18]; }
		reference xwwz() { return static_cast<Derived*>(this)->data[18]; }
		const_reference xwwz() const { return static_cast<const Derived*>(this)->data[18]; }
		reference xwzw() { return static_cast<Derived*>(this)->data[18]; }
		const_reference xwzw() const { return static_cast<const Derived*>(this)->data[18]; }
		reference xzww() { return static_cast<Derived*>(this)->data[18]; }
		const_reference xzww() const { return static_cast<const Derived*>(this)->data[18]; }
		reference wwwx() { return static_cast<Derived*>(this)->data[19]; }
		const_reference wwwx() const { return static_cast<const Derived*>(this)->data[19]; }
		reference wwxw() { return static_cast<Derived*>(this)->data[19]; }
		const_reference wwxw() const { return static_cast<const Derived*>(this)->data[19]; }
		reference wxww() { return static_cast<Derived*>(this)->data[19]; }
		const_reference wxww() const { return static_cast<const Derived*>(this)->data[19]; }
		reference xwww() { return static_cast<Derived*>(this)->data[19]; }
		const_reference xwww() const { return static_cast<const Derived*>(this)->data[19]; }
		reference yyyy() { return static_cast<Derived*>(this)->data[20]; }
		const_reference yyyy() const { return static_cast<const Derived*>(this)->data[20]; }
		reference zyyy() { return static_cast<Derived*>(this)->data[21]; }
		const_reference zyyy() const { return static_cast<const Derived*>(this)->data[21]; }
		reference yzyy() { return static_cast<Derived*>(this)->data[21]; }
		const_reference yzyy() const { return static_cast<const Derived*>(this)->data[21]; }
		reference yyzy() { return static_cast<Derived*>(this)->data[21]; }
		const_reference yyzy() const { return static_cast<const Derived*>(this)->data[21]; }
		reference yyyz() { return static_cast<Derived*>(this)->data[21]; }
		const_reference yyyz() const { return static_cast<const Derived*>(this)->data[21]; }
		reference wyyy() { return static_cast<Derived*>(this)->data[22]; }
		const_reference wyyy() const { return static_cast<const Derived*>(this)->data[22]; }
		reference ywyy() { return static_cast<Derived*>(this)->data[22]; }
		const_reference ywyy() const { return static_cast<const Derived*>(this)->data[22]; }
		reference yywy() { return static_cast<Derived*>(this)->data[22]; }
		const_reference yywy() const { return static_cast<const Derived*>(this)->data[22]; }
		reference yyyw() { return static_cast<Derived*>(this)->data[22]; }
		const_reference yyyw() const { return static_cast<const Derived*>(this)->data[22]; }
		reference zzyy() { return static_cast<Derived*>(this)->data[23]; }
		const_reference zzyy() const { return static_cast<const Derived*>(this)->data[23]; }
		reference zyzy() { return static_cast<Derived*>(this)->data[23]; }
		const_reference zyzy() const { return static_cast<const Derived*>(this)->data[23]; }
		reference zyyz() { return static_cast<Derived*>(this)->data[23]; }
		const_reference zyyz() const { return static_cast<const Derived*>(this)->data[23]; }
		reference yzzy() { return static_cast<Derived*>(this)->data[23]; }
		const_reference yzzy() const { return static_cast<const Derived*>(this)->data[23]; }
		reference yzyz() { return static_cast<Derived*>(this)->data[23]; }
		const_reference yzyz() const { return static_cast<const Derived*>(this)->data[23]; }
		reference yyzz() { return static_cast<Derived*>(this)->data[23]; }
		const_reference yyzz() const { return static_cast<const Derived*>(this)->data[23]; }
		reference wzyy() { return static_cast<Derived*>(this)->data[24]; }
		const_reference wzyy() const { return static_cast<const Derived*>(this)->data[24]; }
		reference wyzy() { return static_cast<Derived*>(this)->data[24]; }
		const_reference wyzy() const { return static_cast<const Derived*>(this)->data[24]; }
		reference wyyz() { return static_cast<Derived*>(this)->data[24]; }
		const_reference wyyz() const { return static_cast<const Derived*>(this)->data[24]; }
		reference zwyy() { return static_cast<Derived*>(this)->data[24]; }
		const_reference zwyy() const { return static_cast<const Derived*>(this)->data[24]; }
		reference zywy() { return static_cast<Derived*>(this)->data[24]; }
		const_reference zywy() const { return static_cast<const Derived*>(this)->data[24]; }
		reference zyyw() { return static_cast<Derived*>(this)->data[24]; }
		const_reference zyyw() const { return static_cast<const Derived*>(this)->data[24]; }
		reference ywzy() { return static_cast<Derived*>(this)->data[24]; }
		const_reference ywzy() const { return static_cast<const Derived*>(this)->data[24]; }
		reference ywyz() { return static_cast<Derived*>(this)->data[24]; }
		const_reference ywyz() const { return static_cast<const Derived*>(this)->data[24]; }
		reference yzwy() { return static_cast<Derived*>(this)->data[24]; }
		const_reference yzwy() const { return static_cast<const Derived*>(this)->data[24]; }
		reference yzyw() { return static_cast<Derived*>(this)->data[24]; }
		const_reference yzyw() const { return static_cast<const Derived*>(this)->data[24]; }
		reference yywz() { return static_cast<Derived*>(this)->data[24]; }
		const_reference yywz() const { return static_cast<const Derived*>(this)->data[24]; }
		reference yyzw() { return static_cast<Derived*>(this)->data[24]; }
		const_reference yyzw() const { return static_cast<const Derived*>(this)->data[24]; }
		reference wwyy() { return static_cast<Derived*>(this)->data[25]; }
		const_reference wwyy() const { return static_cast<const Derived*>(this)->data[25]; }
		reference wywy() { return static_cast<Derived*>(this)->data[25]; }
		const_reference wywy() const { return static_cast<const Derived*>(this)->data[25]; }
		reference wyyw() { return static_cast<Derived*>(this)->data[25]; }
		const_reference wyyw() const { return static_cast<const Derived*>(this)->data[25]; }
		reference ywwy() { return static_cast<Derived*>(this)->data[25]; }
		const_reference ywwy() const { return static_cast<const Derived*>(this)->data[25]; }
		reference ywyw() { return static_cast<Derived*>(this)->data[25]; }
		const_reference ywyw() const { return static_cast<const Derived*>(this)->data[25]; }
		reference yyww() { return static_cast<Derived*>(this)->data[25]; }
		const_reference yyww() const { return static_cast<const Derived*>(this)->data[25]; }
		reference zzzy() { return static_cast<Derived*>(this)->data[26]; }
		const_reference zzzy() const { return static_cast<const Derived*>(this)->data[26]; }
		reference zzyz() { return static_cast<Derived*>(this)->data[26]; }
		const_reference zzyz() const { return static_cast<const Derived*>(this)->data[26]; }
		reference zyzz() { return static_cast<Derived*>(this)->data[26]; }
		const_reference zyzz() const { return static_cast<const Derived*>(this)->data[26]; }
		reference yzzz() { return static_cast<Derived*>(this)->data[26]; }
		const_reference yzzz() const { return static_cast<const Derived*>(this)->data[26]; }
		reference wzzy() { return static_cast<Derived*>(this)->data[27]; }
		const_reference wzzy() const { return static_cast<const Derived*>(this)->data[27]; }
		reference wzyz() { return static_cast<Derived*>(this)->data[27]; }
		const_reference wzyz() const { return static_cast<const Derived*>(this)->data[27]; }
		reference wyzz() { return static_cast<Derived*>(this)->data[27]; }
		const_reference wyzz() const { return static_cast<const Derived*>(this)->data[27]; }
		reference zwzy() { return static_cast<Derived*>(this)->data[27]; }
		const_reference zwzy() const { return static_cast<const Derived*>(this)->data[27]; }
		reference zwyz() { return static_cast<Derived*>(this)->data[27]; }
		const_reference zwyz() const { return static_cast<const Derived*>(this)->data[27]; }
		reference zzwy() { return static_cast<Derived*>(this)->data[27]; }
		const_reference zzwy() const { return static_cast<const Derived*>(this)->data[27]; }
		reference zzyw() { return static_cast<Derived*>(this)->data[27]; }
		const_reference zzyw() const { return static_cast<const Derived*>(this)->data[27]; }
		reference zywz() { return static_cast<Derived*>(this)->data[27]; }
		const_reference zywz() const { return static_cast<const Derived*>(this)->data[27]; }
		reference zyzw() { return static_cast<Derived*>(this)->data[27]; }
		const_reference zyzw() const { return static_cast<const Derived*>(this)->data[27]; }
		reference ywzz() { return static_cast<Derived*>(this)->data[27]; }
		const_reference ywzz() const { return static_cast<const Derived*>(this)->data[27]; }
		reference yzwz() { return static_cast<Derived*>(this)->data[27]; }
		const_reference yzwz() const { return static_cast<const Derived*>(this)->data[27]; }
		reference yzzw() { return static_cast<Derived*>(this)->data[27]; }
		const_reference yzzw() const { return static_cast<const Derived*>(this)->data[27]; }
		reference wwzy() { return static_cast<Derived*>(this)->data[28]; }
		const_reference wwzy() const { return static_cast<const Derived*>(this)->data[28]; }
		reference wwyz() { return static_cast<Derived*>(this)->data[28]; }
		const_reference wwyz() const { return static_cast<const Derived*>(this)->data[28]; }
		reference wzwy() { return static_cast<Derived*>(this)->data[28]; }
		const_reference wzwy() const { return static_cast<const Derived*>(this)->data[28]; }
		reference wzyw() { return static_cast<Derived*>(this)->data[28]; }
		const_reference wzyw() const { return static_cast<const Derived*>(this)->data[28]; }
		reference wywz() { return static_cast<Derived*>(this)->data[28]; }
		const_reference wywz() const { return static_cast<const Derived*>(this)->data[28]; }
		reference wyzw() { return static_cast<Derived*>(this)->data[28]; }
		const_reference wyzw() const { return static_cast<const Derived*>(this)->data[28]; }
		reference zwwy() { return static_cast<Derived*>(this)->data[28]; }
		const_reference zwwy() const { return static_cast<const Derived*>(this)->data[28]; }
		reference zwyw() { return static_cast<Derived*>(this)->data[28]; }
		const_reference zwyw() const { return static_cast<const Derived*>(this)->data[28]; }
		reference zyww() { return static_cast<Derived*>(this)->data[28]; }
		const_reference zyww() const { return static_cast<const Derived*>(this)->data[28]; }
		reference ywwz() { return static_cast<Derived*>(this)->data[28]; }
		const_reference ywwz() const { return static_cast<const Derived*>(this)->data[28]; }
		reference ywzw() { return static_cast<Derived*>(this)->data[28]; }
		const_reference ywzw() const { return static_cast<const Derived*>(this)->data[28]; }
		reference yzww() { return static_cast<Derived*>(this)->data[28]; }
		const_reference yzww() const { return static_cast<const Derived*>(this)->data[28]; }
		reference wwwy() { return static_cast<Derived*>(this)->data[29]; }
		const_reference wwwy() const { return static_cast<const Derived*>(this)->data[29]; }
		reference wwyw() { return static_cast<Derived*>(this)->data[29]; }
		const_reference wwyw() const { return static_cast<const Derived*>(this)->data[29]; }
		reference wyww() { return static_cast<Derived*>(this)->data[29]; }
		const_reference wyww() const { return static_cast<const Derived*>(this)->data[29]; }
		reference ywww() { return static_cast<Derived*>(this)->data[29]; }
		const_reference ywww() const { return static_cast<const Derived*>(this)->data[29]; }
		reference zzzz() { return static_cast<Derived*>(this)->data[30]; }
		const_reference zzzz() const { return static_cast<const Derived*>(this)->data[30]; }
		reference wzzz() { return static_cast<Derived*>(this)->data[31]; }
		const_reference wzzz() const { return static_cast<const Derived*>(this)->data[31]; }
		reference zwzz() { return static_cast<Derived*>(this)->data[31]; }
		const_reference zwzz() const { return static_cast<const Derived*>(this)->data[31]; }
		reference zzwz() { return static_cast<Derived*>(this)->data[31]; }
		const_reference zzwz() const { return static_cast<const Derived*>(this)->data[31]; }
		reference zzzw() { return static_cast<Derived*>(this)->data[31]; }
		const_reference zzzw() const { return static_cast<const Derived*>(this)->data[31]; }
		reference wwzz() { return static_cast<Derived*>(this)->data[32]; }
		const_reference wwzz() const { return static_cast<const Derived*>(this)->data[32]; }
		reference wzwz() { return static_cast<Derived*>(this)->data[32]; }
		const_reference wzwz() const { return static_cast<const Derived*>(this)->data[32]; }
		reference wzzw() { return static_cast<Derived*>(this)->data[32]; }
		const_reference wzzw() const { return static_cast<const Derived*>(this)->data[32]; }
		reference zwwz() { return static_cast<Derived*>(this)->data[32]; }
		const_reference zwwz() const { return static_cast<const Derived*>(this)->data[32]; }
		reference zwzw() { return static_cast<Derived*>(this)->data[32]; }
		const_reference zwzw() const { return static_cast<const Derived*>(this)->data[32]; }
		reference zzww() { return static_cast<Derived*>(this)->data[32]; }
		const_reference zzww() const { return static_cast<const Derived*>(this)->data[32]; }
		reference wwwz() { return static_cast<Derived*>(this)->data[33]; }
		const_reference wwwz() const { return static_cast<const Derived*>(this)->data[33]; }
		reference wwzw() { return static_cast<Derived*>(this)->data[33]; }
		const_reference wwzw() const { return static_cast<const Derived*>(this)->data[33]; }
		reference wzww() { return static_cast<Derived*>(this)->data[33]; }
		const_reference wzww() const { return static_cast<const Derived*>(this)->data[33]; }
		reference zwww() { return static_cast<Derived*>(this)->data[33]; }
		const_reference zwww() const { return static_cast<const Derived*>(this)->data[33]; }
		reference wwww() { return static_cast<Derived*>(this)->data[34]; }
		const_reference wwww() const { return static_cast<const Derived*>(this)->data[34]; }
	};


} // namespace detail

} // namespace lb

#endif // LB_INCLUDED_TENSOR_TENSOR_BASE_HPP
