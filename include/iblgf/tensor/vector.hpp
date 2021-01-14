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

#ifndef MATH_INCLUDED_SIMPLE_VECTOR_HPP
#define MATH_INCLUDED_SIMPLE_VECTOR_HPP

#include <array>
#include <type_traits>
#include <iostream>
#include <boost/serialization/array.hpp>

namespace iblgf
{
namespace math
{
template<typename T, std::size_t N>
class vector
{
  public: // member types
    using data_type = std::array<T, N>;
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
    vector() = default;
    vector(const T& element) { data.fill(element); }
    vector(const vector&) = default;
    vector(vector&&) = default;
    explicit vector(const T* ptr)
    {
        for (unsigned int i = 0; i < N; ++i) data[i] = ptr[i];
    }
    template<typename T2>
    vector(const vector<T2, N>& other)
    {
        for (unsigned int i = 0; i < N; ++i) data[i] = static_cast<T>(other[i]);
    }
    vector(const std::array<T, N>& _data)
    : data(_data)
    {
    }
    vector(std::array<T, N>&& _data)
    : data(std::move(_data))
    {
    }

    vector& operator=(const vector&) & = default;
    vector& operator=(vector&&) & = default;
    vector& operator=(const T& element) &
    {
        data.fill(element);
        return *this;
    }

  public: // static member functions
    static constexpr std::size_t size() noexcept { return N; }

  public: // member functions
    reference       operator[](size_type pos) noexcept { return data[pos]; }
    const_reference operator[](size_type pos) const noexcept
    {
        return data[pos];
    }

    reference       operator()(size_type i) noexcept { return data[i]; }
    const_reference operator()(size_type i) const noexcept { return data[i]; }

    reference       front() { return data.front(); }
    const_reference front() const { return data.front(); }
    reference       back() { return data.back(); }
    const_reference back() const { return data.back(); }
    reference       x() { return data[0]; }
    const_reference x() const { return data[0]; }
    reference       y() { return data[1]; }
    const_reference y() const { return data[1]; }
    reference       z() { return data[2]; }
    const_reference z() const { return data[2]; }
    reference       w() { return data[3]; }
    const_reference w() const { return data[3]; }

    friend std::ostream& operator<<(std::ostream& os, const vector& v)
    {
        os << "";
        for (unsigned int i = 0; i < N; ++i)
        {
            os << v.data[i];
            if (i < N - 1) os << " ";
        }
        os << "";
        return os;
    }

  public: // iterators
    iterator               begin() { return data.begin(); }
    const_iterator         begin() const { return data.begin(); }
    const_iterator         cbegin() const { return data.cbegin(); }
    iterator               end() { return data.end(); }
    const_iterator         end() const { return data.end(); }
    const_iterator         cend() const { return data.cend(); }
    reverse_iterator       rbegin() { return data.rbegin(); }
    const_reverse_iterator rbegin() const { return data.rbegin(); }
    const_reverse_iterator crbegin() const { return data.crbegin(); }
    reverse_iterator       rend() { return data.rend(); }
    const_reverse_iterator rend() const { return data.rend(); }
    const_reverse_iterator crend() const { return data.crend(); }

  public: // arithmetic operator overloads
    vector& operator+=(const vector& other) noexcept
    {
        for (unsigned int i = 0; i < N; ++i) data[i] += other[i];
        return *this;
    }
    vector& operator+=(const T& element) noexcept
    {
        for (unsigned int i = 0; i < N; ++i) data[i] += element;
        return *this;
    }
    vector& operator-=(const vector& other) noexcept
    {
        for (unsigned int i = 0; i < N; ++i) data[i] -= other[i];
        return *this;
    }
    vector& operator-=(const T& element) noexcept
    {
        for (unsigned int i = 0; i < N; ++i) data[i] -= element;
        return *this;
    }
    vector& operator*=(const vector& other) noexcept
    {
        for (unsigned int i = 0; i < N; ++i) data[i] *= other[i];
        return *this;
    }
    vector& operator*=(const T& element) noexcept
    {
        for (unsigned int i = 0; i < N; ++i) data[i] *= element;
        return *this;
    }
    vector& operator/=(const vector& other) noexcept
    {
        for (unsigned int i = 0; i < N; ++i) data[i] /= other[i];
        return *this;
    }
    vector& operator/=(const T& element) noexcept
    {
        for (unsigned int i = 0; i < N; ++i) data[i] /= element;
        return *this;
    }

    friend vector operator-(vector v) noexcept
    {
        for (unsigned int i = 0; i < N; ++i) v.data[i] = -v.data[i];
        return v;
    }

  public: // Rational operator overloads
    friend bool operator==(const vector& lhs, const vector& rhs) noexcept
    {
        return lhs.data == rhs.data;
    }

    friend bool operator!=(const vector& lhs, const vector& rhs) noexcept
    {
        return lhs.data != rhs.data;
    }

    friend bool operator<(const vector& lhs, const vector& rhs) noexcept
    {
        return lhs.data < rhs.data;
    }

    friend bool operator<=(const vector& lhs, const vector& rhs) noexcept
    {
        return lhs.data <= rhs.data;
    }

    friend bool operator>(const vector& lhs, const vector& rhs) noexcept
    {
        return lhs.data > rhs.data;
    }

    friend bool operator>=(const vector& lhs, const vector& rhs) noexcept
    {
        return lhs.data >= rhs.data;
    }

  private: // members
    data_type data;

    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive& ar, const unsigned int version)
    {
        ar& data;
    }
};

/****************************************************************************/
//Binary arithmetic operator overloads

template<typename U, typename V, std::size_t N>
auto
operator+(const vector<U, N>& lhs, const vector<V, N>& rhs)
{
    vector<typename std::common_type<U, V>::type, N> res(lhs);
    return res += rhs;
}
template<typename U, typename V, std::size_t N>
auto
operator+(const vector<U, N>& lhs, const V& rhs)
{
    vector<typename std::common_type<U, V>::type, N> res(lhs);
    return res += rhs;
}
template<typename U, typename V, std::size_t N>
auto
operator+(const V& lhs, const vector<U, N>& rhs)
{
    vector<typename std::common_type<U, V>::type, N> res(rhs);
    return res += lhs;
}

template<typename U, typename V, std::size_t N>
auto
operator-(const vector<U, N>& lhs, const vector<V, N>& rhs)
{
    vector<typename std::common_type<U, V>::type, N> res(lhs);
    return res -= rhs;
}
template<typename U, typename V, std::size_t N>
auto
operator-(const vector<U, N>& lhs, const V& rhs)
{
    vector<typename std::common_type<U, V>::type, N> res(lhs);
    return res -= rhs;
}
template<typename U, typename V, std::size_t N>
auto
operator-(const V& lhs, const vector<U, N>& rhs)
{
    vector<typename std::common_type<U, V>::type, N> res(rhs);
    return res -= lhs;
}

template<typename U, typename V, std::size_t N>
auto operator*(const vector<U, N>& lhs, const vector<V, N>& rhs)
{
    vector<typename std::common_type<U, V>::type, N> res(lhs);
    return res *= rhs;
}
template<typename U, typename V, std::size_t N>
auto operator*(const vector<U, N>& lhs, const V& rhs)
{
    vector<typename std::common_type<U, V>::type, N> res(lhs);
    return res *= rhs;
}
template<typename U, typename V, std::size_t N>
auto operator*(const V& lhs, const vector<U, N>& rhs)
{
    vector<typename std::common_type<U, V>::type, N> res(rhs);
    return res *= lhs;
}

template<typename U, typename V, std::size_t N>
auto
operator/(const vector<U, N>& lhs, const vector<V, N>& rhs)
{
    vector<typename std::common_type<U, V>::type, N> res(lhs);
    return res /= rhs;
}
template<typename U, typename V, std::size_t N>
auto
operator/(const vector<U, N>& lhs, const V& rhs)
{
    vector<typename std::common_type<U, V>::type, N> res(lhs);
    return res /= rhs;
}
template<typename U, typename V, std::size_t N>
auto
operator/(const V& lhs, const vector<U, N>& rhs)
{
    vector<typename std::common_type<U, V>::type, N> res(rhs);
    for (unsigned int i = 0; i < N; ++i) res[i] = lhs / res[i];
    return res;
}

/****************************************************************************/

template<typename U, typename V, std::size_t N>
auto
dot(const vector<U, N>& lhs, const vector<V, N>& rhs)
{
    typename std::common_type<U, V>::type res(lhs[0] * rhs[0]);
    for (unsigned int i = 1; i < N; ++i) res += lhs[i] * rhs[i];
    return res;
}

template<typename U, std::size_t N>
U
norm2(const vector<U, N>& v)
{
    return dot(v, v);
}

template<typename U, std::size_t N>
U
norm_inf(const vector<U, N>& v)
{
    using namespace std;
    auto m = abs(v[0]);
    for (unsigned int i = 1; i < N; ++i) m = max(m, abs(v[i]));
    return m;
}

template<typename U, typename V, std::size_t N>
typename std::enable_if<N == 3,
    vector<typename std::common_type<U, V>::type, N>>::type
cross(const vector<U, N>& lhs, const vector<V, N>& rhs)
{
    return vector<typename std::common_type<U, V>::type, N>(
        {lhs.y() * rhs.z() - lhs.z() * rhs.y(),
            lhs.z() * rhs.x() - lhs.x() * rhs.z(),
            lhs.x() * rhs.y() - lhs.y() * rhs.x()});
}

template<typename U, typename V, std::size_t N>
typename std::enable_if<N == 2, typename std::common_type<U, V>::type>::type
cross(const vector<U, N>& lhs, const vector<V, N>& rhs)
{
    return lhs.x() * rhs.y() - lhs.y() * rhs.x();
}

} // namespace math
} // namespace iblgf

namespace std
{
template<typename U, std::size_t N>
iblgf::math::vector<U, N>
max(const iblgf::math::vector<U, N>& lhs, const iblgf::math::vector<U, N>& rhs)
{
    iblgf::math::vector<U, N> res;
    for (std::size_t n = 0; n < N; ++n) res[n] = std::max(lhs[n], rhs[n]);
    return res;
}

template<typename U, std::size_t N>
iblgf::math::vector<U, N>
min(const iblgf::math::vector<U, N>& lhs, const iblgf::math::vector<U, N>& rhs)
{
    iblgf::math::vector<U, N> res;
    for (std::size_t n = 0; n < N; ++n) res[n] = std::min(lhs[n], rhs[n]);
    return res;
}

} // namespace std

#endif // MATH_INCLUDED_SIMPLE_VECTOR_HPP
