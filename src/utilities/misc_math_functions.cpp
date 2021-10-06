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

#include <iblgf/utilities/misc_math_functions.hpp>
#include <math.h>

namespace iblgf
{
namespace math
{

int nextpow(int a, int x)
{
    int n = int(ceil(log(x)/log(a)));
    int p = int(pow(a, n-1));
    if (p>=x)
        return p;
    else
        return int(pow(a, n));
}

int nextprod(int x)
{
    return nextprod({2,3}, x);
}

int nextprod(std::vector<int> a, int x)
{
    int k = a.size();
    std::vector<int> v(k, 1);
    std::vector<int> mx(k, 0);

    for (int i=0; i<k; i++)
        mx[i] = nextpow(a[i], x);

    v[0] = mx[0];
    int p = mx[0];
    int best = p;
    int icarry = 0;

    while (v[k-1] < mx[k-1]) {
        if (p>= x)
        {
            best = p<best ? p : best;
            bool carrytest = true;
            while (carrytest)
            {
                p = p / v[icarry];
                v[icarry] = 1;
                icarry += 1;
                p *= a[icarry];
                v[icarry] *= a[icarry];
                carrytest = ( v[icarry] > mx[icarry] ) && ( icarry < k-1);
            }
            if (p<x)
                icarry = 0;
        }
        else
        {
            while (p<x)
            {
                p *= a[0];
                v[0] *= a[0];
            }
        }

    }

    return (mx[k-1] < best ) ? mx[k-1] : best;
}

int next_pow_2(int n) noexcept
{
    int p = 1;
    if (n && !(n & (n - 1))) return n;

    while (p < n) p <<= 1;

    return p;
}

int pow2(int n) noexcept
{
    return (1 << n);
}

} // namespace math
} // namespace iblgf
