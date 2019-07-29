#ifndef MATH_INCLUDED_MISC_MATH_HPP
#define MATH_INCLUDED_MISC_MATH_HPP

#include<bits/stdc++.h>
namespace math
{
    int next_pow_2(int n)
    {
        int p = 1;
        if (n && !(n & (n - 1)))
            return n;

        while (p < n)
            p <<= 1;

        return p;
    }

}
#endif // MATH_INCLUDED_MISC_MATH_HPP
