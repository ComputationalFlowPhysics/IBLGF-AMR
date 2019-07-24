#ifndef INCLUDED_LGF_LOOKUP_HPP
#define INCLUDED_LGF_LOOKUP_HPP

#include <algorithm>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <array>
#include <cmath>
#include <global.hpp>


namespace lgf
{
    
//template<class Derived>
//class Lookup_symm_base : crtp::Crtp<Derived>
//{
//
//    template<class Coordinate>
//    static auto get(const Coordinate& _c) noexcept
//    {
//        int x = std::abs(static_cast<int>(_c.x()));
//        int y = std::abs(static_cast<int>(_c.y()));
//        int z = std::abs(static_cast<int>(_c.z()));
//
//        if (x<= N_max && y<=N_max && z<=N_max) // using table
//        {
//            if (x < y)
//                std::swap(x, y);
//            if (x < z)
//                std::swap(x, z);
//            if (y < z)
//                std::swap(y, z);
//
//            return - this->derived().table_[(x * (2 + 3*x + x * x))/6 + y*(y+1)/2 + z]; 
//        } 
//        else
//        {
//            return this->derived().compute_lgf(x,y,z);
//        }
//   }
//    int N_max;
//};    

class GL_Lookup
{
public:

    template<class Coordinate>
    static auto get(const Coordinate& _c) noexcept
    {
        int x = std::abs(static_cast<int>(_c.x()));
        int y = std::abs(static_cast<int>(_c.y()));
        int z = std::abs(static_cast<int>(_c.z()));

        if (x<= N_max && y<=N_max && z<=N_max) // using table
        {
            if (x < y)
                std::swap(x, y);
            if (x < z)
                std::swap(x, z);
            if (y < z)
                std::swap(y, z);

            return - table_[(x * (2 + 3*x + x * x))/6 + y*(y+1)/2 + z]; // indexing

        } else
        {
            return asym(x,y,z);
        }

   }

    //TODO: Vectorize this function. Takes a lot of time!
    static auto asym(int n1, int n2, int n3)
    {
        const float_type n1_2 = n1 * n1, n2_2 = n2 * n2, n3_2 = n3 * n3;
        const float_type n_abs = sqrt(n1_2 + n2_2 + n3_2);

        const float_type n1_6=n1_2*n1_2*n1_2;
        const float_type n2_6=n2_2*n2_2*n2_2;
        const float_type n3_6=n3_2*n3_2*n3_2;

        const float_type n1_8=n1_6*n1_2;
        const float_type n2_8=n2_6*n2_2;
        const float_type n3_8=n3_6*n3_2;

        const float_type n_abs_6 = n_abs*n_abs*n_abs*n_abs*n_abs*n_abs;
        const float_type n_abs_7 = n_abs_6*n_abs;
        const float_type n_abs_13 = n_abs_6*n_abs_6*n_abs;
        
        float_type tmp = -1.0/4.0/M_PI/n_abs; // the first asymp term
        tmp = tmp - (n1_2 * n1_2 + n2_2 * n2_2 + n3_2 * n3_2 
                    - 3.0 * n1_2 * n2_2 
                    - 3.0 * n2_2 * n3_2 
                    - 3.0 * n3_2 * n1_2)/16.0/M_PI/n_abs_7; // the second asymp term

        if (n1<600 || n2<600 || n3<600) // add the third term
        {
            float_type tmp2;
            const float_type coef = 8.0/(768.0 * M_PI);

            tmp2 =-3.0 * ( 23 * (n1_8 + n2_8 + n3_8 ) 
                    - 244 * (n2_6 * n3_2 + n3_6 * n2_2 + 
                        n1_6 * n2_2 + n2_6 * n1_2 + 
                        n1_6 * n3_2 + n3_6 * n1_2 )
                    + 621 * ((n1_2 * n1_2) * (n2_2 * n2_2) 
                        + (n2_2 * n2_2) * (n3_2 * n3_2) 
                        + (n3_2 * n3_2) * (n1_2 * n1_2) )
                    - 228 * ( (n1_2 * n1_2) * n2_2 * n3_2 
                        + n1_2 * (n2_2 * n2_2) * n3_2
                        + n1_2 * n2_2 * (n3_2 * n3_2)));

            tmp2 = tmp2 / n_abs_13 /4.0 * coef;
            tmp += tmp2;
        }

        return tmp;
    }

private:
    static const int N_max;
    static const std::vector<float_type> table_;
};

//decltype(Lookup::N_max) Lookup::N_max(100);

} //namepsace
#endif
