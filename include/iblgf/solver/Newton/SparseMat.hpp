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

#ifndef IBLGF_INCLUDED_SPARSE_MAT_HPP
#define IBLGF_INCLUDED_SPARSE_MAT_HPP

#include <iostream>
#include <algorithm>
#include <vector>
#include <cmath>
#include <array>
#include <set>

// IBLGF-specific
#include <iblgf/global.hpp>
#include <iblgf/simulation.hpp>
#include <iblgf/domain/domain.hpp>
#include <iblgf/IO/parallel_ostream.hpp>
#include <iblgf/solver/poisson/poisson.hpp>
#include <iblgf/solver/Newton/Interpolation_mat.hpp>
#include <iblgf/solver/linsys/linsys.hpp>
#include <iblgf/operators/operators.hpp>
#include <iblgf/utilities/misc_math_functions.hpp>

#include <boost/serialization/map.hpp>

namespace iblgf
{
namespace solver
{
using namespace domain;

class sparse_mat
{
  public:
    sparse_mat() = default;
    void resizing_row(int n) { mat.resize(n); }
    void clean() { mat.resize(0); }
    void add_element(int n, int m, float_type val)
    {
        if (m <= 0) return;
        auto it = mat[n].find(m);
        if (it == mat[n].end()) { mat[n][m] = val; }
        else
        {
            it->second += val;
        }
    }

    sparse_mat operator+(const sparse_mat& b)
    {
        if (this->mat.size() != b.mat.size())
        {
            std::cout << "size are " << this->mat.size() << " and "
                      << b.mat.size() << std::endl;
            throw std::runtime_error(
                "sparse matrix does not have the same dim, cannot add");
        }
        sparse_mat res;
        res.mat.resize(this->mat.size());
        for (int i = 1; i < mat.size(); i++)
        {
            for (const auto& [key, val] : this->mat[i])
            {
                res.mat[i][key] = val;
            }
            for (const auto& [key, val] : b.mat[i])
            {
                res.add_element(i, key, val);
            }
        }
        return res;
    }

    void add_vec(const sparse_mat& b, float_type factor = 1.0)
    {
        if (this->mat.size() != b.mat.size())
        {
            std::cout
                << "sparse matrix does not have the same dim, cannot in place add"
                << std::endl;
            return;
        }
        for (int i = 1; i < mat.size(); i++)
        {
            for (const auto& [key, val] : b.mat[i])
            {
                this->add_element(i, key, (val * factor));
            }
        }
    }

    void print_row(int n)
    {
        for (const auto& [key, val] : this->mat[n])
        {
            //int idx1 = std::get<0>(key1);
            std::cout << n << " " << key << " " << val << " || ";
        }
        std::cout << std::endl;
    }

    void print_row_full(int n)
    {
        for (int i = 0; i < mat.size(); i++)
        {
            auto it = mat[n].find(i);
            if (it == mat[n].end()) { std::cout << 0.00 << " " << std::endl; }
            else
            {
                std::cout << it->second << " ";
            }
        }
        std::cout << std::endl;
    }

    //get the number of element to reserve space for CSR format
    int tot_size(bool include_zero = false)
    {
        //should not include zero but the result should be the same, this is only for debugging
        int res = 0;
        int begin_num = 1;
        if (include_zero) { begin_num = 0; }
        for (int i = begin_num; i < mat.size(); i++) { res += mat[i].size(); }
        return res;
    }
    template<class int_type, class value_type>
    void getCSR(int_type* ia, int_type* ja, value_type* a)
    {
        //for intel Pardiso default CSR format
        int_type counter = 0;
        for (int i = 1; i < mat.size(); i++)
        {
            ia[i - 1] = counter + 1;
            for (const auto& [key, val] : mat[i])
            {
                //int idx1 = std::get<0>(key1);
                //std::cout << n << " " << key << " " << val << " || ";
                if (key < 0) { std::cout << "Negative key" << std::endl; }
                ja[counter] = key;
                a[counter] = val;
                counter++;
            }
        }
        ia[mat.size() - 1] = counter + 1;
    }

    template<class int_type, class value_type>
    void getIdx(int_type* ia, int_type* ja, value_type* a, int_type begin)
    {
        //get the entry idx i and j from (a_ij) 
        //this is beginning with 0, 0
        //this is only used to set matrix for Petsc, cannot be used with Intel Pardiso directly
        int_type counter = 0;
        for (int i = 1; i < mat.size(); i++)
        {
            //ia[i - 1] = counter + 1;
            for (const auto& [key, val] : mat[i])
            {
                //int idx1 = std::get<0>(key1);
                //std::cout << n << " " << key << " " << val << " || ";
                if (key < 0) { std::cout << "Negative key" << std::endl; }
                ja[counter] = key - 1;
                ia[counter] = i - 1 + begin;
                a[counter] = val;
                counter++;
            }
        }
    }

    template<class int_type, class value_type>
    void getCSR_zero_begin(int_type* ia, int_type* ja, value_type* a)
    {
        int_type counter = 0;
        for (int i = 1; i < mat.size(); i++)
        {
            ia[i - 1] = counter;
            for (const auto& [key, val] : mat[i])
            {
                //int idx1 = std::get<0>(key1);
                //std::cout << n << " " << key << " " << val << " || ";
                if (key <= 0) { std::cout << "Negative key" << std::endl; }
                ja[counter] = key - 1;
                a[counter] = val;
                counter++;
            }
        }
        ia[mat.size() - 1] = counter;
    }

    void clean_entry(float_type th_val = 1e-12)
    {
        for (int i = 1; i < mat.size(); i++)
        {
            for (const auto& [key, val] : mat[i])
            {
                if (std::abs(val) < th_val) { mat[i].erase(key); }
            }
        }
    }

    template<class value_type>
    void Apply(value_type* x, value_type* b)
    {
        int counter = 0;
        for (int i = 1; i < mat.size(); i++)
        {
            value_type tmp = 0;

            for (const auto& [key, val] : mat[i])
            {
                //int idx1 = std::get<0>(key1);
                //std::cout << n << " " << key << " " << val << " || ";
                if (key < 0) { std::cout << "Negative key" << std::endl; }
                tmp += x[key - 1] * val;
            }
            b[i - 1] = tmp;
        }
        //ia[mat.size()-1] = counter+1;
    }

    void scale_entries(float_type factor)
    {
        for (int i = 1; i < mat.size(); i++)
        {
            for (const auto& [key, val] : mat[i])
            {
                //int idx1 = std::get<0>(key1);
                //std::cout << n << " " << key << " " << val << " || ";
                if (key < 0) { std::cout << "Negative key" << std::endl; }
                mat[i][key] *= factor;
            }
        }
    }

    int numRow_loc() { return mat.size() - 1; }

  public:
    std::vector<std::map<int, float_type>> mat;
};

} // namespace solver
} // namespace iblgf

#endif // IBLGF_INCLUDED_SPARSE_MAT_HPP
