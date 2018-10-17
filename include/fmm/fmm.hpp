#pragma once

#include <stdio.h>
#include <iostream>
#include <vector>
#include <cmath>

// IBLGF-specific
#include <types.hpp>
#include <fmm/fmm_nli.hpp>

namespace fmm
{

    class Fmm
    {
    public:
        Fmm(int Nb) : lagrange_intrp(Nb)
        {}


        template<template<size_t> class f_source, template<size_t> class f_target>
        void fmm_for_level(auto& domain, int level)
        {

            //std::cout<<"find sub_tree for the whole level"<<std::endl;

            //std::cout<<"anterpolation"<<std::endl;

            //for (int ls = domain.tree()->depth()-1;
            //    ls > simulation_.domain_.tree()->base_level(); --ls)
            //{
            //    for (auto it_s  = simulation_.domain_.begin(ls);
            //        it_s != simulation_.domain_.end(ls); ++it_s)
            //        {
            //            anterpolation(it_s);
            //        }
            //}

            //std::cout<<"convolution"<<std::endl;
            //std::cout<<""<<std::endl;

            //std::pcout<<"find sub_tree for the non-leaf"<<std::endl;
        }

    public:
        Nli lagrange_intrp;
    };

}

