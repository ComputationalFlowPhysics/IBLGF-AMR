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

    //using namespace domain;
    class Nli;

    class Fmm
    {
    public:
        Fmm(int Nb) : lagrange_intrp(Nb)
        {
        }

    private:
        Nli lagrange_intrp;
    };

}

