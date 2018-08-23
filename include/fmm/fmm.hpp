#pragma once

#include <stdio.h>
#include <iostream>
#include <vector>
#include <cmath>

// IBLGF-specific
#include <types.hpp>
#include <fmm/fmm_nli.hpp>

//#include <domain/dataFields/dataBlock.hpp>
//#include <domain/dataFields/datafield.hpp>
//#include <global.hpp>

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
        fmm::Nli lagrange_intrp;
    };

}

