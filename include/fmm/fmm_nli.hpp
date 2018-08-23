#ifndef IBLGF_INCLUDED_FMM_HPP
#define IBLGF_INCLUDED_FMM_HPP
#include <iostream>     // std::cout
#include <algorithm>    // std::max/ min

#include <Eigen/Dense>
#include <vector>

namespace fmm
{
    class Nli
    {

    public:
        Nli(int Nb_) : Nb_(Nb_),
        antrp_coeff_(Nb_, (Nb_ * 2))
        {
            antrp_coeff_calc(antrp_coeff_, Nb_);
            antrp_coeff_sub.push_back(antrp_coeff_.block(0,0,Nb_,Nb_));
            antrp_coeff_sub.push_back(antrp_coeff_.block(0,Nb_,Nb_,Nb_));

            std::cout<<antrp_coeff_sub[0]<<std::endl;
            std::cout<<std::endl;
            std::cout<<antrp_coeff_sub[1]<<std::endl;
        }

        void intrp()
        {
        }

        void antrp()
        {
        }

    private:
        void antrp_coeff_calc(auto& antrp_coeff_, int Nb_)
        {
            antrp_coeff_.setZero();

            for (int c = 0; c < Nb_*2-1; ++c){
                int c_p = c - Nb_;

                if (c % 2 == 0){
                    int p = c / 2;
                    antrp_coeff_(p, c) = 1;
                }
                else{
                    double temp_sum = 0.0;
                    int bd_l = -1;
                    int bd_r = -1;

                    if (c_p < -1){
                        bd_l = std::max(0   , ((c + 1) / 2  - pts_cap / 2 ));
                        bd_r = std::min(Nb_, bd_l + pts_cap);
                    }
                    else{
                        bd_r = std::min(Nb_ , ((c + 1) / 2 + pts_cap / 2));
                        bd_l = std::max(0   , bd_r - pts_cap);
                    }

                    for (int p = bd_l; p<bd_r; ++p){
                        double temp_mult = 1.0;
                        int p_p = p*2 - Nb_ + 1;

                        // the barycentric coefficients
                        for (int l = bd_l; l< bd_r; ++l){
                            if (l != p){
                                temp_mult /= -(2*l - Nb_ + 1 - p_p);
                            }
                        }

                        temp_mult          /= (double)(c_p - p_p + 1);
                        antrp_coeff_(p, c)  = temp_mult;
                        temp_sum           += temp_mult;
                    }

                    antrp_coeff_.col(c) /= temp_sum;
                }

            }

            for (int c = Nb_*2 - 1; c>Nb_-1; --c){
                antrp_coeff_.col(c) = antrp_coeff_.col(c-1);
            }
        }

    private:
        const int pts_cap = 10;

        int Nb_;
        Eigen::Matrix<float_type, Eigen::Dynamic, Eigen::Dynamic> antrp_coeff_;
        std::vector<Eigen::Matrix<float_type, Eigen::Dynamic, Eigen::Dynamic>>
            antrp_coeff_sub;
    };

}

#endif //IBLGF_INCLUDED_FMM_HPP
