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

#ifndef IMMERSED_BOUNDARY_HPP
#define IMMERSED_BOUNDARY_HPP

#include <vector>

#include <iblgf/global.hpp>
#include <iblgf/domain/octree/octant.hpp>

namespace iblgf
{
namespace ib
{

template<int Dim, class DataType>
class IB
{

public: // member types
    static constexpr int dimension = Dim;
    using datablock_t = DataType;
    using ib_points_type = std::vector<std::vector<float_type>>;
    using tree_t = octree::Tree<Dim, datablock_t>;
    using octant_t = typename tree_t::octant_type;
    using coordinate_type = typename tree_t::coordinate_type;
    using ib_infl_type = std::vector<std::vector<octant_t*>>;
    using ib_rank_type = std::vector<int>;

public: // friends

public: // Ctors
    IB() = default;
    IB(const IB& other) = delete;
    IB(IB && other) = default;
    IB& operator=(const IB& other) & = delete;
    IB& operator=(IB && other) & = default;
    ~IB() = default;

    IB(ib_points_type& points, float_type dx_base)
    :N_ib_(size(points[0])),
    ib_points_(points),
    ib_infl_(N_ib_),
    ib_rank_(N_ib_),
    dx_base_(dx_base)
    {
        ddf_radius_ = 2;
        delta_func = [this](float_type x){ return this->yang3(x);};
    }

public:
    ib_points_type& get_ib(){ return ib_points_;}
    const ib_points_type& get_ib() const { return ib_points_;}

    std::vector<float_type> get_ib_coordinate(int i)
    {
        return { ib_points_[0][i], ib_points_[1][i], ib_points_[2][i] };
    }

    ib_infl_type& get_ib_infl(){ return ib_infl_;}
    const ib_infl_type& get_ib_infl() const { return ib_infl_;}

    auto& get_ib_infl(int idx) { return ib_infl_[idx];}
    const auto& get_ib_infl(int idx) const { return ib_infl_[idx];}

    int& rank(int idx) noexcept {return ib_rank_[idx];}
    const int& rank(int idx) const noexcept {return ib_rank_[idx];}

    int ib_tot() {return N_ib_;}

    float_type ddf_radius(){ return ddf_radius_;}

public: // iters

public: // functions
    template<class BlockDscrptr>
    bool ib_block_overlap(int nRef, BlockDscrptr b_dscrptr)
    {
        // if a block overlap with ANY ib point

        for (int i=0; i<N_ib_; i++)
            if (ib_block_overlap(nRef, i, b_dscrptr))
                return true;

        return false;

    }

    template<class BlockDscrptr>
    bool ib_block_overlap(int nRef, int idx, BlockDscrptr b_dscrptr, bool add_radius=true)
    {
        // this function scale the block to the finest level and compare with
        // the influence region of the ib point

        b_dscrptr.extent() += 1;
        auto coor = this->get_ib_coordinate(idx);

        //std::cout<<"--------------------------"<<std::endl << b_dscrptr << std::endl;
        //std::cout<<"ib point ["<< coor[0]<<" "<<coor[1]<<" "<<coor[2] <<"]" <<std::endl;

        b_dscrptr.level_scale(nRef);

        if (add_radius)
        {
            b_dscrptr.extent() += ddf_radius_*2+safety_dis_*2;
            b_dscrptr.base()   -= ddf_radius_+safety_dis_;
        }

        float_type factor = std::pow(2, nRef)/dx_base_;
        //std::cout<<"block_descriptor" << b_dscrptr << std::endl;
        //std::cout<<"factor" << factor << std::endl;
        for (std::size_t d = 0; d < Dim; ++d)
        {
            if (b_dscrptr.max()[d]+1<ib_points_[d][idx]* factor || b_dscrptr.min()[d]>=ib_points_[d][idx] * factor)
                return false;
        }
        return true;
    }


public: // ddfs
    float_type yang3(float_type x)
    {
        float_type r = abs(x);
        float_type ddf = 0;
        if (r>ddf_radius_)
            return 0;

        float_type r2 = r*r;
        if (r<=1.0)
            ddf = 17/48+sqrt(3)*M_PI/108+r/4-r2/4+(1-2*r)/16.*sqrt(-12*r2+12*r+1)
                        -sqrt(3)/12*std::asin(sqrt(3)/2*(2*r-1));
        else
            ddf = 55/48-sqrt(3)*M_PI/108-13*r/12+r2/4+(2*r-3)/48.*sqrt(-12*r2+36*r-23)
                        +sqrt(3)/36*std::asin(sqrt(3)/2*(2*r-3));

        return ddf;
    }

private:

    int N_ib_;
    int ib_level_;
    int safety_dis_=0;
    float_type dx_base_;

    std::shared_ptr<tree_t> t_;

    ib_points_type ib_points_;
    ib_infl_type ib_infl_;
    ib_rank_type ib_rank_;

    float_type ddf_radius_;
    std::function<float_type(float_type x)> delta_func ;
};



}}

#endif


