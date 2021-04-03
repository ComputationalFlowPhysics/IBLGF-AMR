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
#include <functional>
#include <iblgf/global.hpp>
#include <iblgf/domain/octree/octant.hpp>
#include <iblgf/domain/ib_communicator.hpp>

namespace iblgf
{
namespace ib
{
template<int Dim, class DataBlock>
class IB
{
  public: // member types
    using datablock_t = DataBlock;
    using node_t = typename datablock_t::node_t;
    using tree_t = octree::Tree<Dim, datablock_t>;
    using octant_t = typename tree_t::octant_type;

    using real_coordinate_type = typename tree_t::real_coordinate_type;
    using coordinate_type = typename tree_t::coordinate_type;
    using force_type = std::vector<real_coordinate_type>;

    using delta_func_type = std::function<float_type(real_coordinate_type)>;

    using communicator_t = ib_communicator<IB>;

  public: // Ctors
    IB(const IB& other) = default;
    IB(IB&& other) = default;
    IB& operator=(const IB& other) & = default;
    IB& operator=(IB&& other) & = default;
    ~IB() = default;

    IB()
    : ib_comm_(this)
    {
    }

  public: // init functions
    template<class DictionaryPtr>
    void init(DictionaryPtr d, float_type dx_base, int nRef)
    {

        ibph_ = d->template get_or<float_type>("ibph", 1.5);
        geometry_ = d->template get_or<std::string>("geometry", "plate");

        nRef_ = nRef;
        dx_base_ = dx_base;

        read_points();

        // will add more, default is yang4
        ddf_radius_ = 1.5;
        std::function<float_type(float_type x)> delta_func_1d_ =
            [this](float_type x) { return this->roma(x); };

        this->delta_func_ = [this, delta_func_1d_](real_coordinate_type x) {
            return delta_func_1d_(x[0]) * delta_func_1d_(x[1]) *
                   delta_func_1d_(x[2]);
        };

        //temp variables
        ib_infl_.resize(coordinates_.size());
        ib_infl_pts_.resize(coordinates_.size());
        ib_rank_.resize(coordinates_.size());
        forces_.resize(
            coordinates_.size(), real_coordinate_type((float_type)0));

        forces_prev_.resize(4);
        for (auto& f:forces_prev_)
            f.resize(coordinates_.size(), real_coordinate_type((float_type)0));
    }

    void read_points()
    {
        //coordinates_.emplace_back(real_coordinate_type({0.01, 0.01, 0.01}));

        if (geometry_=="plate")
        {
            float_type L = 1.0;
            float_type AR = 1.0;
            float_type Ly= L*AR;
            //int        nx = 2;
            int        nx = int(L/dx_base_/ibph_*pow(2,nRef_));
            int        ny = nx*2;


            for (int ix = 0; ix < nx; ++ix)
                for (int iy = 0; iy < ny; ++iy)
                {
                    float_type w = (ix * L)/(nx-1)- L/2.0;
                    float_type angle = M_PI/2;

                    coordinates_.emplace_back(
                            real_coordinate_type(
                                { w * std::cos(angle), (iy * Ly) / (ny-1) - Ly/2.0, -w * std::sin(angle) }));
                }
        }
        else if (geometry_=="sphere")
        {
            float_type R = 0.5;

            float_type dx = dx_base_/pow(2,nRef_)*ibph_;
            int n = floor(M_PI / (dx * dx * 0.86602540378) )+1;

            if (comm_.rank()==1)
                std::cout<< " Geometry = sphere, n = "<< n << std::endl;

            float_type lambda = (1. + sqrt(5) ) / 2.0;
            for (int i=0; i<n; ++i)
            {
                float_type x = i+0.5;
                float_type phi = std::acos(1.0 - 2 * x/n);
                float_type theta = 2*M_PI * x / lambda;
                coordinates_.emplace_back( real_coordinate_type({R*cos(theta)*sin(phi), R*sin(theta)*sin(phi), R*cos(phi)}));
            }
        }
        else
        {
            std::fstream file(geometry_, std::ios_base::in);
            int n; file >> n;

            if (comm_.rank()==1)
                std::cout<< " Geometry = read from text "<< geometry_ << std::endl;

            for (int i=0; i<n; i++)
            {
                real_coordinate_type p;
                for (int field_idx = 0; field_idx<p.size(); field_idx++)
                    file>>p[field_idx];

                coordinates_.emplace_back(p);
            }
        }
    }

    /** @{ @brief Get the force vector of  all immersed boundary points */
    void communicate_test(bool send_locally_owned) noexcept
    {
        ib_comm_.compute_indices();
        ib_comm_.communicate(send_locally_owned);
    }

  public: //Access
    /** @{ @brief Get the force vector of  all immersed boundary points */
    auto&       force() noexcept { return forces_; }

    auto&       force(std::size_t _i) noexcept { return forces_[_i]; }
    const auto& force(std::size_t _i) const noexcept { return forces_[_i]; }

    /** @{ @brief Get the force of ith  immersed boundary point, dimension idx = idx */
    auto&       force(std::size_t _i, std::size_t _idx) noexcept { return forces_[_i][_idx]; }
    const auto& force(std::size_t _i, std::size_t _idx) const noexcept { return forces_[_i][_idx]; }

    /** @{ @brief Get the force of previous timestep */
    auto&       force_prev(std::size_t _i) noexcept { return forces_prev_[_i]; }
    const auto& force_prev(std::size_t _i) const noexcept { return forces_prev_[_i]; }

    /** @{ @brief Get the coordinates of the ith ib points */
    auto&       coordinate(std::size_t _i) noexcept { return coordinates_[_i]; }
    const auto& coordinate(std::size_t _i) const noexcept
    {return coordinates_[_i];}
    /** @} */

    /** @{ @brief Get the coordinates of the ith ib points scaled by level */
    auto       scaled_coordinate(std::size_t _i) noexcept { return coordinates_[_i] * std::pow(2, nRef_) / dx_base_; }
    /** @} */

    /** @{ @brief Get the influence lists of the ib points */
    auto&       influence_pts(std::size_t i) noexcept { return ib_infl_pts_[i]; }
    const auto& influence_pts(std::size_t i) const noexcept { return ib_infl_pts_[i]; }

    /** @{ @brief Get the influence lists of the ib points */
    auto&       influence_pts(std::size_t i, std::size_t oct_i ) noexcept { return ib_infl_pts_[i][oct_i]; }
    const auto& influence_pts(std::size_t i, std::size_t oct_i ) const noexcept { return ib_infl_pts_[i][oct_i]; }

    /** @{ @brief Get the influence lists of the ib points */
    auto&       influence_list() noexcept { return ib_infl_; }
    const auto& influence_list() const noexcept { return ib_infl_; }
    /** @} */
    /** @{ @brief Get the influence list of the ith ib points */
    auto&       influence_list(std::size_t _i) noexcept { return ib_infl_[_i]; }
    const auto& influence_list(std::size_t _i) const noexcept
    {
        return ib_infl_[_i];
    }
    /** @} */

    /** @{ @brief Get the rank of the ith ib points */
    auto&       rank(std::size_t _i) noexcept { return ib_rank_[_i]; }
    const auto& rank(std::size_t _i) const noexcept { return ib_rank_[_i]; }
    /** @} */

    /** @brief Get number of ib points */
    auto size() const noexcept { return coordinates_.size(); }
    /** @brief Get delta function radius */
    auto ddf_radius() const noexcept { return ddf_radius_; }

    /** @{ @brief Get ib mpi communicator  */
    auto&       communicator() noexcept { return ib_comm_; }
    const auto& communicator() const noexcept { return ib_comm_; }
    /** @} */

    bool locally_owned(std::size_t _i) noexcept { return ib_rank_[_i] == comm_.rank();}

    float_type force_scale()
    {
        float_type tmp = dx_base_ / std::pow(2, nRef_);
        return tmp*tmp*tmp;
    }

    void scales(const float_type a)
    {
        for (std::size_t i = 0; i < size(); i++)
            for (std::size_t field_idx = 0; field_idx<Dim; field_idx++)
            this->force(i, field_idx) *= a;
    }

  public: // iters
  public: // functions

    void clean_non_local()
    {
        for (std::size_t i = 0; i < size(); i++)
            if (!this->locally_owned(i))
                this->force(i)=0.0;
    }

    template<class BlockDscrptr>
    bool ib_block_overlap(BlockDscrptr b_dscrptr, int radius_level = 0)
    {

        for (std::size_t i = 0; i < size(); i++)
            if (ib_block_overlap(i, b_dscrptr, radius_level)) return true;

        return false;
    }

    template<class BlockDscrptr>
    bool ib_block_overlap(
        int idx, BlockDscrptr b_dscrptr, int radius_level = 0)
    {
        // this function scale the block to the finest level and compare with
        // the influence region of the ib point

        b_dscrptr.extent() += 1;
        b_dscrptr.level_scale(nRef_);

        float_type added_radius = 0;
        if (radius_level == 2)
            added_radius += ddf_radius_+safety_dis_+1.0;
        else if (radius_level == 1)
            added_radius += ddf_radius_+1.0;

        b_dscrptr.extent() += 2*added_radius;
        b_dscrptr.base() -= added_radius;

        float_type factor = std::pow(2, nRef_) / dx_base_;

        for (std::size_t d = 0; d < Dim; ++d)
        {
            if (b_dscrptr.max()[d] < coordinates_[idx][d] * factor ||
                b_dscrptr.min()[d] >= coordinates_[idx][d] * factor)
                return false;
        }
        return true;
    }

  public: // ddfs
    const auto& delta_func() const noexcept { return delta_func_; }
    auto&       delta_func() noexcept { return delta_func_; }

    float_type yang4(float_type x)
    {
        float_type r = std::fabs(x);
        if (r>2.5) return 0;

        float_type r2 = r * r;
        float_type ddf = 0;

        if (r<=0.5)
            ddf = 3.0/8+M_PI/32.0-r2/4;
        else if (r<=1.5)
            ddf = 1.0/4+(1.0-r)/8.0 * sqrt(-2.0+8*r-4*r2) - 1.0/8 * asin(sqrt(2)*(r-1) );

        else if (r<=2.5)
            ddf = 17.0/16-M_PI/64.0-3.0/4*r+r2/8+(r-2.0)/16.0*sqrt(-14.0+16*r-4*r2)
                    +1/16*asin(sqrt(2)*(r-2));

        return ddf;

    }

    float_type yang3(float_type x)
    {
        float_type r = std::fabs(x);
        float_type ddf = 0;
        if (r > 2) return 0;

        float_type r2 = r * r;
        if (r <= 1.0)
            ddf = 17.0 / 48.0 + sqrt(3) * M_PI / 108.0 + r / 4.0 - r2 / 4.0 +
                  (1.0 - 2.0 * r) / 16 * sqrt(-12.0 * r2 + 12.0 * r + 1.0) -
                  sqrt(3) / 12.0 * std::asin(sqrt(3) / 2.0 * (2.0 * r - 1.0));
        else
            ddf = 55.0 / 48.0 - sqrt(3) * M_PI / 108.0 - 13.0 * r / 12.0 + r2 / 4.0 +
                  (2.0 * r - 3.0) / 48.0 * sqrt(-12.0 * r2 + 36.0 * r - 23.0) +
                  sqrt(3) / 36.0 * std::asin(sqrt(3) / 2.0 * (2 * r - 3.0));

        return ddf;
    }
    float_type roma(float_type x)
    {
        float_type r = std::fabs(x);
        float_type ddf = 0;
        if (r > 1.5) return 0;

        float_type r2 = r*r;

        if (r <= 0.5)
            ddf = (1.0+sqrt(-3.*r2 + 1.0))/3.0;
        else if (r <=1.5)
            ddf = (5.0 - 3.0*r - sqrt( 1.0 - 3.0*(1-r)*(1-r) ) )/6.0;
        return ddf;
    }

public:

    boost::mpi::communicator comm_;

    int        nRef_ = 0;
    int        safety_dis_ = 7;
    float_type dx_base_ = 1;
    float_type ibph_;

    std::vector<real_coordinate_type>   coordinates_;
    force_type                          forces_;
    std::vector<force_type>             forces_prev_;
    std::vector<std::vector<octant_t*>> ib_infl_;
    std::vector<std::vector<std::vector<node_t>>> ib_infl_pts_;
    std::vector<int>                    ib_rank_;

    delta_func_type delta_func_;
    float_type      ddf_radius_ = 0;

    std::string geometry_;

    communicator_t ib_comm_;
};

} // namespace ib
} // namespace iblgf

#endif

