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
    using tree_t = octree::Tree<Dim, datablock_t>;
    using octant_t = typename tree_t::octant_type;

    using real_coordinate_type = typename tree_t::real_coordinate_type;
    using coordinate_type = typename tree_t::coordinate_type;

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
    void init(float_type dx_base)
    {
        read_points();
        ddf_radius_ = 2;
        dx_base_ = dx_base;

        // will add more, default is yang3
        std::function<float_type(float_type x)> delta_func_1d_ =
            [this](float_type x) { return this->yang3(x); };

        // ddf 3D
        this->delta_func_ = [this, delta_func_1d_](real_coordinate_type x) {
            return delta_func_1d_(x[0]) * delta_func_1d_(x[1]) *
                   delta_func_1d_(x[2]);
        };
        ib_infl_.resize(coordinates_.size());
        ib_rank_.resize(coordinates_.size());
        forces_.resize(
            coordinates_.size(), real_coordinate_type((float_type)0));
    }

    void read_points() noexcept
    {
        coordinates_.emplace_back(real_coordinate_type({0.01, 0.01, 0.01}));

        int        nx = 20;
        int        nyz = nx;
        float_type L = 0.7555555555555555;
        for (int ix = 0; ix < nx; ++ix)
            for (int iyz = 0; iyz < nyz; ++iyz)
            {
                coordinates_.emplace_back(
                    real_coordinate_type({(ix * L) / nx - L / 2.0,
                        (iyz * L) / nyz - L / 2.0, (iyz * L) / nyz - L / 2.0}));
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
    const auto& force() const noexcept { return forces_; }
    /** @} */

    /** @{ @brief Get the force of ith  immersed boundary point */
    auto&       force(std::size_t _i) noexcept { return forces_[_i]; }
    const auto& force(std::size_t _i) const noexcept { return forces_[_i]; }

    /** @{ @brief Get the coordinates of the ib points */
    auto&       coordinate() noexcept { return coordinates_; }
    const auto& coordinate() const noexcept { return coordinates_; }
    /** @} */
    /** @{ @brief Get the coordinates of the ith ib points */
    auto&       coordinate(std::size_t _i) noexcept { return coordinates_[_i]; }
    const auto& coordinate(std::size_t _i) const noexcept
    {
        return coordinates_[_i];
    }
    /** @} */

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

  public: // iters
  public: // functions
    template<class BlockDscrptr>
    bool ib_block_overlap(int nRef, BlockDscrptr b_dscrptr)
    {
        // if a block overlap with ANY ib point

        for (std::size_t i = 0; i < size(); i++)
            if (ib_block_overlap(nRef, i, b_dscrptr)) return true;

        return false;
    }

    template<class BlockDscrptr>
    bool ib_block_overlap(
        int nRef, int idx, BlockDscrptr b_dscrptr, bool add_radius = true)
    {
        // this function scale the block to the finest level and compare with
        // the influence region of the ib point

        b_dscrptr.extent() += 1;
        //auto coor = this->coordinate(idx);

        //std::cout<<"--------------------------"<<std::endl << b_dscrptr << std::endl;
        //std::cout<<"ib point ["<< coor[0]<<" "<<coor[1]<<" "<<coor[2] <<"]" <<std::endl;

        b_dscrptr.level_scale(nRef);

        if (add_radius)
        {
            b_dscrptr.extent() += ddf_radius_ * 2 + safety_dis_ * 2;
            b_dscrptr.base() -= ddf_radius_ + safety_dis_;
        }

        float_type factor = std::pow(2, nRef) / dx_base_;
        //std::cout<<"block_descriptor" << b_dscrptr << std::endl;
        //std::cout<<"factor" << factor << std::endl;

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

    float_type yang3(float_type x)
    {
        float_type r = std::fabs(x);
        float_type ddf = 0;
        if (r > ddf_radius_) return 0;

        float_type r2 = r * r;
        if (r <= 1.0)
            ddf = 17 / 48 + sqrt(3) * M_PI / 108 + r / 4 - r2 / 4 +
                  (1 - 2 * r) / 16. * sqrt(-12 * r2 + 12 * r + 1) -
                  sqrt(3) / 12 * std::asin(sqrt(3) / 2 * (2 * r - 1));
        else
            ddf = 55 / 48 - sqrt(3) * M_PI / 108 - 13 * r / 12 + r2 / 4 +
                  (2 * r - 3) / 48. * sqrt(-12 * r2 + 36 * r - 23) +
                  sqrt(3) / 36 * std::asin(sqrt(3) / 2 * (2 * r - 3));

        return ddf;
    }

  public:
    int        safety_dis_ = 1;
    float_type dx_base_ = 1;

    std::vector<real_coordinate_type>   coordinates_;
    std::vector<real_coordinate_type>   forces_;
    std::vector<std::vector<octant_t*>> ib_infl_;
    std::vector<int>                    ib_rank_;

    delta_func_type delta_func_;
    float_type      ddf_radius_ = 1;

    communicator_t ib_comm_;
};

} // namespace ib
} // namespace iblgf

#endif

