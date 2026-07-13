#ifndef IBLGF_INCLUDED_IFHERK_VORTEX_RINGS_HPP
#define IBLGF_INCLUDED_IFHERK_VORTEX_RINGS_HPP

#include <boost/serialization/vector.hpp>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <unordered_map>
#include <vector>

namespace iblgf
{
namespace solver
{

template<class Setup>
float_type Ifherk<Setup>::body_force_amplitude() const noexcept
{
    if (b_f_num_pulse == 0)
        return 0.0;

    const auto pulse_index = body_force_pulse_index();

    if (b_f_freq <= 0.0)
    {
        const auto tau = body_force_parameter(
            b_f_tau_values_, b_f_tau, pulse_index);
        const auto mag = body_force_parameter(
            b_f_mag_values_, b_f_mag, pulse_index);
        return (T_ <= tau) ? mag/tau : 0.0;
    }

    const auto period = float_type(1) / b_f_freq;
    if (b_f_num_pulse > 0 && pulse_index >= b_f_num_pulse)
        return 0.0;

    const auto phase_time = std::fmod(T_, period);
    const auto tau = body_force_parameter(
        b_f_tau_values_, b_f_tau, pulse_index);
    const auto mag = body_force_parameter(
        b_f_mag_values_, b_f_mag, pulse_index);
    return (phase_time <= tau) ? mag/tau : 0.0;
}

template<class Setup>
void Ifherk<Setup>::body_force_profile_2d(
    float_type x,
    float_type y,
    float_type& force_x) const noexcept
{
    force_x = 0.0;

    if constexpr (domain_type::dims != 2)
        return;

    const auto A = body_force_amplitude();
    if (std::abs(A) <= 1.0e-14)
        return;

    const auto pulse_index = body_force_pulse_index();

    /*
    const auto beta = body_force_parameter(
        b_f_beta_values_, b_f_beta, pulse_index);
    const auto lx = body_force_parameter(
        b_f_lx_values_, b_f_lx, pulse_index);
    const auto ly = body_force_parameter(
        b_f_ly_values_, b_f_ly, pulse_index);
    */

    const auto x0 = body_force_parameter(
        b_f_x0_values_, b_f_x0, pulse_index);
    const auto y0 = body_force_parameter(
        b_f_y0_values_, b_f_y0, pulse_index);

    const auto R = body_force_parameter(
        b_f_R_values_, b_f_R, pulse_index);
    const auto alpha = b_f_alpha;
    const auto alpha_r = b_f_alpha_r;
    const auto alpha_x = b_f_alpha_x;

    //-----------

    /*
    const auto x_shift = x - x0;
    const auto y_shift = y - y0;
    const auto arg_x =
        beta * (lx / float_type(2) - std::abs(x_shift));
    const auto arg_y =
        beta * (ly / float_type(2) - std::abs(y_shift));
    const auto tanh_x = std::tanh(arg_x);
    const auto tanh_y = std::tanh(arg_y);
    const auto g_x =
        float_type(0.5) * (float_type(1) + tanh_x);
    const auto g_y =
        float_type(0.5) * (float_type(1) + tanh_y);
    */

    //G(x) - Gaussian smoothing function
    // H(y) - Shape
    const auto x_shift = x - x0;
    const auto y_shift = y - y0;
    const auto alpha_sqrt = std::sqrt(alpha);

    const auto g = alpha_sqrt / (alpha_x * std::sqrt(M_PI)) *
                   std::exp(-alpha * std::pow(x_shift / alpha_x, 2.0));
    const auto h = 0.5 * std::erfc(alpha_sqrt * (std::abs(y_shift) - R) / alpha_r);

    //force_x = A * g_x * g_y;
    force_x = A * g * h;
}

template<class Setup>
void Ifherk<Setup>::body_force_profile_3d(
    float_type x,
    float_type y,
    float_type z,
    float_type& force_x) const noexcept
{
    force_x = 0.0;

    if constexpr (domain_type::dims != 3)
        return;

    const auto A = body_force_amplitude();
    if (std::abs(A) <= 1.0e-14)
        return;

    const auto pulse_index = body_force_pulse_index();

    const auto x0 = body_force_parameter(
        b_f_x0_values_, b_f_x0, pulse_index);
    const auto y0 = body_force_parameter(
        b_f_y0_values_, b_f_y0, pulse_index);
    const auto z0 = body_force_parameter(
        b_f_z0_values_, b_f_z0, pulse_index);
    const auto R = body_force_parameter(
        b_f_R_values_, b_f_R, pulse_index);
    const auto alpha = b_f_alpha;
    const auto alpha_r = b_f_alpha_r;
    const auto alpha_x = b_f_alpha_x;

    //-----------

    //G(x) - Gaussian smoothing function
    // H(y) - Shape
    const auto x_shift = x - x0;
    const auto y_shift = y - y0;
    const auto z_shift = z - z0;
    const auto r_shift = std::sqrt(y_shift * y_shift + z_shift * z_shift);
    const auto alpha_sqrt = std::sqrt(alpha);

    const auto g = alpha_sqrt / (alpha_x * std::sqrt(M_PI)) *
                   std::exp(-alpha * std::pow(x_shift / alpha_x, 2.0));
    const auto h = 0.5 * std::erfc(alpha_sqrt * (std::abs(r_shift) - R) / alpha_r);

    //force_x = A * g_x * g_y;
    force_x = A * g * h;
}

template<class Setup>
typename Ifherk<Setup>::vortex_diagnostics_t
Ifherk<Setup>::vortex_diagnostics()
{
    struct vortex_cell_t
    {
        long long i0 = 0;
        long long j0 = 0;
        long long extent = 1;
        float_type x = 0.0;
        float_type y = 0.0;
        float_type omega = 0.0;
        float_type dA = 0.0;
    };

    struct fine_cell_key_t
    {
        long long i = 0;
        long long j = 0;

        bool operator==(const fine_cell_key_t& other) const noexcept
        {
            return i == other.i && j == other.j;
        }
    };

    struct fine_cell_key_hash_t
    {
        std::size_t operator()(const fine_cell_key_t& key) const noexcept
        {
            const auto h_i = std::hash<long long>{}(key.i);
            const auto h_j = std::hash<long long>{}(key.j);
            return h_i ^ (h_j + 0x9e3779b97f4a7c15ULL +
                (h_i << 6) + (h_i >> 2));
        }
    };

    struct vortex_region_t
    {
        float_type gamma = 0.0;
        float_type area = 0.0;
        float_type x_moment = 0.0;
        float_type y_moment = 0.0;
        float_type x_center = std::numeric_limits<float_type>::quiet_NaN();
        float_type y_center = std::numeric_limits<float_type>::quiet_NaN();
        float_type max_x = -std::numeric_limits<float_type>::infinity();
    };

    float_type gamma_total_local = 0.0;
    float_type gamma_top_local = 0.0;
    float_type gamma_bottom_local = 0.0;
    float_type omega_max_local = -std::numeric_limits<float_type>::infinity();
    float_type omega_min_local = std::numeric_limits<float_type>::infinity();
    float_type x_omega_max_local =
        std::numeric_limits<float_type>::quiet_NaN();
    float_type y_omega_max_local =
        std::numeric_limits<float_type>::quiet_NaN();
    float_type x_omega_min_local =
        std::numeric_limits<float_type>::quiet_NaN();
    float_type y_omega_min_local =
        std::numeric_limits<float_type>::quiet_NaN();
    std::vector<long long> local_index_data;
    std::vector<float_type> local_value_data;
    boost::mpi::communicator world;
    vortex_diagnostics_t diagnostics;
    const bool use_vorticity_threshold =
        vortex_detection_mode_ ==
        vortex_detection_mode_t::vorticity_threshold;

    if constexpr (domain_type::dims == 2)
    {
        if (!domain_->is_server())
        {
            const auto center = coordinate_origin_index();
            long long finest_scaling = 1;
            if (!use_vorticity_threshold)
            {
                const auto tree_depth =
                    static_cast<int>(domain_->tree()->depth());
                const auto finest_refinement_level =
                    std::max(0, tree_depth - 1);
                finest_scaling =
                    static_cast<long long>(math::pow2(finest_refinement_level));
            }

            for (auto it = domain_->begin(); it != domain_->end(); ++it)
            {
                if (!it->locally_owned()) continue;
                if (!it->is_leaf()) continue;
                if (it->is_correction()) continue;

                const auto scaling = math::pow2(it->refinement_level());
                const auto dx_level = dx_base_ / scaling;

                for (auto& n : it->data())
                {
                    const auto& coord = n.level_coordinate();
                    const auto x = static_cast<float_type>(
                        coord[0] - center[0] * scaling) * dx_level;
                    const auto y = static_cast<float_type>(
                        coord[1] - center[1] * scaling) * dx_level;
                    const auto omega = n(edge_aux_type::tag(), 0);

                    if (omega > omega_max_local)
                    {
                        omega_max_local = omega;
                        x_omega_max_local = x;
                        y_omega_max_local = y;
                    }
                    if (omega < omega_min_local)
                    {
                        omega_min_local = omega;
                        x_omega_min_local = x;
                        y_omega_min_local = y;
                    }

                    const auto dA = dx_level * dx_level;
                    const auto weighted_omega = omega * dA;
                    gamma_total_local += weighted_omega;
                    if (y > 0.0)
                        gamma_top_local += weighted_omega;
                    else if (y < 0.0)
                        gamma_bottom_local += weighted_omega;

                    if (use_vorticity_threshold)
                        continue;

                    const auto q = n(q_criterion_type::tag());
                    if (q <= 0.0)
                        continue;

                    const auto fine_extent = std::max<long long>(
                        1, finest_scaling / static_cast<long long>(scaling));
                    const auto i0 =
                        static_cast<long long>(coord[0]) * fine_extent;
                    const auto j0 =
                        static_cast<long long>(coord[1]) * fine_extent;

                    local_index_data.push_back(i0);
                    local_index_data.push_back(j0);
                    local_index_data.push_back(fine_extent);
                    local_value_data.push_back(x);
                    local_value_data.push_back(y);
                    local_value_data.push_back(omega);
                    local_value_data.push_back(dA);
                }
            }
        }
    }

    std::vector<float_type> gathered_omega_max;
    std::vector<float_type> gathered_x_omega_max;
    std::vector<float_type> gathered_y_omega_max;
    std::vector<float_type> gathered_omega_min;
    std::vector<float_type> gathered_x_omega_min;
    std::vector<float_type> gathered_y_omega_min;
    boost::mpi::all_gather(world, omega_max_local, gathered_omega_max);
    boost::mpi::all_gather(world, x_omega_max_local, gathered_x_omega_max);
    boost::mpi::all_gather(world, y_omega_max_local, gathered_y_omega_max);
    boost::mpi::all_gather(world, omega_min_local, gathered_omega_min);
    boost::mpi::all_gather(world, x_omega_min_local, gathered_x_omega_min);
    boost::mpi::all_gather(world, y_omega_min_local, gathered_y_omega_min);
    boost::mpi::all_reduce(
        world, gamma_total_local, diagnostics.gamma_total,
        std::plus<float_type>());
    boost::mpi::all_reduce(
        world, gamma_top_local, diagnostics.gamma_top,
        std::plus<float_type>());
    boost::mpi::all_reduce(
        world, gamma_bottom_local, diagnostics.gamma_bottom,
        std::plus<float_type>());

    for (std::size_t i = 0; i < gathered_omega_max.size(); ++i)
    {
        if (gathered_omega_max[i] > diagnostics.omega_max)
        {
            diagnostics.omega_max = gathered_omega_max[i];
            diagnostics.x_omega_max = gathered_x_omega_max[i];
            diagnostics.y_omega_max = gathered_y_omega_max[i];
        }
        if (gathered_omega_min[i] < diagnostics.omega_min)
        {
            diagnostics.omega_min = gathered_omega_min[i];
            diagnostics.x_omega_min = gathered_x_omega_min[i];
            diagnostics.y_omega_min = gathered_y_omega_min[i];
        }
    }

    const auto omega_threshold =
        std::isfinite(diagnostics.omega_max) &&
        std::isfinite(diagnostics.omega_min) ?
        std::max(
            std::abs(diagnostics.omega_max),
            std::abs(diagnostics.omega_min)) *
            vortex_identification_threshold_factor_ :
        0.0;
    diagnostics.omega_positive_threshold = omega_threshold;
    diagnostics.omega_negative_threshold = -omega_threshold;

    if (use_vorticity_threshold)
    {
        // Two streaming sweeps are intentional here: the first finds the
        // global threshold, and this one accumulates only scalar moments.
        // That avoids storing or gathering thresholded cells across MPI.
        float_type gamma_positive_local = 0.0;
        float_type gamma_negative_local = 0.0;
        float_type area_positive_local = 0.0;
        float_type area_negative_local = 0.0;
        float_type x_moment_positive_local = 0.0;
        float_type y_moment_positive_local = 0.0;
        float_type x_moment_negative_local = 0.0;
        float_type y_moment_negative_local = 0.0;

        if constexpr (domain_type::dims == 2)
        {
            if (!domain_->is_server())
            {
                const auto center = coordinate_origin_index();
                for (auto it = domain_->begin(); it != domain_->end(); ++it)
                {
                    if (!it->locally_owned()) continue;
                    if (!it->is_leaf()) continue;
                    if (it->is_correction()) continue;

                    const auto scaling = math::pow2(it->refinement_level());
                    const auto dx_level = dx_base_ / scaling;
                    const auto dA = dx_level * dx_level;

                    for (auto& n : it->data())
                    {
                        const auto& coord = n.level_coordinate();
                        const auto x = static_cast<float_type>(
                            coord[0] - center[0] * scaling) * dx_level;
                        const auto y = static_cast<float_type>(
                            coord[1] - center[1] * scaling) * dx_level;
                        const auto omega = n(edge_aux_type::tag(), 0);
                        const auto weighted_omega = omega * dA;

                        if (omega >= omega_threshold)
                        {
                            gamma_positive_local += weighted_omega;
                            area_positive_local += dA;
                            x_moment_positive_local += x * weighted_omega;
                            y_moment_positive_local += y * weighted_omega;
                        }
                        else if (omega <= -omega_threshold)
                        {
                            gamma_negative_local += weighted_omega;
                            area_negative_local += dA;
                            x_moment_negative_local += x * weighted_omega;
                            y_moment_negative_local += y * weighted_omega;
                        }
                    }
                }
            }
        }

        vortex_region_t positive_region;
        vortex_region_t negative_region;
        boost::mpi::all_reduce(
            world, gamma_positive_local, positive_region.gamma,
            std::plus<float_type>());
        boost::mpi::all_reduce(
            world, gamma_negative_local, negative_region.gamma,
            std::plus<float_type>());
        boost::mpi::all_reduce(
            world, area_positive_local, positive_region.area,
            std::plus<float_type>());
        boost::mpi::all_reduce(
            world, area_negative_local, negative_region.area,
            std::plus<float_type>());
        boost::mpi::all_reduce(
            world, x_moment_positive_local, positive_region.x_moment,
            std::plus<float_type>());
        boost::mpi::all_reduce(
            world, y_moment_positive_local, positive_region.y_moment,
            std::plus<float_type>());
        boost::mpi::all_reduce(
            world, x_moment_negative_local, negative_region.x_moment,
            std::plus<float_type>());
        boost::mpi::all_reduce(
            world, y_moment_negative_local, negative_region.y_moment,
            std::plus<float_type>());

        if (positive_region.area >= vortex_min_region_area_ &&
            std::abs(positive_region.gamma) > 1.0e-14)
        {
            positive_region.x_center =
                positive_region.x_moment / positive_region.gamma;
            positive_region.y_center =
                positive_region.y_moment / positive_region.gamma;
        }
        if (negative_region.area >= vortex_min_region_area_ &&
            std::abs(negative_region.gamma) > 1.0e-14)
        {
            negative_region.x_center =
                negative_region.x_moment / negative_region.gamma;
            negative_region.y_center =
                negative_region.y_moment / negative_region.gamma;
        }

        std::vector<vortex_region_t> regions;
        if (positive_region.area >= vortex_min_region_area_ &&
            std::isfinite(positive_region.x_center))
        {
            diagnostics.gamma_positive.push_back(positive_region.gamma);
            diagnostics.x_center_positive.push_back(positive_region.x_center);
            diagnostics.y_center_positive.push_back(positive_region.y_center);
            positive_region.max_x = positive_region.x_center;
            regions.push_back(positive_region);
        }
        if (negative_region.area >= vortex_min_region_area_ &&
            std::isfinite(negative_region.x_center))
        {
            diagnostics.gamma_negative.push_back(negative_region.gamma);
            diagnostics.x_center_negative.push_back(negative_region.x_center);
            diagnostics.y_center_negative.push_back(negative_region.y_center);
            negative_region.max_x = negative_region.x_center;
            regions.push_back(negative_region);
        }

        std::sort(regions.begin(), regions.end(),
            [](const vortex_region_t& lhs, const vortex_region_t& rhs) {
                return lhs.x_center > rhs.x_center;
            });
        for (const auto& region : regions)
        {
            diagnostics.vortex_circulation.push_back(region.gamma);
            diagnostics.vortex_center_x.push_back(region.x_center);
            diagnostics.vortex_center_y.push_back(region.y_center);
            diagnostics.vortex_area.push_back(region.area);
            diagnostics.vortex_sign.push_back(
                region.gamma >= 0.0 ? 1.0 : -1.0);
        }

        return diagnostics;
    }

    std::vector<std::vector<long long>> gathered_index_data;
    std::vector<std::vector<float_type>> gathered_value_data;
    boost::mpi::all_gather(world, local_index_data, gathered_index_data);
    boost::mpi::all_gather(world, local_value_data, gathered_value_data);

    std::vector<vortex_cell_t> cells;
    for (std::size_t rank = 0; rank < gathered_index_data.size(); ++rank)
    {
        const auto& index_data = gathered_index_data[rank];
        const auto& value_data = gathered_value_data[rank];
        const auto n_cells =
            std::min(index_data.size() / 3, value_data.size() / 4);
        for (std::size_t i = 0; i < n_cells; ++i)
        {
            vortex_cell_t cell;
            cell.i0 = index_data[3 * i];
            cell.j0 = index_data[3 * i + 1];
            cell.extent = index_data[3 * i + 2];
            cell.x = value_data[4 * i];
            cell.y = value_data[4 * i + 1];
            cell.omega = value_data[4 * i + 2];
            cell.dA = value_data[4 * i + 3];
            cells.push_back(cell);
        }
    }

    std::vector<std::size_t> parent(cells.size());
    std::iota(parent.begin(), parent.end(), 0);

    auto find_root = [&parent](std::size_t index) {
        while (parent[index] != index)
        {
            parent[index] = parent[parent[index]];
            index = parent[index];
        }
        return index;
    };

    auto unite = [&parent, &find_root](std::size_t a, std::size_t b) {
        const auto root_a = find_root(a);
        const auto root_b = find_root(b);
        if (root_a != root_b)
            parent[root_b] = root_a;
    };

    std::unordered_map<fine_cell_key_t, std::size_t, fine_cell_key_hash_t>
        fine_cell_to_vortex_cell;
    for (std::size_t cell_idx = 0; cell_idx < cells.size(); ++cell_idx)
    {
        const auto& cell = cells[cell_idx];
        for (long long di = 0; di < cell.extent; ++di)
        {
            for (long long dj = 0; dj < cell.extent; ++dj)
            {
                const fine_cell_key_t key{cell.i0 + di, cell.j0 + dj};
                const fine_cell_key_t left{key.i - 1, key.j};
                const fine_cell_key_t down{key.i, key.j - 1};

                auto neighbor = fine_cell_to_vortex_cell.find(left);
                if (neighbor != fine_cell_to_vortex_cell.end())
                    unite(cell_idx, neighbor->second);
                neighbor = fine_cell_to_vortex_cell.find(down);
                if (neighbor != fine_cell_to_vortex_cell.end())
                    unite(cell_idx, neighbor->second);

                auto inserted =
                    fine_cell_to_vortex_cell.emplace(key, cell_idx);
                if (!inserted.second)
                    unite(cell_idx, inserted.first->second);
            }
        }
    }

    std::unordered_map<std::size_t, vortex_region_t> region_by_root;
    for (std::size_t cell_idx = 0; cell_idx < cells.size(); ++cell_idx)
    {
        const auto root = find_root(cell_idx);
        const auto& cell = cells[cell_idx];
        auto& region = region_by_root[root];
        const auto weighted_omega = cell.omega * cell.dA;
        region.gamma += weighted_omega;
        region.area += cell.dA;
        region.x_moment += cell.x * weighted_omega;
        region.y_moment += cell.y * weighted_omega;
        region.max_x = std::max(region.max_x, cell.x);
    }

    std::vector<vortex_region_t> regions;
    regions.reserve(region_by_root.size());
    for (auto& entry : region_by_root)
    {
        auto region = entry.second;
        if (region.area < vortex_min_region_area_)
            continue;

        if (std::abs(region.gamma) > 1.0e-14)
        {
            region.x_center = region.x_moment / region.gamma;
            region.y_center = region.y_moment / region.gamma;
        }
        regions.push_back(region);
    }

    const auto sort_x = [](const vortex_region_t& region) {
        return std::isfinite(region.x_center) ?
            region.x_center : region.max_x;
    };
    std::sort(regions.begin(), regions.end(),
        [&sort_x](const vortex_region_t& lhs, const vortex_region_t& rhs) {
            return sort_x(lhs) > sort_x(rhs);
        });

    for (const auto& region : regions)
    {
        diagnostics.vortex_circulation.push_back(region.gamma);
        diagnostics.vortex_center_x.push_back(region.x_center);
        diagnostics.vortex_center_y.push_back(region.y_center);
        diagnostics.vortex_area.push_back(region.area);

        if (region.gamma > 1.0e-14)
        {
            diagnostics.vortex_sign.push_back(1.0);
            diagnostics.gamma_positive.push_back(region.gamma);
            diagnostics.x_center_positive.push_back(region.x_center);
            diagnostics.y_center_positive.push_back(region.y_center);
        }
        else if (region.gamma < -1.0e-14)
        {
            diagnostics.vortex_sign.push_back(-1.0);
            diagnostics.gamma_negative.push_back(region.gamma);
            diagnostics.x_center_negative.push_back(region.x_center);
            diagnostics.y_center_negative.push_back(region.y_center);
        }
        else
        {
            diagnostics.vortex_sign.push_back(0.0);
        }
    }

    return diagnostics;
}

template<class Setup>
void Ifherk<Setup>::write_circulation_attributes(
    const std::string& flow_path,
    const typename Ifherk<Setup>::vortex_diagnostics_t& diagnostics)
{
    if constexpr (domain_type::dims != 2)
        return;

    hdf5_file<domain_type::dims> flow_file;
    flow_file.open_file_rw(flow_path);
    auto root = flow_file.get_root();
    flow_file.create_attribute(
        root, "circulation_total", diagnostics.gamma_total);
    flow_file.create_attribute(
        root, "circulation_positive", diagnostics.gamma_positive);
    flow_file.create_attribute(
        root, "circulation_negative", diagnostics.gamma_negative);
    flow_file.create_attribute(
        root, "circulation_top", diagnostics.gamma_top);
    flow_file.create_attribute(
        root, "circulation_bottom", diagnostics.gamma_bottom);
    flow_file.create_attribute(
        root, "vortex_center_positive_x", diagnostics.x_center_positive);
    flow_file.create_attribute(
        root, "vortex_center_positive_y", diagnostics.y_center_positive);
    flow_file.create_attribute(
        root, "vortex_center_negative_x", diagnostics.x_center_negative);
    flow_file.create_attribute(
        root, "vortex_center_negative_y", diagnostics.y_center_negative);
    flow_file.create_attribute(
        root, "vortex_circulation", diagnostics.vortex_circulation);
    flow_file.create_attribute(
        root, "vortex_center_x", diagnostics.vortex_center_x);
    flow_file.create_attribute(
        root, "vortex_center_y", diagnostics.vortex_center_y);
    flow_file.create_attribute(
        root, "vortex_area", diagnostics.vortex_area);
    flow_file.create_attribute(
        root, "vortex_sign", diagnostics.vortex_sign);
    flow_file.create_attribute(
        root, "vortex_count",
        static_cast<int>(diagnostics.vortex_circulation.size()));
    flow_file.create_attribute(
        root, "vortex_count_positive",
        static_cast<int>(diagnostics.gamma_positive.size()));
    flow_file.create_attribute(
        root, "vortex_count_negative",
        static_cast<int>(diagnostics.gamma_negative.size()));
    flow_file.create_attribute(root, "omega_max", diagnostics.omega_max);
    flow_file.create_attribute(root, "omega_min", diagnostics.omega_min);
    flow_file.create_attribute(
        root, "omega_positive_threshold",
        diagnostics.omega_positive_threshold);
    flow_file.create_attribute(
        root, "omega_negative_threshold",
        diagnostics.omega_negative_threshold);
    flow_file.create_attribute(
        root, "vortex_identification_threshold_factor",
        vortex_identification_threshold_factor_);
    flow_file.create_attribute(
        root, "vortex_min_region_area",
        vortex_min_region_area_);
    flow_file.create_attribute(
        root, "vortex_detection_mode",
        static_cast<int>(vortex_detection_mode_));
    flow_file.create_attribute(root, "omega_max_x", diagnostics.x_omega_max);
    flow_file.create_attribute(root, "omega_max_y", diagnostics.y_omega_max);
    flow_file.create_attribute(root, "omega_min_x", diagnostics.x_omega_min);
    flow_file.create_attribute(root, "omega_min_y", diagnostics.y_omega_min);
    flow_file.create_attribute(
        root, "formation_length", formation_length_);
    flow_file.create_attribute(
        root, "body_force_active_time", body_force_active_time_);
    flow_file.create_attribute(
        root, "body_force_center_velocity_average",
        body_force_center_velocity_average());
    flow_file.close_group(root);
}

template<class Setup>
float_type Ifherk<Setup>::body_force_center_velocity_average() const noexcept
{
    if (body_force_active_time_ <= 1.0e-14)
        return 0.0;
    return formation_length_ / body_force_active_time_;
}

template<class Setup>
template<class Velocity>
void Ifherk<Setup>::accumulate_body_force_center_velocity()
{
    if constexpr (domain_type::dims != 2)
        return;

    if (std::abs(body_force_amplitude()) <= 1.0e-14)
        return;

    boost::mpi::communicator world;
    float_type velocity_x_local = 0.0;
    int velocity_count_local = 0;

    if (domain_->is_client())
    {
        for (auto it = domain_->begin(); it != domain_->end(); ++it)
        {
            if (!it->locally_owned() || !it->has_data() ||
                !it->is_leaf() || it->is_correction())
                continue;

            const auto dx_level =
                dx_base_ / math::pow2(it->refinement_level());
            const auto pulse_index = body_force_pulse_index();
            const auto x0 = body_force_parameter(
                b_f_x0_values_, b_f_x0, pulse_index);
            const auto y0 = body_force_parameter(
                b_f_y0_values_, b_f_y0, pulse_index);
            const auto target_x =
                static_cast<int>(std::llround(x0 / dx_level));
            const auto target_y =
                static_cast<int>(std::llround(y0 / dx_level));

            for (auto& node : it->data())
            {
                const auto& coord = node.level_coordinate();
                if (coord[0] == target_x && coord[1] == target_y)
                {
                    velocity_x_local += node(Velocity::tag(), 0);
                    velocity_count_local++;
                }
            }
        }
    }

    float_type velocity_x_global = 0.0;
    int velocity_count_global = 0;
    boost::mpi::all_reduce(
        world, velocity_x_local, velocity_x_global, std::plus<float_type>());
    boost::mpi::all_reduce(
        world, velocity_count_local, velocity_count_global, std::plus<int>());

    if (velocity_count_global > 0)
    {
        const auto velocity_x =
            velocity_x_global / static_cast<float_type>(velocity_count_global);
        formation_length_ += velocity_x * dt_;
        body_force_active_time_ += dt_;
    }
}

} // namespace solver
} // namespace iblgf

#endif // IBLGF_INCLUDED_IFHERK_VORTEX_RINGS_HPP
