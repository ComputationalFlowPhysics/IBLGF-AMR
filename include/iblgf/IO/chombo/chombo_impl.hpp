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

#ifndef CHOMBOIMPL_HPP
#define CHOMBOIMPL_HPP

#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>
#include <array>
#include <map>
#include <tuple>
#include <set>
#include <cmath>
#include <vector>

#include <iblgf/IO/chombo/h5_file.hpp>
#include <iblgf/IO/chombo/h5_io.hpp>
#include <iblgf/domain/dataFields/blockDescriptor.hpp>
#include <iblgf/domain/dataFields/datafield.hpp>
#include <iblgf/global.hpp>

#include <boost/unordered_map.hpp>
#include <boost/serialization/vector.hpp>

namespace iblgf
{
namespace chombo_writer
{
template<std::size_t Dim, class BlockDescriptor, class FieldData, class Domain,
    class HDF5File>
class Chombo
{
  public: //member type:
    using value_type = float_type;
    using hsize_type = typename HDF5File::hsize_type;
    using field_type_iterator_t = typename Domain::field_type_iterator_t;

    using octant_type = typename Domain::octant_t;
    using write_info_t = typename std::pair<BlockDescriptor, FieldData*>;
    using block_info_t = typename std::tuple<int, BlockDescriptor, FieldData*>;
    using index_list_t = typename HDF5File::index_list_t;

    using key_type = typename Domain::key_t;
    using offset_type = int;
    using offset_vector = std::vector<offset_type>;

    using map_type = typename boost::unordered_map<key_type, offset_type>;
    using domain_t = Domain;
    using datablock_t = typename domain_t::datablock_t;
    using block_descriptor_t = typename datablock_t::block_descriptor_type;
    using extent_t = typename block_descriptor_t::extent_t;

    using MeshObject = domain::MeshObject;

  public: //Ctors:
    Chombo() = default;
    ~Chombo() = default;

    Chombo(const Chombo& rhs) = default;
    Chombo& operator=(const Chombo& rhs) = default;

    Chombo(Chombo&& rhs) = default;
    Chombo& operator=(Chombo& rhs) = default;

    Chombo(const std::vector<octant_type*>& _octant_blocks)
    {
        init(_octant_blocks);
    }

    struct LevelInfo
    {
        BlockDescriptor                        probeDomain;
        int                                    level;
        std::vector<octant_type*>              octants;
        std::vector<std::vector<octant_type*>> octant_groups;

        bool operator<(const LevelInfo& _other) const
        {
            return level < _other.level;
        }
    };

  private:
    void init(const std::vector<octant_type*>& _octant_blocks)
    {
        boost::mpi::communicator world;

        //Blocks per level:
        for (auto& p : _octant_blocks)
        {
            auto b = p->data_ref().descriptor(); // block descriptor
            auto it = level_map_.find(b.level());

            // FIXME: Some blocks have -1 level
            if (b.level() < 0) continue;

            if (it !=
                level_map_.end()) // if found somewhere (does not return end())
            {
                it->second.octants.push_back(p);

                //Adapt probeDomain if there is a block beyond it
                for (std::size_t d = 0; d < Dim; ++d)
                {
                    if (it->second.probeDomain.min()[d] > b.min()[d])
                        it->second.probeDomain.min()[d] = b.min()[d];

                    if (it->second.probeDomain.max()[d] < b.max()[d])
                        it->second.probeDomain.extent()[d] =
                            b.max()[d] - it->second.probeDomain.base()[d] + 1;
                }
            }
            else // if not found, create level and insert the block
            {
                LevelInfo l;
                l.level = b.level();
                l.octants.push_back(p);
                l.probeDomain = b;
                level_map_.insert(std::make_pair(b.level(), l));
            }
        } // octant blocks

        // Once level_map_ is initialized, loop through and group blocks
        for (auto it = level_map_.begin(); it != level_map_.end(); it++)
        {
            auto& l = it->second;

            unsigned int track = 0;     // beginning of new group
            int          group_no = -1; // keep track of group number
            std::vector<octant_type*> v;

            // Loop through blocks in level map.
            for (unsigned int i = 0; i != l.octants.size(); ++i)
            {
                auto& block = l.octants[i];

                if (i == track) // Beginning of new group
                {
                    l.octant_groups.push_back(v);
                    ++group_no;
                    l.octant_groups[group_no].push_back(block);
                    //int group_rank = block->rank();

                    // Check how big group is need to add subsequent blocks to the group (how
                    // Currently just grouping if all children exist (group cubes).
                    unsigned int recurse = 0;
                    unsigned int recurse_lim =
                        block->level(); // don't go beyond tree structure
                    auto* p = block;
                    while (
                        i == track &&
                        recurse <
                            recurse_lim) // only enters for first block in group
                    {
                        unsigned int shift = pow(pow(2, Dim), recurse + 1) - 1;
                        unsigned int shift_final = pow(pow(2, Dim), recurse);

                        if ((p->key().child_number() == 0) &&
                            (i + shift <= l.octants.size() - 1)) // contained?
                        {
                            //auto& block_end = l.octants[i+shift];

                            if (block_consecutive(
                                    l.octants, block->key(), i, shift))
                            {
                                ++recurse;
                                // Get parent to check bigger group.
                                p = p->parent();
                            }
                            else
                            {
                                track += shift_final;
                            }
                        }
                        else
                        {
                            track += shift_final;
                        }
                    } // recursion
                }
                else if (i < track) // add to existing group
                {
                    l.octant_groups[group_no].push_back(block);
                }
            }
        }
    } // init

    bool block_consecutive(std::vector<octant_type*>& octs, key_type k_start,
        int idx_start, int shift)
    {
        auto k_ = k_start;
        for (int i = 0; i <= shift; ++i)
        {
            if (octs[i + idx_start]->key() != k_ ||
                octs[i + idx_start]->rank() != octs[idx_start]->rank())
                return false;
            k_++;
        }
        return true;
    }

  public:
    template<typename Field, typename Block_list_t>
    void read_u(HDF5File* _file, Block_list_t& blocklist, Domain* domain)
    {
        // Cleaning ---------------------------------------------------------
        std::size_t nFields = Field::nFields;
        for (std::size_t field_idx = 0; field_idx < nFields; ++field_idx)
        {
            for (auto& b : blocklist)
            {
                if (!b->locally_owned()) continue;
                auto& cp2 = b->data_r(Field::tag(), field_idx).linalg_data();
                std::fill(cp2.begin(), cp2.end(), 0.0);
            }
        }

        // Read hdf5 file ---------------------------------------------------

        float_type dx_base = domain->dx_base();
        //int base_level = domain->tree()->base_level();

        boost::mpi::communicator world;
        if (world.rank() == 0) return;

        auto root = _file->get_root();

        // check the number for the needed attri
        int num_components = static_cast<int>(
            _file->template read_attribute<int>(root, "num_components"));
        int num_levels = static_cast<int>(
            _file->template read_attribute<int>(root, "num_levels"));

        int         component_idx = 0;
        std::string read_in_name{"u"};
        for (int i = 0; i < num_components; ++i)
        {
            auto attribute = _file->template read_attribute<std::string>(
                root, "component_" + std::to_string(i));

            if ((nFields > 1 && attribute == read_in_name + "_0") ||
                (nFields == 1 && attribute == read_in_name))
            {
                component_idx = i;
                break;
            }
        }

        // Find the imainary tree base level for the read file ----
        auto       level_group = _file->get_group("level_0");
        float_type file_dx_base = static_cast<float_type>(
            _file->template read_attribute<float_type>(level_group, "dx"));

        float_type base_level_diff_f = log2(dx_base / file_dx_base);
        if (fabs(round(base_level_diff_f) - base_level_diff_f) > 1e-10)
            throw std::runtime_error("base dx doesn't match!");
        int base_level_diff = (int)(round(base_level_diff_f));
        H5Gclose(level_group);

        //Start reading
        for (int l = 0; l < num_levels; ++l)
        {
            auto level_group = _file->get_group("level_" + std::to_string(l));
            // Read box descriptors
            auto box_dataset_id = H5Dopen2(level_group, "boxes", H5P_DEFAULT);

            // l+base_level+base_level_diff is the imaginary refinement level as if
            // everything is in the runtime octree
            int  fake_level = l + base_level_diff;
            auto file_boxes =
                _file->template read_box_descriptors<BlockDescriptor>(
                    box_dataset_id, fake_level);

            auto dataset_id =
                H5Dopen2(level_group, "data:datatype=0", H5P_DEFAULT);

            std::vector<float_type> data;
            int                     box_offset = 0;
            for (auto& file_b_dscriptr : file_boxes)
            {
                auto copy_b_dscriptr = file_b_dscriptr;

                for (auto& b : blocklist)
                {
                    if (!b->locally_owned()) continue;
                    if (!b->is_leaf()) continue;

                    auto b_dscrptr = b->data_ref().descriptor();
                    int  level = b_dscrptr.level();
                    if (level > fake_level) continue;

                    int             factor = pow(2, abs(fake_level - level));
                    BlockDescriptor overlap_fake_level;
                    BlockDescriptor overlap_local;
                    auto            has_overlap =
                        file_b_dscriptr.overlap(b_dscrptr, overlap_fake_level);
                    has_overlap =
                        b_dscrptr.overlap(file_b_dscriptr, overlap_local);

                    auto scale_up_overlap_local = overlap_local;
                    scale_up_overlap_local.level_scale(fake_level);
                    //auto sub_block_shift = overlap_fake_level.base()-scale_up_overlap_local.base();

                    std::array<int, 2> single{0, 0};
                    std::array<int, 2> avg{(factor - 1) / 2, factor / 2};
                    std::array<std::array<int, 2>, 3> FV_avg{avg, avg, avg};

                    if (has_overlap)
                    {
                        for (std::size_t field_idx = 0; field_idx < nFields;
                             ++field_idx)
                        {
                            // Finte Volume requires differnet averaging for
                            // differnt mesh objects
                            FV_avg = {avg, avg, avg};
                            if (Field::mesh_type == MeshObject::face)
                                FV_avg[field_idx] = single;
                            else if (Field::mesh_type == MeshObject::edge)
                            {
                                FV_avg = {single, single, single};
                                FV_avg[field_idx] = avg;
                            }
                            else if (Field::mesh_type == MeshObject::vertex)
                                FV_avg = {single, single, single};
                            else if (Field::mesh_type == MeshObject::cell)
                            {
                            }
                            else
                                throw std::runtime_error("Mesh object wrong");

                            //for (auto tmp: sub_block_shift)
                            //    std::cout<<tmp;

                            float_type total_avg =
                                (FV_avg[2][1] - FV_avg[2][0] + 1) *
                                (FV_avg[1][1] - FV_avg[1][0] + 1) *
                                (FV_avg[0][1] - FV_avg[0][0] + 1);

                            for (int shift_k = FV_avg[2][0];
                                 shift_k <= FV_avg[2][1]; ++shift_k)
                                for (int k = 0; k < overlap_local.extent()[2];
                                     ++k)
                                    for (int shift_j = FV_avg[1][0];
                                         shift_j <= FV_avg[1][1]; ++shift_j)
                                        for (int j = 0;
                                             j < overlap_local.extent()[1]; ++j)
                                            for (int shift_i = FV_avg[0][0];
                                                 shift_i <= FV_avg[0][1];
                                                 ++shift_i)
                                            {
                                                int file_idx_2 =
                                                    scale_up_overlap_local
                                                        .base()[2] -
                                                    file_b_dscriptr.base()[2] +
                                                    k * factor + shift_k;
                                                int file_idx_1 =
                                                    scale_up_overlap_local
                                                        .base()[1] -
                                                    file_b_dscriptr.base()[1] +
                                                    j * factor + shift_j;
                                                if (file_idx_2 >=
                                                    file_b_dscriptr.extent()[2])
                                                    continue;
                                                if (file_idx_2 < 0) continue;
                                                if (file_idx_1 >=
                                                    file_b_dscriptr.extent()[1])
                                                    continue;
                                                if (file_idx_1 < 0) continue;

                                                int offset =
                                                    box_offset +
                                                    (component_idx +
                                                        field_idx) *
                                                        file_b_dscriptr
                                                            .extent()[0] *
                                                        file_b_dscriptr
                                                            .extent()[1] *
                                                        file_b_dscriptr
                                                            .extent()[2] +
                                                    (file_idx_2)*file_b_dscriptr
                                                            .extent()[0] *
                                                        file_b_dscriptr
                                                            .extent()[1] +
                                                    (file_idx_1)*file_b_dscriptr
                                                        .extent()[0] +
                                                    (scale_up_overlap_local
                                                            .base()[0] -
                                                        file_b_dscriptr
                                                            .base()[0]) +
                                                    shift_i;

                                                std::vector<hsize_t> base(
                                                    1, offset),
                                                    extent(1, overlap_local
                                                                  .extent()[0]),
                                                    stride(1, factor);
                                                _file->template read_hyperslab<
                                                    float_type>(dataset_id,
                                                    base, extent, stride, data);

                                                for (int i = 0;
                                                     i <
                                                     overlap_local.extent()[0];
                                                     ++i)
                                                {
                                                    int file_idx_0 =
                                                        scale_up_overlap_local
                                                            .base()[0] -
                                                        file_b_dscriptr
                                                            .base()[0] +
                                                        i * factor + shift_i;
                                                    if (file_idx_0 >=
                                                        file_b_dscriptr
                                                            .extent()[0])
                                                        continue;
                                                    if (file_idx_0 < 0)
                                                        continue;

                                                    coordinate_type<int, Dim>
                                                        coffset({i + overlap_local
                                                                         .base()
                                                                             [0],
                                                            j + overlap_local
                                                                    .base()[1],
                                                            k + overlap_local
                                                                    .base()
                                                                        [2]});
                                                    b->data_r(Field::tag(),
                                                        coffset, field_idx) +=
                                                        data[i] / total_avg;
                                                }
                                            }
                        }
                    }
                }

                box_offset += copy_b_dscriptr.extent()[0] *
                              copy_b_dscriptr.extent()[1] *
                              copy_b_dscriptr.extent()[2] * num_components;
            }

            H5Dclose(box_dataset_id);
            H5Dclose(dataset_id);
            H5Gclose(level_group);
        }

        H5Gclose(root);
    }

    void write_global_metaData(HDF5File* _file, value_type _dx = 1,
        value_type _time = 0.0, int _dt = 1, int _ref_ratio = 2)
    {
        boost::mpi::communicator world;
        auto                     root = _file->get_root();

        // iterate through field to get field names
        std::vector<std::string> components;
        field_type_iterator_t::for_types(
            component_push_back_for_each(components));

        // rite components, number of components and number of levels
        for (std::size_t i = 0; i < components.size(); ++i)
        {
            _file->template create_attribute<std::string>(
                root, "component_" + std::to_string(i), components[i]);
        }

        // Get number of levels from server and communicate to all ranks
        int num_levels = 0;
        if (world.rank() == 0) { num_levels = level_map_.size(); }

        world.barrier();
        boost::mpi::broadcast(world, num_levels, 0); // send from server
        _file->template create_attribute<int>(root, "num_levels", num_levels);

        // num_components
        const int num_components = components.size();
        _file->template create_attribute<int>(
            root, "num_components", static_cast<int>(num_components));

        // Ordering of the dataspaces
        _file->template create_attribute<std::string>(
            root, "filetype", "VanillaAMRFileType");
        // Node-centering=7, zone-centering=6
        _file->template create_attribute<int>(root, "data_centering", 6);

        auto global_id = _file->create_group(root, "Chombo_global");
        _file->template create_attribute<int>(
            global_id, "SpaceDim", static_cast<int>(Dim));
        _file->close_group(global_id);

        // *******************************************************************
        // Write level structure and collective calls (everything except data

        // Use # of levels to write each level_* group
        for (int lvl = 0; lvl < num_levels; ++lvl)
        {
            auto group_id_lvl =
                _file->create_group(root, "level_" + std::to_string(lvl));

            // Write Attributes -----------------------------------------------
            // dt
            _file->template create_attribute<int>(group_id_lvl, "dt", _dt);

            // dx
            value_type dx = _dx / (std::pow(2, lvl)); // dx = 1/(2^i)
            _file->template create_attribute<value_type>(
                group_id_lvl, "dx", dx);

            // ref_ratio
            _file->template create_attribute<int>(
                group_id_lvl, "ref_ratio", _ref_ratio);

            // time
            _file->template create_attribute<int>(group_id_lvl, "time", _time);

            // data attributes
            auto group_id_dattr =
                _file->create_group(group_id_lvl, "data_attributes");
            _file->template create_attribute<std::string>(
                group_id_dattr, "objectType", "CArrayBox");

            // prob_domain ****************************************************
            // 1: Server get prob_domain
            index_list_t min_cellCentered = 0;
            index_list_t max_cellCentered = 0;
            if (world.rank() == 0)
            {
                auto it_lvl = level_map_.find(lvl);
                auto l = it_lvl->second;
                min_cellCentered = l.probeDomain.min();
                max_cellCentered = l.probeDomain.max();
                //Cell-centered data: ??? Seems fine without
                //max_cellCentered-=1;
            }
            // 2: Server broadcast prob_domain
            boost::mpi::broadcast(world, min_cellCentered[0], 0);
            boost::mpi::broadcast(world, min_cellCentered[1], 0);
            boost::mpi::broadcast(world, min_cellCentered[2], 0);
            boost::mpi::broadcast(world, max_cellCentered[0], 0);
            boost::mpi::broadcast(world, max_cellCentered[1], 0);
            boost::mpi::broadcast(world, max_cellCentered[2], 0);

            // 3: All write prob_domain
            _file->template write_boxCompound<Dim>(group_id_lvl, "prob_domain",
                min_cellCentered,
                max_cellCentered); // write prob_domain as attribute

            // Create "boxes" dataset: ****************************************
            hsize_type boxes_size = 0;

            // Get size of boxes on server
            if (world.rank() == 0)
            {
                auto it_lvl = level_map_.find(lvl);
                auto l = it_lvl->second;

                boxes_size = l.octant_groups.size();
            }
            boost::mpi::broadcast(
                world, boxes_size, 0); // send from server to all others

            _file->template create_boxCompound<Dim>(
                group_id_lvl, "boxes", boxes_size, false);

            // Create dataset for data and "offsets" **************************
            // Server gets size:
            hsize_type offsets_size = 0;
            hsize_type dset_size = 0; // Count size of dataset for this level

            // Get dataset size with octant_groups
            if (world.rank() == 0)
            {
                // Write DATA for different components-----------------------------
                auto it_lvl = level_map_.find(lvl);
                auto l = it_lvl->second;

                // Determine size of dataset for this level to create dataset
                for (std::size_t g = 0; g < l.octant_groups.size(); ++g)
                {
                    for (std::size_t b = 0; b < l.octant_groups[g].size(); ++b)
                    {
                        hsize_type  nElements_patch = 1;
                        const auto& p = l.octant_groups[g][b];
                        const auto& block_desc = p->data_ref().descriptor();
                        for (std::size_t d = 0; d < Dim;
                             ++d) // for node-centered
                        { nElements_patch *= block_desc.extent()[d]; }
                        dset_size += nElements_patch * num_components;
                    }
                }
                offsets_size = l.octant_groups.size() + 1;
            }

            // Scatter size of "offsets"
            boost::mpi::broadcast(
                world, offsets_size, 0); // send from server to all others

            // Create "offsets" dataset
            auto space_attr =
                _file->create_simple_space(1, &offsets_size, NULL);
            auto dset_Attr = _file->template create_dataset<int>(
                group_id_dattr, "offsets", space_attr);
            _file->close_space(space_attr);
            _file->close_dset(dset_Attr);
            _file->close_group(group_id_dattr);

            // send size to all clients
            world.barrier();
            boost::mpi::broadcast(
                world, dset_size, 0); // send from server to all others

            // Create full empty dataset with all processes
            auto space = _file->create_simple_space(1, &dset_size, NULL);
            auto dset_id = _file->template create_dataset<value_type>(
                group_id_lvl, "data:datatype=0", space); //collective
            _file->close_space(space);
            _file->close_dset(dset_id);
        }
        _file->close_group(root);
    } // write_global_metaData_________________________________________________

    struct component_push_back_for_each
    {
        component_push_back_for_each(std::vector<std::string>& components)
        : components_(components)
        {
            //components_=&components;
        }

        template<class T>
        auto operator()() const
        {
            std::string name = std::string(T::name());
            if (!T::output) return;
            if (T::nFields == 1) components_.push_back(name);
            else
                for (std::size_t fidx = 0; fidx < T::nFields; ++fidx)
                    components_.push_back(name + "_" + std::to_string(fidx));
        }

        std::vector<std::string>& components_;
    };

    struct write_data_for_each
    {
        write_data_for_each(std::vector<value_type>& _single_block_data,
            boost::mpi::communicator _world, extent_t _block_extent,
            int _group_extent, std::vector<octant_type*>& _ordered_group)
        : single_block_data(_single_block_data)
        , ordered_group(_ordered_group)
        {
            world = _world;
            block_extent = _block_extent;
            group_extent = _group_extent;
        }

        template<class T>
        auto operator()() const
        {
            if (!T::output) return;
            for (std::size_t fidx = 0; fidx < T::nFields; ++fidx)
            {
                double                   field_value = 0.0;
                boost::mpi::communicator world;
                if (world.rank() != 0)
                    for (auto z = 0; z < group_extent; ++z)
                        // base and max should be based on 0 (*block_extent)
                        for (auto k = 0; k < block_extent[2]; ++k)
                            for (auto y = 0; y < group_extent; ++y)
                                for (auto j = 0; j < block_extent[1]; ++j)
                                    for (auto x = 0; x < group_extent; ++x)
                                    {
                                        int group_coord =
                                            x + y * group_extent +
                                            z * pow(group_extent, 2);
                                        const auto& octant =
                                            (ordered_group)[group_coord];
                                        const auto& block_desc =
                                            octant->data_ref().descriptor();
                                        const auto& field =
                                            &(octant->data_ref().node_field());

                                        auto base = block_desc.base();

                                        for (auto i = 0; i < block_extent[0];
                                             ++i)
                                        {
                                            auto n = field->get(i + base[0],
                                                j + base[1], k + base[2]);

                                            field_value = 0.0;
                                            field_value = static_cast<value_type>( n(T::tag(),fidx)); 
                                            single_block_data.push_back( field_value);
                                        }
                                    }
            }
        }

        std::vector<value_type>&   single_block_data;
        boost::mpi::communicator   world;
        extent_t                   block_extent;
        int                        group_extent;
        std::vector<octant_type*>& ordered_group;
    };

    void write_level_info(HDF5File* _file, value_type _time = 0.0, int _dt = 1,
        int _ref_ratio = 2)
    {
        boost::mpi::communicator world;

        std::vector<std::string> components;
        //using hsize_type =H5size_type;
        auto root = _file->get_root();

        // get field names and number of components
        field_type_iterator_t::for_types(
            component_push_back_for_each(components));
        const int num_components = components.size();

        std::vector<std::vector<std::vector<offset_type>>> offset_vector;
        for (int i = 0; i < world.size(); i++)
        {
            std::vector<std::vector<offset_type>> w;
            offset_vector.push_back(w);
            for (int j = 0; j < 19; j++)
            {
                std::vector<int> v;
                offset_vector[i].push_back(v);
            }
        }

        if (world.rank() == 0)
        {
            // Loop initially to get offsets from rank==0
            for (auto& lp : level_map_)
            {
                auto        l = lp.second; // LevelInfo
                int         lvl = l.level;
                offset_type offset = 0; // offset in dataset

                // Calculate offsets by group
                for (std::size_t g = 0; g < l.octant_groups.size(); ++g)
                {
                    for (std::size_t b = 0; b < l.octant_groups[g].size(); ++b)
                    {
                        const auto& octant = l.octant_groups[g][b];
                        int         rank = octant->rank();
                        // Loop through components in block
                        hsize_type  nElements_patch = 1;
                        const auto& block_desc =
                            octant->data_ref().descriptor();
                        for (std::size_t d = 0; d < Dim;
                             ++d) // for node-centered
                        { nElements_patch *= block_desc.extent()[d]; }

                        int data_block_size = nElements_patch * num_components;

                        if (b == 0)
                        { offset_vector[rank][lvl].push_back(offset); }
                        offset += data_block_size;
                    }
                }
            }
        }

        std::vector<std::vector<offset_type>> offsets_for_rank;
        boost::mpi::scatter(world, offset_vector, offsets_for_rank, 0);

        world.barrier();

        // Parallel Write -----------------------------------------------------
        // Loop over levels and write patches
        for (auto& lp : level_map_)
        {
            // lp iterates over each pair of <int, LevelInfo> in the full map
            auto l = lp.second; // LevelInfo
            int  lvl = l.level;

            auto group_id =
                _file->create_group(root, "level_" + std::to_string(lvl));

            /*****************************************************************/
            // Write level data
            // Write "offsets" ------------------------------------------------
            std::vector<int> patch_offsets_vec;
            patch_offsets_vec.push_back(0);

            // Gather patch_offsets for each group
            for (std::size_t g = 0; g < l.octant_groups.size(); ++g)
            {
                hsize_type patch_offset = 0;
                for (std::size_t b = 0; b < l.octant_groups[g].size(); ++b)
                {
                    hsize_type  nElements_patch = 1;
                    const auto& p = l.octants[b];
                    const auto& block_desc = p->data_ref().descriptor();

                    for (std::size_t d = 0; d < Dim; ++d) // for node-centered
                    { nElements_patch *= block_desc.extent()[d]; }
                    patch_offset += nElements_patch * num_components;
                }
                patch_offsets_vec.push_back(patch_offset);
            }

            // data attributes (open because already created)
            auto group_id_dattr =
                _file->create_group(group_id, "data_attributes");
            // offsets (Written only by server)
            auto dset_Attr = _file->open_dataset(group_id_dattr, "offsets");
            if (world.rank() == 0)
            {
                hsize_type size = patch_offsets_vec.size();
                _file->template write<int>(
                    dset_Attr, size, &patch_offsets_vec[0]);
            }
            _file->close_dset(dset_Attr);
            _file->close_group(group_id_dattr);

            // ----------------------------------------------------------------
            // Write DATA for different components-----------------------------
            auto dset_id = _file->open_dataset(group_id, "data:datatype=0");

            std::vector<value_type> single_block_data;

            // GROUP ITERATOR
            for (std::size_t g = 0; g < l.octant_groups.size(); ++g)
            {
                single_block_data.clear();
                const auto& octant0 = l.octant_groups[g][0];
                const auto& block_desc0 = octant0->data_ref().descriptor();

                auto base0 = block_desc0.base();
                auto block_extent = block_desc0.extent();
                // for cubic blocks and cubic extents
                int group_extent = std::cbrt(l.octant_groups[g].size());

                // Order octants:
                std::vector<octant_type*> ordered_group(
                    l.octant_groups[g].size());

                // BLOCK ITERATOR
                for (std::size_t b = 0; b < l.octant_groups[g].size(); ++b)
                {
                    const auto& p = l.octant_groups[g][b];
                    const auto& block_desc = p->data_ref().descriptor();

                    // Get group coordinate based on shift
                    auto base_shift0 = block_desc.base() - base0;

                    // Order: Get flattened index and insert
                    // coord = k*maxI*maxJ + j*maxI + I
                    int group_coord =
                        base_shift0[0] / block_extent[0] +
                        base_shift0[1] / block_extent[1] * group_extent +
                        base_shift0[2] / block_extent[2] * group_extent *
                            group_extent;

                    ordered_group[group_coord] = p;
                }

                // Iterate and print
                // Assumes all blocks have same extents
                // Iterate over blocks in group: x,y,z
                // Iterate over cells in block: i,j,k

                // ------------------------------------------------------------
                // COMPONENT ITERATOR
                field_type_iterator_t::for_types(
                    write_data_for_each(single_block_data, world, block_extent,
                        group_extent, ordered_group));

                // Write single block data
                hsize_type block_data_size = single_block_data.size();
                hsize_type start = -1;
                if (world.rank() != 0)
                {
                    start = offsets_for_rank[lvl][g];
                    _file->template write<value_type>(
                        dset_id, block_data_size, start, &single_block_data[0]);
                }
            } // GROUP ITERATOR

            // Write "boxes" --------------------------------------------------
            if (world.rank() == 0)
            {
                // Determine and write "boxes"
                auto  p = l.octant_groups[0][0];
                auto& block_desc = p->data_ref().descriptor();

                auto pmin = block_desc.min(); // vector of ints size 3
                auto pmax = block_desc.max();
                std::vector<decltype(pmin)> mins(l.octant_groups.size());
                std::vector<decltype(pmax)> maxs(l.octant_groups.size());
                decltype(pmin)              mins_temp;
                decltype(pmax)              maxs_temp;

                for (std::size_t g = 0; g < l.octant_groups.size(); ++g)
                {
                    for (std::size_t b = 0; b < l.octant_groups[g].size(); ++b)
                    {
                        const auto  p = l.octant_groups[g][b];
                        const auto& block_desc = p->data_ref().descriptor();
                        if (b == 0)
                        {
                            mins_temp = block_desc.min();
                            maxs_temp = block_desc.max();
                        }
                        else
                        {
                            for (std::size_t d = 0; d < Dim; ++d)
                            {
                                if (mins_temp[d] > block_desc.min()[d])
                                { mins_temp[d] = block_desc.min()[d]; }
                                if (maxs_temp[d] < block_desc.max()[d])
                                { maxs_temp[d] = block_desc.max()[d]; }
                            }
                        }
                    }
                    mins[g] = mins_temp;
                    maxs[g] = maxs_temp;
                }

                // 2 Write boxes with just rank 0
                _file->template open_write_boxCompound<Dim>(
                    group_id, "boxes", mins, maxs, false);
            }

            // Close spaces:
            _file->close_dset(dset_id);
            _file->close_group(group_id);
        } // level iterator

        _file->close_group(root);
    }

    void write_mesh(HDF5File* _file)
    {
        this->write_global_metaData(_file);
        this->write_level_info(_file);
    }

    void write_mesh(std::string _filename)
    {
        HDF5File f(_filename);
        this->write_global_metaData(&f);
        this->write_level_info(&f);
    }

    std::map<int, LevelInfo> level_map_;
};
} // namespace chombo_writer
} // namespace iblgf

#endif //Chombo
