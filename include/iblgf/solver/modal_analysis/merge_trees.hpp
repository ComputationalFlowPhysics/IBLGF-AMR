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

#ifndef IBLGF_MERGE_TREES_HPP
#define IBLGF_MERGE_TREES_HPP

#include <algorithm>
#include <cmath>
#include <fstream>
#include <memory>
#include <set>
#include <vector>
#include <boost/mpi/collectives.hpp>
#include <iblgf/dictionary/dictionary.hpp>
#include <iblgf/IO/parallel_ostream.hpp>
#include <iblgf/domain/domain.hpp>
#include <iblgf/types.hpp>

namespace iblgf
{
using namespace domain;
using namespace types;
using namespace dictionary;
template<class Setup>
class MergeTrees
{
  public:
    struct CommonTreeRestartState
    {
        int         next_idx = 0;
        std::string ref_tree_file;
        std::string ref_flow_file;
    };

    struct AdaptRestartState
    {
        int next_idx = 0;
    };

    // using tree_t = typename Setup::tree_t;
    using key_id_t = typename Setup::key_id_t;
    using key_t = typename Setup::domain_t::key_t;
    // using tree_t = typename Setup::domain_t::tree_t;

    MergeTrees() = default;
    MergeTrees(const MergeTrees&) = delete;
    MergeTrees(MergeTrees&&) = default;
    MergeTrees& operator=(const MergeTrees&) & = delete;
    MergeTrees& operator=(MergeTrees&&) & = default;
    ~MergeTrees() = default;
    MergeTrees(dictionary::Dictionary* _d)
    : dict_ref_(_d)
    {
        auto dict_ref = _d->get_dictionary("simulation_parameters");
        auto dict_out = dict_ref->get_dictionary("output");
        dir_out_ = dict_out->get_or<std::string>("directory",
            "output"); // where common tree or adapted snapshots are stored
        dir_in_ = dict_out->get_or<std::string>("field_dir", dir_out_); // where original snapshots are stored
        nStart_ = dict_ref->get_or<int>("nStart", 0);                   // starting index of snapshots to merge
        nTotal_ = dict_ref->get_or<int>("nTotal", 1);                   // total number of snapshots to merge
        nSkip_ = dict_ref->get_or<int>("nskip", 1); // skip factor between snapshots to merge
        tree_file_prefix_ = dict_ref->get_or<std::string>("tree_file_prefix", "tree_info_");
        flow_file_prefix_ = dict_ref->get_or<std::string>("flow_file_prefix", "flowTime_");
    }

  public:
    static bool load_common_tree_restart_state(
        const std::string& path, CommonTreeRestartState& state)
    {
        std::ifstream in(path);
        if (!in.good()) { return false; }

        std::string line;
        while (std::getline(in, line))
        {
            const auto pos = line.find('=');
            if (pos == std::string::npos) continue;
            const auto key = line.substr(0, pos);
            const auto val = line.substr(pos + 1);
            if (key == "next_idx") state.next_idx = std::stoi(val);
            else if (key == "ref_tree_file")
                state.ref_tree_file = val;
            else if (key == "ref_flow_file")
                state.ref_flow_file = val;
        }
        return state.next_idx > 0 && !state.ref_tree_file.empty() && !state.ref_flow_file.empty();
    }

    static void write_common_tree_restart_state(
        const std::string& path, const CommonTreeRestartState& state)
    {
        std::ofstream out(path, std::ios::trunc);
        out << "next_idx=" << state.next_idx << "\n";
        out << "ref_tree_file=" << state.ref_tree_file << "\n";
        out << "ref_flow_file=" << state.ref_flow_file << "\n";
    }

    static bool load_adapt_restart_state(const std::string& path, AdaptRestartState& state)
    {
        std::ifstream in(path);
        if (!in.good()) { return false; }

        std::string line;
        while (std::getline(in, line))
        {
            const auto pos = line.find('=');
            if (pos == std::string::npos) continue;
            const auto key = line.substr(0, pos);
            const auto val = line.substr(pos + 1);
            if (key == "next_idx") state.next_idx = std::stoi(val);
        }
        return state.next_idx > 0;
    }

    static void write_adapt_restart_state(
        const std::string& path, const AdaptRestartState& state)
    {
        std::ofstream out(path, std::ios::trunc);
        out << "next_idx=" << state.next_idx << "\n";
    }

    std::unique_ptr<Setup> get_common_tree()
    {
        boost::mpi::communicator world;
        pcout << "\nCollecting tree keys from all domains...\n" << std::endl;

        auto read_tree_file_keys = [&](const std::string& tree_file) {
            std::set<key_id_t> keys;
            if (world.rank() == 0)
            {
                std::ifstream ifs(tree_file, std::ios::binary);
                if (!ifs.is_open())
                {
                    throw std::runtime_error("Could not open file: " + tree_file);
                }
                ifs.seekg(0, std::ios::end);
                const auto file_size = ifs.tellg();
                ifs.seekg(0, std::ios::beg);
                const std::size_t n_entries = static_cast<std::size_t>(file_size) /
                    (sizeof(key_id_t) + sizeof(bool));
                for (std::size_t i = 0; i < n_entries; ++i)
                {
                    key_id_t id{};
                    ifs.read(reinterpret_cast<char*>(&id), sizeof(id));
                    keys.insert(id);
                }
            }
            std::vector<key_id_t> keys_vec;
            if (world.rank() == 0) keys_vec.assign(keys.begin(), keys.end());
            boost::mpi::broadcast(world, keys_vec, 0);
            return std::set<key_id_t>(keys_vec.begin(), keys_vec.end());
        };

        // Start with snapshot 0 as the reference domain, but compute overlap
        // keys from raw tree files to avoid adaptation side effects.
        std::string tree_file_0 = tree_file_path(dir_in_, nStart_);
        std::string flow_file_0 = flow_file_path(dir_in_, nStart_);
        auto        ref_domain_ = std::make_unique<Setup>(dict_ref_, tree_file_0, flow_file_0);
        auto        overlap_keys = read_tree_file_keys(tree_file_0);

        for (int i = 1; i < nTotal_; i++)
        {
            std::string tree_file_i = tree_file_path(dir_in_, nStart_ + i * nSkip_);
            auto        keys_i = read_tree_file_keys(tree_file_i);

            std::set<key_id_t> intersection;
            std::set_intersection(overlap_keys.begin(), overlap_keys.end(),
                keys_i.begin(), keys_i.end(),
                std::inserter(intersection, intersection.begin()));
            overlap_keys.swap(intersection);

            pcout << "Merged tree from snapshot " << i << "/" << nTotal_ - 1 << std::endl;
            if (overlap_keys.empty())
            {
                pcout << "Warning: common-tree key intersection is empty after snapshot " << i
                      << std::endl;
                break;
            }
        }

        auto ref_tree = ref_domain_->tree();
        std::vector<key_t> to_delete;
        to_delete.reserve(1024);
        for (auto it = ref_tree->begin(); it != ref_tree->end(); ++it)
        {
            if (!it->has_data()) continue;
            if (overlap_keys.find(it->key().id()) != overlap_keys.end()) continue;
            to_delete.emplace_back(it->key());
        }

        std::sort(to_delete.begin(), to_delete.end(), [](const key_t& a, const key_t& b) {
            return a.level() > b.level();
        });
        for (const auto& k : to_delete)
        {
            auto oct = ref_tree->find_octant(k);
            if (!oct || !oct->has_data()) continue;
            ref_tree->delete_oct(oct);
        }

        ref_tree->construct_level_maps();
        ref_tree->construct_leaf_maps(false);
        ref_tree->construct_lists();
        ref_domain_->save_common_tree_ref();
        return ref_domain_;
    }

    void adapt_to_ref_with_restart()
    { adapt_to_ref(); }

    std::unique_ptr<Setup> ref_to_symmetric_ref_with_symfield()
    {
        auto sim_dict = dict_ref_->get_dictionary("simulation_parameters");
        const bool do_symfield = sim_dict->get_or<bool>("do_symfield", true);
        const int  symfield_time_idx = sim_dict->get_or<int>("symfield_time_idx", 2);

        auto ref_domain = ref_to_symmetric_ref();
        if (do_symfield)
        {
            ref_domain->initialize();
            ref_domain->template symfield<typename Setup::u_type, typename Setup::u_type>(
                symfield_time_idx);
        }
        return ref_domain;
    }

    void adapt_to_ref()
    {
        boost::mpi::communicator world;
        pcout << "\nAdapting trees to common tree...\n" << std::endl;
        auto sim_dict = dict_ref_->get_dictionary("simulation_parameters");
        std::string tree_ref_file =
            sim_dict->get<std::string>("tree_ref_file");
        std::string flow_ref_file =
            sim_dict->get<std::string>("flow_ref_file");
        int  ref_levels = sim_dict->get_or<int>("nLevels", 0);
        const bool resume = sim_dict->get_or<bool>("resume", false);
        const bool do_symfield = sim_dict->get_or<bool>("do_symfield", false);
        const std::string restart_file =
            sim_dict->get_or<std::string>("restart_file", "./adapt_to_ref_restart.txt");

        const int idx_end = nStart_ + (nTotal_ - 1) * nSkip_;
        int       idx_begin = nStart_;
        AdaptRestartState restart_state;
        const bool has_restart = resume && load_adapt_restart_state(restart_file, restart_state);
        if (has_restart && restart_state.next_idx <= idx_end)
        {
            idx_begin = restart_state.next_idx;
            if (world.rank() == 0)
            {
                std::cout << "Resuming adapt_to_ref from restart file: " << restart_file
                          << " (next_idx=" << idx_begin << ")" << std::endl;
            }
        }

        auto ref_domain = std::make_unique<Setup>(dict_ref_, tree_ref_file, flow_ref_file);
        auto ref_tree = ref_domain->tree();
        std::vector<key_t> octs;
        std::vector<int>   level_change;
        for (int idx = idx_begin; idx <= idx_end; idx += nSkip_)
        {
            std::string tree_file_i = tree_file_path(dir_in_, idx);
            std::string flow_file_i = flow_file_path(dir_in_, idx);
            auto        domain_i = std::make_unique<Setup>(dict_ref_, tree_file_i, flow_file_i);
            auto        tree_i = domain_i->tree();
            for (int level = ref_levels; level >= 0; --level)
            {
                std::set<std::pair<typename key_t::value_type, int>> prev_request_sig;
                for (int pass = 0; pass < 16;
                    ++pass) // iterate until no more octs to adapt at this level, to handle chained +1 refinements
                {
                    octs.clear();
                    level_change.clear();
                    get_level_changes_distributed(ref_tree, tree_i, level, octs, level_change, world);
                    if (octs.empty()) break;

                    std::set<std::pair<typename key_t::value_type, int>> cur_request_sig;
                    for (std::size_t i = 0; i < octs.size(); ++i)
                    {
                        cur_request_sig.emplace(octs[i].id(), level_change[i]);
                    }

                    const auto before_keys = globalize_key_set(
                        world, collect_physical_leaf_key_ids(*domain_i));
                    run_adapt_from_keys_distributed(*domain_i, -1, octs, level_change, world);
                    const auto after_keys = globalize_key_set(
                        world, collect_physical_leaf_key_ids(*domain_i));

                    const bool repeated_request = (cur_request_sig == prev_request_sig);
                    const bool no_tree_change = (before_keys == after_keys);
                    if (repeated_request && no_tree_change)
                    {
                        pcout << "Warning: non-progress adapt loop detected at level "
                              << level << ", pass " << pass
                              << " (request repeated and key set unchanged). Breaking."
                              << std::endl;
                        break;
                    }
                    prev_request_sig.swap(cur_request_sig);
                }
            }
            if (do_symfield)
            {
                domain_i->template symfield<typename Setup::u_type, typename Setup::u_type>(idx);
            }
            domain_i->save_adapted(idx);
            if (world.rank() == 0)
            {
                AdaptRestartState out_state;
                out_state.next_idx = idx + nSkip_;
                write_adapt_restart_state(restart_file, out_state);
            }
            pcout << "interpolated snapshot " << idx << "/" << idx_end
                  << " to common tree reference levels." << std::endl;
        }
    }
    std::unique_ptr<Setup> adapt_to_ref(int idx)
    {
        boost::mpi::communicator world;
        pcout << "\nAdapting trees to common tree...\n" << std::endl;
        std::string tree_ref_file =
            dict_ref_->get_dictionary("simulation_parameters")->get<std::string>("tree_ref_file");
        std::string flow_ref_file =
            dict_ref_->get_dictionary("simulation_parameters")->get<std::string>("flow_ref_file");
        int  ref_levels = dict_ref_->get_dictionary("simulation_parameters")->get_or<int>("nLevels", 0);
        auto ref_domain = std::make_unique<Setup>(dict_ref_, tree_ref_file, flow_ref_file);
        auto ref_tree = ref_domain->tree();
        std::vector<key_t> octs;
        std::vector<int>   level_change;

        std::string tree_file_i = tree_file_path(dir_in_, idx);
        std::string flow_file_i = flow_file_path(dir_in_, idx);
        auto        domain_i = std::make_unique<Setup>(dict_ref_, tree_file_i, flow_file_i);
        auto        tree_i = domain_i->tree();
        for (int level = ref_levels; level >= 0; --level)
        {
            std::set<std::pair<typename key_t::value_type, int>> prev_request_sig;
            for (int pass = 0; pass < 16;
                ++pass) // iterate until no more octs to adapt at this level, to handle chained +1 refinements
            {
                octs.clear();
                level_change.clear();
                get_level_changes_distributed(ref_tree, tree_i, level, octs, level_change, world);
                if (octs.empty()) break;

                std::set<std::pair<typename key_t::value_type, int>> cur_request_sig;
                for (std::size_t i = 0; i < octs.size(); ++i)
                {
                    cur_request_sig.emplace(octs[i].id(), level_change[i]);
                }

                const auto before_keys = globalize_key_set(
                    world, collect_physical_leaf_key_ids(*domain_i));
                run_adapt_from_keys_distributed(*domain_i, -1, octs, level_change, world);
                const auto after_keys = globalize_key_set(
                    world, collect_physical_leaf_key_ids(*domain_i));

                const bool repeated_request = (cur_request_sig == prev_request_sig);
                const bool no_tree_change = (before_keys == after_keys);
                if (repeated_request && no_tree_change)
                {
                    pcout << "Warning: non-progress adapt loop detected at level "
                          << level << ", pass " << pass
                          << " (request repeated and key set unchanged). Breaking."
                          << std::endl;
                    break;
                }
                prev_request_sig.swap(cur_request_sig);
            }
        }
        domain_i->save_adapted(idx);
        pcout << "interpolated snapshot " << idx << "/" << nTotal_ - 1 << " to common tree reference levels."
              << std::endl;
        return domain_i;
    }

    std::unique_ptr<Setup> ref_to_symmetric_ref()
    {
        boost::mpi::communicator world;
        auto                     sim_dict = dict_ref_->get_dictionary("simulation_parameters");
        std::string              tree_ref_file = sim_dict->get<std::string>("tree_ref_file");
        std::string              flow_ref_file = sim_dict->get<std::string>("flow_ref_file");
        auto       domain_dict = dict_ref_->get_dictionary("simulation_parameters")->get_dictionary("domain");
        const auto bd_base = domain_dict->get<int, Setup::u_type::nFields()>("bd_base");
        const auto block_extent = domain_dict->get<int>("block_extent");
        const int  mirror_span = (-2 * bd_base[1]) / block_extent;

        auto               ref_domain = std::make_unique<Setup>(dict_ref_, tree_ref_file, flow_ref_file);
        std::vector<key_t> octs;
        std::vector<int>   level_change;
        auto               tree_for_levels = ref_domain->tree();
        const int max_ref_level = static_cast<int>(tree_for_levels->depth()) - 1 - tree_for_levels->base_level();
        for (int i = max_ref_level; i >= 0; --i)
        {
            for (int pass = 0; pass < 16; ++pass)
            {
                octs.clear();
                level_change.clear();
                auto ref_tree = ref_domain->tree();
                symgrid(ref_tree, i, octs, level_change, mirror_span, false);
                bool no_changes = octs.empty();
                boost::mpi::broadcast(world, no_changes, 0);
                if (no_changes) break; // all ranks take the same branch
                run_adapt_from_keys_distributed(*ref_domain, -1, octs, level_change, world);
            }
        }
        ref_domain->save_symmetric_ref();
        return ref_domain;
    }

  public:
    std::string flow_file_path(const std::string& dir, const int idx)
    { return "./" + dir + "/" + flow_file_prefix_ + std::to_string(idx) + ".hdf5"; }

    std::string tree_file_path(const std::string& dir, const int idx)
    { return "./" + dir + "/" + tree_file_prefix_ + std::to_string(idx) + ".bin"; }
    static void merge()
    {
        boost::mpi::communicator world;
        if (world.rank() != 0) return;
        std::cout << "Merging trees..." << std::endl;
    }
    template<class SetupType>
    static std::set<key_id_t> get_tree_keys(SetupType& domain_setup)
    {
        std::cout << "Collecting tree keys..." << std::endl;
        boost::mpi::communicator world;
        std::set<key_id_t>       local_keys;
        std::set<key_id_t>       global_keys;
        auto                     tree = domain_setup.tree();
        for (auto it = tree->begin(); it != tree->end(); ++it)
        {
            if (!it->has_data()) continue;
            if (!it->locally_owned()) continue;
            local_keys.insert(it->key().id());
        }
        std::cout << "Local keys collected, size: " << local_keys.size() << std::endl;
        std::vector<key_id_t>              local_vec(local_keys.begin(), local_keys.end());
        std::vector<std::vector<key_id_t>> gathered;
        boost::mpi::gather(world, local_vec, gathered, 0);
        std::cout << "Keys gathered from all ranks, size: " << gathered.size() << std::endl;
        if (world.rank() == 0)
        {
            std::set<key_id_t> merged;
            for (const auto& v : gathered) { merged.insert(v.begin(), v.end()); }
            global_keys = std::set<key_id_t>(merged.begin(), merged.end());
        }
        boost::mpi::broadcast(world, global_keys, 0);

        return global_keys;
    }

    template<class SetupType>
    static std::set<key_id_t> collect_physical_leaf_key_ids(SetupType& domain_setup)
    {
        std::set<key_id_t> keys;
        auto               tree = domain_setup.tree();
        for (auto it = tree->begin(); it != tree->end(); ++it)
        {
            if (!it->has_data()) continue;
            if (!it->is_leaf()) continue;
            if (it->is_correction()) continue;
            if (!it->physical()) continue;
            keys.insert(it->key().id());
        }
        return keys;
    }

    static std::set<key_id_t> globalize_key_set(const boost::mpi::communicator& world, const std::set<key_id_t>& local)
    {
        std::vector<key_id_t>              local_vec(local.begin(), local.end());
        std::vector<std::vector<key_id_t>> gathered;
        boost::mpi::gather(world, local_vec, gathered, 0);

        std::vector<key_id_t> global_vec;
        if (world.rank() == 0)
        {
            std::set<key_id_t> merged;
            for (const auto& v : gathered) { merged.insert(v.begin(), v.end()); }
            global_vec.assign(merged.begin(), merged.end());
        }
        boost::mpi::broadcast(world, global_vec, 0);
        return std::set<key_id_t>(global_vec.begin(), global_vec.end());
    }

    template<class tree_t, class key_t>
    static void getDelOcts_distributed(tree_t& t1, tree_t& t2, std::vector<key_t>& keys_to_del,
        std::vector<int>& level_change, const boost::mpi::communicator& world)
    {
        getDelOcts(t1, t2, keys_to_del, level_change);
        boost::mpi::broadcast(world, keys_to_del, 0);
        boost::mpi::broadcast(world, level_change, 0);
    }

    template<class SetupType, class key_t>
    static void run_adapt_del_distributed(SetupType& setup, std::vector<key_t>& keys_to_del,
        std::vector<int>& level_change, const boost::mpi::communicator& world)
    {
        boost::mpi::broadcast(world, keys_to_del, 0);
        boost::mpi::broadcast(world, level_change, 0);
        setup.run_adapt_del(keys_to_del, level_change);
    }

    template<class tree_t, class key_t>
    static void get_level_changes_distributed(tree_t& ref_tree, tree_t& old_tree, int ref_level,
        std::vector<key_t>& octs, std::vector<int>& level_change, const boost::mpi::communicator& world)
    {
        get_level_changes(ref_tree, old_tree, ref_level, octs, level_change);
        boost::mpi::broadcast(world, octs, 0);
        boost::mpi::broadcast(world, level_change, 0);
    }

    template<class SetupType, class key_t>
    static void run_adapt_from_keys_distributed(SetupType& setup, int timeIdx, std::vector<key_t>& octs,
        std::vector<int>& level_change, const boost::mpi::communicator& world)
    {
        boost::mpi::broadcast(world, octs, 0);
        boost::mpi::broadcast(world, level_change, 0);
        if (octs.empty()) return;
        setup.template run_adapt_from_keys<typename SetupType::u_type>(timeIdx, octs, level_change);
    }

    template<class SetupType>
    static void initialize_linear_u_field(SetupType& domain_setup)
    {
        auto tree = domain_setup.tree();
        for (auto it = tree->begin(); it != tree->end(); ++it)
        {
            if (!it->locally_owned()) continue;
            if (!it->has_data()) continue;
            for (auto& node : it->data())
            {
                const auto gc = node.global_coordinate();
                for (std::size_t d = 0; d < SetupType::u_type::nFields(); ++d)
                {
                    node(SetupType::u, static_cast<int>(d)) = gc[d];
                }
            }
        }
    }

    template<class SetupType>
    static float_type max_linear_u_error(SetupType& domain_setup)
    {
        float_type local_max = 0.0;
        auto              tree = domain_setup.tree();
        for (auto it = tree->begin(); it != tree->end(); ++it)
        {
            if (!it->locally_owned()) continue;
            if (!it->has_data()) continue;
            if (!it->physical()) continue;

            for (auto& node : it->data())
            {
                const auto gc = node.global_coordinate();
                for (std::size_t d = 0; d < SetupType::u_type::nFields(); ++d)
                {
                    node(SetupType::u_s, static_cast<int>(d)) = node(SetupType::u, static_cast<int>(d)) - gc[d];
                    local_max = std::max(local_max, std::abs(node(SetupType::u_s, static_cast<int>(d))));
                }
            }
        }
        return local_max;
    }

    template<class tree_t>
    static void merger(tree_t& t1, tree_t& t2, std::vector<key_id_t>& octs, std::vector<int>& leafs)
    {
        boost::mpi::communicator world;
        // std::cout << "Merging trees..." << std::endl;
        octs.clear();
        leafs.clear();
        if (world.rank() != 0) return;
        std::cout << "Merging trees..." << std::endl;
        for (auto it1 = t1->begin(); it1 != t1->end(); ++it1)
        {
            if (!it1->has_data()) continue;
            auto it2 = t2->find_octant(it1->key());
            if (it2 && (it2->has_data() && it1->has_data()))
            {
                int is_leaf = 0;
                if ((it1->is_correction() || it2->is_correction()) && it1->refinement_level() > 0)
                {
                    // if(it1->is_correction()&& it2->is_correction()) continue;
                    // if(it1->is_correction()|| it2->is_correction()) continue;
                    // if(it1->refinement_level()>0&&!(it1->is_correction()|| it2->is_correction())) continue;
                    is_leaf = 0;
                }
                // if(it1->is_leaf_search()&& it2->is_leaf_search())
                // {
                //     is_leaf=1;
                // }
                // else if(it1->is_leaf_search()|| it2->is_leaf_search())
                // {
                //     if(it1->refinement_level()>0&&!(it1->is_correction()|| it2->is_correction())) is_leaf=0;
                //     else
                //     {
                //         is_leaf=1;
                //     }
                // }
                else if (it1->is_leaf() || it2->is_leaf()) { is_leaf = 1; }
                // else if(it1->refinement_level()==0 && (it1->is_leaf() || it2->is_leaf()))
                // {
                //     is_leaf=1;
                // }
                // else if (it1->is_leaf() && !it2->is_correction())
                // {
                //     is_leaf=1;
                // }
                // else if (it2->is_leaf() && !it1->is_correction())
                // {
                //     is_leaf=1;
                // }

                // if((it1->is_leaf()|| it2->is_leaf()))
                // {
                //     if(it1->refinement_level()>0&&(it1->is_correction()|| it2->is_correction())) continue;
                //     // if(it1->is_leaf() && it2->is_leaf()) is_leaf=1;
                //     is_leaf=0;
                //     // if(it1->refinement_level()>=0&&(it1->is_correction()|| it2->is_correction())) is_leaf=false;
                //     // std::cout << "Common Leaf: " << it1->tree_coordinate() << std::endl;
                //     // std::cout<<"level: "<<it1->refinement_level()<<std::endl;
                // }
                octs.emplace_back(it2->key().id());
                leafs.emplace_back(is_leaf);
            }
        }
    }

    template<class tree_t>
    static void get_ranks(tree_t& t1, std::vector<key_id_t>& octs, std::vector<int>& leafs)
    {
        boost::mpi::communicator world;
        // server contains all octs and leafs and client has empty vector. use function to get ranks from server and fill in vectors
        std::vector<std::vector<key_id_t>> send_octs(world.size()); //outer vector is rank
        std::vector<std::vector<int>>      send_leafs(world.size());

        if (world.rank() == 0)
        {
            std::cout << "Getting ranks for octs..." << std::endl;
            for (auto& k : octs)
            {
                auto nn = t1->find_octant(k);
                if (!nn) continue;
                if (nn->rank() > 0)
                {
                    send_octs[nn->rank()].emplace_back(k);
                    send_leafs[nn->rank()].emplace_back(leafs[&k - &octs[0]]);
                }
            }

            world.barrier();
            for (int i = 1; i < world.size(); ++i)
            {
                world.send(i, i * 2, send_octs[i]);
                world.send(i, i * 2 + 1, send_leafs[i]);
            }
            world.barrier();
        }
        else
        {
            octs.clear();
            leafs.clear();
            world.barrier();
            world.recv(0, world.rank() * 2, octs);
            world.recv(0, world.rank() * 2 + 1, leafs);
            world.barrier();
        }
    }

    template<class tree_t, class key_t>
    static void getDelOcts(tree_t& t1, tree_t& t2, std::vector<key_t>& keys_to_del, std::vector<int>& level_change)
    {
        boost::mpi::communicator world;
        // get octs from t1 which are not in t2 and have data so that we can delete them in next step to adapt tree
        keys_to_del.clear();
        level_change.clear();
        if (world.rank() != 0) return;
        std::cout << "Getting octs to delete..." << std::endl;
        for (auto it1 = t1->begin(); it1 != t1->end(); ++it1)
        {
            if (!it1->has_data()) continue;
            if (it1->refinement_level() < 0) continue; //
            if (!it1->is_leaf() && it1->is_correction())
            {
                keys_to_del.emplace_back(it1->key());
                level_change.emplace_back(-1);
                continue;
            }
            if (!it1->is_leaf()) continue;
            // if(it1->refinement_level()<=0) continue; // only leafs with refinement level >=0 need to be adapted
            auto it2 = t2->find_octant(it1->key());
            // if(!it2 || !it2->has_data()||(it2->is_correction() && it2->refinement_level()>0)||(it1->is_correction()&&it1->refinement_level()==0))
            // if(!it2 || !it2->has_data()||(it2->is_correction())||(it1->is_correction()&&it1->refinement_level()==0))
            if (!it2 || !it2->has_data() || !it2->physical() || (it2->is_correction()))
            {
                keys_to_del.emplace_back(it1->key());
                level_change.emplace_back(-1);
            }
            else if (it1->refinement_level() == 0 && (it1->is_correction() && !it2->is_leaf()))
            {
                keys_to_del.emplace_back(it1->key());
                level_change.emplace_back(-1);
            }
        }
    }

    template<class tree_t, class key_t>
    static void get_level_changes(tree_t& ref_tree, tree_t& old_tree, int ref_level, std::vector<key_t>& octs,
        std::vector<int>& level_change)
    {
        boost::mpi::communicator world;
        // adapt old_tree to ref_tree
        // need to add octs not in old_tree and delete blocks not in ref_tree from certain level
        octs.clear();
        level_change.clear();
        if (world.rank() != 0) return;
        std::cout << "Getting level changes..." << std::endl;
        std::set<std::pair<typename key_t::value_type, int>> seen_changes;
        auto add_change = [&](const key_t& k, int lc) {
            auto sig = std::make_pair(k.id(), lc);
            if (seen_changes.insert(sig).second)
            {
                octs.emplace_back(k);
                level_change.emplace_back(lc);
            }
        };
        int old_level = ref_level + old_tree->base_level();
        int new_level = ref_level + ref_tree->base_level();
        // first gets octs in old_tree which of not in ref_tree
        for (auto it1 = old_tree->begin(old_level); it1 != old_tree->end(old_level); ++it1)
        {
            if (!it1->has_data()) continue;
            if (!it1->is_leaf()) continue;
            auto it2 = ref_tree->find_octant(it1->key());
            if (!it2 || !it2->has_data() || !it2->physical() || (it2->is_correction()))
            {
                add_change(it1->key(), -1);
            }
        }

        // second gets octs in ref_tree which are not in old_tree
        for (auto it1 = ref_tree->begin(new_level); it1 != ref_tree->end(new_level); ++it1)
        {
            if (!it1->has_data()) continue;
            if (!it1->is_leaf()) continue;
            auto it2 = old_tree->find_octant(it1->key());
            if (!it2 || !it2->has_data() || !it2->physical() || (it2->is_correction()))
            {
                auto ptag = it1->key().parent();
                auto tt = old_tree->find_octant(ptag);
                int  climb_guard = static_cast<int>(ptag.level()) + 2;
                while ((!tt || !tt->has_data() || !tt->is_leaf()) && climb_guard-- > 0)
                {
                    const auto parent = ptag.parent();
                    if (parent == ptag || parent.is_end()) break;
                    ptag = parent;
                    tt = old_tree->find_octant(ptag);
                }
                if (!tt || !tt->has_data() || !tt->is_leaf()) continue;

                // Refine the nearest existing ancestor that contains the
                // missing reference leaf. Using its parent under-refines by
                // one level and can miss required reference keys.
                add_change(tt->key(), 1);
            }
        }
    }

    template<class tree_t, class key_t>
    static void symgrid(tree_t& ref_tree, int ref_level, std::vector<key_t>& octs, std::vector<int>& level_change,
        int mirror_span, const bool toPrint = false)
    {
        boost::mpi::communicator world;
        // adapt old_tree to ref_tree
        // need to add octs not in old_tree and delete blocks not in ref_tree from certain level
        octs.clear();
        level_change.clear();
        if (world.rank() != 0) return;
        std::cout << "Getting symmetry changes..." << std::endl;
        int new_level = ref_level + ref_tree->base_level();
        for (auto it1 = ref_tree->begin(new_level); it1 != ref_tree->end(new_level); ++it1)
        {
            if (!it1->has_data()) continue;
            if (!it1->is_leaf() || it1->is_correction()) continue;
            // check if opposite block is also a leaf. if its not add it to be deleted
            auto coord = it1->tree_coordinate();
            auto key = it1->key();
            auto level = it1->key().level();
            // std::cout << "Checking symmetry for block: " << key << std::endl;
            // std::cout << "Coordinate: " << coord << std::endl;
            // std::cout<< "Level: " << level << std::endl;
            // std::array<int, 3> shift = {0, std::pow2(level), 0};
            auto opposite_coord = coord;
            // opposite_coord[0] = coord[0];
            // opposite_coord[1] = std::pow2(level)-coord[1];
            // Mirror around y=0 in tree-index space.
            // mirror_span = (-2 * bd_base_y) / block_extent, provided by caller.
            opposite_coord[1] = mirror_span * (1 << ref_level) - (coord[1] + 1);

            //   std::cout << "Checking symmetry for block: " << key << std::endl;
            //     std::cout << "Coordinate: " << coord << std::endl;
            //     std::cout<< "Level: " << level << std::endl;
            //     std::cout << "Shifted coordinate: " << opposite_coord << std::endl;
            auto it2 = ref_tree->find_octant(key_t(opposite_coord, level));
            if (!it2)
            {
                if (toPrint)
                {
                    std::cout << "No opposite block found for: " << it1->key() << std::endl;
                    std::cout << "shifted coord: " << opposite_coord << std::endl;
                }
                // std::cout << "No opposite block found for: " << it1->key() << std::endl;
                // std::cout<<"shifted coord: " << opposite_coord << std::endl;
                octs.emplace_back(it1->key());
                level_change.emplace_back(-1);
                // std::cout << "Found opposite block: " << it2->key() << std::endl;
                continue;
            }
            if (!it2->has_data() || !it2->is_leaf() || it2->is_correction())
            {
                // std::cout << "Opposite block is not a leaf: " << it2->key() << std::endl;
                if (toPrint)
                {
                    std::cout << "No opposite leaf block found for: " << it1->key() << std::endl;
                    std::cout << "shifted coord: " << opposite_coord << std::endl;
                    std::cout << "key:" << it2->key() << std::endl;
                }
                octs.emplace_back(it1->key());
                level_change.emplace_back(-1);
                continue;
            }
        }
    }

  private:
    std::string                       dir_out_;
    std::string                       dir_in_;
    int                               nStart_;
    int                               nTotal_;
    int                               nSkip_;
    parallel_ostream::ParallelOstream pcout = parallel_ostream::ParallelOstream(1);
    dictionary::Dictionary*           dict_ref_;
    std::string                       tree_file_prefix_;
    std::string                       flow_file_prefix_;
};

} // namespace iblgf
#endif // IBLGF_MERGE_TREES_HPP
