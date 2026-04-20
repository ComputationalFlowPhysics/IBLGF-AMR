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
#include <memory>
#include <set>
#include <vector>
#include <boost/mpi/collectives.hpp>
#include <iblgf/dictionary/dictionary.hpp>
#include <iblgf/IO/parallel_ostream.hpp>

namespace iblgf
{
using namespace domain;
template<class Setup>
class MergeTrees
{
  public:
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
    MergeTrees(Dictionary* _d)
    : dict_ref_(_d)
    {
        auto dict_ref = _d->get_dictionary("simulation_parameters");
        auto dict_out = dict_ref->get_dictionary("output");
        dir_out_ = dict_out->template get_or<std::string>("directory",
            "output"); // where common tree or adapted snapshots are stored
        dir_in_ = dict_out->template get_or<std::string>("field_dir", dir_out_); // where original snapshots are stored
        nStart_ = dict_ref->template get_or<int>("nStart", 0);                   // starting index of snapshots to merge
        nTotal_ = dict_ref->template get_or<int>("nTotal", 1);                   // total number of snapshots to merge
        nSkip_ = dict_ref->template get_or<int>("nskip", 1); // skip factor between snapshots to merge
        tree_file_prefix_ = dict_ref->template get_or<std::string>("tree_file_prefix", "tree_info_");
        flow_file_prefix_ = dict_ref->template get_or<std::string>("flow_file_prefix", "flowTime_");
    }

  public:
    std::unique_ptr<Setup> get_common_tree()
    {
        boost::mpi::communicator world;
        pcout << "\nCollecting tree keys from all domains...\n" << std::endl;
        std::vector<key_t> octs;
        std::vector<int>   level_change;
        // start with the first snapshot as the initial common tree
        std::string tree_file_0 = tree_file_path(dir_in_, nStart_);
        std::string flow_file_0 = flow_file_path(dir_in_, nStart_);
        auto        ref_domain_ = std::make_unique<Setup>(dict_ref_, tree_file_0, flow_file_0);
        auto        ref_tree = ref_domain_->tree();
        octs.clear();
        level_change.clear();
        for (int i = 1; i < nTotal_; i++)
        {
            std::string tree_file_i = tree_file_path(dir_in_, nStart_ + i * nSkip_);
            std::string flow_file_i = flow_file_path(dir_in_, nStart_ + i * nSkip_);
            auto        domain_i = std::make_unique<Setup>(dict_ref_, tree_file_i, flow_file_i);
            auto        tree_i = domain_i->tree();
            getDelOcts(ref_tree, tree_i, octs, level_change);
            ref_domain_->run_adapt_del(octs, level_change);
            pcout << "Merged tree from snapshot " << i << "/" << nTotal_ - 1 << std::endl;
        }
        return ref_domain_;
    }

    void adapt_to_ref()
    {
        boost::mpi::communicator world;
        pcout << "\nAdapting trees to common tree...\n" << std::endl;
        std::string tree_ref_file =
            dict_ref_->get_dictionary("simulation_parameters")->template get<std::string>("tree_ref_file");
        std::string flow_ref_file =
            dict_ref_->get_dictionary("simulation_parameters")->template get<std::string>("flow_ref_file");
        int  ref_levels = dict_ref_->get_dictionary("simulation_parameters")->template get_or<int>("nLevels", 0);
        auto ref_domain = std::make_unique<Setup>(dict_ref_, tree_ref_file, flow_ref_file);
        auto ref_tree = ref_domain->tree();
        std::vector<key_t> octs;
        std::vector<int>   level_change;
        for (int i = 0; i < nTotal_; ++i)
        {
            std::string tree_file_i = tree_file_path(dir_in_, nStart_ + i * nSkip_);
            std::string flow_file_i = flow_file_path(dir_in_, nStart_ + i * nSkip_);
            auto        domain_i = std::make_unique<Setup>(dict_ref_, tree_file_i, flow_file_i);
            auto        tree_i = domain_i->tree();
            for (int level = ref_levels; level >= 0; --level)
            {
                for (int pass = 0; pass < 16;
                    ++pass) // iterate until no more octs to adapt at this level, to handle chained +1 refinements
                {
                    octs.clear();
                    level_change.clear();
                    get_level_changes_distributed(ref_tree, tree_i, level, octs, level_change, world);
                    if (octs.empty()) break;
                    run_adapt_from_keys_distributed(*domain_i, -1, octs, level_change, world);
                }
            }
            pcout << "interpolated snapshot " << i << "/" << nTotal_ - 1 << " to common tree reference levels."
                  << std::endl;
        }
    }
    std::unique_ptr<Setup> adapt_to_ref(int idx)
    {
        boost::mpi::communicator world;
        pcout << "\nAdapting trees to common tree...\n" << std::endl;
        std::string tree_ref_file =
            dict_ref_->get_dictionary("simulation_parameters")->template get<std::string>("tree_ref_file");
        std::string flow_ref_file =
            dict_ref_->get_dictionary("simulation_parameters")->template get<std::string>("flow_ref_file");
        int  ref_levels = dict_ref_->get_dictionary("simulation_parameters")->template get_or<int>("nLevels", 0);
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
            for (int pass = 0; pass < 16;
                ++pass) // iterate until no more octs to adapt at this level, to handle chained +1 refinements
            {
                octs.clear();
                level_change.clear();
                get_level_changes_distributed(ref_tree, tree_i, level, octs, level_change, world);
                if (octs.empty()) break;
                run_adapt_from_keys_distributed(*domain_i, -1, octs, level_change, world);
            }
        }
        pcout << "interpolated snapshot " << idx << "/" << nTotal_ - 1 << " to common tree reference levels."
              << std::endl;
        return domain_i;
    }

    std::unique_ptr<Setup> ref_to_symmetric_ref()
    {
        boost::mpi::communicator world;
        auto                     sim_dict = dict_ref_->get_dictionary("simulation_parameters");
        std::string              tree_ref_file = sim_dict->template get<std::string>("tree_ref_file");
        std::string              flow_ref_file = sim_dict->template get<std::string>("flow_ref_file");
        auto       domain_dict = dict_ref_->get_dictionary("simulation_parameters")->get_dictionary("domain");
        const auto bd_base = domain_dict->template get<int, Dim>("bd_base");
        const auto block_extent = domain_dict->template get<int>("block_extent");
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
                run_adapt_from_keys_distributed(*ref_domain, i == 0 ? 1 : -1, octs, level_change, world);
            }
        }
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
    static types::float_type max_linear_u_error(SetupType& domain_setup)
    {
        types::float_type local_max = 0.0;
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
                octs.emplace_back(it1->key());
                level_change.emplace_back(-1);
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
                while (!tt)
                {
                    ptag = ptag.parent();
                    tt = old_tree->find_octant(ptag);
                }

                // Refine the nearest existing ancestor that contains the
                // missing reference leaf. Using its parent under-refines by
                // one level and can miss required reference keys.
                octs.emplace_back(tt->key());
                level_change.emplace_back(1);
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
    Dictionary*                       dict_ref_;
    std::string                       tree_file_prefix_;
    std::string                       flow_file_prefix_;
};

} // namespace iblgf
#endif // IBLGF_MERGE_TREES_HPP
