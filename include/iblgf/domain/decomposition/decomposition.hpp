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

#ifndef DOMAIN_INCLUDED_DECOMPOSITION_HPP
#define DOMAIN_INCLUDED_DECOMPOSITION_HPP

#include <vector>
#include <stdexcept>
#include <unordered_map>
#include <boost/mpi/communicator.hpp>
#include <boost/serialization/set.hpp>

#include <iblgf/global.hpp>
#include <iblgf/domain/octree/tree.hpp>
#include <iblgf/dictionary/dictionary.hpp>
#include <iblgf/domain/decomposition/client.hpp>
#include <iblgf/domain/decomposition/server.hpp>
#include <iblgf/fmm/fmm.hpp>

namespace domain
{
/** @brief Domain decomposition.
 *  Splits octree according to a server/client model
 */
template<class Domain>
class Decomposition
{
  public:
    using domain_type = Domain;
    using key_t = typename domain_type::key_t;
    using datablock_t = typename domain_type::datablock_t;
    using client_type = Client<domain_type>;
    using server_type = Server<domain_type>;
    using communicator_type = typename domain_type::communicator_type;
    using octant_t = typename domain_type::octant_t;
    using MASK_LIST = typename octant_t::MASK_LIST;
    using fmm_mask_builder_t = typename fmm::FmmMaskBuilder<domain_type>;

  public:
    //server rank within communicator comm_
    static constexpr int server_rank = 0;

  public:
    Decomposition(const Decomposition& other) = default;
    Decomposition(Decomposition&& other) = default;
    Decomposition& operator=(const Decomposition& other) & = default;
    Decomposition& operator=(Decomposition&& other) & = default;
    ~Decomposition() = default;
    Decomposition() = default;

    Decomposition(domain_type* _d)
    : domain_(_d)
    , comm_(communicator_type())
    , baseBlockBufferNumber_(_d->baseBlockBufferNumber())
    {
        if (comm_.size() < 2)
        { throw std::runtime_error("Minimum world size is 2."); }
        if (comm_.rank() == server_rank)
            server_ = std::make_shared<server_type>(domain_, comm_);
        else
            client_ = std::make_shared<client_type>(domain_, comm_);
    }

  public: //memeber functions
    const bool& subtract_non_leaf() const noexcept
    {
        return subtract_non_leaf_;
    }
    bool& subtract_non_leaf() noexcept { return subtract_non_leaf_; }

    void sync_decomposition()
    {
        if (server())
        {
            server()->rank_query();
            server()->flag_query();
            server()->mask_query();
        }
        else if (client())
        {
            client()->query_octants();
            client()->disconnect();

            client()->query_flags();
            client()->disconnect();

            client()->query_masks();
            client()->disconnect();

            client()->halo_reset();
        }
    }

    template<class LoadCalculator, class FmmMaskBuilder>
    void distribute()
    {
        if (server())
        {
            std::cout << "Initialization of masks start" << std::endl;
            FmmMaskBuilder::fmm_lgf_mask_build(domain_, subtract_non_leaf_);
            std::cout << "Initialization of masks done" << std::endl;
            FmmMaskBuilder::fmm_vortex_streamfun_mask(domain_);
            //FmmMaskBuilder::fmm_if_load_build(domain_);
            // it's together with fmmMaskBuild for now
            //LoadCalculator::calculate();
        }

        //Send the construction keys back and forth
        if (server()) { server()->send_keys(); }
        else if (client())
        {
            client()->receive_keys();
        }

        //Construct neighborhood and influence list:
        sync_decomposition();
    }

    template<class... Field>
    void balance()
    {
        if (server()) { server()->update_decomposition(); }
        else if (client())
        {
            auto update = client()->update_decomposition();
            (client()->template update_field<Field>(update), ...);
            client()->finish_decomposition_update(update);
            client()->halo_reset();
        }
        sync_decomposition();
    }

    template<class Field>
    auto adapt_decoposition(float_type source_max)
    {
        std::vector<octant_t*> interpolation_list;

        if (server())
        {
            std::vector<key_t> octs_all;
            std::vector<int>   level_change_all;

            std::vector<octant_t*>          refinement_server;
            std::vector<std::vector<key_t>> refinement(comm_.size());
            std::vector<std::vector<key_t>> deletion(comm_.size());

            auto base_level = domain_->tree()->base_level();

            // --------------------------------------------------------------
            // mark correction to be deleted

            for (auto it = domain_->begin(); it != domain_->end(); ++it)
            {
                if (!it->data()) continue;

                if (it->is_correction() && it->refinement_level() > 0)
                    it->aim_deletion(true);
                else
                    it->aim_deletion(false);
            }

            // --------------------------------------------------------------
            // 0. receive attempts
            server()->recv_adapt_attempts(octs_all, level_change_all);
            for (auto it = domain_->begin(); it != domain_->end(); ++it)
            {
                if (!it->data()) continue;
                it->aim_level_change() = 0;
            }

            for (std::size_t c = 0; c < octs_all.size(); ++c)
            {
                if (level_change_all[c] != 0)
                {
                    auto key = octs_all[c];
                    auto it = domain_->tree()->find_octant(key);
                    if (!it || !it->data())
                        throw std::runtime_error("can't find oct on server");
                    it->aim_level_change() = level_change_all[c];
                }
            }

            std::unordered_map<key_t, bool> checklist;
            for (auto it = domain_->begin_leafs(); it != domain_->end_leafs();
                 ++it)
            {
                int l_change = it->aim_level_change();

                if (l_change != 0)
                {
                    if (l_change < 0) { it->aim_deletion(true); }
                    else
                    {
                        if (!domain_->tree()->try_2to1(it->key(),
                                domain_->key_bounding_box(), checklist))
                            continue;
                        refinement_server.emplace_back(it.ptr());
                    }
                }
            }

            // Unmark deletion distance_N blocks from the solution domain
            std::unordered_set<key_t> listNBlockAway;

            auto f = [&](octant_t* _o) {
                auto level = _o->refinement_level();
                level = level >= 0 ? level : 0;
                auto bbase = domain_->tree()->octant_to_level_coordinate(
                    _o->tree_coordinate(), level);
                _o->data() = std::make_shared<datablock_t>(
                    bbase, domain_->block_extent(), level, false);
            };

            for (auto it = domain_->begin(base_level);
                 it != domain_->end(base_level); ++it)
            {
                if (it->aim_deletion() || it->is_correction()) continue;

                auto k = it->key();
                auto nks = k.get_neighbor_keys(baseBlockBufferNumber_);
                for (auto nk : nks)
                {
                    if (nk.is_end()) continue;
                    listNBlockAway.emplace(nk);
                }
            }

            for (auto it = listNBlockAway.begin(); it != listNBlockAway.end();
                 ++it)
            {
                auto nk = *it;
                auto oct = domain_->tree()->find_octant(nk);
                if (!domain_->key_bounding_box(false).is_inside(
                        nk.coordinate()))
                    continue;
                if (!oct || !oct->data())
                {
                    oct = domain_->tree()->insert_td(nk);
                    f(oct);
                    oct->flag_leaf(true);
                    oct->flag_correction(false);
                    oct->leaf_boundary() = false;
                    oct->aim_deletion(false);
                }

                oct->aim_deletion(false);
            }

            domain_->tree()->construct_leaf_maps(true);
            domain_->tree()->construct_level_maps();

            // refine those allow 2to1 ratio
            for (auto& oct : refinement_server)
            {
                if (!oct->is_leaf()) continue;

                //domain_->refine_with_exisitng_correction(oct, deletion);
                domain_->refine(oct);
            }

            // dynmaic Programming to rduce repeated checks
            std::unordered_set<octant_t*> checklist_reset_2to1_aim_deletion;
            for (auto it = domain_->begin_leafs(); it != domain_->end_leafs();
                 ++it)
            {
                if (it->refinement_level() > 0)
                    domain_->tree()->deletionReset_2to1(
                        it->parent(), checklist_reset_2to1_aim_deletion);
            }

            // --------------------------------------------------------------
            // 3. try delete

            domain_->tree()->construct_leaf_maps(true);
            domain_->tree()->construct_level_maps();

            // Base level
            for (auto it = domain_->begin(base_level);
                 it != domain_->end(base_level); ++it)
            {
                if (it->aim_deletion() && it->is_leaf())
                {
                    deletion[it->rank()].emplace_back(it->key());
                    domain_->tree()->delete_oct(it.ptr());
                }
            }

            domain_->tree()->construct_level_maps();
            domain_->delete_all_children(deletion);
            domain_->tree()->construct_level_maps();
            domain_->mark_correction();
            domain_->mark_leaf_boundary();
            //domain_->delete_all_children(deletion,false);

            for (auto it = domain_->begin(); it != domain_->end(); ++it)
            {
                if (it->aim_deletion() && it->refinement_level() > 0)
                {
                    deletion[it->rank()].emplace_back(it->key());
                    domain_->tree()->delete_oct(it.ptr());
                }
            }

            for (int l = domain_->tree()->base_level() - 1; l > 0; --l)
            {
                for (auto it = domain_->begin(l); it != domain_->end(l); ++it)
                {
                    if (it->is_leaf_search())
                    {
                        if (it->data())
                            deletion[it->rank()].emplace_back(it->key());
                        domain_->tree()->delete_oct(it.ptr());
                    }
                }
            }

            // --------------------------------------------------------------
            // 4. set rank of new octants

            for (auto it = domain_->begin(); it != domain_->end(); ++it)
            {
                if (it->rank() == -1 && it->data())
                {
                    auto pa = it->parent();
                    while (pa->rank() <= 0 && pa != pa->parent())
                    { pa = pa->parent(); }

                    if (pa->rank() > 0)
                    {
                        it->rank() = pa->rank();
                        refinement[it->rank()].emplace_back(it->key());
                    }
                    else
                    {
                        auto k = it->key();
                        auto nks = k.get_neighbor_keys();
                        for (auto nk : nks)
                        {
                            auto oct = domain_->tree()->find_octant(nk);
                            if (oct && oct->data() && oct->rank() > 0)
                            {
                                it->rank() = oct->rank();
                                refinement[it->rank()].emplace_back(it->key());
                                break;
                            }
                        }
                    }

                    pa = it->parent();
                    while (!pa->data())
                    {
                        f(pa);
                        pa->rank() = it->rank();
                        refinement[pa->rank()].emplace_back(pa->key());
                        pa = pa->parent();
                    }
                }
            }

            // --------------------------------------------------------------
            // 5. update depth

            int depth = 0;
            for (auto it = domain_->begin(); it != domain_->end(); ++it)
            {
                if (it->data() && it->level() + 1 > depth)
                    depth = it->level() + 1;
            }
            domain_->tree()->depth() = depth;

            // --------------------------------------------------------------
            // 6. send back new locally owned octs

            for (int i = 1; i < comm_.size(); ++i)
            {
                comm_.send(i, i + 0 * comm_.size(), refinement[i]);
                comm_.send(i, i + 1 * comm_.size(), deletion[i]);
            }

            for (int i = 1; i < comm_.size(); ++i)
                comm_.send(i, 0, domain_->tree()->depth());

            // --------------------------------------------------------------
            // 7. construct maps / masks / loads

            domain_->tree()->construct_leaf_maps(true);
            domain_->tree()->construct_level_maps();
            domain_->tree()->construct_lists();

            fmm_mask_builder_t::fmm_clean_load(domain_);
            fmm_mask_builder_t::fmm_lgf_mask_build(domain_, subtract_non_leaf_);
            fmm_mask_builder_t::fmm_vortex_streamfun_mask(domain_);

            // --------------------------------------------------------------
            // 8. sync ghosts
            sync_decomposition();
        }
        else if (client())
        {
            std::vector<key_t> refinement_local;
            std::vector<key_t> deletion_local;

            client()->template send_adapt_attempts<Field>(
                domain_->register_adapt_condition(), source_max);

            comm_.recv(0, comm_.rank() + 0 * comm_.size(), refinement_local);
            comm_.recv(0, comm_.rank() + 1 * comm_.size(), deletion_local);

            int depth;
            comm_.recv(0, 0, depth);
            domain_->tree()->depth() = depth;

            // Local deletion
            std::sort(deletion_local.begin(), deletion_local.end(),
                [](key_t k1, key_t k2) -> bool {
                    return k1.level() > k2.level();
                });
            boost::mpi::communicator world;

            for (auto k : refinement_local)
            {
                auto oct = domain_->tree()->find_octant(k);
                if (oct && oct->data() && oct->locally_owned())
                    std::cout << " wrong : oct already exists\n " << oct->key();
            }

            for (auto& key : deletion_local)
            {
                //find the octant
                auto it = domain_->tree()->find_octant(key);
                if (!it)
                    throw std::runtime_error(
                        "can't find the octant to be deleted");

                domain_->tree()->delete_oct(it);
            }

            // Local refinement
            domain_->tree()->insert_keys(
                refinement_local,
                [&](octant_t* _o) {
                    auto level = _o->refinement_level();
                    level = level >= 0 ? level : 0;
                    auto bbase = domain_->tree()->octant_to_level_coordinate(
                        _o->tree_coordinate(), level);
                    if (!_o->data() || !_o->data()->is_allocated())
                    {
                        _o->data() = std::make_shared<datablock_t>(
                            bbase, domain_->block_extent(), level, true);
                    }
                    _o->rank() = comm_.rank();
                },
                false);

            // ghost ranks are sync
            auto old_leaf_map = domain_->tree()->leaf_map();
            sync_decomposition();

            for (auto it = domain_->begin_leafs(); it != domain_->end_leafs();
                 ++it)
            {
                if (!it->locally_owned()) continue;
                if (it->refinement_level() > 0 &&
                    old_leaf_map.find(it->key()) == old_leaf_map.end())
                { interpolation_list.emplace_back(it->parent()); }
            }
        }

        return interpolation_list;
    }

  public: //access memebers:
    auto        client() { return client_; }
    auto        server() { return server_; }
    inline bool is_server() const noexcept
    {
        if (server_) return true;
        return false;
    }
    inline bool is_client() const noexcept
    {
        if (client_) return true;
        return false;
    }

    const auto& domain() const { return domain_; }
    auto&       domain() { return domain_; }

  private:
    domain_type*                 domain_;
    boost::mpi::communicator     comm_;
    std::shared_ptr<client_type> client_ = nullptr;
    std::shared_ptr<server_type> server_ = nullptr;
    int                          baseBlockBufferNumber_;
    bool                         subtract_non_leaf_ = false;
};

} // namespace domain

#endif
