#ifndef DOMAIN_INCLUDED_DECOMPOSITION_HPP
#define DOMAIN_INCLUDED_DECOMPOSITION_HPP

#include <vector>
#include <stdexcept>

#include <global.hpp>
#include <domain/octree/tree.hpp>
#include <dictionary/dictionary.hpp>

#include <boost/mpi/communicator.hpp>
#include <domain/decomposition/client.hpp>
#include <domain/decomposition/server.hpp>

#include <fmm/fmm.hpp>

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
    using key_t  = typename  domain_type::key_t;
    using datablock_t = typename domain_type::datablock_t;
    using client_type = Client<domain_type>;
    using server_type = Server<domain_type>;
    using communicator_type  = typename  domain_type::communicator_type;
    using octant_t = typename domain_type::octant_t;
    using MASK_LIST = typename octant_t::MASK_LIST;
    using fmm_mask_builder_t = typename fmm::FmmMaskBuilder<domain_type>;

public:

    //server rank within communicator comm_
    static constexpr int server_rank = 0;

public:

    Decomposition(const Decomposition&  other) = default;
    Decomposition(Decomposition&& other) = default;
    Decomposition& operator=(const Decomposition&  other) & = default;
    Decomposition& operator=(Decomposition&& other) & = default;
    ~Decomposition() = default;
    Decomposition()=default;


    Decomposition( domain_type* _d )
    :domain_(_d), comm_(communicator_type())
    {
        if(comm_.size() <2)
        {
            throw std::runtime_error("Minimum world size is 2.");
        }
        if(comm_.rank()==server_rank)
            server_=std::make_shared<server_type>(domain_, comm_);
        else
            client_=std::make_shared<client_type>(domain_, comm_);
    }


public: //memeber functions

    void sync_decomposition()
    {
        if(server())
        {
            server()->rank_query();
            server()->leaf_query();
            server()->correction_query();
            server()->mask_query();
        }
        else if(client())
        {
            client()->query_octants();
            client()->disconnect();

            client()->query_leafs();
            client()->disconnect();

            client()->query_corrections();
            client()->disconnect();

            client()->query_masks();
            client()->disconnect();

        }
    }


    template<class LoadCalculator, class FmmMaskBuilder>
    void distribute()
    {
        if(server())
        {
            FmmMaskBuilder::fmm_lgf_mask_build(domain_);
            FmmMaskBuilder::fmm_vortex_streamfun_mask(domain_);
            //FmmMaskBuilder::fmm_if_load_build(domain_);
            // it's together with fmmMaskBuild for now
            //LoadCalculator::calculate();
        }

        //Send the construction keys back and forth
        if(server())
        {
            server()->send_keys();
        }
        else if(client())
        {
            client()->receive_keys();
        }

        //Construct neighborhood and influence list:
        sync_decomposition();
    }

    template<class... Field>
    void balance()
    {
        if(server())
        {
            server()->update_decomposition();
        }
        else if(client())
        {
            auto update=client()->update_decomposition();
            (client()->template update_field<Field>(update), ...);
            client()->finish_decomposition_update(update);
        }
        sync_decomposition();
    }

    template<class Field>
    std::vector<octant_t*> adapt_decoposition()
    {
        std::vector<octant_t*> interpolation_list{};

        if (server())
        {
            std::vector<key_t> octs_all;
            std::vector<int>   level_change_all;

            std::vector<octant_t*> refinement_server;
            std::vector<std::vector<key_t>> refinement(comm_.size());
            std::vector<std::vector<key_t>> deletion(comm_.size());

            // 0. receive level_chagne
            server()->recv_adapt_attempts(octs_all, level_change_all);

            // 1. mark all delete attempts
            for (auto it = domain_->begin(); it != domain_->end(); ++it)
            {
                if (it->is_correction())
                    it->aim_deletion(true);
                else
                    it->aim_deletion(false);
            }

            for (int c=0; c<octs_all.size(); ++c)
            {
                if (level_change_all[c]<0)
                {
                    auto key = octs_all[c];
                    auto it = domain_->tree()->find_octant(key);
                    if (!it)
                        throw std::runtime_error("can't find oct on server");
                    it->aim_deletion(true);
                }
            }

            // 2. refine attempts
            //      let children's rank = parent's rank
            //      2to1 ratio remove some delete marks

            // try refine

            // Dynmaic Programming to rduce repeated checks
            std::map<key_t, bool> checklist;

            for (int c=0; c<octs_all.size(); ++c)
            {
                if (level_change_all[c]>0)
                {
                    auto key = octs_all[c];

                    if (!domain_->tree()->try_2to1(key, checklist)) continue;

                    auto oct = domain_->tree()->find_octant(key);
                    if (oct->rank()==0)
                        throw std::runtime_error("shouldn't try to refine rank 0 oct");

                    refinement_server.emplace_back(oct);
                }
            }

            // refine those allow 2to1 ratio
            for (auto oct:refinement_server)
            {
                auto key = oct->key();
                std::cout<<oct->key()<<std::endl;


                //check if there is a correction child
                for (int i = 0; i < oct->num_children(); ++i)
                {
                    auto child = oct->child(i);
                    if (child && child->is_correction() && child->rank()!=oct->rank())
                    {
                        deletion[child->rank()].emplace_back(child->key());
                        oct->delete_child(i);
                    }
                }

                domain_->refine(oct);

                for (int i = 0; i < oct->num_children(); ++i)
                {
                    auto child = oct->child(i);
                    child->rank() = oct->rank();
                }

                // this only stores the parent of all new leafs
                refinement[oct->rank()].emplace_back(key);
            }
            std::cout<< " start removing octants " << std::endl;

            // unmark all new corrections
            domain_->mark_correction();

            // 4. check all that can be removed
            // remove leafs
            for (auto it = domain_->begin(); it != domain_->end(); ++it)
            {
                if (it->is_correction()) continue;
                bool delete_all_children=true;

                for (int i = 0; i < it->num_children(); ++i)
                {
                    auto child = it->child(i);
                    if (!child )
                        continue;

                    if (!child->aim_deletion() || !child->is_leaf())
                    {
                        delete_all_children=false;
                        break;
                    }
                }

                if (delete_all_children)
                {
                    for (int i = 0; i < it->num_children(); ++i)
                    {
                        auto child = it->child(i);
                        if (!child )
                            continue;
                        std::cout<<child->rank()<<std::endl;
                        deletion[child->rank()].emplace_back(child->key());

                        //child->deallocate_data();
                        it->delete_child(i);
                    }
                    it->flag_leaf(true);
                }
            }

            // remove remove corrections
            for (auto it = domain_->begin(); it != domain_->end(); ++it)
            {
                if (it->is_correction() && it->aim_deletion())
                {
                    deletion[it->rank()].emplace_back(it->key());
                }
            }

            // update the tree structure
            domain_->tree()->construct_leaf_maps(true);
            domain_->tree()->construct_level_maps();
            domain_->tree()->construct_influence_lists();
            domain_->tree()->construct_neighbor_lists();

            fmm_mask_builder_t::fmm_lgf_mask_build(domain_);
            fmm_mask_builder_t::fmm_vortex_streamfun_mask(domain_);

            // 5. send back refine/delete keys
            std::cout<< " Sending refinement/deletion lists back" << std::endl;
            for(int i=1;i<comm_.size();++i)
            {
                comm_.send(i,i+0*comm_.size(), refinement[i] );
                comm_.send(i,i+1*comm_.size(), deletion[i] );
            }

            //Send global tree depth
            for(int i=1;i<comm_.size();++i)
                comm_.send(i,0, domain_->tree()->depth()) ;

        }
        else if (client())
        {
            std::vector<key_t> refinement_local;
            std::vector<key_t> deletion_local;

            client()->template send_adapt_attempts<Field>(domain_->register_refinement_condition());

            comm_.recv(0,comm_.rank()+0*comm_.size(),refinement_local);
            comm_.recv(0,comm_.rank()+1*comm_.size(),deletion_local);
            int depth;
            comm_.recv(0,0,depth);
            domain_->tree()->depth()=depth;

            for (auto k:deletion_local)
                std::cout<<k<<std::endl;

            // Local refinement
            for (auto k:refinement_local)
            {
                auto oct =domain_->tree()->find_octant(k);
                domain_->refine(oct, false);
                interpolation_list.emplace_back(oct);
            }

            // Local deletion
            for(auto& key : deletion_local)
            {
                //find the octant
                auto it =domain_->tree()->find_octant(key);
                if(!it)
                    throw std::runtime_error("can't find the octant to be deleted");

                if(it->is_leaf_search())
                {
                    //delete
                    int cnumber=it->key().child_number();
                    it->deallocate_data();
                    auto p=it->parent();
                    if(p) p->delete_child(cnumber);
                }
                else
                {
                    it->deallocate_data();
                }
            }

        }

        std::cout<<" Sync from server ----------------------- start" << std::endl;
        sync_decomposition();
        std::cout<<" Sync from server ----------------------- end "<< std::endl;

        return interpolation_list;

    }


public: //access memebers:

    auto client(){ return client_; }
    auto server(){ return server_; }
    inline bool is_server() const noexcept
    {
        if(server_)return true;
        return false;
    }
    inline bool is_client() const noexcept
    {
        if(client_)return true;
        return false;
    }

    const auto& domain() const{ return domain_; }
    auto& domain() { return domain_; }

private:
    domain_type* domain_;
    boost::mpi::communicator comm_;
    std::shared_ptr<client_type> client_=nullptr;
    std::shared_ptr<server_type> server_=nullptr;
};

}

#endif
