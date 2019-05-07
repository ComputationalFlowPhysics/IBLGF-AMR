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
    using client_type = Client<domain_type>;
    using server_type = Server<domain_type>;
    using communicator_type  = typename  domain_type::communicator_type;
    using octant_t = typename domain_type::octant_t;
    using MASK_LIST = typename octant_t::MASK_LIST;

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

    template<class LoadCalculator, class FmmMaskBuilder>
    void distribute()
    {
        if(server())
        {
            FmmMaskBuilder::build(domain_);
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
        if(server())
        {
            server()->rank_query();
            server()->leaf_query();
            server()->mask_query();
        }
        else if(client())
        {
            client()->query_octants();
            client()->disconnect();

            client()->query_leafs();
            client()->disconnect();

            client()->query_masks();
            client()->disconnect();
        }
    }

private:

    //void poisson_dry()
    //{
    //    for (int l  = domain_->tree()->base_level()+0;
    //            l < domain_->tree()->depth(); ++l)
    //    {
    //        FMM::fmm_dry(l, false);
    //        fmm_dry(l, true);
    //    }
    //}

    //void fmm_dry(int base_level, bool non_leaf_as_source)
    //{
    //    fmm_dry_init_base_level_masks(base_level, non_leaf_as_source);
    //    fmm_upward_pass_masks(base_level,
    //            MASK_LIST::Mask_FMM_Source,
    //            non_leaf_as_source);

    //    fmm_upward_pass_masks(base_level,
    //            MASK_LIST::Mask_FMM_Target,
    //            non_leaf_as_source);
    //}

    //void fmm_upward_pass_masks(int base_level,
    //        int mask_id,
    //        bool non_leaf_as_source)
    //{

    //    int refinement_level = base_level-domain_->tree()->base_level();
    //    int fmm_mask_idx_ = refinement_level*2+non_leaf_as_source;

    //    // for all levels
    //    for (int level=base_level-1; level>=0; --level)
    //    {
    //        // parent's mask is true if any of its child's mask is true
    //        for (auto it = domain_->begin(level);
    //                it != domain_->end(level);
    //                ++it)
    //        {
    //            // including ghost parents
    //            it->fmm_mask(fmm_mask_idx_,mask_id,false);
    //            for ( int c = 0; c < it->num_children(); ++c )
    //            {
    //                if ( it->child(c) && it->child(c)->fmm_mask(fmm_mask_idx_,mask_id) )
    //                {
    //                    it->fmm_mask(fmm_mask_idx_,mask_id, true);
    //                    it->add_load(it->influence_number());
    //                    break;
    //                }
    //            }
    //        }

    //    }
    //}

    //void fmm_dry_init_base_level_masks(int base_level, bool non_leaf_as_source)
    //{
    //    int refinement_level = base_level-domain_->tree()->base_level();
    //    int fmm_mask_idx_ = refinement_level*2+non_leaf_as_source;

    //    if (non_leaf_as_source)
    //    {
    //        for (auto it = domain_->begin(base_level);
    //                it != domain_->end(base_level); ++it)
    //        {
    //            if ( it->is_leaf() )
    //            {
    //                it->fmm_mask(fmm_mask_idx_,MASK_LIST::Mask_FMM_Source, false);
    //                it->fmm_mask(fmm_mask_idx_,MASK_LIST::Mask_FMM_Target, false);
    //            } else
    //            {
    //                it->fmm_mask(fmm_mask_idx_,MASK_LIST::Mask_FMM_Source, true);
    //                it->fmm_mask(fmm_mask_idx_,MASK_LIST::Mask_FMM_Target, true);

    //                it->add_load(it->influence_number());
    //                it->add_load(it->neighbor_number());
    //            }
    //        }
    //    } else
    //    {
    //        for (auto it = domain_->begin(base_level);
    //                it != domain_->end(base_level); ++it)
    //        {
    //            it->fmm_mask(fmm_mask_idx_,MASK_LIST::Mask_FMM_Source, true);
    //            it->fmm_mask(fmm_mask_idx_,MASK_LIST::Mask_FMM_Target, true);
    //            it->add_load(it->influence_number());
    //            it->add_load(it->neighbor_number());
    //        }
    //    }
    //}


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
