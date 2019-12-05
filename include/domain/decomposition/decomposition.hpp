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

    template<class Field,class Field2,class LoadCalculator, class FmmMaskBuilder>
    void balance()
    {
        std::cout<<"Balancing "<<std::endl;
        if(server())
        {
            server()->update_decomposition();

            domain_->tree()->construct_leaf_maps();
            domain_->tree()->construct_level_maps();
            domain_->tree()->construct_neighbor_lists();
            //this->tree()->construct_influence_lists();

            server()->rank_query();
            server()->leaf_query();
            server()->correction_query();
            server()->mask_query();

            //std::ofstream ofs("master.txt");
            //for(auto it  = domain_->begin(); it != domain_->end(); ++it)
            //{
            //    {
            //        ofs<<it->key().id()<<" "<<it->rank()<<std::endl;
            //    }

            //}

        }
        else if(client())
        {
            auto update=client()->update_decomposition();
            client()->template update_field<Field>(update);
            //client()->template update_field<Field2>(update);
            client()->query_octants();
            client()->disconnect();

            client()->query_leafs();
            client()->disconnect();

            client()->query_corrections();
            client()->disconnect();

            client()->query_masks();
            client()->disconnect();

            //std::ofstream ofs("master_"+std::to_string(comm_.rank())+".txt");
            //for(auto it  = domain_->begin(); it != domain_->end(); ++it)
            //{
            //    {
            //        ofs<<it->key().id()<<" "<<it->rank()<<std::endl;
            //    }

            //}

        }
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
