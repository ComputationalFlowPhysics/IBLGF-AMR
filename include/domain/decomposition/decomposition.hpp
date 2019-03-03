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

    void distribute()
    {

        //Send the construction keys back and forth
        if(server())
        {
            domain_->tree()->construct_leaf_maps();
            domain_->tree()->construct_level_maps();

            auto center = (domain_->bounding_box().max() -
                           domain_->bounding_box().min()) / 2.0 +
                           domain_->bounding_box().min();

            int nRef = 0;

            for(int l=0;l<nRef;++l)
            {
                for (auto it  = domain_->begin_leafs();
                        it != domain_->end_leafs(); ++it)
                {
                    auto b=it->data()->descriptor();

                    const auto lower((center )/2-2 ), upper((center )/2+2 - b.extent());
                    b.grow(lower, upper);
                    if(b.is_inside( center * pow(2.0,l))
                       && it->refinement_level()==l
                      )
                    {
                        domain_->refine(it);
                    }
                }
            }
            //domain_->tree()->construct_leaf_maps();

            server()->send_keys();
        }
        else if(client())
            client()->receive_keys();

        //Construct neighborhood and influence list:
        if(server())
        {
            server()->rank_query();
            std::cout<<" Server done rank queries" << std::endl;
            server()->leaf_query();
        }
        else if(client())
        {
            client()->query_octants();
            client()->disconnect();
            sleep(3);

            client()->query_leaves();
            client()->disconnect();
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
