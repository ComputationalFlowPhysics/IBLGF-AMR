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

#ifndef IBLGF_INCLUDED_COMMON_TREE_HPP
#define IBLGF_INCLUDED_COMMON_TREE_HPP

#include <iostream>
#include <iblgf/dictionary/dictionary.hpp>

#include "../../setups/setup_base.hpp"
#include <iblgf/operators/operators.hpp>
#include <iblgf/solver/time_integration/ifherk.hpp>
namespace iblgf
{
using namespace domain;
using namespace types;
using namespace dictionary;

const int Dim = 2;

struct parameters
{
    static constexpr std::size_t Dim = 2;
    // clang-format off
    REGISTER_FIELDS
    (
    Dim,
        (
            //name, type, nFields, l/h-buf,mesh_obj, output(optional)
            (tlevel,       float_type, 1, 1, 1, cell, true),
            (u,float_type, 2, 1, 1, face, true),
            (p,float_type, 1, 1, 1, cell, true),
            (test             , float_type, 1,    1,       1,     cell,true )
        )
    )
    // clang-format on
};

struct CommonTree : public SetupBase<CommonTree, parameters>
{
    using super_type = SetupBase<CommonTree, parameters>;

    //Timings
    using clock_type = std::chrono::high_resolution_clock;
    using milliseconds = std::chrono::milliseconds;
    using duration_type = typename clock_type::duration;
    using time_point_type = typename clock_type::time_point;
    using key_id_t= typename domain_t::tree_t::key_type::value_type;
    CommonTree(Dictionary* _d)
    : super_type(_d)
    {
        if (domain_->is_client()) client_comm_ = client_comm_.split(1);
        else
            client_comm_ = client_comm_.split(0);
        nLevelRefinement_ = simulation_.dictionary_->template get_or<int>("nLevels", 0);
        std::cout << "Number of refinement levels: " << nLevelRefinement_ << std::endl;
        // std::cout << "Restarting list construction..." << std::endl;
        domain_->register_adapt_condition()=
            [this]( std::vector<float_type> source_max, auto& octs, std::vector<int>& level_change )
                {return this->template adapt_level_change(source_max, octs, level_change);};

        domain_->register_refinement_condition() = [this](auto octant,
                                                    int diff_level) {
            return false;
        };
        domain_->init_refine(nLevelRefinement_, 0, 0);

        
        domain_->distribute<fmm_mask_builder_t, fmm_mask_builder_t>();
        this->initialize();
        
    }
    CommonTree(Dictionary* _d,
                             std::vector<key_id_t>&_keys,
               std::vector<int>& _leafs)
    : super_type(_d,_keys,_leafs)
    {
        if (domain_->is_client()) client_comm_ = client_comm_.split(1);
        else
            client_comm_ = client_comm_.split(0);
        nLevelRefinement_ = simulation_.dictionary_->template get_or<int>("nLevels", 0);
        // std::cout << "Number of refinement levels: " << nLevelRefinement_ << std::endl;
        // std::cout << "Restarting list construction..." << std::endl;
        domain_->register_adapt_condition()=
            [this]( std::vector<float_type> source_max, auto& octs, std::vector<int>& level_change )
                {return this->template adapt_level_change(source_max, octs, level_change);};

        domain_->register_refinement_condition() = [this](auto octant,
                                                    int diff_level) {
            return false;
        };
        // domain_->init_refine(nLevelRefinement_, 0, 0);

        domain_->restart_list_construct();
        domain_->distribute<fmm_mask_builder_t, fmm_mask_builder_t>();
        std::cout << "CommonTree: Distributed domain with " << std::endl;
        this->initialize();
        
    }

    CommonTree(Dictionary* _d,std::string restart_tree_dir,
                             std::string restart_field_dir)
                             :super_type(_d,
            [this](auto _d, auto _domain){
                return this->initialize_domain(_d, _domain); },
                restart_tree_dir),
    restart_tree_dir_(restart_tree_dir),
    restart_field_dir_(restart_field_dir)
    {
        if (domain_->is_client()) client_comm_ = client_comm_.split(1);
        else
            client_comm_ = client_comm_.split(0);
        nLevelRefinement_ = simulation_.dictionary_->template get_or<int>("nLevels", 0);
        // std::cout << "Number of refinement levels: " << nLevelRefinement_ << std::endl;
        // std::cout << "Restarting list construction..." << std::endl;
        domain_->register_adapt_condition()=
            [this]( std::vector<float_type> source_max, auto& octs, std::vector<int>& level_change )
                {return this->template adapt_level_change(source_max, octs, level_change);};

        domain_->register_refinement_condition() = [this](auto octant,
                                                    int diff_level) {
            return false;
        };
        // domain_->init_refine(nLevelRefinement_, 0, 0);

        domain_->restart_list_construct();
        domain_->distribute<fmm_mask_builder_t, fmm_mask_builder_t>();
        simulation_.template read_h5<u_type>(restart_field_dir,"u");

        // simulation_.read(restart_tree_dir,"tree");
        // simulation_.read(restart_field_dir,"fields");
        
    }
    

    void initialize()
    {
        boost::mpi::communicator world;
        if(domain_->is_server()) return;
        auto center = (domain_->bounding_box().max() -
                       domain_->bounding_box().min()+1) / 2.0 +
                       domain_->bounding_box().min();
        
        for(auto it=domain_->begin(); it!=domain_->end(); ++it)
        {
            if(!it->locally_owned()) continue;
            int ref_level_=it->refinement_level();
            for(auto& n: it->data())
            {
                n(tlevel)=ref_level_+0.5;
                n(u,0)=ref_level_+0.5;
                n(u,1)=ref_level_+0.5;
            }
        }


    }
    void run(int i,bool adapt_=true)
    {
        boost::mpi::communicator world;
        time_integration_t ifherk(&this->simulation_);
        if(world.rank()== 0)
        {
            std::cout << "Running common tree test with i = " << i << std::endl;
        }
        if (adapt_)
        {
            ifherk.adapt(true,false);
        }
        
        if(world.rank()== 0)
        {
            std::cout << "after adapt " << i << std::endl;
        }
        this->initialize();
        if(world.rank()== 0)
        {
            std::cout << "after initialize " << i << std::endl;
        }


        std::string filename = "common_tree_" + std::to_string(i) + ".hdf5";
        // simulation_.write("common_tree.hdf5");
        simulation_.write(filename);
    }

    //register adapt condition based on reference keys and run adaptation
    void run_adapt_to_ref(std::vector<key_id_t>& _ref_keys,
                      std::vector<int>& _ref_leafs)
    {
        ref_keys_ = _ref_keys;
        ref_leafs_ = _ref_leafs;

        domain_->register_adapt_condition()=
            [this]( std::vector<float_type> source_max, auto& octs, std::vector<int>& level_change )
                {return this->template adapt_to_ref(source_max, octs, level_change);};
        time_integration_t ifherk(&this->simulation_);
        ifherk.adapt(true,false);
        ifherk.adapt(true,false);

        simulation_.write("adapted_to_ref");

    }
    template< class key_t >
    void adapt_to_ref(std::vector<float_type> source_max,
                            std::vector<key_t>& octs,
                            std::vector<int>&   level_change )
    {
        octs.clear();
        level_change.clear();
        for (auto it = domain_->begin(); it != domain_->end(); ++it)
        {
            if (!it->locally_owned()) continue;
            // if (!it->is_leaf() && !it->is_correction()) continue;
            // if(!it->has_data()) continue;
            //try to find it in reference keys
            auto it_ref = std::find(ref_keys_.begin(), ref_keys_.end(), it->key().id());
            if (it_ref != ref_keys_.end())
            {
                if (it->is_leaf())
                {
                    if(ref_leafs_[it_ref - ref_keys_.begin()]==1)
                    {
                        // it is a leaf
                        // octs.emplace_back(it->key());
                        // level_change.emplace_back(0);
                        continue;
                    }
                    else
                    {
                        // it is a correction
                        octs.emplace_back(it->key());
                        level_change.emplace_back(-1);
                        continue;
                    }
                }
                else{
                    if(ref_leafs_[it_ref - ref_keys_.begin()]==1)
                    {
                        // it is a leaf
                        // octs.emplace_back(it->key().parent());
                        // level_change.emplace_back(0);
                        continue;
                    }
                    else
                    {
                        // // it is a correction
                        octs.emplace_back(it->key());
                        level_change.emplace_back(-1);
                        continue;
                    }
                }
                continue;
            }
            else
            {
                // continue;
                octs.emplace_back(it->key());
                level_change.emplace_back(-1);
            }
        }
    }

    void print_keys()
    {
        // t= domain_->tree();
        std::cout << "Tree keys: " << std::endl;
        for (auto it = domain_->begin(); it != domain_->end(); ++it)
        {
            // is leaf
            if (it->is_leaf_search())
            {
                // std::cout << "Leaf: " << it->key() << std::endl;
                if(it->is_leaf())
                {
                    std::cout << "Leaf: " << it->key() << std::endl;
                    std::cout<<"flagged";
                }
                // else
                // {
                //     std::cout<<"not flagged"<<std::endl;
                // }
            }
            else
            {
               if(it->is_leaf())
                {
                    std::cout << "Leaf: " << it->key() << std::endl;
                    std::cout<<"not seacrhed"<<std::endl;
                }
            }
        }
    }
    template< class key_t >
    void adapt_level_change(std::vector<float_type> source_max,
                            std::vector<key_t>& octs,
                            std::vector<int>&   level_change )
    {
        octs.clear();
        level_change.clear();
        for (auto it = domain_->begin(); it != domain_->end(); ++it)
        {

            if (!it->locally_owned()) continue;
            if (!it->is_leaf() && !it->is_correction()) continue;
            if (it->is_leaf() && it->is_correction()) continue;
            if (it->is_leaf()&&!it->is_correction())
            {
                octs.emplace_back(it->key());
                level_change.emplace_back(1);
            }
        }
    }


     /** @brief  Refienment conditon for octants.  */
    template<class OctantType>
    bool refinement(OctantType* it, int diff_level, bool use_all = false) const
        noexcept
    {
        // if(it->is_leaf()&& !it->is_correction())
        // {
        //     // std::cout<<"refinement condition for leaf"<<std::endl;
        //     return true;
        // }
        return false;
    }
    // function to access tree
    auto tree()
    {
        return domain_->tree();
    }
        /** @brief  Initialization of the domain blocks. This is registered in the
     *          domain through the base setup class, passing it to the domain ctor.
     */
    std::vector<coordinate_t> initialize_domain(
        Dictionary* _d, domain_t* _domain)
    {
        auto res =
            _domain->construct_basemesh_blocks(_d, _domain->block_extent());
        domain_->read_parameters(_d);

        return res;
    }
    private:
    int nLevelRefinement_ = 0; // Number of refinement levels
    boost::mpi::communicator client_comm_;
    std::vector<key_id_t> ref_keys_; //referfecne keys local to rank
    std::vector<int> ref_leafs_; //reference leafs local to rank
    std::string restart_tree_dir_;
    std::string restart_field_dir_;
};
}
#endif // IBLGF_INCLUDED_OPERATORTEST_HPP