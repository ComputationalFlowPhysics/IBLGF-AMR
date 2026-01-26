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
#include <iblgf/solver/modal_analysis/reflect_field.hpp>
namespace iblgf
{
using namespace domain;
using namespace types;
using namespace dictionary;

const int Dim = 3;

struct parameters
{
    static constexpr std::size_t Dim = 3;
    // clang-format off
    REGISTER_FIELDS
    (
    Dim,
        (
            //name, type, nFields, l/h-buf,mesh_obj, output(optional)
            (tlevel,        float_type, 1, 1, 1, cell, true),
            (u,             float_type, Dim, 1, 1, face, true),
            (u_s,             float_type, Dim, 1, 1, face, true),
            (u_a,             float_type, Dim, 1, 1, face, true),
            (u_sym,             float_type, Dim, 1, 1, face, true),
            (rf_s,             float_type, Dim, 1, 1, face, true),
            (rf_t,             float_type, Dim, 1, 1, face, true),
            (p,             float_type, 1, 1, 1, cell, true),
            (test,          float_type, 1,    1,       1,     cell,true )
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
        
        const float_type dx_base = domain_->dx_base();
        for(auto it=domain_->begin(); it!=domain_->end(); ++it)
        {
            if(!it->locally_owned()) continue;
            auto dx_level =  dx_base/std::pow(2,it->refinement_level());
            auto scaling =  std::pow(2,it->refinement_level());
            int ref_level_=it->refinement_level();
            // auto coord = it->tree_coordinate();
            for(auto& n: it->data())
            {
                const auto& coord = n.level_coordinate();

                float_type x = static_cast<float_type>
                (coord[0]-center[0]*scaling)*dx_level; //bottom left corner of cell
                float_type y = static_cast<float_type>
                (coord[1]-center[1]*scaling)*dx_level;
                n(tlevel)=ref_level_+0.5;
                n(u,0)=x;
                n(u,1)=y;
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
    template< class key_t >
    void run_adapt_del(std::vector<key_t>& keys_to_del,
                      std::vector<int>& level_change)
    {
        //register adapt condition based on reference keys and run adaptation
        domain_->decomposition().adapt_del_leafs(keys_to_del, level_change,true);
        this->initialize();
        simulation_.write("adapted_to_ref");
    }
    // //register adapt condition based on reference keys and run adaptation
    // void run_adapt_to_ref(std::vector<key_id_t>& _ref_keys,
    //                   std::vector<int>& _ref_leafs)
    // {
    //     ref_keys_ = _ref_keys;
    //     ref_leafs_ = _ref_leafs;

    //     domain_->register_adapt_condition()=
    //         [this]( std::vector<float_type> source_max, auto& octs, std::vector<int>& level_change )
    //             {return this->template adapt_to_ref(source_max, octs, level_change);};
    //     time_integration_t ifherk(&this->simulation_);
    //     ifherk.adapt(true,false);
    //     ifherk.adapt(true,false);

    //     simulation_.write("adapted_to_ref");

    // }
    template<class Field,class key_t>
    void run_adapt_from_keys(int timeIdx,std::vector<key_t>& octs,
                            std::vector<int>& level_change)
    {
        poisson_solver_t psolver(&this->simulation_);
        boost::mpi::communicator world;
        auto client = domain_->decomposition().client();
        //up to correction
        if(domain_->is_client())
        {
            clean<Field>(true);
            for (std::size_t _field_idx=0; _field_idx<Field::nFields(); ++_field_idx)
                psolver.template source_coarsify<Field,Field>(_field_idx, _field_idx, Field::mesh_type(), false, false, false, false);

        }

        world.barrier();
        auto intrp_list=domain_->decomposition().adapt_del_leafs(octs, level_change, true);
        world.barrier();
        pcout << "Adapt - intrp" << std::endl;
        if (client)
        {
            // Intrp
            for (std::size_t _field_idx=0; _field_idx<Field::nFields(); ++_field_idx)
            {
                for (int l = domain_->tree()->depth() - 2;
                     l >= domain_->tree()->base_level(); --l) // finest level is l=depth-1 and we only 
                {
                    client->template buffer_exchange<Field>(l);

                    domain_->decomposition().client()->
                    template communicate_updownward_assign
                    <Field, Field>(l,false,false,-1,_field_idx);
                }

                for (auto& oct : intrp_list)
                {
                    if (!oct || !oct->has_data()) continue;
                    psolver.c_cntr_nli().template nli_intrp_node<Field, Field>(oct, Field::mesh_type(), _field_idx, _field_idx, false, false);
                }
            }
        }
        world.barrier();
        pcout << "Adapt - done" << std::endl;
        //get interpolation list
        if(timeIdx>0) simulation_.write("adapted_to_ref_"+std::to_string(timeIdx));
        // interpolate
    }
    template<class Field, class Target>
    void symfield(int timeIdx=-1)
    {
        //loop through all blocks
        //get ranks of left and right block
        boost::mpi::communicator world;
        // this->initialize();
        clean<u_sym_type>();
        // up_and_down<u_type>();   
        solver::ReflectField<SetupBase> rf(&this->simulation_); //u has field, u_sym has reflected field 
        // make u_s and u_a fields
        rf.combine_reflection<u_type,u_sym_type,u_s_type,u_a_type>();

        // simulation_.write("symfield"); 
        if(timeIdx>0) simulation_.write("adapted_to_ref_"+std::to_string(timeIdx));

    }
    float_type read_write_test()
    {
        boost::mpi::communicator world;
        pcout_c<<"Testing read and write functionality"<<std::endl;
        std::string f_suffix=simulation_.dictionary()->template get<std::string>("read_write_filename");
        std::string filename = "./flowTime_" + f_suffix + ".hdf5";
        std::string filename2="flowTime_rw_" + f_suffix + ".hdf5";
        simulation_.template  read_h5<u_type>(filename,"u");
        simulation_.template read_h5<edge_aux_type>(filename,"edge_aux");
        world.barrier();
        // std::string filename2="restart_"+filename;
        simulation_.write(filename2);
        return 0.0;

    }

    template <typename F>
    void clean(bool non_leaf_only=false, int clean_width=1) noexcept
    {
        for (auto it = domain_->begin(); it != domain_->end(); ++it)
        {
            if (!it->has_data()) continue;
            if (!it->data().is_allocated()) continue;

            for (std::size_t field_idx = 0; field_idx < F::nFields(); ++field_idx)
            {
                auto& lin_data = it->data_r(F::tag(), field_idx).linalg_data();

                if (non_leaf_only && it->is_leaf() && it->locally_owned())
                {
                    int N = it->data().descriptor().extent()[0];
		    if(domain_->dimension() == 3) {
                    view(lin_data, xt::all(), xt::all(),
                        xt::range(0, clean_width)) *= 0.0;
                    view(lin_data, xt::all(), xt::range(0, clean_width),
                        xt::all()) *= 0.0;
                    view(lin_data, xt::range(0, clean_width), xt::all(),
                        xt::all()) *= 0.0;
                    view(lin_data, xt::range(N + 2 - clean_width, N + 3),
                        xt::all(), xt::all()) *= 0.0;
                    view(lin_data, xt::all(),
                        xt::range(N + 2 - clean_width, N + 3), xt::all()) *=
                        0.0;
                    view(lin_data, xt::all(), xt::all(),
                        xt::range(N + 2 - clean_width, N + 3)) *= 0.0;
		    }
		    else {
                    view(lin_data, xt::all(), xt::range(0, clean_width)) *= 0.0;
                    view(lin_data, xt::range(0, clean_width), xt::all()) *= 0.0;
                    view(lin_data, xt::range(N + 2 - clean_width, N + 3),xt::all()) *= 0.0;
                    view(lin_data, xt::all(),xt::range(N + 2 - clean_width, N + 3)) *=0.0;
		    }
                }
                else
                {
                    //TODO whether to clean base_level correction?
                    std::fill(lin_data.begin(), lin_data.end(), 0.0);
                }
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