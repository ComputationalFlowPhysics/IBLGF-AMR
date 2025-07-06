#include <iblgf/dictionary/dictionary.hpp>
#include <iblgf/types.hpp>
#include <iblgf/domain/octree/tree.hpp>
#include <iblgf/domain/domain.hpp>
#include <iblgf/simulation.hpp>
namespace iblgf
{
using namespace types;
using namespace dictionary;
// static constexpr int Dim = 2;
const int Dim = 2;

struct CommonTree
{
    struct block
    {
        int dat = -1;
    };
    using tree_type = octree::Tree<Dim, block>;
    using coordinate_type = typename tree_type::coordinate_type;
    using key_type = typename tree_type::key_type;
    using domain_t = domain::Domain<Dim, block>;
    using simulation_t = Simulation<domain_t>;
    using block_descriptor_t=int; // for simplicity, using int as block descriptor
    // CommonTree(int _extent = 5)
    // : ext(_extent)
    // , t(ext)
    // {
    //     // Initialize the tree with a base level
       
    // }
    CommonTree(Dictionary* _d)
    : simulation_(_d->get_dictionary("simulation_parameters"))
    , domain_(simulation_.domain())
    {
        domain_->initialize(simulation_.dictionary()->get_dictionary("domain"));
    }
    void print_keys()
    {
        // t= domain_->tree();
        std::cout << "Tree keys: " << std::endl;
        for (auto it = domain_->>begin(); it != domain_->end(); ++it)
        {
            // is leaf
            if (it->is_leaf_search())
            {
                std::cout << "Leaf: " << it->key() << std::endl;
                if(it->is_leaf())
                {
                    std::cout<<"flagged";
                }
                else
                {
                    std::cout<<"not flagged"<<std::endl;
                }
            }
            // else
            // {
            //     std::cout << "Node: " << it->key() << std::endl;
            // }
        }
    }
    private:
    coordinate_type ext;
    tree_type t;
    simulation_t              simulation_; ///< simulation
    std::shared_ptr<domain_t> domain_ =
        nullptr;   

};
}