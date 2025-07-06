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
namespace iblgf
{
using namespace domain;
template<class Setup>
class MergeTrees
{
  public:
    // using tree_t = typename Setup::tree_t;
    using key_id_t= typename Setup::key_id_t;
    // using tree_t = typename Setup::domain_t::tree_t;

    MergeTrees() = default;
    MergeTrees(const MergeTrees&) = delete;
    MergeTrees(MergeTrees&&) = default;
    MergeTrees& operator=(const MergeTrees&) & = delete;
    MergeTrees& operator=(MergeTrees&&) & = default;
    ~MergeTrees() = default;

  public:
    static void merge()
    {
        boost::mpi::communicator world;
        if (world.rank() != 0) return;
        std::cout << "Merging trees..." << std::endl;
    }
    template<class tree_t>
    static void merger( tree_t& t1,  tree_t& t2,std::vector<key_id_t>& octs, std::vector<int>& leafs)
    {

        boost::mpi::communicator world;
        // std::cout << "Merging trees..." << std::endl;
        octs.clear();
        leafs.clear();
        if(world.rank()!=0) return;
        for (auto it1 = t1->begin(); it1 != t1->end(); ++it1)
        {
            if(!it1->has_data()) continue;
            auto it2= t2->find_octant(it1->key());
            if(it2&&(it2->has_data()&&it1->has_data()))
            {
                bool is_leaf=false;
                if((it1->is_leaf()|| it2->is_leaf()))
                {
                    if(it1->refinement_level()>0&&(it1->is_correction()|| it2->is_correction())) continue;
                    is_leaf=true;
                    // if(it1->refinement_level()>=0&&(it1->is_correction()|| it2->is_correction())) is_leaf=false;
                    std::cout << "Common Leaf: " << it1->tree_coordinate() << std::endl;
                    std::cout<<"level: "<<it1->refinement_level()<<std::endl;   
                }
                octs.emplace_back(it2->key().id());
                leafs.emplace_back(is_leaf);
            }
        }
    }

    template<class tree_t>
    static void get_ranks(tree_t& t1,std::vector<key_id_t>& octs, std::vector<int>& leafs)
    {
        boost::mpi::communicator world;
        // server contains all octs and leafs and client has empty vector. use function to get ranks from server and fill in vectors
        std::vector<std::vector<key_id_t>> send_octs(world.size()); //outer vector is rank
        std::vector<std::vector<int>> send_leafs(world.size());

        if(world.rank()==0)
        {
            std::cout<< "Getting ranks for octs..." << std::endl;
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
                world.send(i,i*2, send_octs[i]);
                world.send(i,i*2+1, send_leafs[i]);
            }
            world.barrier();
        }
        else
        {
            octs.clear();
            leafs.clear();
            world.barrier();
            world.recv(0, world.rank()*2, octs);
            world.recv(0, world.rank()*2+1, leafs);
            world.barrier();
        }

    }


};

}// namespace iblgf
#endif // IBLGF_MERGE_TREES_HPP