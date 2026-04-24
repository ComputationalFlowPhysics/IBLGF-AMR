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

#ifndef IBLGF_SOLVER_MODAL_ANALYSIS_REFLECT_FIELD_HPP
#define IBLGF_SOLVER_MODAL_ANALYSIS_REFLECT_FIELD_HPP
#include <iblgf/domain/mpi/task_manager.hpp>
namespace iblgf
{
namespace domain
{
template<class Setup>
class ReflectField
{
  public: //member types
    using simulation_type = typename Setup::simulation_t;
    using domain_type = typename simulation_type::domain_type;
    using datablock_type = typename domain_type::datablock_t;
    using tree_t = typename domain_type::tree_t;
    using octant_t = typename tree_t::octant_type;
    using MASK_TYPE = typename octant_t::MASK_TYPE;
    using block_type = typename datablock_type::block_descriptor_type;
    using real_coordinate_type = typename domain_type::real_coordinate_type;
    using coordinate_type = typename domain_type::coordinate_type;
    using poisson_solver_t = typename Setup::poisson_solver_t;
    using linsys_solver_t = typename Setup::linsys_solver_t;
    using time_integration_t = typename Setup::time_integration_t;
    // using key_query_t = sr_mpi::Task<tags::key_query, std::vector<int>>;
    // using task_manager_t = sr_mpi::TaskManager<key_query_t>;
    using ib_t = typename domain_type::ib_t;
    using force_type = typename ib_t::force_type;

    //FMM
    using Fmm_t = typename Setup::Fmm_t;
    using key_t = typename tree_t::key_type;
    using test_type = typename Setup::test_type;
    // using idx_u_type = typename Setup::idx_u_type;
    using edge_aux_type = typename Setup::edge_aux_type;
    using u_type = typename Setup::u_type;
    using u_sym_type = typename Setup::u_sym_type;
    // using u_mean_type = typename Setup::u_mean_type;
    using stream_f_type = typename Setup::stream_f_type;
    using cell_aux_type = typename Setup::cell_aux_type;
    // using cell_aux_tmp_type = typename Setup::cell_aux_tmp_type;
    // using face_aux_tmp_type = typename Setup::face_aux_tmp_type;
    // using trait_t = ServerClientTraits<Domain>;
    using face_aux_type = typename Setup::face_aux_type;
    using rf_s_type = typename Setup::rf_s_type;
    using rf_t_type = typename Setup::rf_t_type;
    // using balance_task = typename trait_t::balance_task;
    using balance_task = Task<tags::balance, std::vector<double>, Inplace, octant_t>;
    static constexpr int lBuffer = 1; ///< Lower left buffer for interpolation
    static constexpr int rBuffer = 1; ///< Lower left buffer for interpolation

    static constexpr int Dim = Setup::Dim; ///< Number of dimensions
  public:
    struct ServerUpdate
    {
        ServerUpdate(int _worldsize)
        : send_octs(_worldsize)
        , dest_ranks(_worldsize)
        , recv_octs(_worldsize)
        , src_ranks(_worldsize)
        , src_octs(_worldsize)
        , dest_gids(_worldsize)
        , src_gids(_worldsize)
        {
        }

        void insert(int _current_rank, int _new_rank, key_t _current_key, key_t _new_key, int _gid)
        {
            send_octs[_current_rank].emplace_back(_current_key);
            dest_ranks[_current_rank].emplace_back(_new_rank);
            dest_gids[_current_rank].emplace_back(_gid);

            recv_octs[_new_rank].emplace_back(_new_key);
            src_octs[_new_rank].emplace_back(_current_key);
            src_ranks[_new_rank].emplace_back(_current_rank);
            src_gids[_new_rank].emplace_back(_gid);
        }

        //octant key and dest rank,outer vector in current  rank
        std::vector<std::vector<key_t>> send_octs;
        std::vector<std::vector<int>>   dest_ranks;
        std::vector<std::vector<int>>   dest_gids;
        //octant key and src rank, outer vector in current  rank
        std::vector<std::vector<key_t>> recv_octs;
        std::vector<std::vector<key_t>> src_octs;
        std::vector<std::vector<int>>   src_ranks;
        std::vector<std::vector<int>>   src_gids;
    };

    struct ClientUpdate
    {
        std::vector<key_t> send_octs;
        std::vector<int>   dest_ranks;
        std::vector<int>   dest_gids;

        std::vector<key_t> recv_octs;
        std::vector<key_t> src_octs;
        std::vector<int>   src_ranks;
        std::vector<int>   src_gids;
    };

    ReflectField(simulation_type* _simulation)
    : simulation_(_simulation)
    , domain_(_simulation->domain_.get())
    , psolver(_simulation)
    {
        boost::mpi::communicator world;
        // this->init_idx<idx_u_type>();
        std::cout << "ReflectField::ReflectField()" << std::endl;
        world.barrier();
        if (domain_->is_server()) { send_tasks(); }
        else
        {
            auto update = receive_tasks();

            update_field<u_type, u_sym_type>(update);
        }
        world.barrier();
    }

    template<class send_field, class recv_field>
    void update_field(ClientUpdate& _update)
    {
        auto& send_comm = domain_->decomposition().client()->task_manager()->template send_communicator<balance_task>();
        auto& recv_comm = domain_->decomposition().client()->task_manager()->template recv_communicator<balance_task>();
        std::cout << "update_field: send_comm.size() " << std::endl;
        copy<u_type, rf_s_type>(); //flip rf_s blocks and then send them

        up_and_down<rf_s_type>();
        for (int l = domain_->tree()->base_level(); l < domain_->tree()->depth(); ++l)
        {
            domain_->decomposition().client()->template buffer_exchange<rf_s_type>(l);
        }
        for (int field_idx = 0; field_idx < static_cast<int>(rf_s_type::nFields()); ++field_idx)
        {
            int count = 0;
            for (auto& key : _update.send_octs)
            {
                if(Dim==2)
                {
                    //find the octant
                    auto        it = domain_->tree()->find_octant(key);
                    auto        lin_data = it->data_r(rf_s_type::tag(), field_idx).linalg_data();
                    std::size_t N0 = lin_data.shape()[0];
                    std::size_t N1 = lin_data.shape()[1];
                    if (field_idx == 1)
                    {
                        // for (std::size_t i = N0 - 2; i >= 2; --i)
                        // { // interior rows
                        //     for (std::size_t j = 1; j < N1 - 1; ++j)
                        //     {                                        // interior columns
                        //         lin_data(i, j) = lin_data(i , j+1); // shift down
                        // //     }
                        // // }
                        for (std::size_t i = 1; i < N0 - 1; ++i) // interior rows
                        {
                            for (std::size_t j = 1; j < N1 - 1; ++j) // interior columns
                            {
                                lin_data(i, j) = lin_data(i, j + 1); // shift left
                            }
                        }
                        // for (std::size_t i = 1; i < N0 - 1; ++i) // interior rows
                        // {
                        //     for (int j = static_cast<int>(N1) - 2; j >= 1; --j) // interior columns right-to-left
                        //     {
                        //         lin_data(i, j) = lin_data(i, j - 1); // shift right
                        //     }
                        // }
                    }

                    for (std::size_t i = 0; i < N0; ++i)
                    {
                        for (std::size_t j = 0, k = N1 - 1; j < k; ++j, --k) { std::swap(lin_data(i, j), lin_data(i, k)); }
                    }
                }
                else if(Dim==3)
                {
                    auto        it = domain_->tree()->find_octant(key);
                    auto        lin_data = it->data_r(rf_s_type::tag(), field_idx).linalg_data();
                    std::size_t N0 = lin_data.shape()[0];
                    std::size_t N1 = lin_data.shape()[1];
                    std::size_t N2 = lin_data.shape()[2];
                    if(field_idx==1)
                    {
                        for(std::size_t i=1;i<N0-1;++i)
                        {
                            for(std::size_t j=1;j<N1-1;++j)
                            {
                                for(std::size_t k=1;k<N2-1;++k)
                                {
                                    lin_data(i,j,k)=lin_data(i,j+1,k); //shift left
                                }
                            }
                        }
                    }
                    for(std::size_t i=0;i<N0;++i)
                    {
                        for(std::size_t k=0;k<N2; ++k)
                        {
                            for(std::size_t j=0, m=N1-1; j<m; ++j,--m)
                            {
                                std::swap(lin_data(i,j,k),lin_data(i,m,k));
                            }
                        }
                    }

                }
            }
        }
        // return;
        //send the actuall octants
        //for (int field_idx=Field::nFields()-1; field_idx>=0; --field_idx)
        for (int field_idx = 0; field_idx < static_cast<int>(rf_s_type::nFields()); ++field_idx)
        {
            int count = 0;
            for (auto& key : _update.send_octs)
            {
                //find the octant
                auto it = domain_->tree()->find_octant(key);
                // it->rank() = _update.dest_ranks[count];
                // const auto idx = get_octant_idx(it, field_idx); //change thisiisisiiss
                const auto idx = get_tag_idx(_update.dest_gids[count], field_idx);

                auto send_ptr = it->data_r(rf_s_type::tag(), field_idx).data_ptr();

                auto task = send_comm.post_task(send_ptr, _update.dest_ranks[count], true, idx);
                task->attach_data(send_ptr);
                task->rank_other() = _update.dest_ranks[count];
                task->requires_confirmation() = true;
                task->octant() = it;
                ++count;
            }
            send_comm.pack_reflect_messages();

            count = 0;
            for (auto& key : _update.recv_octs)
            {
                auto it = domain_->tree()->find_octant(key);
                // auto it_src= domain_->tree()->find_octant(_update.src_octs[count]);
                // // it->rank() = comm_.rank();
                // const auto idx = get_octant_idx(it, field_idx); //change thisiisisiiss
                // const auto idx=_update.src_gids[count];
                const auto idx = get_tag_idx(_update.src_gids[count], field_idx);
                const auto recv_ptr = it->data_r(rf_t_type::tag(), field_idx).data_ptr();
                auto       task = recv_comm.post_task(recv_ptr, _update.src_ranks[count], true, idx);

                task->attach_data(recv_ptr);
                task->rank_other() = _update.src_ranks[count];
                task->requires_confirmation() = true;
                task->octant() = it;

                ++count;
            }
            recv_comm.pack_reflect_messages();
            std::cout << "send_comm.size() " << domain_->client_communicator().size() << std::endl;

            while (true)
            {
                send_comm.unpack_reflect_messages();
                recv_comm.unpack_reflect_messages();
                if ((recv_comm.done() && send_comm.done())) break;
            }
            recv_comm.clear();
            send_comm.clear();
            domain_->client_communicator().barrier();
            copy<rf_t_type, u_sym_type>();
        }
        for (int l = domain_->tree()->base_level(); l < domain_->tree()->depth(); ++l)
        {
            domain_->decomposition().client()->template buffer_exchange<u_sym_type>(l);
        }
        // multiply by u_1 by -1 to get correct direction
        for (auto it = domain_->begin_leaves(); it != domain_->end_leaves(); ++it)
        {
            if(!it->locally_owned()) continue;
            if (!it->has_data()) continue;
            auto lin_data = it->data_r(u_sym_type::tag(), 1).linalg_data();
            lin_data *= -1.0;
        }
    }
    template<class field, class field_reflect, class field_s, class field_a>
    void combine_reflection()
    {
        //take field and field_reflect and combine them into field_s and field_a (symmetric and antisymmetric)
        //field_s=field+field_reflect
        //field_a=field-field_reflect
        for (auto it = domain_->begin_leaves(); it != domain_->end_leaves(); ++it)
        {   
            if(!it->locally_owned()) continue;
            if (!it->has_data()) continue;
            for (std::size_t field_idx = 0; field_idx < field::nFields(); ++field_idx)
            {
                auto lin_data = it->data_r(field::tag(), field_idx).linalg_data();
                auto lin_data_reflect = it->data_r(field_reflect::tag(), field_idx).linalg_data();
                auto lin_data_s = it->data_r(field_s::tag(), field_idx).linalg_data();
                auto lin_data_a = it->data_r(field_a::tag(), field_idx).linalg_data();
                lin_data_s = lin_data + lin_data_reflect;
                lin_data_a = lin_data - lin_data_reflect;
            }
        }
        // up_and_down<field_s>();
        // up_and_down<field_a>();


    }
    template<class T>
    auto get_octant_idx(T it, int field_idx = 0) const noexcept
    {
        int sep = 3 * 21; //CC: change to 3*21 for 21 frequnecies
                          //(not sure if this is right, runnning out of tags in 3D)

        // if (helmholtz) sep = N_modes * 9;
        int max_id = (boost::mpi::environment::max_tag() / sep) - 1;
        if (it->global_id() < 0)
        {
            std::cout << "error: trying to get oct idx of -1" << std::endl;
            return -1;
        }

        int tmp = (it->global_id() % max_id) + max_id * field_idx;
        if (tmp > boost::mpi::environment::max_tag())
        {
            throw std::runtime_error("Computed MPI tag exceeds max allowable value");
        }
        return tmp;
    }

    template<class T>
    auto get_tag_idx(T gid, int field_idx = 0) const noexcept
    {
        int sep = 3 * 21; //CC: change to 3*21 for 21 frequnecies
                          //(not sure if this is right, runnning out of tags in 3D)

        // if (helmholtz) sep = N_modes * 9;
        int max_id = (boost::mpi::environment::max_tag() / sep) - 1;
        if (gid < 0)
        {
            std::cout << "error: trying to get oct idx of -1" << std::endl;
            return -1;
        }

        int tmp = (gid % max_id) + max_id * field_idx;
        if (tmp > boost::mpi::environment::max_tag())
        {
            throw std::runtime_error("Computed MPI tag exceeds max allowable value");
        }
        return tmp;
    }

    void send_tasks()
    {
        auto                     update = server_get_tasks();
        boost::mpi::communicator world;
        for (int i = 1; i < world.size(); ++i)
        {
            world.send(i, i + 0 * world.size(), update.send_octs[i]);
            world.send(i, i + 1 * world.size(), update.dest_ranks[i]);
            world.send(i, i + 2 * world.size(), update.dest_gids[i]);

            world.send(i, i + 3 * world.size(), update.recv_octs[i]);
            world.send(i, i + 4 * world.size(), update.src_ranks[i]);
            world.send(i, i + 5 * world.size(), update.src_gids[i]);
            world.send(i, i + 6 * world.size(), update.src_octs[i]);
        }
    }

    ClientUpdate receive_tasks()
    {
        ClientUpdate             update;
        boost::mpi::communicator world;
        world.recv(0, world.rank() + 0 * world.size(), update.send_octs);
        world.recv(0, world.rank() + 1 * world.size(), update.dest_ranks);
        world.recv(0, world.rank() + 2 * world.size(), update.dest_gids);
        world.recv(0, world.rank() + 3 * world.size(), update.recv_octs);
        world.recv(0, world.rank() + 4 * world.size(), update.src_ranks);
        world.recv(0, world.rank() + 5 * world.size(), update.src_gids);
        world.recv(0, world.rank() + 6 * world.size(), update.src_octs);
        std::cout << "Received tasks for rank: " << world.rank() << std::endl;
        //number of tasks
        std::cout << "Number of tasks: " << update.recv_octs.size() << std::endl;
        return update;
    }

    ServerUpdate server_get_tasks()
    {
        boost::mpi::communicator world;
        // if (!domain_->is_server()) return;
        ServerUpdate update(world.size());
        // --------------------------------------------------------------

        for (auto it1 = domain_->begin_leaves(); it1 != domain_->end_leaves(); ++it1)
        {
            if (!it1->has_data()) continue;
            if (!it1->is_leaf() || it1->is_correction()) continue;
            auto coord = it1->tree_coordinate();
            auto key = it1->key();
            auto level = it1->key().level();
            auto opposite_coord = coord;
            // 128 = 1792/14 =extent on baselevel
            auto ref_level = it1->key().level() - domain_->tree()->base_level();
            opposite_coord[1] = 32 * (1 << ref_level) - (coord[1] + 1);

            auto it2 = domain_->tree()->find_octant(key_t(opposite_coord, level));
            if (!it2)
            {
                std::cout << "No opposite block found for: " << it1->key() << std::endl;

                continue;
            }
            if (!it2->is_leaf())
            {
                std::cout << "No opposite leaf block found for: " << it1->key() << std::endl;

                continue;
            }

            // // it1 and 2 are valid and across from eachother

            auto send_rank = it1->rank();
            auto recv_rank = it2->rank();
            auto send_gid = it1->global_id();
            auto send_key = it1->key();
            auto recv_key = it2->key();
            // auto recv_gid = it2->global_id();
            // if (send_rank == recv_rank) continue; //no need to send if same rank
            //                                       // update.insert(send_rank, recv_rank, it1->key(), send_gid);
            update.insert(send_rank, recv_rank, send_key, recv_key, send_gid);
        }
        return update;
    }

  private:
    template<class Field>
    void up_and_down()
    {
        //claen non leafs
        clean<Field>(true);
        this->up<Field>();
        this->down_to_correction<Field>();
    }

    template<class Field>
    void up(bool leaf_boundary_only = false)
    {
        //Coarsification:
        for (std::size_t _field_idx = 0; _field_idx < Field::nFields(); ++_field_idx)
            psolver.template source_coarsify<Field, Field>(_field_idx, _field_idx, Field::mesh_type(), false, false,
                false, leaf_boundary_only);
    }

    template<class Field>
    void down_to_correction()
    {
        // Interpolate to correction buffer
        for (std::size_t _field_idx = 0; _field_idx < Field::nFields(); ++_field_idx)
            psolver.template intrp_to_correction_buffer<Field, Field>(_field_idx, _field_idx, Field::mesh_type(), true,
                false);
    }
    template<class from, class to>
    void copy()
    {
        static_assert(from::nFields() == to::nFields(), "number of fields doesn't match when copy");

        for (auto it = domain_->begin(); it != domain_->end(); ++it)
        {
            if (!it->locally_owned()) continue;
            if (!it->has_data()) continue;
            for (int f = 0; f < from::nFields(); ++f)
            {
                auto from_data = it->data_r(from::tag(), f).linalg_data();
                auto to_data = it->data_r(to::tag(), f).linalg_data();
                xt::noalias(to_data) = from_data;
            }
        }
    }
    template<typename F>
    void clean(bool non_leaf_only = false, int clean_width = 1) noexcept
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
                    if (domain_->dimension() == 3)
                    {
                        view(lin_data, xt::all(), xt::all(), xt::range(0, clean_width)) *= 0.0;
                        view(lin_data, xt::all(), xt::range(0, clean_width), xt::all()) *= 0.0;
                        view(lin_data, xt::range(0, clean_width), xt::all(), xt::all()) *= 0.0;
                        view(lin_data, xt::range(N + 2 - clean_width, N + 3), xt::all(), xt::all()) *= 0.0;
                        view(lin_data, xt::all(), xt::range(N + 2 - clean_width, N + 3), xt::all()) *= 0.0;
                        view(lin_data, xt::all(), xt::all(), xt::range(N + 2 - clean_width, N + 3)) *= 0.0;
                    }
                    else
                    {
                        view(lin_data, xt::all(), xt::range(0, clean_width)) *= 0.0;
                        view(lin_data, xt::range(0, clean_width), xt::all()) *= 0.0;
                        view(lin_data, xt::range(N + 2 - clean_width, N + 3), xt::all()) *= 0.0;
                        view(lin_data, xt::all(), xt::range(N + 2 - clean_width, N + 3)) *= 0.0;
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

  private:
    simulation_type* simulation_;
    domain_type*     domain_; ///< domain
    poisson_solver_t psolver;
};
} // namespace domain
} // namespace iblgf

#endif // IBLGF_SOLVER_MODAL_ANALYSIS_POD_PETSC_HPP