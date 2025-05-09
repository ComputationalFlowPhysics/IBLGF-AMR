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

#ifndef IBLGF_INCLUDED_SIMULATION_HPP
#define IBLGF_INCLUDED_SIMULATION_HPP

#include <iblgf/dictionary/dictionary.hpp>
#include <iblgf/IO/chombo/h5_io.hpp>
#include <iblgf/IO/output_directory.hpp>

namespace iblgf
{
using namespace dictionary;

template<class Domain>
class Simulation
{
  public:
    using domain_type = Domain;
    static constexpr int  N_modes = Domain::N_modes_val;
    static constexpr bool  helmholtz = Domain::helmholtz_bool;
    static constexpr std::size_t Dim = domain_type::dims;
    using h5io_t = typename io::H5_io<Dim, Domain>;

  public:
    Simulation(const Simulation& other) = delete;
    Simulation(Simulation&& other) = default;
    Simulation& operator=(const Simulation& other) & = delete;
    Simulation& operator=(Simulation&& other) & = default;
    ~Simulation() = default;

    Simulation(std::shared_ptr<Dictionary> _dictionary)
    : dictionary_(_dictionary)
    , domain_(std::make_shared<domain_type>())
    , io_init_(_dictionary.get())
    {
        intrp_order_ = dictionary_->template get_or<int>("intrp_order", 5);
    }

    friend std::ostream& operator<<(std::ostream& os, Simulation& s)
    {
        os << "Domain: \n" << *(s.domain()) << " " << std::endl;
        return os;
    }

    void copy_restart()
    {
        boost::filesystem::path backupdir_tmp(restart_write_dir()+"/backup_tmp/");
        boost::filesystem::create_directories(backupdir_tmp);

        boost::filesystem::path backupdir(restart_write_dir()+"/backup/");
        boost::filesystem::create_directories(backupdir);

        // in case it stops when moving files, add another backup
        move_file(restart_write_dir()+"/backup/"+restart_field_file_,      restart_write_dir()+"/backup_tmp/"+restart_field_file_ );
        move_file(restart_write_dir()+"/backup/"+tree_info_file_+".bin",   restart_write_dir()+"/backup_tmp/"+tree_info_file_+".bin");
        move_file(restart_write_dir()+"/backup/"+restart_info_file_,       restart_write_dir()+"/backup_tmp/"+restart_info_file_  );

        move_file(restart_write_dir()+"/"+restart_field_file_,  restart_write_dir()+"/backup/"+restart_field_file_ );
        move_file(restart_write_dir()+"/"+tree_info_file_+".bin", restart_write_dir()+"/backup/"+tree_info_file_+".bin");
        move_file(restart_write_dir()+"/"+restart_info_file_,   restart_write_dir()+"/backup/"+restart_info_file_  );
    }

    void move_file(std::string f_in, std::string f_out )
    {

        std::ifstream  src(f_in,  std::ios::binary);
        std::ofstream  dst(f_out, std::ios::binary);

        if (!src.good()) return;
        if (!dst.good()) return;

        if(!src.is_open())
            throw std::runtime_error("Could not open file: " + f_in);
        if(!dst.is_open())
            throw std::runtime_error("Could not open file: " + f_in);
        boost::filesystem::rename(f_in, f_out);
    }

    void write_tree(std::string _filename, bool restart_file=false)
    {

        if (restart_file==true)
            domain_->tree()->write(io::output().restart_save_dir()+"/"+tree_info_file_+".bin");
        else
            domain_->tree()->write(io::output().dir()+"/"+tree_info_file_+_filename+".bin");

    }

    void write(std::string _filename, bool restart_file=false)
    {
        if (restart_file)
        {
            io_h5.write_h5(io::output().restart_save_dir()+"/"+restart_field_file_, domain_.get(), true, true);
            if (domain_->is_server())
                write_tree("", true);
        }
        else
        {
            io_h5.write_h5(io::output().dir()+"/flow_"+_filename+".hdf5", domain_.get(), true, true);
            //io_h5.write_h5_withTime(io::output().dir()+"/flowTime_"+_filename+".hdf5", domain_.get(), true, true);
            if (domain_->is_server())
                write_tree("_"+_filename, false);

            if (helmholtz) {
                float_type c_z = dictionary_->template get_or<float_type>("L_z", 1);
                float_type dz = c_z / static_cast<float_type>(N_modes*3);
                io_h5.write_helm_3D(io::output().dir()+"/vort_"+_filename+".hdf5", domain_.get(), dz, false, false);
                io_h5.write_helm_3D_shallow(io::output().dir()+"/vort_shallow_"+_filename+".hdf5", domain_.get(), dz, false, false);
            }
        }
    }

    void write_test(std::string _filename, bool restart_file=false)
    {
        if (restart_file)
        {
            // io_h5.write_h5(io::output().restart_save_dir()+"/"+restart_field_file_, domain_.get(), true, true);
            // if (domain_->is_server())
            //     write_tree("", true);
        }
        else
        {
            io_h5.write_h5(io::output().dir()+"/flow_"+_filename+".hdf5", domain_.get(), true, true);
            // io_h5.write_h5_swithTime(io::output().dir()+"/flowTime_"+_filename+".hdf5", domain_.get(), true, true);
            // if (domain_->is_server())
            //     write_tree("_"+_filename, false);

            // if (helmholtz) {
            //     float_type c_z = dictionary_->template get_or<float_type>("L_z", 1);
            //     float_type dz = c_z / static_cast<float_type>(N_modes*3);
            //     io_h5.write_helm_3D(io::output().dir()+"/vort_"+_filename+".hdf5", domain_.get(), dz, false, false);
            // }
        }
    }

    void writeWithCorr(std::string _filename, bool restart_file=false)
    {
        
            io_h5.write_h5(io::output().dir()+"/flowWithCorr_"+_filename+".hdf5", domain_.get(), true, false);
            //io_h5.write_h5_withTime(io::output().dir()+"/flowTime_"+_filename+".hdf5", domain_.get(), true, true);
            if (domain_->is_server())
                write_tree("_"+_filename, false);

            /*if (helmholtz) {
                float_type c_z = dictionary_->template get_or<float_type>("L_z", 1);
                float_type dz = c_z / static_cast<float_type>(N_modes*3);
                io_h5.write_helm_3D(io::output().dir()+"/vort_"+_filename+".hdf5", domain_.get(), dz, false, false);
            }*/
        
    }

    void writeWithTime(std::string _filename, float_type _time, float_type dt_)
    {
        io_h5.write_h5_withTime(io::output().dir() + "/flowTime_" + _filename +
                                    ".hdf5",
            domain_.get(), _time, dt_, true, true);
        if (domain_->is_server()) write_tree("_" + _filename, false);

        if (helmholtz)
        {
            float_type c_z = dictionary_->template get_or<float_type>("L_z", 1);
            float_type dz = c_z / static_cast<float_type>(N_modes * 3);
            io_h5.write_helm_3D(io::output().dir() + "/vort_" + _filename +
                                    ".hdf5",
                domain_.get(), dz, false, false);
        }
    }

    auto restart_load_dir()
    {return io::output().restart_load_dir();}

    auto restart_write_dir()
    {return io::output().restart_save_dir();}

    auto restart_field_dir()
    {return io::output().restart_load_dir()+"/"+restart_field_file_;}

    auto restart_tree_info_dir()
    {
            return io::output().restart_load_dir()+"/"+tree_info_file_+".bin";
    }
    bool restart_dir_exist()
    {
        std::ifstream f(restart_tree_info_dir());
        if (!f.good() && domain_->is_server())
            std::cout << " restart file doesn't exist yet" << std::endl;
        return f.good();
    }

    template<typename Field>
    void read_h5(std::string _filename, std::string field_name)
    {
       io_h5.template read_h5<Field>(_filename, field_name, domain_.get());
    }
    template<typename Field>
    void read_h5_test(std::string _filename, std::string field_name)
    {
       io_h5.template read_h5_test<Field>(_filename, field_name, domain_.get());
    }
    template<typename Field>
    void read_h5_DiffNmode(std::string _filename, std::string field_name, int N_input_mode)
    {
       io_h5.template read_h5_DiffNmode<Field>(_filename, field_name, domain_.get(), N_input_mode);
    }


    template<typename Field>
    void read_h5_2D(std::string _filename, std::string field_name)
    {
       io_h5.template read_h5_2D<Field>(_filename, field_name, domain_.get());
    }

    auto& domain()noexcept{return domain_;}
    const auto& domain()const noexcept{return domain_;}
    auto& dictionary()noexcept{return dictionary_;}
    const auto& dictionary()const noexcept{return dictionary_;}

    int intrp_order()noexcept{return intrp_order_;}

    auto& frame_vel() {return frame_vel_;}
    auto& bc_vel() {return bc_vel_;}

public:
  std::shared_ptr<Dictionary> dictionary_=nullptr;
  std::shared_ptr<Domain> domain_=nullptr;
  boost::mpi::communicator world_;
  //io::Vtk_io<Domain> writer;
  h5io_t io_h5;
  io::IO_init io_init_;
  std::string restart_info_file_="restart_info";
  std::string tree_info_file_="tree_info";
  std::string restart_field_file_="restart_field.hdf5";

  std::function<float_type(std::size_t idx, float_type t, typename domain_type::real_coordinate_type coord)> frame_vel_;
  std::function<float_type(std::size_t idx, float_type t, typename domain_type::real_coordinate_type coord)> bc_vel_;
  int intrp_order_;

};

} // namespace iblgf
#endif
