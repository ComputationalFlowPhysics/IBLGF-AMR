#ifndef IBLGF_INCLUDED_SIMULATION_HPP
#define IBLGF_INCLUDED_SIMULATION_HPP

#include <dictionary/dictionary.hpp>
#include <IO/vtk_io.hpp>
#include <IO/chombo/h5_io.hpp>
#include <IO/output_directory.hpp>
#include <boost/filesystem.hpp>

using namespace dictionary;

template<class Domain>
class Simulation
{

public:
    using domain_type = Domain;

public:


    Simulation(const Simulation& other) = delete;
    Simulation(Simulation&& other) = default;
    Simulation& operator=(const Simulation& other) & = delete;
    Simulation& operator=(Simulation&& other) & = default;
    ~Simulation() = default;

    Simulation(std::shared_ptr<Dictionary> _dictionary)
    :dictionary_(_dictionary),
     domain_(std::make_shared<domain_type>()),
     io_init_(_dictionary.get())
    {
        intrp_order_ = dictionary_->template get_or<int>("intrp_order",3);
    }

    friend std::ostream& operator<<(std::ostream& os, Simulation& s)
    {
        os
        <<"Domain: \n"<<*(s.domain())<<" "
        <<std::endl;
        return os;
    }

    void copy_restart()
    {
        // in case it stops when moving files, add another backup
        boost::filesystem::remove_all(restart_write_dir()+"/backup_tmp");
        move_file(restart_write_dir()+"/backup",      restart_write_dir()+"/backup_tmp");

        boost::filesystem::path backupdir(restart_write_dir()+"/backup");
        boost::filesystem::create_directories(backupdir);

        move_file(restart_write_dir()+"/"+restart_field_file_+".hdf5",    restart_write_dir()+"/backup/"+restart_field_file_+".hdf5" );
        move_file(restart_write_dir()+"/"+tree_info_file_+".bin", restart_write_dir()+"/backup/"+tree_info_file_+".bin");
        move_file(restart_write_dir()+"/"+restart_info_file_,     restart_write_dir()+"/backup/"+restart_info_file_  );
    }

    void move_file(std::string f_in, std::string f_out )
    {

        if ( boost::filesystem::exists( f_in ) )
            boost::filesystem::rename(f_in, f_out);

    }

    void write_tree(int nstep, bool restart_file=false)
    {

        if (restart_file==true)
            domain_->tree()->write(restart_tree_info_dir(true, nstep));
        else
            domain_->tree()->write(io::output().dir()+"/"+tree_info_file_+std::to_string(nstep)+".bin");

    }

    void write(std::string _filename)
    {
        writer.write_vtk(io::output().dir()+"/"+_filename, domain_.get());
    }

    void write2(std::string prefix, int nstep=-1, bool restart_file=false)
    {
        if (restart_file)
        {
            io_h5.write_h5(restart_field_dir(true, nstep), domain_.get());
            if (domain_->is_server())
                write_tree(nstep, true);
        }
        else
        {
            io_h5.write_h5(io::output().dir()+"/"+prefix +"_flow_"+std::to_string(nstep)+".hdf5", domain_.get());
            if (domain_->is_server())
                write_tree(nstep, false);
        }
    }

    auto restart_load_dir()
    {return io::output().restart_load_dir();}

    auto restart_write_dir()
    {return io::output().restart_save_dir();}

    // ----------------------------------------------------------------------
    auto restart_laststep_info_dir()
    {
        return restart_load_dir()+"/restart_last.fdt";
    }

    int restart_N()
    {
        std::string fdir = restart_laststep_info_dir();
        std::ifstream f(fdir);

        if (!f.good())
        {
            return 0;
        }
        else
        {
            Dictionary d(fdir);
            return d.template get<int>("restart_n_last");
        }
    }

    auto restart_simulation_info_dir(bool to_wrt, int nstep=-1)
    {
        if (to_wrt)
            return io::output().restart_load_dir()+"/info_"+std::to_string(nstep);
        else
            return io::output().restart_load_dir()+"/info_"+std::to_string(restart_N());
    }

    auto restart_field_dir(bool to_wrt, int nstep=-1)
    {
        if (to_wrt)
            return io::output().restart_load_dir()+"/"+restart_field_file_+std::to_string(nstep)+".hdf5";
        else
            return io::output().restart_load_dir()+"/"+restart_field_file_+std::to_string(restart_N())+".hdf5";
    }

    auto restart_tree_info_dir(bool to_wrt, int nstep=-1)
    {
        if (to_wrt)
            return io::output().restart_load_dir()+"/"+tree_info_file_+std::to_string(nstep)+".bin";
        else
            return io::output().restart_load_dir()+"/"+tree_info_file_+std::to_string(restart_N())+".bin";
    }

    bool restart_dir_exist()
    {
        std::ifstream f(restart_tree_info_dir(false));
        if (!f.good() && domain_->is_server())
            std::cout<< " restart file doesn't exist yet" <<std::endl;

        return f.good();
    }

    template<typename Field>
    void read_h5(std::string _filename, std::string field_name)
    {
       io_h5.template read_h5<Field>(_filename, field_name, domain_.get());
    }

    auto& domain()noexcept{return domain_;}
    const auto& domain()const noexcept{return domain_;}
    auto& dictionary()noexcept{return dictionary_;}
    const auto& dictionary()const noexcept{return dictionary_;}

    int intrp_order()noexcept{return intrp_order_;}

public:
  std::shared_ptr<Dictionary> dictionary_=nullptr;
  std::shared_ptr<Domain> domain_=nullptr;
  boost::mpi::communicator world_;
  io::Vtk_io<Domain> writer;
  io::H5_io<3, Domain> io_h5;
  io::IO_init io_init_;
  std::string restart_info_file_="restart_info";
  std::string tree_info_file_="tree_info_";
  std::string restart_field_file_="field_";

  int intrp_order_=3;

};





#endif
