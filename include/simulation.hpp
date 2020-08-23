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

        move_file(restart_write_dir()+"/"+restart_field_file_,    restart_write_dir()+"/backup/"+restart_field_file_ );
        move_file(restart_write_dir()+"/"+tree_info_file_+".bin", restart_write_dir()+"/backup/"+tree_info_file_+".bin");
        move_file(restart_write_dir()+"/"+restart_info_file_,     restart_write_dir()+"/backup/"+restart_info_file_  );
    }

    void move_file(std::string f_in, std::string f_out )
    {

        if ( boost::filesystem::exists( f_in ) )
            boost::filesystem::rename(f_in, f_out);

    }

    void write_tree(std::string _filename, bool restart_file=false)
    {

        if (restart_file==true)
            domain_->tree()->write(io::output().restart_save_dir()+"/"+tree_info_file_+".bin");
        else
            domain_->tree()->write(io::output().dir()+"/"+tree_info_file_+_filename+".bin");

    }

    void write(std::string _filename)
    {
        writer.write_vtk(io::output().dir()+"/"+_filename, domain_.get());
    }

    void write2(std::string _filename, bool restart_file=false)
    {
        if (restart_file)
        {
            io_h5.write_h5(io::output().restart_save_dir()+"/"+restart_field_file_, domain_.get());
            if (domain_->is_server())
                write_tree("", true);
        }
        else
        {
            io_h5.write_h5(io::output().dir()+"/flow_"+_filename+".hdf5", domain_.get());
            if (domain_->is_server())
                write_tree("_"+_filename, false);
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
  std::string tree_info_file_="tree_info";
  std::string restart_field_file_="restart_field.hdf5";

  int intrp_order_=3;

};





#endif
