#ifndef IBLGF_INCLUDED_SIMULATION_HPP
#define IBLGF_INCLUDED_SIMULATION_HPP

#include <dictionary/dictionary.hpp>
#include <IO/vtk_io.hpp>
#include <IO/chombo/h5_io.hpp>
#include <IO/output_directory.hpp>

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
        copy_file(restart_write_dir()+"/"+restart_field_file_,  restart_write_dir()+"/backup/"+restart_field_file_ );
        copy_file(restart_write_dir()+"/"+restart_domain_file_, restart_write_dir()+"/backup/"+restart_domain_file_);
        copy_file(restart_write_dir()+"/"+restart_info_file_,   restart_write_dir()+"/backup/"+restart_info_file_  );
    }

    void copy_file(std::string f_in, std::string f_out )
    {
        std::ifstream  src(f_in,  std::ios::binary);
        std::ofstream  dst(f_out, std::ios::binary);

        if(!src.is_open())
            throw std::runtime_error("Could not open file: " + f_in);
        if(!dst.is_open())
            throw std::runtime_error("Could not open file: " + f_in);

        dst << src.rdbuf();
    }

    void write_tree()
    {
        domain_->tree()->write(io::output().restart_save_dir()+"/"+restart_domain_file_);
    }

    void write(std::string _filename)
    {
        writer.write_vtk(io::output().dir()+"/"+_filename, domain_.get());
    }

    void write2(std::string _filename, bool to_restart=false)
    {
        if (to_restart)
            io_h5.write_h5(io::output().restart_save_dir()+"/"+restart_field_file_, domain_.get());
        else
            io_h5.write_h5(io::output().dir()+"/"+_filename, domain_.get());
    }

    auto restart_load_dir()
    {return io::output().restart_load_dir();}
    auto restart_write_dir()
    {return io::output().restart_save_dir();}
    auto restart_field_dir()
    {return io::output().restart_load_dir()+"/"+restart_field_file_;}
    auto restart_domain_dir()
    {return io::output().restart_load_dir()+"/"+restart_domain_file_;}
    bool restart_dir_exist()
    {
        std::ifstream f(restart_domain_dir());
        if (!f.good() && domain_->is_server())
            std::cout<< " restart file doesn't exist yet" <<std::endl;
        return f.good();
    }

    template<typename Field>
    void read_h5(std::string _filename)
    {
       io_h5.template read_h5<Field>(_filename, domain_.get());
    }

    auto& domain()noexcept{return domain_;}
    const auto& domain()const noexcept{return domain_;}
    auto& dictionary()noexcept{return dictionary_;}
    const auto& dictionary()const noexcept{return dictionary_;}

public:
  std::shared_ptr<Dictionary> dictionary_=nullptr;
  std::shared_ptr<Domain> domain_=nullptr;
  boost::mpi::communicator world_;
  io::Vtk_io<Domain> writer;
  io::H5_io<3, Domain> io_h5;
  io::IO_init io_init_;
  std::string restart_info_file_="restart_info";
  std::string restart_domain_file_="restart_domain.bin";
  std::string restart_field_file_="restart_field.hdf5";

};





#endif
