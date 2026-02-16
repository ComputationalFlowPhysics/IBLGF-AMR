                        ▄▄▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄▄▄   ▄            ▄▄▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄▄▄▄
                        ▐░░░░░░░░░░░▌▐░░░░░░░░░░▌ ▐░▌          ▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌
                        ▀▀▀▀█░█▀▀▀▀ ▐░█▀▀▀▀▀▀▀█░▌▐░▌          ▐░█▀▀▀▀▀▀▀▀▀ ▐░█▀▀▀▀▀▀▀▀▀
                            ▐░▌     ▐░▌       ▐░▌▐░▌          ▐░▌          ▐░▌          
                            ▐░▌     ▐░█▄▄▄▄▄▄▄█░▌▐░▌          ▐░▌ ▄▄▄▄▄▄▄▄ ▐░█▄▄▄▄▄▄▄▄▄
                            ▐░▌     ▐░░░░░░░░░░▌ ▐░▌          ▐░▌▐░░░░░░░░▌▐░░░░░░░░░░░▌
                            ▐░▌     ▐░█▀▀▀▀▀▀▀█░▌▐░▌          ▐░▌ ▀▀▀▀▀▀█░▌▐░█▀▀▀▀▀▀▀▀▀
                            ▐░▌     ▐░▌       ▐░▌▐░▌          ▐░▌       ▐░▌▐░▌          
                        ▄▄▄▄█░█▄▄▄▄ ▐░█▄▄▄▄▄▄▄█░▌▐░█▄▄▄▄▄▄▄▄▄ ▐░█▄▄▄▄▄▄▄█░▌▐░▌          
                        ▐░░░░░░░░░░░▌▐░░░░░░░░░░▌ ▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░▌          
                        ▀▀▀▀▀▀▀▀▀▀▀  ▀▀▀▀▀▀▀▀▀▀   ▀▀▀▀▀▀▀▀▀▀▀  ▀▀▀▀▀▀▀▀▀▀▀  ▀           

Immersed boundary lattice Green's function flow solver.

## Getting Started

This code solves the incompressible Naiver-Stokes equations on unbounded domains
using a mimetic finite volume approach and lattice Green's functions on
adaptively refined meshes.

#### References:
* [1] Liska S. & Colonius T. (2016) [A fast lattice Green’s function method for solving viscous
incompressible flows on unbounded domains](https://doi.org/10.1016/j.jcp.2016.04.023).
Journal of Computational Physics 316: 360-384.
* [2] Dorschner B., Yu K., Mengaldo G. & Colonius T. (2020). [A fast multi-resolution lattice Green's function method for
elliptic difference equations](https://doi.org/10.1016/j.jcp.2020.109270). Journal of Computational Physics: 109270.

### Prerequisites and external dependencies

The following external libraries need to be installed on your system.
All packages should be available through standard package manager. 
Be sure to install all libraries with mpi support. 

* C++ compiler with c++17-support or above (tested with gcc-7 or above, intel 2018 or above)
* Mpi implementation such as [OpenMpi](https://www.open-mpi.org/)
* [Cmake](https://cmake.org/)
* [Boost](https://www.boost.org/) (required libraries: system, filesystem, serialization,mpi)
* [FFTW](http://www.fftw.org/)
* [HDF5](https://www.hdfgroup.org/solutions/hdf5/) (Mpi support needed. The cxx binding are NOT needed.)
* [Blas](https://www.openblas.net/)
* [xTensor](https://github.com/xtensor-stack/xtensor)
* [xTensor-Blas](https://github.com/xtensor-stack/xtensor-blas)
* [xtl](https://github.com/xtensor-stack/xtl)
* [xsimd](https://github.com/xtensor-stack/xsimd)

##### Optional GPU support:
* [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) (11.0 or higher)
* NVIDIA GPU with compute capability 3.5 or higher

##### Notes to install Boost:

    $ ./bootstrap.sh --prefix=/opt/boost
In the file inproject-config.jam, insert

    $ using mpi ;
or

    $ using mpi : CC : <define>DAMNITB2JUSTUSECC ;
Then

    $ ./b2  -j  --prefix=/opt/boost --target=shared,static
    $ ./b2 install

### Running the project inside Docker

For convenience, this repository provides a helper script docker_iblgf.sh that
launches the project inside the prebuilt Docker image used by the lab
(ccardina/my-app:cpu or ccardina/my-app:gpu) and adds a lightweight Python environment on top.

The script automatically:
* pulls the base Docker image if needed,
* builds a local Python-enabled image (cached after the first run),
* mounts the current repository into the container,
* optionally limits the number of CPUs available to Docker.

#### Basic Usage
From the repository root:

    $ ./docker_iblgf.sh

This launches an interactive shell inside the Docker container with the
repository mounted at /workspace2.

For GPU support:

    $ ./docker_iblgf.sh -g

This uses the GPU Docker image and provides access to NVIDIA GPUs.

Once inside Docker, you can run the usual workflow.

### Limiting the number of CPUs used by Docker
On laptops or shared machines, it is often desirable to limit how many CPU cores
Docker can use.

To restrict the container to N CPU cores, use:

    $ ./docker_iblgf.sh -c N

Example (limit Docker to 4 cores):

    $ ./docker_iblgf.sh -c 4

You can verify the limit inside Docker with:

    $ nproc

### Building and Running the tests using the iblgf.sh helper script

For convenience, this repository provides a wrapper script iblgf.sh that
automates configuration, building, and testing.
It replaces the need to manually create a build directory and invoke CMake
commands by hand.

#### Basic usage

From the repository root:

    $ ./iblgf.sh build

Builds the project using all available CPU cores by default.

    $ ./iblgf.sh test

Builds the project (if needed) and then runs the test suite.

#### Controlling the number of cores

The number of CPU cores used for building and testing can be controlled
independently.

Build parallelism (compilation)

    $ ./iblgf.sh build -j N

Build using N parallel compilation jobs.
If -j is not specified, the script automatically uses the number of cores
available on the machine.

Test parallelism (ctest)

    $ ./iblgf.sh test -j N

Run up to N tests concurrently.

If -j is not specified, tests are run sequentially (safe default).

#### Running a single test manually
To run a specific test with a chosen configuration:

    $ ./iblgf.sh run-test <test_name> <config_name_or_path>

Example:

    $ ./iblgf.sh run-test ns_amr_lgf configFile_0

By default, the test is run with a small number of MPI ranks.
This can be overridden explicitly:

    $ ./iblgf.sh run-test ns_amr_lgf configFile_0 -n N

Each run is executed in a timestamped directory under runs/, and
standard output, error logs, and metadata are recorded for reproducibility.

### Restarting a test run

When running a test, you can set the options 'write-restart' and 'use-restart' 
to true in its config file to make the test support restart/checkpointing. 
This allows a simulation to be interrupted and later resumed were if left off. 
This is useful for long-running tests, debugging, or situations where a run
is stopped due to time limits.

When a test is running, it periodically writes restart files (e.g.
restart_field.hdf5, tree_info.bin) into its run directory.

#### Resuming the most recent run

To resume the most recent run of a specific test:

    $ ./iblgf.sh run-test <test_name> <config_name_or_path> --resume

Example:

    $ ./iblgf.sh run-test ns_amr_lgf configFile_0 --resume -n 4

This reuses the latest run directory under runs/<test_name>/ and continues
the simulation from the last available restart checkpoint.

#### Resuming a specific run directory

To resume a specific previous run explicitly:

    $ ./iblgf.sh run-test <test_name> <config_name_or_path> --resume runs/<test_name>/<timestamp>

Example:

    $ ./iblgf.sh run-test ns_amr_lgf configFile_0 --resume runs/ns_amr_lgf/2026-01-25_20-12-09 -n 4

### Configuring and building the library

IBLGF uses the [CMake](https://cmake.org/) integrated configuration and build system.
Within the parent directory do the following steps:

    $ mkdir build
    $ cd build
    $ ccmake ..  (adjust system depended paths to external dependencies if not detected automatically)
    $ make   (alternatively $ make -j<N> )

#### Building with GPU support

To enable GPU acceleration via CUDA:

    $ mkdir build
    $ cd build
    $ cmake -DUSE_GPU=ON ..
    $ make -j<N>

This enables GPU-accelerated convolution operations and FFTs using cuFFT.

The GPU-enabled Navier-Stokes solver is available in `tests/ns_amr_lgf_gpu/`.
See that directory's README for specific usage instructions.

### Verify installation and running the tests

    $ make test

## Authors
Benedikt Dorschner  
Ke Yu  
Marcus Lee  
Wei Hou
