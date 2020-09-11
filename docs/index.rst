.. IBLGF-AMR documentation master file, created by
   sphinx-quickstart on Fri Sep  4 13:30:20 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

The IBLGF-AMR Project
=====================

Immersed boundary lattice Green's function flow solver.

This code solves the incompressible Naiver-Stokes equations on unbounded domains
using a mimetic finite volume approach and lattice Green's functions on
adaptively refined meshes.

References
++++++++++

* [1] Liska S. & Colonius T. (2016) [A fast lattice Green’s function method for solving viscous
incompressible flows on unbounded domains](https://doi.org/10.1016/j.jcp.2016.04.023).
Journal of Computational Physics 316: 360-384.
* [2] Dorschner B., Yu K., Mengaldo G. & Colonius T. (2020). [A fast multi-resolution lattice Green's function method for
elliptic difference equations](https://doi.org/10.1016/j.jcp.2020.109270). Journal of Computational Physics: 109270.

Prerequisites and external dependencies
+++++++++++++++++++++++++++++++++++++++

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

Notes to install Boost:
+++++++++++++++++++++++

    $ ./bootstrap.sh --prefix=/opt/boost
In the file inproject-config.jam, insert

    $ using mpi ;
or

    $ using mpi : CC : <define>DAMNITB2JUSTUSECC ;
Then

    $ ./b2  -j  --prefix=/opt/boost --target=shared,static
    $ ./b2 install



### Configuring and building the library

IBLGF uses the [CMake](https://cmake.org/) integrated configuration and build system.
Within the parent directory do the following steps:

    $ mkdir build
    $ cd build
    $ ccmake ..  (adjust system depended paths to external dependencies if not detected automatically)
    $ make   (alternatively $ make -j<N> )

### Verify installation and running the tests

make test

## Authors
Benedikt Dorschner  
Ke Yu  
Marcus Lee

.. doxygenclass:: Afterdoc_Test
   :project: iblgf
   :members:

.. toctree::
   :maxdepth: 2

   source/getting_started.rst

