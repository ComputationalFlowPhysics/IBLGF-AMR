ARG BASE_IMAGE

FROM ${BASE_IMAGE}
ARG TARGET

ARG DEBIAN_FRONTEND=noninteractive
ARG CORES=4
LABEL maintainer="ccardina@caltech.edu"

WORKDIR /workspace

RUN if [ "$TARGET" = "cpu" ] ; then \
        apt-get update && apt-get install -y --no-install-recommends \
        time \
        lcov \
        build-essential \
        ca-certificates \
        wget \
        curl \
        git \
        cmake \
        ninja-build \
        pkg-config \
        python3 \
        python3-pip \
        python3-venv \
        python3-dev \
        autoconf \
        automake \
        libtool \
        gfortran \
        vim \
        # MPI
        openmpi-bin \
        libopenmpi-dev \
        # FFTW
        libfftw3-dev \
        libfftw3-mpi-dev \
        # HDF5 with MPI
        # libhdf5-openmpi-dev \
        # BLAS/LAPACK
        libblas-dev \
        liblapack-dev \
        libopenblas-dev \
        && apt-get clean && rm -rf /var/lib/apt/lists/* ; \
    elif [ "$TARGET" = "gpu" ] ; then \
        apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        wget \
        curl \
        git \
        cmake \
        ninja-build \
        pkg-config \
        python3 \
        python3-pip \
        autoconf \
        automake \
        libtool \
        gfortran \
        vim \
        sudo \
        # CUDA Math Libraries (standard for scientific computing)
        libcublas-dev-12-1 \
        libcufft-dev-12-1 \
        libcurand-dev-12-1 \
        libcusolver-dev-12-1 \
        libcusparse-dev-12-1 \
        # MPI
        openmpi-bin \
        libopenmpi-dev \
        # FFTW
        libfftw3-dev \
        libfftw3-mpi-dev \
        # BLAS/LAPACK
        libblas-dev \
        liblapack-dev \
        libopenblas-dev \
        && apt-get clean && rm -rf /var/lib/apt/lists/* ; \
    else \
        echo "TARGET must be 'cpu' or 'gpu'" ; exit 1 ; \
    fi

# ------------------------------
# Python packages
# ------------------------------
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

# ------------------------------
# Environment Variables for CUDA
# ------------------------------
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:/usr/local/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
# ------------------------------
# Boost 1.68 with MPI support
# ------------------------------
ENV BOOST_VERSION=1.68.0
ENV BOOST_DIR=/workspace/boost_1_68_0
ENV BOOST_URL=https://sourceforge.net/projects/boost/files/boost/1.68.0/boost_1_68_0.tar.gz/download

# RUN cd /workspace \
#     && wget -O boost.tar.gz $BOOST_URL \
#     && tar -xzf boost.tar.gz \
#     && cd $BOOST_DIR \
#     && echo "using mpi : /usr/bin/mpicxx ;" >> project-config.jam \
#     && ./bootstrap.sh --with-libraries=system,filesystem,serialization,mpi \
#     && ./b2 -j${CORES} --with-mpi --target=shared,static install \
#     && cd /workspace && rm -rf $BOOST_DIR boost.tar.gz
RUN cd /workspace \
    && wget -O boost.tar.gz $BOOST_URL \
    && tar -xzf boost.tar.gz \
    && cd $BOOST_DIR \
    && echo "using mpi ;" >> project-config.jam \
    && echo "using mpi : /usr/bin/mpicxx ;">> project-config.jam \
    && ./bootstrap.sh  --with-libraries=system,filesystem,serialization,mpi \
    && echo "using mpi ;" >> project-config.jam \
    && ./b2 -j${CORES} --with-mpi --target=shared,static \
    && echo "using mpi ;" >> project-config.jam \
    && ./b2 install \
    && cd /workspace && rm -rf boost.tar.gz
# ------------------------------
# xtensor stack
# ------------------------------
ENV XTL_VERSION=0.7.5
ENV XSIMD_VERSION=7.4.9
ENV XTENSOR_VERSION=0.23.10
ENV XTENSOR_BLAS_VERSION=0.19.2

RUN mkdir -p /workspace/src && cd /workspace/src \
    # xtl
    && wget -q https://github.com/xtensor-stack/xtl/archive/refs/tags/${XTL_VERSION}.tar.gz -O xtl.tar.gz \
    && tar -xzf xtl.tar.gz && cd xtl-${XTL_VERSION} && mkdir build && cd build \
    && cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local -DBUILD_TESTS=OFF && make install -j${CORES} && cd ../.. \
    # xsimd
    && wget -q https://github.com/xtensor-stack/xsimd/archive/refs/tags/${XSIMD_VERSION}.tar.gz -O xsimd.tar.gz \
    && tar -xzf xsimd.tar.gz && cd xsimd-${XSIMD_VERSION} && mkdir build && cd build \
    && cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local -DBUILD_TESTS=OFF && make install -j${CORES} && cd ../.. \
    # xtensor
    && wget -q https://github.com/xtensor-stack/xtensor/archive/refs/tags/${XTENSOR_VERSION}.tar.gz -O xtensor.tar.gz \
    && tar -xzf xtensor.tar.gz && cd xtensor-${XTENSOR_VERSION} && mkdir build && cd build \
    && cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local -DBUILD_TESTS=OFF && make install -j${CORES} && cd ../.. \
    # xtensor-blas
    && wget -q https://github.com/xtensor-stack/xtensor-blas/archive/refs/tags/${XTENSOR_BLAS_VERSION}.tar.gz -O xtensor-blas.tar.gz \
    && tar -xzf xtensor-blas.tar.gz && cd xtensor-blas-${XTENSOR_BLAS_VERSION} && mkdir build && cd build \
    && cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local -DBUILD_TESTS=OFF && make install -j${CORES} \
    && rm -rf /workspace/src

# ------------------------------
# HDF5 with Parallel (MPI) support
# ------------------------------
ENV HDF5_VERSION=1.12.2
RUN cd /workspace \
    && wget https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.12/hdf5-${HDF5_VERSION}/src/hdf5-${HDF5_VERSION}.tar.gz \
    && tar -xzf hdf5-${HDF5_VERSION}.tar.gz \
    && cd hdf5-${HDF5_VERSION} \
    && ./configure --enable-parallel --enable-shared --prefix=/usr/local CC=mpicc CXX=mpicxx \
    && make -j${CORES} && make install \
    && cd /workspace && rm -rf hdf5-${HDF5_VERSION}*

# ------------------------------
# Final Configuration
# ------------------------------
RUN useradd -m cppuser && \
    echo "cppuser ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

USER cppuser
WORKDIR /home/cppuser

# # Setup environment for user
# RUN echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc && \
#     echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/lib:$LD_LIBRARY_PATH' >> ~/.bashrc

CMD ["/bin/bash"]