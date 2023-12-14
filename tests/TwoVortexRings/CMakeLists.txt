#Bring the headers, such as Student.h into the project
include_directories(${IBLGF_SOURCE_DIR}/src )

find_package(PETSc)
#find_package(SLEPc)

INCLUDE_DIRECTORIES(${PETSC_INCLUDES})
INCLUDE_DIRECTORIES(${SLEPC_INC})
ADD_DEFINITIONS(${PETSC_DEFINITIONS})

set(NAME TwoVortRings)


# add the executable
add_executable(TwoVortRings.x TwoVortRings.cpp ${SOURCES})
target_link_libraries(TwoVortRings.x
                      ${MPI_LIBRARIES}
                      ${Boost_LIBRARIES}
                      ${FFTW_LIBRARIES}
                      ${HDF5_LIBRARIES} hdf5
                      ${BLAS_LIBRARIES}
                      ${PETSC_LIBRARIES}
                      xtensor
                      xsimd)


