#Bring the headers, such as Student.h into the project
include_directories(${IBLGF_SOURCE_DIR}/src )

find_package(PETSc)
#find_package(SLEPc)

INCLUDE_DIRECTORIES(${PETSC_INCLUDES})
INCLUDE_DIRECTORIES(${SLEPC_INC})
ADD_DEFINITIONS(${PETSC_DEFINITIONS})

set(NAME ns_amr_lgf_helm_oseen)

set(Ptest_list ns_amr_lgf_Test_oseen)

# add the executable
add_executable(ns_amr_lgf_helm_oseen.x ns_amr_lgf_oseen.cpp ${SOURCES})
target_link_libraries(ns_amr_lgf_helm_oseen.x
                      ${MPI_LIBRARIES}
                      ${Boost_LIBRARIES}
                      ${FFTW_LIBRARIES}
                      ${HDF5_LIBRARIES} hdf5
                      ${BLAS_LIBRARIES}
                      ${PETSC_LIBRARIES}
                      xtensor
                      xsimd)
