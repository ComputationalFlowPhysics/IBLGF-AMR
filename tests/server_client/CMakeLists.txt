#      ▄▄▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄▄▄   ▄            ▄▄▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄▄▄▄
#     ▐░░░░░░░░░░░▌▐░░░░░░░░░░▌ ▐░▌          ▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌
#      ▀▀▀▀█░█▀▀▀▀ ▐░█▀▀▀▀▀▀▀█░▌▐░▌          ▐░█▀▀▀▀▀▀▀▀▀ ▐░█▀▀▀▀▀▀▀▀▀
#          ▐░▌     ▐░▌       ▐░▌▐░▌          ▐░▌          ▐░▌
#          ▐░▌     ▐░█▄▄▄▄▄▄▄█░▌▐░▌          ▐░▌ ▄▄▄▄▄▄▄▄ ▐░█▄▄▄▄▄▄▄▄▄
#          ▐░▌     ▐░░░░░░░░░░▌ ▐░▌          ▐░▌▐░░░░░░░░▌▐░░░░░░░░░░░▌
#          ▐░▌     ▐░█▀▀▀▀▀▀▀█░▌▐░▌          ▐░▌ ▀▀▀▀▀▀█░▌▐░█▀▀▀▀▀▀▀▀▀
#          ▐░▌     ▐░▌       ▐░▌▐░▌          ▐░▌       ▐░▌▐░▌
#      ▄▄▄▄█░█▄▄▄▄ ▐░█▄▄▄▄▄▄▄█░▌▐░█▄▄▄▄▄▄▄▄▄ ▐░█▄▄▄▄▄▄▄█░▌▐░▌
#     ▐░░░░░░░░░░░▌▐░░░░░░░░░░▌ ▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░▌
#      ▀▀▀▀▀▀▀▀▀▀▀  ▀▀▀▀▀▀▀▀▀▀   ▀▀▀▀▀▀▀▀▀▀▀  ▀▀▀▀▀▀▀▀▀▀▀  ▀

# *added*
execute_process(
  COMMAND nproc
  OUTPUT_VARIABLE DOCKER_CPUS
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

if (NOT DOCKER_CPUS)
  include(ProcessorCount)
  ProcessorCount(DOCKER_CPUS)
endif()

# Ensure DOCKER_CPUS is at least 2 for server client test
if (DOCKER_CPUS LESS 2)
  set(DOCKER_CPUS 2)
endif()

set(IBLGF_MPI_NP "${DOCKER_CPUS}" CACHE STRING
    "MPI ranks for tests (minimum 2)")
# *end of added*

set(server_client_test server_client)

foreach (item ${server_client_test})
    add_executable(${item} ${item}.cpp ${SOURCES})
    target_link_libraries(${item} iblgf_test_main)
    add_test(
        NAME ${item}
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${IBLGF_MPI_NP} ${MPIEXEC_PREFLAGS} $<TARGET_FILE:${item}>
        #COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 8 ${MPIEXEC_PREFLAGS} $<TARGET_FILE:${item}>
    )
endforeach()
