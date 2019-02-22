# CMake generated Testfile for 
# Source directory: /home/mlee/IBLGF-AMR-parallel-new
# Build directory: /home/mlee/IBLGF-AMR-parallel-new/builds
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(DomainTest "/usr/bin/mpirun.openmpi" "-np" "2" "bin/decompTest.x" "/home/mlee/IBLGF-AMR-parallel-new/tests/domain_decomposition/configFile")
subdirs("tests/sever_client")
subdirs("tests/domain_decomposition")
subdirs("setups/poissonProblem")
