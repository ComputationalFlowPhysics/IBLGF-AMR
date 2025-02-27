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

add_subdirectory (server_client)
add_subdirectory (datafields)
add_subdirectory (tree)
add_subdirectory (poisson)
add_subdirectory (operators)
add_subdirectory (ns_amr_lgf)
add_subdirectory (ns_amr_lgf2D)
add_subdirectory (compute_mean2D)
#add_subdirectory (norm_Poisson2D)
add_subdirectory (lns_amr_lgf2D)
add_subdirectory (lns_test_petsc)
#add_subdirectory (ifherk_linear_test)
add_subdirectory (postProcessIntrp)
# add_subdirectory (ibsolve)
#add_subdirectory (helmholtz_lgf)
#add_subdirectory (helmholtz)
add_subdirectory (Poisson2D)
#add_subdirectory (helmholtz_FFT)
#add_subdirectory (operators_helmholtz)
#add_subdirectory (helmholtzFFT_operatorTest)
#add_subdirectory (operators_helmholtz_complex)
#add_subdirectory (ns_amr_lgf_helm)
#add_subdirectory (NewtonOperatorTest)
#add_subdirectory (NewtonJacobianSolveTest)
#add_subdirectory (NewtonIteration)
find_package(MKL CONFIG REQUIRED)
#add_subdirectory (MKLSolverTest)
#add_subdirectory (MKLPoissonTest)
#add_subdirectory (MKLJacobian)
# add_subdirectory (MKLNewton)
#add_subdirectory (PETscPoissonTest)
#add_subdirectory (PETscJacobian)
#add_subdirectory (SLEPcTest)
#add_subdirectory (SLEPcTestLNS)
#add_subdirectory (ScalapackDirectSolverTest)
#add_subdirectory (SLEPcTestLNS_helm)
#add_subdirectory (SLEPcTestRes)
#add_subdirectory (OMP_MPI_test)
#add_subdirectory (operators_helmholtz_complex_OMP)
#add_subdirectory (helmholtz_OMP)
#add_subdirectory (ns_amr_lgf_helm_OMP)
#add_subdirectory (PETScSpectralMatrixTest)
#add_subdirectory (TwoVortexRings)
#add_subdirectory (OscillatingAirfoil)
#add_subdirectory (operators_linear)
#add_subdirectory (op_test2)
add_subdirectory (QR_2D)


