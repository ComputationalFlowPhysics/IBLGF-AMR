ADD_LIBRARY(SpatialDomain
bitmasks.h
key.h
leaf.h
linearTree.cpp
types.h
utils.h
vector.h)

INSTALL(TARGETS SpatialDomain
        DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
