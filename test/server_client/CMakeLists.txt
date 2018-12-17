cmake_minimum_required (VERSION 2.6)
if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/CMakeLists.txt.user")
    include(${CMAKE_CURRENT_LIST_DIR}/CMakeLists.txt.user)
else()
endif()
project (ServerClient)
 
#Bring the headers, such as Student.h into the project
include_directories(include)
 
#Can manually add the sources using the set command as follows:
#set(SOURCES src/main.cpp )
 
#However, the file(GLOB...) allows for wildcard additions:
file(GLOB SOURCES "src/*.cpp")

#Output compile commands
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
 
# add the binary tree to the search path for include files
# so that we will find TutorialConfig.h
include_directories("${PROJECT_BINARY_DIR}")

#include(${CMAKE_CURRENT_SOURCE_DIR}/../CMakeLists.txt.user)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -Wall -pedantic -O3")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)
#LIBS

find_package(MPI REQUIRED)
include_directories(SYSTEM ${MPI_INCLUDE_PATH})
link_directories(${MPI_LIBRARIES} )

find_package(Boost COMPONENTS serialization mpi REQUIRED)
include_directories(${Boost_INCLUDE_DIR})
link_directories(${Boost_LIBRARY_DIR})

# add the executable
add_executable(project.x ${SOURCES})
target_link_libraries(project.x ${Boost_LIBRARIES} boost_mpi ${MPI_LIBRARIES} )
