## DOXYGEN
find_package(Doxygen REQUIRED)

# Find all headers we wish to document 
get_target_property(IBLGF_AMR_HEADER_DIR src INTERFACE_INCLUDE_DIRECTORIES)
file(GLOB_RECURSE IBLGF_AMR_HEADER_DIR ${IBLGF_AMR_HEADER_DIR}/*.hpp)

# some doxygen related variables 
set(DOXYGEN_INPUT_DIR ${PROJECT_SOURCE_DIR}/src) # directory where headers are extracted
set(DOXYGEN_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/docs/doxygen) # doxygen build result path
set(DOXYGEN_INDEX_FILE ${DOXYGEN_OUTPUT_DIR}/html/index.html) # doc entry file 
set(DOXYFILE_IN ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile)
set(DOXYFILE_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

configure_file(${DOXYFILE_IN} ${DOXYFILE_OUT} @ONLY)

file(MAKE_DIRECTORY ${DOXYGEN_OUTPUT_DIR})
add_custom_command(OUTPUT ${DOXYGEN_INDEX_FILE}
                DEPENDS ${IBLGF_AMR_HEADER_DIR}
                COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYFILE_OUT}
                MAIN_DEPENDENCY ${DOXYFILE_OUT} ${DOXYFILE_IN}
                COMMENT "Generating docs")

add_custom_target(Doxygen ALL DEPENDS ${DOXYGEN_INDEX_FILE})