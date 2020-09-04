## DOXYGEN
find_package(Doxygen REQUIRED)

# some doxygen related variables 
set(DOXYGEN_INPUT_DIR ${PROJECT_SOURCE_DIR}/src) # directory where headers are extracted
set(DOXYGEN_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/docs/doxygen) # doxygen build result path
set(DOXYGEN_INDEX_FILE ${DOXYGEN_OUTPUT_DIR}/html/index.html) # doc entry file 
set(DOXYFILE_IN ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile)
set(DOXYFILE_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

configure_file(${DOXYFILE_IN} ${DOXYFILE_OUT} @ONLY)

file(MAKE_DIRECTORY ${DOXYGEN_OUTPUT_DIR})
add_custom_command(OUTPUT ${DOXYGEN_INDEX_FILE}
                COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYFILE_OUT}
                DEPENDS ${IBLGF_AMR_HEADER_DIR}
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                MAIN_DEPENDENCY ${DOXYFILE_OUT} ${DOXYFILE_IN}
                COMMENT "Generating docs")

## Sphinx 
find_package(Python REQUIRED)
find_package(Sphinx REQUIRED)

set(SPHINX_SOURCE ${CMAKE_CURRENT_SOURCE_DIR})
set(SPHINX_BUILD ${CMAKE_CURRENT_BINARY_DIR}/docs/sphinx)
set(SPHINX_INDEX_FILE ${SPHINX_BUILD}/index.html)

add_custom_target(Sphinx ALL
                COMMAND
                ${SPHINX_EXECUTABLE} -b html
                    -Dbreathe_projects.iblgf=${DOXYGEN_OUTPUT_DIR}/xml
                ${SPHINX_SOURCE} ${SPHINX_BUILD}
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                DEPENDS 
                    ${CMAKE_CURRENT_SOURCE_DIR}/index.rst
                    ${DOXYGEN_INDEX_FILE}
                    ${SPHINX_SOURCE}/conf.py
                COMMENT "Generating documentation with Sphinx")

