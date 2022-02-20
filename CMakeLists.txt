cmake_minimum_required(VERSION 3.5)
project(pybinding CXX)

set(CMAKE_CXX_STANDARD 11)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build" FORCE)
endif()

find_program(ccache_found ccache)
if(ccache_found)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
endif()

add_subdirectory(cppcore)
add_subdirectory(cppmodule)

if(EXISTS ${CMAKE_SOURCE_DIR}/docs)
    add_subdirectory(docs EXCLUDE_FROM_ALL)
endif()

if(TARGET catch)
    add_custom_target(tests COMMAND $<TARGET_FILE:catch> COMMAND ${PYTHON_EXECUTABLE} -m pytest
                      DEPENDS _pybinding WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
endif()
