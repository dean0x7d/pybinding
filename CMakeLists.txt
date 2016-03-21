cmake_minimum_required(VERSION 3.0)
project(pybinding)

if(NOT PYTHON_VERSION)
    set(PYTHON_VERSION 3.4)  # minimum required version
endif()
add_subdirectory(cppwrapper)

if(NOT CMAKE_LIBRARY_OUTPUT_DIRECTORY)
    set_target_properties(_pybinding PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR})
    foreach(config ${CMAKE_CONFIGURATION_TYPES})
        string(TOUPPER ${config} config)
        set_target_properties(_pybinding PROPERTIES 
                              LIBRARY_OUTPUT_DIRECTORY_${config} ${CMAKE_CURRENT_LIST_DIR})
    endforeach()
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build" FORCE)
endif()

if(EXISTS ${CMAKE_SOURCE_DIR}/docs)
    add_subdirectory(docs EXCLUDE_FROM_ALL)
endif()

add_custom_target(pytest COMMAND py.test \${ARGS} DEPENDS _pybinding
                  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

if(TARGET catch)
    add_custom_target(tests COMMAND $<TARGET_FILE:catch> COMMAND py.test
                      DEPENDS _pybinding WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
endif()
