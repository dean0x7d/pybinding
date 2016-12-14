set(PYBIND11_CPP_STANDARD ${PB_CPP_STANDARD} CACHE STRING "Match core library")
set(PYBIND11_PYTHON_VERSION 3.4 CACHE STRING "Minimum required Python version")
add_subdirectory(deps/pybind11 EXCLUDE_FROM_ALL)

pybind11_add_module(_pybinding
    include/cast.hpp
    include/resolve.hpp
    include/wrappers.hpp
    src/kpm.cpp
    src/lattice.cpp
    src/leads.cpp
    src/main.cpp
    src/model.cpp
    src/modifiers.cpp
    src/parallel.cpp
    src/shape.cpp
    src/solver.cpp
    src/system.cpp
    src/wrapper_tests.cpp
)
set_higher_warning_level(_pybinding)
target_include_directories(_pybinding PRIVATE include)

target_link_libraries(_pybinding PRIVATE pybinding_cppcore)
if(NOT MSVC)  # match default visibility of core library and extension module
    target_compile_options(pybinding_cppcore PUBLIC -fvisibility=hidden)
endif()

if(NOT CMAKE_LIBRARY_OUTPUT_DIRECTORY)
    set_target_properties(_pybinding PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})
    foreach(config ${CMAKE_CONFIGURATION_TYPES})
        string(TOUPPER ${config} config)
        set_target_properties(_pybinding PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY_${config} ${CMAKE_SOURCE_DIR})
    endforeach()
endif()

add_custom_target(pytest COMMAND ${PYTHON_EXECUTABLE} -m pytest DEPENDS _pybinding
                  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
