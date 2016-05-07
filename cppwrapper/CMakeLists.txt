cmake_minimum_required(VERSION 3.0)
project(pybinding_cppwrapper)

# project to be wrapped
add_subdirectory(../cppcore ../cppcore)
# wrap using boost_python
add_subdirectory(deps/boost_python EXCLUDE_FROM_ALL)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
find_package(NumPy REQUIRED)

boost_python_add_module(_pybinding
    include/eigen3_converters.hpp
    include/python_support.hpp
    src/greens.cpp
    src/hamiltonian.cpp
    src/leads.cpp
    src/main.cpp
    src/model.cpp
    src/parallel.cpp
    src/solver.cpp
    src/system.cpp
)

target_include_directories(_pybinding PRIVATE include)
target_include_directories(_pybinding SYSTEM PRIVATE ${NUMPY_INCLUDE_DIRS})

set_target_properties(pybinding_cppcore PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
target_link_libraries(_pybinding PUBLIC pybinding_cppcore)

set_higher_warning_level(_pybinding)
