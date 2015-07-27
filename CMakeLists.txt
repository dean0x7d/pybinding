cmake_minimum_required(VERSION 3.0)
project(pybinding)

# Python 3.x is required, 3.4 recommended
set(PYTHON_VERSION 3)
add_subdirectory(dependencies/boost_python)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
find_package(NumPy REQUIRED)
find_package(Eigen3orDownload 3.2.4 REQUIRED)

boost_python_add_module(_pybinding
    src/Model.cpp
    src/greens/Greens.cpp
    src/greens/KPM.cpp
    src/hamiltonian/Hamiltonian.cpp
    src/hamiltonian/HamiltonianModifiers.cpp
    src/solver/Solver.cpp
    src/solver/FEAST.cpp
    src/system/Foundation.cpp
    src/system/Lattice.cpp
    src/system/Shape.cpp
    src/system/Symmetry.cpp
    src/system/System.cpp
    src/system/SystemModifiers.cpp
    src/utils/Chrono.cpp
    src/utils/Log.cpp
    src/wrapper/greens_.cpp
    src/wrapper/hamiltonian_.cpp
    src/wrapper/main_.cpp
    src/wrapper/model_.cpp
    src/wrapper/parallel_.cpp
    src/wrapper/solver_.cpp
    src/wrapper/system_.cpp
)

target_include_directories(_pybinding
    PUBLIC include
    PRIVATE ${NUMPY_INCLUDE_DIRS}
    PRIVATE ${EIGEN3_INCLUDE_DIR}
)

target_compile_options(_pybinding PUBLIC -std=c++11)

if(${CMAKE_CXX_COMPILER_ID} STREQUAL Clang)
elseif(${CMAKE_CXX_COMPILER_ID} STREQUAL GNU)
elseif(${CMAKE_CXX_COMPILER_ID} STREQUAL Intel)
    target_compile_options(_pybinding PRIVATE -xHost)
    find_package(MKL REQUIRED)
    if(MKL_INCLUDE_DIR)
        target_compile_definitions(_pybinding PRIVATE TBM_USE_MKL)
        target_compile_options(_pybinding PRIVATE -mkl=parallel)
        target_link_libraries(_pybinding -mkl=parallel)
    endif()
elseif(${CMAKE_CXX_COMPILER_ID} STREQUAL MSVC)
endif()

if(NOT CMAKE_LIBRARY_OUTPUT_DIRECTORY)
    set_target_properties(_pybinding PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR})
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build" FORCE)
endif()

# temporary evil
file(GLOB_RECURSE headers include/*.hpp src/wrapper/*.hpp)
add_custom_target(_header_dummy SOURCES ${headers})
