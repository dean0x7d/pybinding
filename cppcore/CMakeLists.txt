cmake_minimum_required(VERSION 3.0)
project(pybinding_cppcore)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
find_package(Eigen3orDownload 3.2.7 REQUIRED)

add_library(pybinding_cppcore
    include/compute/eigen3/kernel_polynomial.hpp
    include/compute/eigen3/lanczos.hpp
    include/compute/eigen3/linear_algebra.hpp
    include/compute/mkl/kernel_polynomial.hpp
    include/compute/mkl/lanczos.hpp
    include/compute/mkl/linear_algebra.hpp
    include/compute/kernel_polynomial.hpp
    include/compute/lanczos.hpp
    include/compute/linear_algebra.hpp
    include/detail/slice.hpp
    include/detail/strategy.hpp
    include/detail/typelist.hpp
    include/greens/Greens.hpp
    include/greens/KPM.hpp
    include/hamiltonian/Hamiltonian.hpp
    include/hamiltonian/HamiltonianModifiers.hpp
    include/solver/FEAST.hpp
    include/solver/Solver.hpp
    include/support/arrayref.hpp
    include/support/config.hpp
    include/support/cpp14.hpp
    include/support/dense.hpp
    include/support/format.hpp
    include/support/physics.hpp
    include/support/sparse.hpp
    include/support/sparseref.hpp
    include/support/thread.hpp
    include/support/traits.hpp
    include/system/Foundation.hpp
    include/system/Lattice.hpp
    include/system/Lead.hpp
    include/system/Shape.hpp
    include/system/Symmetry.hpp
    include/system/System.hpp
    include/system/SystemModifiers.hpp
    include/utils/Chrono.hpp
    include/utils/Log.hpp
    include/Model.hpp
    src/greens/Greens.cpp
    src/greens/KPM.cpp
    src/hamiltonian/Hamiltonian.cpp
    src/hamiltonian/HamiltonianModifiers.cpp
    src/solver/FEAST.cpp
    src/solver/Solver.cpp
    src/system/Foundation.cpp
    src/system/Lattice.cpp
    src/system/Lead.cpp
    src/system/Shape.cpp
    src/system/Symmetry.cpp
    src/system/System.cpp
    src/system/SystemModifiers.cpp
    src/utils/Chrono.cpp
    src/utils/Log.cpp
    src/Model.cpp
)

target_include_directories(pybinding_cppcore PUBLIC include)
target_include_directories(pybinding_cppcore SYSTEM PUBLIC ${EIGEN3_INCLUDE_DIR})

if(NOT WIN32)
    target_compile_options(pybinding_cppcore PUBLIC -std=c++11)
endif()

if(${CMAKE_CXX_COMPILER_ID} STREQUAL Clang)
elseif(${CMAKE_CXX_COMPILER_ID} STREQUAL GNU)
elseif(${CMAKE_CXX_COMPILER_ID} STREQUAL Intel)
    target_compile_options(pybinding_cppcore PUBLIC -xHost)
    find_package(MKL REQUIRED)
    if(MKL_INCLUDE_DIR)
        target_compile_definitions(pybinding_cppcore PUBLIC TBM_USE_MKL)
        target_compile_options(pybinding_cppcore PUBLIC -mkl=parallel)
        target_link_libraries(pybinding_cppcore PUBLIC -mkl=parallel)
    endif()
elseif(${CMAKE_CXX_COMPILER_ID} STREQUAL MSVC)
endif()


option(PB_WERROR "Make all warnings into errors" OFF)

function(set_higher_warning_level TARGET)
    if(WIN32)
        target_compile_options(${TARGET} PRIVATE /W3)
    else()
        target_compile_options(${TARGET} PRIVATE -Wall -Wextra)
    endif()

    if(PB_WERROR OR DEFINED ENV{PB_WERROR})
        if(WIN32)
            target_compile_options(${TARGET} PRIVATE /WX)
        else()
            target_compile_options(${TARGET} PRIVATE -Werror)
        endif()
    endif()
endfunction()

set_higher_warning_level(pybinding_cppcore)
