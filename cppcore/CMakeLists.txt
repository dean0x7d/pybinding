cmake_minimum_required(VERSION 3.0)
project(pybinding_cppcore)

option(PB_WERROR "Make all warnings into errors" OFF)
option(PB_TESTS "Enable testing" ON)
option(PB_NATIVE_SIMD "Enable all instruction sets supported by the local machine" ON)
option(PB_MKL "Use Intel's Math Kernel Library" OFF)
option(PB_CUDA "Enable compilation of components written in CUDA" OFF)

add_library(pybinding_cppcore
    include/compute/eigen3/lanczos.hpp
    include/compute/eigen3/linear_algebra.hpp
    include/compute/mkl/lanczos.hpp
    include/compute/mkl/linear_algebra.hpp
    include/compute/mkl/wrapper.hpp
    include/compute/detail.hpp
    include/compute/kernel_polynomial.hpp
    include/compute/lanczos.hpp
    include/compute/linear_algebra.hpp
    include/detail/config.hpp
    include/detail/macros.hpp
    include/detail/slice.hpp
    include/detail/strategy.hpp
    include/detail/thread.hpp
    include/detail/typelist.hpp
    include/greens/kpm/Bounds.hpp
    include/greens/kpm/calc_moments.hpp
    include/greens/kpm/OptimizedHamiltonian.hpp
    include/greens/kpm/OptimizedSizes.hpp
    include/greens/kpm/Moments.hpp
    include/greens/kpm/Stats.hpp
    include/greens/Greens.hpp
    include/greens/KPM.hpp
    include/hamiltonian/Hamiltonian.hpp
    include/hamiltonian/HamiltonianModifiers.hpp
    include/leads/HamiltonianPair.hpp
    include/leads/Leads.hpp
    include/leads/Spec.hpp
    include/leads/Structure.hpp
    include/numeric/arrayref.hpp
    include/numeric/constant.hpp
    include/numeric/dense.hpp
    include/numeric/ellmatrix.hpp
    include/numeric/random.hpp
    include/numeric/sparse.hpp
    include/numeric/sparseref.hpp
    include/numeric/traits.hpp
    include/solver/FEAST.hpp
    include/solver/Solver.hpp
    include/support/cppfuture.hpp
    include/support/format.hpp
    include/support/simd.hpp
    include/support/thrust.hpp
    include/support/variant.hpp
    include/system/Foundation.hpp
    include/system/Lattice.hpp
    include/system/Shape.hpp
    include/system/Symmetry.hpp
    include/system/System.hpp
    include/system/Generators.hpp
    include/system/SystemModifiers.hpp
    include/utils/Chrono.hpp
    include/Model.hpp
    src/greens/kpm/Bounds.cpp
    src/greens/kpm/OptimizedHamiltonian.cpp
    src/greens/Greens.cpp
    src/greens/KPM.cpp
    src/hamiltonian/Hamiltonian.cpp
    src/hamiltonian/HamiltonianModifiers.cpp
    src/leads/Leads.cpp
    src/leads/Spec.cpp
    src/leads/Structure.cpp
    src/solver/FEAST.cpp
    src/solver/Solver.cpp
    src/system/Foundation.cpp
    src/system/Lattice.cpp
    src/system/Shape.cpp
    src/system/Symmetry.cpp
    src/system/System.cpp
    src/system/SystemModifiers.cpp
    src/utils/Chrono.cpp
    src/Model.cpp
)

target_include_directories(pybinding_cppcore PUBLIC include)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
include(download)
download_dependency(eigen3 3.2.8
                    https://bitbucket.org/eigen/eigen/get
                    \${VERSION}.tar.gz */Eigen)
target_include_directories(pybinding_cppcore SYSTEM PUBLIC ${EIGEN3_INCLUDE_DIR})
download_dependency(variant 1.1.0
                    https://raw.githubusercontent.com/mapbox/variant/v\${VERSION}
                    variant.hpp recursive_wrapper.hpp)
target_include_directories(pybinding_cppcore SYSTEM PUBLIC ${VARIANT_INCLUDE_DIR})
download_dependency(simdpp 2.0-rc2
                    https://github.com/p12tic/libsimdpp/archive
                    /v\${VERSION}.tar.gz */simdpp)
target_include_directories(pybinding_cppcore SYSTEM PUBLIC ${SIMDPP_INCLUDE_DIR})

include(cppformat)
target_link_libraries(pybinding_cppcore PUBLIC cppformat)

if(NOT WIN32)
    target_compile_options(pybinding_cppcore PUBLIC -std=c++11)
endif()

include(warnings)
set_higher_warning_level(pybinding_cppcore)

if(PB_NATIVE_SIMD AND NOT MSVC) # MSVC does not have anything like a /arch:native flag
    target_compile_options(pybinding_cppcore PUBLIC -march=native)
endif()

if(PB_MKL)
    include(mkl)
    target_link_mkl(pybinding_cppcore PUBLIC)
    target_compile_definitions(pybinding_cppcore PUBLIC CPB_USE_MKL)
endif()

if(PB_CUDA)
    find_package(CUDA REQUIRED)
    set(CUDA_NVCC_FLAGS "--std c++11;${CUDA_NVCC_FLAGS}")
    cuda_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

    cuda_add_library(pybinding_cuda src/greens/kpm/calc_moments.cu)
    target_link_libraries(pybinding_cppcore PUBLIC pybinding_cuda ${CUDA_cublas_LIBRARY})
    target_compile_definitions(pybinding_cppcore PUBLIC CPB_USE_CUDA)
    target_include_directories(pybinding_cppcore SYSTEM PUBLIC ${CUDA_INCLUDE_DIRS})
endif()

if(PB_TESTS)
    set(catch_url https://raw.githubusercontent.com/philsquared/Catch/v\${VERSION}/single_include)
    download_dependency(catch 1.4.0 ${catch_url} catch.hpp)
    add_subdirectory(tests)
endif()
