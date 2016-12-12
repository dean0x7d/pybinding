find_package(CUDA REQUIRED)

set(CUDA_NVCC_FLAGS "--std c++11;-Wno-deprecated-gpu-targets;${CUDA_NVCC_FLAGS}")

cuda_add_library(pybinding_cuda
    kpm/calc_moments.hpp
    kpm/calc_moments.cu
    thrust.hpp
    traits.cuh
)

target_include_directories(pybinding_cuda PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include
                                          PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/..)
cuda_include_directories($<TARGET_PROPERTY:pybinding_cuda,INCLUDE_DIRECTORIES>)

target_compile_definitions(pybinding_cuda PUBLIC CPB_USE_CUDA)
target_include_directories(pybinding_cuda SYSTEM PUBLIC ${CUDA_INCLUDE_DIRS})
