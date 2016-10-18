set(PYBIND11_CPP_STANDARD -std=c++11)
add_subdirectory(deps/pybind11 EXCLUDE_FROM_ALL)

pybind11_add_module(_pybinding
    include/cast.hpp
    include/wrappers.hpp
    src/greens.cpp
    src/lattice.cpp
    src/leads.cpp
    src/main.cpp
    src/model.cpp
    src/modifiers.cpp
    src/parallel.cpp
    src/shape.cpp
    src/solver.cpp
    src/system.cpp
)
set_higher_warning_level(_pybinding)
target_include_directories(_pybinding PRIVATE include)

target_link_libraries(_pybinding PRIVATE pybinding_cppcore)
if(NOT MSVC)  # match default visibility of core library and extension module
    target_compile_options(pybinding_cppcore PUBLIC -fvisibility=hidden)
endif()
