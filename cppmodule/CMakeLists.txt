add_subdirectory(deps/pybind11 EXCLUDE_FROM_ALL)

pybind11_add_module(_pybinding2
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
set_higher_warning_level(_pybinding2)
target_include_directories(_pybinding2 PRIVATE include)

target_link_libraries(_pybinding2 PUBLIC pybinding_cppcore)
