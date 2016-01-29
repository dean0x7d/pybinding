#include "eigen3_converters.hpp"
#include "python_support.hpp"

#include <boost/python/module.hpp>
#include <boost/python/def.hpp>
#include <boost/python/class.hpp>
#include <boost/python/tuple.hpp>

using namespace boost::python;
using namespace tbm;

void export_core();
void export_system();
void export_solver();
void export_greens();
void export_modifiers();
void export_parallel();

BOOST_PYTHON_MODULE(_pybinding) {
    // init numpy and register converters
    import_array1(void());
    eigen3_numpy_register_type<ArrayXf>();
    eigen3_numpy_register_type<ArrayXd>();
    eigen3_numpy_register_type<ArrayXcf>();
    eigen3_numpy_register_type<ArrayX<bool>>();
    eigen3_numpy_register_type<Cartesian>();
    eigen3_numpy_register_type<Index3D>();

    register_arrayref_converter<ArrayRef>();
    register_arrayref_converter<RealArrayRef>();
    register_arrayref_converter<ComplexArrayRef>();

    class_<CartesianArray>{
        "CartesianArray",
        init<ArrayXf const&, ArrayXf const&, ArrayXf const&>{args("self", "x", "y", "z")}
    }
    .add_property("x", dense_uref(&CartesianArray::x), &CartesianArray::x)
    .add_property("y", dense_uref(&CartesianArray::y), &CartesianArray::y)
    .add_property("z", dense_uref(&CartesianArray::z), &CartesianArray::z)
    ;

    class_<SparseURef> {"SparseURef", no_init}
    .add_property("shape", [](SparseURef const& s) { return make_tuple(s.rows, s.cols); })
    .add_property("data", internal_ref(&SparseURef::values))
    .add_property("indices", internal_ref(&SparseURef::inner_indices))
    .add_property("indptr", internal_ref(&SparseURef::outer_starts))
    ;
    
    // export all classes
    export_core();
    export_system();
    export_solver();
    export_greens();
    export_modifiers();
    export_parallel();

#ifdef TBM_USE_MKL
    // export some helper functions
    def("get_max_threads", MKL_Get_Max_Threads,
        "Get the maximum number of MKL threads. (<= logical theads)");
    def("set_num_threads", MKL_Set_Num_Threads, arg("number"),
        "Set the number of MKL threads.");
    def("get_max_cpu_frequency", MKL_Get_Max_Cpu_Frequency);
    def("get_cpu_frequency", MKL_Get_Cpu_Frequency);
#endif
}
