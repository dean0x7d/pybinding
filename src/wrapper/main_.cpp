#include <thread>
#include <boost/python/module.hpp>
#include <boost/python/def.hpp>
#include <boost/python/class.hpp>
#include "converters/eigen3.hpp"
#include "python_support.hpp"
using namespace boost::python;

void export_core();
void export_system();
void export_solver();
void export_greens();
void export_modifiers();
void export_results();
void export_parallel_sweep();

BOOST_PYTHON_MODULE(_pybinding)
{
    // init numpy and register converters
    import_array1();
    eigen3_numpy_register_type<ArrayXf>();
    eigen3_numpy_register_type<ArrayXd>();
    eigen3_numpy_register_type<ArrayXcf>();
    eigen3_numpy_register_type<ArrayX<short>>();
    eigen3_numpy_register_type<Cartesian>();
    eigen3_numpy_register_type<Index3D>();
    to_python_converter<DenseURef, denseuref_to_python, true>{};

    // sparse matrix class
    class_<SparseURef> {"SparseURef", no_init}
    .add_property("rows", &SparseURef::rows)
    .add_property("cols", &SparseURef::cols)
    .add_property("inner_indices", internal_ref(&SparseURef::inner_indices))
    .add_property("outer_starts", internal_ref(&SparseURef::outer_starts))
    .add_property("values", internal_ref(&SparseURef::values))
    ;
    
    // export all classes
    export_core();
    export_system();
    export_solver();
    export_greens();
    export_modifiers();
    export_parallel_sweep();

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
