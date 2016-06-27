#include "system/Lattice.hpp"

#include "support/simd.hpp"

#include "eigen3_converters.hpp"
#include "python_support.hpp"

#include <boost/python/module.hpp>
#include <boost/python/def.hpp>
#include <boost/python/class.hpp>
#include <boost/python/docstring_options.hpp>

using namespace boost::python;
using namespace cpb;

void export_core();
void export_system();
void export_leads();
void export_solver();
void export_greens();
void export_modifiers();
void export_parallel();

BOOST_PYTHON_MODULE(_pybinding) {
    docstring_options doc_options;
    doc_options.disable_cpp_signatures();

    // init numpy and register converters
    import_array1(void());
    eigen3_numpy_register_type<ArrayXf>();
    eigen3_numpy_register_type<ArrayXd>();
    eigen3_numpy_register_type<ArrayXcf>();
    eigen3_numpy_register_type<ArrayXcd>();
    eigen3_numpy_register_type<ArrayXi>();
    eigen3_numpy_register_type<ArrayX<bool>>();
    eigen3_numpy_register_type<ArrayX<hop_id>>();
    eigen3_numpy_register_type<Cartesian>();
    eigen3_numpy_register_type<Index3D>();

    register_arrayref_converter<ArrayConstRef>();
    register_arrayref_converter<RealArrayConstRef>();
    register_arrayref_converter<ComplexArrayConstRef>();
    register_arrayref_converter<ArrayRef>();
    register_arrayref_converter<RealArrayRef>();
    register_arrayref_converter<ComplexArrayRef>();

    register_csr_converter<hop_id>();
    register_csrref_converter<ComplexCsrConstRef>();

    class_<CartesianArray, noncopyable>{
        "CartesianArray",
        init<ArrayXf const&, ArrayXf const&, ArrayXf const&>(args("self", "x", "y", "z"))
    }
    .add_property("x", return_arrayref(&CartesianArray::x), &CartesianArray::x)
    .add_property("y", return_arrayref(&CartesianArray::y), &CartesianArray::y)
    .add_property("z", return_arrayref(&CartesianArray::z), &CartesianArray::z)
    .enable_pickling()
    .def("__getinitargs__", [](object o) {
        return make_tuple(o.attr("x"), o.attr("y"), o.attr("z"));
    })
    ;

    // export all classes
    export_core();
    export_system();
    export_leads();
    export_solver();
    export_greens();
    export_modifiers();
    export_parallel();

#ifdef CPB_USE_MKL
    def("get_max_threads", MKL_Get_Max_Threads,
        "Get the maximum number of MKL threads. (<= logical theads)");
    def("set_num_threads", MKL_Set_Num_Threads, arg("number"),
        "Set the number of MKL threads.");
    def("get_max_cpu_frequency", MKL_Get_Max_Cpu_Frequency);
    def("get_cpu_frequency", MKL_Get_Cpu_Frequency);
    // The max threads count may change later but at init time it's usually
    // equal to the physical core count which is useful information.
    scope().attr("physical_core_count") = MKL_Get_Max_Threads();
#endif

    def("simd_info", []() -> std::string {
#if SIMDPP_USE_AVX
        auto const bits = std::to_string(simd::detail::basic_traits::size_bytes * 8);
        return "AVX-" + bits;
#elif SIMDPP_USE_SSE3
        return "SSE3";
#elif SIMDPP_USE_SSE2
        return "SSE2";
#else
        return "x87";
#endif
    });
}
