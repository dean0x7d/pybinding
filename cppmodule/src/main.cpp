#include "wrappers.hpp"
#include "support/simd.hpp"
#ifdef CPB_USE_MKL
# include <mkl.h>
#endif
using namespace cpb;


PYBIND11_MODULE(_pybinding, m) {
    wrap_greens(m);
    wrap_lattice(m);
    wrap_leads(m);
    wrap_model(m);
    wrap_modifiers(m);
    wrap_parallel(m);
    wrap_shape(m);
    wrap_solver(m);
    wrap_system(m);

    wrapper_tests(m);

    m.def("simd_info", []() -> std::string {
#if SIMDPP_USE_AVX
        auto const bits = std::to_string(simd::basic_traits::register_size_bytes * 8);
        return (SIMDPP_USE_AVX2 ? "AVX2-" : "AVX-") + bits;
#elif SIMDPP_USE_SSE3
        return "SSE3";
#elif SIMDPP_USE_SSE2
        return "SSE2";
#else
        return "x87";
#endif
    });

#ifdef CPB_USE_MKL
    m.def("get_max_threads", MKL_Get_Max_Threads,
          "Get the maximum number of MKL threads. (<= logical theads)");
    m.def("set_num_threads", MKL_Set_Num_Threads, "number"_a, "Set the number of MKL threads.");
    m.def("get_max_cpu_frequency", MKL_Get_Max_Cpu_Frequency);
    m.def("get_cpu_frequency", MKL_Get_Cpu_Frequency);
    // The max threads count may change later but at init time it's usually
    // equal to the physical core count which is useful information.
    m.attr("physical_core_count") = MKL_Get_Max_Threads();
#endif

#ifdef CPB_VERSION
    m.attr("__version__") = CPB_VERSION;
#else
    m.attr("__version__") = "dev";
#endif
}
