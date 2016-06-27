#include <catch.hpp>
#include <complex>

#include "detail/typelist.hpp"
using namespace cpb;

namespace static_test_typelist {
    using List = TypeList<float, double, std::complex<float>, std::complex<double>>;

    static_assert(tl::AnyOf<List, float>::value, "");
    static_assert(!tl::AnyOf<List, int>::value, "");
}
