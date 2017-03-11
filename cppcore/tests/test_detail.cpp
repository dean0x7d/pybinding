#include <catch.hpp>
#include <complex>

#include "Model.hpp"
using namespace cpb;

namespace static_test_typelist {
    using List = TypeList<float, double, std::complex<float>, std::complex<double>>;

    static_assert(tl::AnyOf<List, float>::value, "");
    static_assert(!tl::AnyOf<List, int>::value, "");
}

TEST_CASE("Symmetry masks") {
    SECTION("1") {
        auto const masks = detail::make_masks({true, false, false}, 1);
        REQUIRE_THAT(masks, Catch::Equals(std::vector<Index3D>{{0, 0, 0}, {1, 0, 0}}));
    }
    SECTION("2-1") {
        auto const masks = detail::make_masks({false, true, false}, 2);
        REQUIRE_THAT(masks, Catch::Equals(std::vector<Index3D>{{0, 0, 0}, {0, 1, 0}}));
    }
    SECTION("2-2") {
        auto const masks = detail::make_masks({true, true, false}, 2);
        REQUIRE_THAT(masks, Catch::Equals(std::vector<Index3D>{
            {0, 0, 0}, {0, 1, 0}, {1, 0, 0}, {1, 1, 0}
        }));
    }
    SECTION("3") {
        auto const masks = detail::make_masks({true, true, true}, 3);
        REQUIRE_THAT(masks, Catch::Equals(std::vector<Index3D>{
            {0, 0, 0}, {0, 0, 1}, {0, 1, 0}, {0, 1, 1},
            {1, 0, 0}, {1, 0, 1}, {1, 1, 0}, {1, 1, 1}
        }));
    }
}
