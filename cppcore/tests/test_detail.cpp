#include <catch.hpp>
#include <complex>

#include "Model.hpp"
#include "detail/algorithm.hpp"
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

TEST_CASE("sliced") {
    auto const v = []{
        auto result = std::vector<int>(10);
        std::iota(result.begin(), result.end(), 0);
        return result;
    }();
    REQUIRE_THAT(v, Catch::Equals(std::vector<int>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}));

    auto vectors = std::vector<std::vector<int>>();
    for (auto const& slice : sliced(v, 3)) {
        auto tmp = std::vector<int>();
        std::copy(slice.begin(), slice.end(), std::back_inserter(tmp));
        vectors.push_back(tmp);
    }
    REQUIRE(vectors.size() == 4);
    REQUIRE_THAT(vectors[0], Catch::Equals(std::vector<int>{0, 1, 2}));
    REQUIRE_THAT(vectors[1], Catch::Equals(std::vector<int>{3, 4, 5}));
    REQUIRE_THAT(vectors[2], Catch::Equals(std::vector<int>{6, 7, 8}));
    REQUIRE_THAT(vectors[3], Catch::Equals(std::vector<int>{9}));
}
