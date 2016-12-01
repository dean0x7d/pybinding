#include <catch.hpp>
#include <complex>

#include "Model.hpp"
using namespace cpb;

namespace static_test_typelist {
    using List = TypeList<float, double, std::complex<float>, std::complex<double>>;

    static_assert(tl::AnyOf<List, float>::value, "");
    static_assert(!tl::AnyOf<List, int>::value, "");
}

TEST_CASE("Nonzeros per row of a triangular hopping matrix") {
    SparseMatrixX<hop_id> sm(5, 5);
    sm.insert(0, 3) = 1;
    sm.insert(0, 4) = 1;
    sm.insert(2, 0) = 1;
    sm.makeCompressed();

    auto expected0 = ArrayXi(sm.rows());
    expected0 << 3, 0, 1, 1, 1;
    REQUIRE((nonzeros_per_row(sm, false) == expected0).all());

    auto expected1 = ArrayXi(sm.rows());
    expected1 << 4, 1, 2, 2, 2;
    REQUIRE((nonzeros_per_row(sm, true) == expected1).all());
}

TEST_CASE("Symmetry masks") {
    SECTION("1") {
        auto const masks = detail::make_masks({true, false, false}, 1);
        auto const expected = std::vector<Index3D>{{0, 0, 0}, {1, 0, 0}};
        REQUIRE(masks == expected);
    }
    SECTION("2-1") {
        auto const masks = detail::make_masks({false, true, false}, 2);
        auto const expected = std::vector<Index3D>{{0, 0, 0}, {0, 1, 0}};
        REQUIRE(masks == expected);
    }
    SECTION("2-2") {
        auto const masks = detail::make_masks({true, true, false}, 2);
        auto const expected = std::vector<Index3D>{{0, 0, 0}, {0, 1, 0}, {1, 0, 0}, {1, 1, 0}};
        REQUIRE(masks == expected);
    }
    SECTION("3") {
        auto const masks = detail::make_masks({true, true, true}, 3);
        auto const expected = std::vector<Index3D>{
            {0, 0, 0}, {0, 0, 1}, {0, 1, 0}, {0, 1, 1},
            {1, 0, 0}, {1, 0, 1}, {1, 1, 0}, {1, 1, 1}
        };
        REQUIRE(masks == expected);
    }
}
