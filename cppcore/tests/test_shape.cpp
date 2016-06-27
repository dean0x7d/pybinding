#include <catch.hpp>

#include "system/Shape.hpp"
using namespace cpb;

TEST_CASE("FreeformShape", "[shape]") {
    auto const contains = [](CartesianArray const& p) -> ArrayX<bool> { return p.x > 0.5f; };
    auto const shape = FreeformShape(contains, {1, 1, 1}, {0.5f, 0.5f, 0.5f});

    SECTION("Bounding box") {
        auto expected_vertices = Shape::Vertices{
            {0, 0, 0},
            {1, 0, 0},
            {0, 1, 0},
            {1, 1, 0},
            {0, 0, 1},
            {1, 0, 1},
            {0, 1, 1},
            {1, 1, 1},
        };
        REQUIRE(shape.vertices == expected_vertices);
    }

    SECTION("Contains") {
        auto const size = 4;
        auto const v = ArrayXf::LinSpaced(size, 0, 1).eval();
        auto const p = CartesianArray(v, v, v);
        auto expected = ArrayX<bool>(size);
        expected << false, false, true, true;
        REQUIRE(all_of(shape.contains(p) == expected));
    }
}
