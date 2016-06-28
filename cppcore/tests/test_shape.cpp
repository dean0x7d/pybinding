#include <catch.hpp>

#include "fixtures.hpp"
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

TEST_CASE("Shape-imposed lattice offset") {
    auto shape = shape::rectangle(2.4f, 2.4f);
    
    auto model = Model(lattice::square());
    model.set_shape(shape);
    auto const& system = *model.system();
    
    auto offset_model = Model(lattice::square());
    shape.lattice_offset = {-0.1f, 0.5f, .0f};
    offset_model.set_shape(shape);
    auto const& offset_system = *offset_model.system();
    
    REQUIRE(system.lattice.offset.isZero());
    REQUIRE(offset_system.lattice.offset.isApprox(shape.lattice_offset));
    REQUIRE(system.positions.x.minCoeff() > offset_system.positions.x.minCoeff());
    REQUIRE(system.num_sites() > offset_system.num_sites());
}
