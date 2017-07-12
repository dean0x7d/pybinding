#include <catch.hpp>

#include "fixtures.hpp"
using namespace cpb;

TEST_CASE("Primitive") {
    REQUIRE_THROWS_WITH(Model(lattice::square(), Primitive(2, 2, 2)),
                        Catch::Contains("more dimensions than the lattice"));
}

TEST_CASE("FreeformShape", "[shape]") {
    auto const contains = [](CartesianArrayConstRef p) -> ArrayX<bool> { return p.x() > 0.5f; };
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
    
    auto const model = Model(lattice::square(), shape);
    auto const& system = *model.system();

    shape.lattice_offset = {-0.1f, 0.5f, .0f};
    auto const offset_model = Model(lattice::square(), shape);
    auto const& offset_system = *offset_model.system();
    
    REQUIRE(system.lattice.get_offset().isZero());
    REQUIRE(offset_system.lattice.get_offset().isApprox(shape.lattice_offset));
    REQUIRE(system.positions.x.minCoeff() > offset_system.positions.x.minCoeff());
    REQUIRE(system.num_sites() > offset_system.num_sites());
}
