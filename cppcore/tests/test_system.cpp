#include <catch.hpp>

#include "system/Lattice.hpp"
using namespace tbm;

TEST_CASE("Sublattice", "[lattice]") {
    auto sublattice = Sublattice{};
    sublattice.add_hopping({0, 0, 0}, 0, 0, false);
    REQUIRE_THROWS(sublattice.add_hopping({0, 0, 0}, 0, 0, false));
}

TEST_CASE("Lattice", "[lattice]") {
    auto lattice = Lattice({1, 0, 0}, {0, 1, 0});
    REQUIRE(lattice.vectors.size() == 2);
    REQUIRE(lattice.vectors.capacity() == 2);
    REQUIRE(lattice.max_hoppings() == 0);

    SECTION("Add sublattices") {
        lattice.add_sublattice("A");
        REQUIRE_FALSE(lattice.has_onsite_energy);
        REQUIRE_THROWS(lattice.add_sublattice("A"));

        lattice.add_sublattice("B", {0, 0, 0}, 1.0);
        REQUIRE(lattice.has_onsite_energy);

        lattice.sublattices.resize(std::numeric_limits<sub_id>::max());
        REQUIRE_THROWS(lattice.add_sublattice("overflow"));
    }

    SECTION("Register hoppings") {
        lattice.register_hopping_energy("t1", 1.0);
        REQUIRE_FALSE(lattice.has_complex_hopping);
        REQUIRE_THROWS(lattice.register_hopping_energy("t1", 1.0));

        lattice.register_hopping_energy("t2", {0, 1.0});
        REQUIRE(lattice.has_complex_hopping);

        lattice.hopping_energies.resize(std::numeric_limits<hop_id>::max());
        REQUIRE_THROWS(lattice.register_hopping_energy("overflow", 1.0));
    }

    SECTION("Add hoppings") {
        auto const a = lattice.add_sublattice("A");
        auto const b = lattice.add_sublattice("B");
        auto const t1 = lattice.register_hopping_energy("t1", 1.0);

        REQUIRE_THROWS(lattice.add_registered_hopping({0, 0, 0}, a, a, t1));
        REQUIRE_THROWS(lattice.add_registered_hopping({0, 0, 0},  -1, a, t1));
        REQUIRE_THROWS(lattice.add_registered_hopping({0, 0, 0}, b+1, a, t1));
        REQUIRE_THROWS(lattice.add_registered_hopping({0, 0, 0}, a, a,   -1));
        REQUIRE_THROWS(lattice.add_registered_hopping({0, 0, 0}, a, a, t1+1));

        lattice.add_registered_hopping({1, 0, 0}, a, a, t1);
        REQUIRE_THROWS(lattice.add_registered_hopping({1, 0, 0}, a, a, t1));
        REQUIRE(lattice[a].hoppings[1].relative_index == Index3D(-1, 0, 0));
        REQUIRE(lattice.max_hoppings() == 2);

        lattice.add_registered_hopping({1, 0, 0}, a, b, t1);
        REQUIRE(lattice[b].hoppings[0].relative_index == Index3D(-1, 0, 0));
        REQUIRE(lattice.max_hoppings() == 3);

        lattice.add_registered_hopping({1, 0, 0}, b, b, t1);
        REQUIRE(lattice.max_hoppings() == 3);

        auto const t2 = lattice.add_hopping({1, 1, 0}, a, a, 2.0);
        REQUIRE(lattice.hopping_energies.size() == 2);
        REQUIRE(lattice.add_hopping({1, 1, 0}, a, b, 2.0) == t2);
    }

    SECTION("Calculate position") {
        auto const a = lattice.add_sublattice("A", {0, 0, 0.5});
        REQUIRE(lattice.calc_position({1, 2, 0}, {0.5, 0, 0}, a).isApprox(Cartesian(1.5, 2, 0.5)));
    }
}
