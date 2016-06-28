#include <catch.hpp>

#include "Model.hpp"
using namespace cpb;

Lattice square_lattice() {
    auto lattice = Lattice({1, 0, 0}, {0, 1, 0});
    auto const a = lattice.add_sublattice("A", {0, 0, 0});
    auto const b = lattice.add_sublattice("B", {0.5, 0.5, 0});
    auto const t1 = lattice.register_hopping_energy("t1", 1.0);
    auto const t2 = lattice.register_hopping_energy("t2", 2.0);
    lattice.add_registered_hopping({0, 0, 0}, a, b, t1);
    lattice.add_registered_hopping({1, 1, 0}, a, b, t1);
    lattice.add_registered_hopping({1, 0, 0}, a, a, t2);
    return lattice;
}

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
        REQUIRE(lattice.calc_position({1, 2, 0}, a).isApprox(Cartesian(1, 2, 0.5)));
    }

    SECTION("Set offset") {
        REQUIRE_NOTHROW(lattice.set_offset({0.5f, 0.5f, 0}));
        REQUIRE_THROWS(lattice.set_offset({0.6f, 0, 0}));
        REQUIRE_THROWS(lattice.set_offset({0, -0.6f, 0}));

        auto const copy = lattice.with_offset({0.5f, 0, 0});
        REQUIRE(copy.calc_position({1, 2, 0}).isApprox(Cartesian(1.5f, 2, 0)));
    }

    SECTION("Min neighbors") {
        auto const copy = lattice.with_min_neighbors(3);
        REQUIRE(copy.min_neighbors == 3);
    }
}

TEST_CASE("Lattice translate coordinates", "[lattice]") {
    auto const lattice = Lattice({1, 0, 0}, {1, 1, 0});

    REQUIRE(lattice.translate_coordinates({1, 0, 0}).isApprox(Vector3f(1, 0, 0)));
    REQUIRE(lattice.translate_coordinates({1.5, 0.5, 0}).isApprox(Vector3f(1, 0.5, 0)));
    REQUIRE(lattice.translate_coordinates({0, 0, 1}).isApprox(Vector3f(0, 0, 0)));
}

TEST_CASE("SiteStateModifier", "[modifier]") {
    auto model = Model(square_lattice());
    model.set_primitive({2});
    REQUIRE(model.system()->num_sites() == 4);

    auto remove_site = [](ArrayX<bool>& state, CartesianArray const&, SubIdRef) {
        state[0] = false;
    };
    model.add_site_state_modifier({remove_site});
    REQUIRE(model.system()->num_sites() == 3);
    model.add_site_state_modifier({remove_site, 1});
    REQUIRE(model.system()->num_sites() == 2);
    model.add_site_state_modifier({remove_site, 2});
    REQUIRE_THROWS(model.system());
}

TEST_CASE("SitePositionModifier", "[modifier]") {
    auto model = Model(square_lattice());
    REQUIRE(model.system()->positions.y[1] == Approx(0.5));

    model.add_position_modifier({[](CartesianArray& position, SubIdRef) {
        position.y[1] = 1;
    }});
    REQUIRE(model.system()->positions.y[1] == Approx(1));
}

struct OnsiteEnergyOp {
    template<class Array>
    void operator()(Array energy) {
        energy.setConstant(1);
    }
};

TEST_CASE("OnsiteEnergyModifier", "[modifier]") {
    auto model = Model(square_lattice());
    auto const& h_init = model.hamiltonian();
    REQUIRE(h_init.rows() == 2);
    REQUIRE(h_init.non_zeros() == 2);

    model.add_onsite_modifier({[](ComplexArrayRef energy, CartesianArray const&, SubIdRef) {
        num::match<ArrayX>(energy, OnsiteEnergyOp{});
    }});
    auto const& h = model.hamiltonian();
    REQUIRE(h.rows() == 2);
    REQUIRE(h.non_zeros() == 4);
}

struct HoppingEnergyOp {
    template<class Array>
    void operator()(Array energy) {
        energy.setZero();
    }
};

TEST_CASE("HoppingEnergyModifier", "[modifier]") {
    auto model = Model(square_lattice());
    auto const& h_init = model.hamiltonian();
    REQUIRE(h_init.rows() == 2);
    REQUIRE(h_init.non_zeros() == 2);

    model.add_hopping_modifier({[](ComplexArrayRef energy, CartesianArray const&,
                                   CartesianArray const&, HopIdRef) {
        num::match<ArrayX>(energy, HoppingEnergyOp{});
    }});
    auto const& h = model.hamiltonian();
    REQUIRE(h.rows() == 2);
    REQUIRE(h.non_zeros() == 0);
}

TEST_CASE("HoppingGenerator", "[generator]") {
    auto model = Model([]{
        auto lattice = Lattice({1, 0, 0}, {0, 1, 0});
        lattice.add_sublattice("A");
        lattice.add_sublattice("B");
        lattice.register_hopping_energy("t1", 1.0);
        return lattice;
    }());
    REQUIRE_FALSE(model.is_complex());
    REQUIRE(model.get_lattice().hopping_energies.size() == 1);
    REQUIRE(model.system()->hoppings.isCompressed());
    REQUIRE(model.system()->hoppings.rows() == 2);
    REQUIRE(model.system()->hoppings.nonZeros() == 0);

    SECTION("Add real generator") {
        model.add_hopping_family({"t2", 2.0, [](CartesianArray const&, SubIdRef) {
            auto r = HoppingGenerator::Result{ArrayXi(1), ArrayXi(1)};
            r.from << 0;
            r.to   << 1;
            return r;
        }});

        REQUIRE_FALSE(model.is_complex());
        REQUIRE(model.get_lattice().hopping_energies.size() == 2);
        REQUIRE(model.system()->hoppings.isCompressed());
        REQUIRE(model.system()->hoppings.rows() == 2);
        REQUIRE(model.system()->hoppings.nonZeros() == 1);

        auto const hopping_it = model.get_lattice().hop_name_map.find("t2");
        REQUIRE(hopping_it != model.get_lattice().hop_name_map.end());
        auto const hopping_id = hopping_it->second;
        REQUIRE(model.system()->hoppings.coeff(0, 1) == hopping_id);
    }

    SECTION("Add complex generator") {
        model.add_hopping_family({"t2", {0.0, 1.0}, [](CartesianArray const&, SubIdRef) {
            return HoppingGenerator::Result{ArrayXi(), ArrayXi()};
        }});

        REQUIRE(model.is_complex());
        REQUIRE(model.system()->hoppings.isCompressed());
        REQUIRE(model.system()->hoppings.rows() == 2);
        REQUIRE(model.system()->hoppings.nonZeros() == 0);
    }

    SECTION("Upper triangular form should be preserved") {
        model.add_hopping_family({"t2", 2.0, [](CartesianArray const&, SubIdRef) {
            auto r = HoppingGenerator::Result{ArrayXi(2), ArrayXi(2)};
            r.from << 0, 1;
            r.to   << 1, 0;
            return r;
        }});

        REQUIRE(model.system()->hoppings.rows() == 2);
        REQUIRE(model.system()->hoppings.nonZeros() == 1);
        REQUIRE(model.system()->hoppings.coeff(0, 1) == 1);
        REQUIRE(model.system()->hoppings.coeff(1, 0) == 0);
    }
}

TEST_CASE("Nonzeros per row of a triangular hopping matrix", "[unit]") {
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

TEST_CASE("Symmetry masks", "[unit]") {
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
