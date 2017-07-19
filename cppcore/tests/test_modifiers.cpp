#include <catch.hpp>

#include "fixtures.hpp"
using namespace cpb;

TEST_CASE("SiteStateModifier") {
    auto model = Model(lattice::square_2atom(), Primitive(2));
    REQUIRE(model.system()->num_sites() == 4);

    auto count = std::unordered_map<std::string, idx_t>();

    auto remove_site = [&](Eigen::Ref<ArrayX<bool>> state, CartesianArrayConstRef, string_view s) {
        count[s] = state.size();
        if (s == "A" && state.size() != 0) {
            state[0] = false;
        }
    };

    SECTION("Apply to foundation") {
        model.add(SiteStateModifier(remove_site));
        REQUIRE(model.system()->num_sites() == 3);
        REQUIRE(count["A"] == 2);
        REQUIRE(count["B"] == 2);

        model.add(SiteStateModifier(remove_site, 1));
        REQUIRE(model.system()->num_sites() == 2);
        REQUIRE(count["A"] == 2);
        REQUIRE(count["B"] == 2);

        model.add(SiteStateModifier(remove_site, 2));
        REQUIRE_THROWS(model.system());
        REQUIRE(count["A"] == 2);
        REQUIRE(count["B"] == 2);
    }

    SECTION("Apply to system") {
        model.add(SiteStateModifier(remove_site));
        model.add(generator::do_nothing_hopping());

        model.add(SiteStateModifier(remove_site));
        REQUIRE(model.system()->num_sites() == 2);
        REQUIRE(count["A"] == 1);
        REQUIRE(count["B"] == 2);

        model.add(generator::do_nothing_hopping("_t2"));
        model.add(SiteStateModifier(remove_site));
        REQUIRE(model.system()->num_sites() == 2);
        REQUIRE(count["A"] == 0);
        REQUIRE(count["B"] == 2);

        model.add(SiteStateModifier(remove_site, 1));
        REQUIRE_THROWS_WITH(model.system(), Catch::Contains("has not been implemented yet"));
        REQUIRE(count["A"] == 0);
        REQUIRE(count["B"] == 2);
    }
}

TEST_CASE("SitePositionModifier") {
    auto model = Model(lattice::square_2atom(), shape::rectangle(2, 2));
    REQUIRE(model.system()->num_sites() == 6);
    REQUIRE(model.system()->positions.y[1] == Approx(-1));

    auto count = std::unordered_map<std::string, idx_t>();
    constexpr auto moved_pos = 10.0f;

    auto move_site = PositionModifier([&](CartesianArrayRef position, string_view sublattice) {
        count[sublattice] = position.size();
        if (sublattice == "B") {
            position.y().setConstant(moved_pos);
        }
    });

    SECTION("Apply to foundation") {
        model.add(move_site);
        model.eval();
        REQUIRE(count["A"] == 25);
        REQUIRE(count["B"] == 25);
        REQUIRE(model.system()->num_sites() == 6);
        REQUIRE(model.system()->positions.y.segment(4, 2).isApproxToConstant(moved_pos));
    }

    SECTION("Apply to system") {
        model.add(generator::do_nothing_hopping());
        model.add(move_site);
        model.eval();
        REQUIRE(count["A"] == 4);
        REQUIRE(count["B"] == 2);
        REQUIRE(model.system()->num_sites() == 6);
        REQUIRE(model.system()->positions.y.segment(4, 2).isApproxToConstant(moved_pos));
    }
}

TEST_CASE("State and position modifier ordering") {
    auto model = Model(lattice::square_2atom(), Primitive(2));

    auto delete_site = SiteStateModifier([](Eigen::Ref<ArrayX<bool>> state,
                                            CartesianArrayConstRef position,
                                            string_view sublattice) {
        if (sublattice == "A" && position.x()[0] < 0) {
            state[0] = false;
        }
    });

    auto move_site = PositionModifier([](CartesianArrayRef position, string_view sublattice) {
        if (sublattice == "A") {
            position.x()[0] = 10;
        }
    });

    SECTION("State before position") {
        REQUIRE(model.system()->num_sites() == 4);
        REQUIRE(model.system()->positions.x[0] == Approx(-1));
        REQUIRE(model.system()->positions.x[1] == Approx(0));

        model.add(delete_site);
        model.add(move_site);

        REQUIRE(model.system()->num_sites() == 3);
        REQUIRE(model.system()->positions.x[0] == Approx(0));
    }

    SECTION("Position before state") {
        REQUIRE(model.system()->num_sites() == 4);
        REQUIRE(model.system()->positions.x[0] == Approx(-1));
        REQUIRE(model.system()->positions.x[1] == Approx(0));

        model.add(move_site);
        model.add(delete_site);

        REQUIRE(model.system()->num_sites() == 4);
        REQUIRE(model.system()->positions.x[0] == Approx(10));
        REQUIRE(model.system()->positions.x[1] == Approx(0));

    }
}

struct OnsiteEnergyOp {
    template<class Array>
    void operator()(Array energy) {
        energy.setConstant(1);
    }
};

TEST_CASE("OnsiteEnergyModifier") {
    auto model = Model(lattice::square_2atom());
    auto const& h_init = model.hamiltonian();
    REQUIRE(h_init.rows() == 2);
    REQUIRE(h_init.non_zeros() == 2);

    model.add(OnsiteModifier([](ComplexArrayRef energy, CartesianArrayConstRef, string_view) {
        num::match<ArrayX>(energy, OnsiteEnergyOp{});
    }));
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

TEST_CASE("HoppingEnergyModifier") {
    auto model = Model(lattice::square_2atom());
    auto const& h_init = model.hamiltonian();
    REQUIRE(h_init.rows() == 2);
    REQUIRE(h_init.non_zeros() == 2);

    model.add(HoppingModifier([](ComplexArrayRef energy, CartesianArrayConstRef,
                                 CartesianArrayConstRef, string_view) {
        num::match<ArrayX>(energy, HoppingEnergyOp{});
    }));
    auto const& h = model.hamiltonian();
    REQUIRE(h.rows() == 2);
    REQUIRE(h.non_zeros() == 0);
}

TEST_CASE("SiteGenerator") {
    auto model = Model([]{
        auto lattice = Lattice({1, 0, 0}, {0, 1, 0});
        lattice.add_sublattice("A", {0, 0, 0});
        lattice.add_sublattice("B", {0, 0, 0});
        lattice.register_hopping_energy("t1", 1.0);
        return lattice;
    }());
    REQUIRE_FALSE(model.is_complex());
    REQUIRE(model.get_lattice().get_hoppings().size() == 1);
    REQUIRE(model.system()->hopping_blocks.nnz() == 0);

    SECTION("Errors") {
        auto const noop = [](System const&) { return CartesianArray(); };

        auto const complex_vector = MatrixXcd::Constant(1, 2, 2.0);
        REQUIRE_THROWS_WITH(model.add(SiteGenerator("C", complex_vector, noop)),
                            Catch::Contains("must be a real vector or a square matrix"));

        auto const complex_matrix = MatrixXcd::Constant(2, 2, {1.0, 1.0});
        REQUIRE_THROWS_WITH(model.add(SiteGenerator("C", complex_matrix, noop)),
                            Catch::Contains("diagonal of the onsite hopping term must be real"));
    }

    SECTION("Structure") {
        auto const energy = MatrixXcd::Constant(1, 1, 2.0);
        model.add(SiteGenerator("C", energy, [](System const&) {
            auto const size = 5;
            auto x = ArrayXf::Constant(size, 1);
            auto y = ArrayXf::LinSpaced(size, 1, 5);
            auto z = ArrayXf::Constant(size, 0);
            return CartesianArray(x, y, z);
        }));

        REQUIRE_FALSE(model.is_complex());
        REQUIRE(model.get_lattice().get_sublattices().size() == 2);
        REQUIRE(model.get_site_registry().size() == 3);
        REQUIRE(model.system()->compressed_sublattices.alias_ids().size() == 3);
        REQUIRE(model.system()->num_sites() == 7);

        REQUIRE(model.system()->positions[0].isApprox(Cartesian{0, 0, 0}));
        REQUIRE(model.system()->positions[1].isApprox(Cartesian{0, 0, 0}));
        REQUIRE(model.system()->positions[2].isApprox(Cartesian{1, 1, 0}));
        REQUIRE(model.system()->positions[3].isApprox(Cartesian{1, 2, 0}));
        REQUIRE(model.system()->positions[4].isApprox(Cartesian{1, 3, 0}));
        REQUIRE(model.system()->positions[5].isApprox(Cartesian{1, 4, 0}));
        REQUIRE(model.system()->positions[6].isApprox(Cartesian{1, 5, 0}));

        auto const names = model.get_site_registry().name_map();
        auto const it = names.find("C");
        REQUIRE(it != names.end());

        auto const id = it->second;
        auto const& cs = model.system()->compressed_sublattices;
        REQUIRE(cs.alias_ids()[2] == id);
        REQUIRE(cs.orbital_counts()[2] == 1);
        REQUIRE(cs.site_counts()[2] == 5);
    }
}

TEST_CASE("HoppingGenerator") {
    auto model = Model([]{
        auto lattice = Lattice({1, 0, 0}, {0, 1, 0});
        lattice.add_sublattice("A", {0, 0, 0});
        lattice.add_sublattice("B", {0, 0, 0});
        lattice.register_hopping_energy("t1", 1.0);
        return lattice;
    }());
    REQUIRE_FALSE(model.is_complex());
    REQUIRE(model.get_lattice().get_hoppings().size() == 1);
    REQUIRE(model.system()->hopping_blocks.nnz() == 0);

    SECTION("Add real generator") {
        model.add(HoppingGenerator("t2", 2.0, [](System const&) {
            auto r = HoppingGenerator::Result{ArrayXi(1), ArrayXi(1)};
            r.from << 0;
            r.to   << 1;
            return r;
        }));

        REQUIRE_FALSE(model.is_complex());
        REQUIRE(model.get_lattice().get_hoppings().size() == 1);
        REQUIRE(model.get_hopping_registry().size() == 2);
        REQUIRE(model.system()->hopping_blocks.nnz() == 1);

        auto const hop_names = model.get_hopping_registry().name_map();
        auto const hopping_it = hop_names.find("t2");
        REQUIRE(hopping_it != hop_names.end());
        auto const hopping_id = hopping_it->second;
        REQUIRE(model.system()->hopping_blocks.tocsr().coeff(0, 1) == hopping_id);
    }

    SECTION("Add complex generator") {
        model.add(HoppingGenerator("t2", std::complex<double>{0.0, 1.0}, [](System const&) {
            return HoppingGenerator::Result{ArrayXi(), ArrayXi()};
        }));

        REQUIRE(model.is_complex());
        REQUIRE(model.system()->hopping_blocks.nnz() == 0);
    }

    SECTION("Upper triangular form should be preserved") {
        model.add(HoppingGenerator("t2", 2.0, [](System const&) {
            auto r = HoppingGenerator::Result{ArrayXi(2), ArrayXi(2)};
            r.from << 0, 1;
            r.to   << 1, 0;
            return r;
        }));

        REQUIRE(model.system()->hopping_blocks.nnz() == 1);
        auto const csr = model.system()->hopping_blocks.tocsr();
        REQUIRE(csr.coeff(0, 1) == 1);
        REQUIRE(csr.coeff(1, 0) == 0);
    }
}
