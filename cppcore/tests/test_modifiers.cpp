#include <catch.hpp>

#include "fixtures.hpp"
using namespace cpb;

TEST_CASE("SiteStateModifier") {
    auto model = Model(lattice::square_2atom(), Primitive(2));
    REQUIRE(model.system()->num_sites() == 4);

    auto remove_site = [](Eigen::Ref<ArrayX<bool>> state, CartesianArrayConstRef, string_view s) {
        if (s == "A") {
            state[0] = false;
        }
    };
    model.add(SiteStateModifier(remove_site));
    REQUIRE(model.system()->num_sites() == 3);
    model.add(SiteStateModifier(remove_site, 1));
    REQUIRE(model.system()->num_sites() == 2);
    model.add(SiteStateModifier(remove_site, 2));
    REQUIRE_THROWS(model.system());
}

TEST_CASE("SitePositionModifier") {
    auto model = Model(lattice::square_2atom());
    REQUIRE(model.system()->positions.y[1] == Approx(0.5));

    model.add(PositionModifier([](CartesianArrayRef position, string_view sublattice) {
        if (sublattice == "B") {
            position.y()[0] = 1;
        }
    }));
    REQUIRE(model.system()->positions.y[1] == Approx(1));
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
        model.add(HoppingGenerator("t2", 2.0, [](CartesianArray const&, SubIdRef) {
            auto r = HoppingGenerator::Result{ArrayXi(1), ArrayXi(1)};
            r.from << 0;
            r.to   << 1;
            return r;
        }));

        REQUIRE_FALSE(model.is_complex());
        REQUIRE(model.get_lattice().get_hoppings().size() == 2);
        REQUIRE(model.system()->hopping_blocks.nnz() == 1);

        auto const hop_names = model.get_lattice().hop_name_map();
        auto const hopping_it = hop_names.find("t2");
        REQUIRE(hopping_it != hop_names.end());
        auto const hopping_id = hopping_it->second;
        REQUIRE(model.system()->hopping_blocks.tocsr().coeff(0, 1) == hopping_id);
    }

    SECTION("Add complex generator") {
        model.add(HoppingGenerator("t2", {0.0, 1.0}, [](CartesianArray const&, SubIdRef) {
            return HoppingGenerator::Result{ArrayXi(), ArrayXi()};
        }));

        REQUIRE(model.is_complex());
        REQUIRE(model.system()->hopping_blocks.nnz() == 0);
    }

    SECTION("Upper triangular form should be preserved") {
        model.add(HoppingGenerator("t2", 2.0, [](CartesianArray const&, SubIdRef) {
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
