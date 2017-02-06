#include <catch.hpp>

#include "Lattice.hpp"
using namespace cpb;

TEST_CASE("Lattice") {
    auto lattice = Lattice({1, 0, 0}, {0, 1, 0});
    REQUIRE(lattice.ndim() == 2);
    REQUIRE(lattice.get_vectors().capacity() == 2);
    REQUIRE(lattice.max_hoppings() == 0);

    SECTION("Add sublattices") {
        REQUIRE_THROWS(lattice.add_sublattice("", {0, 0, 0}));

        lattice.add_sublattice("A", {0, 0, 0});
        REQUIRE_FALSE(lattice.has_onsite_energy());
        REQUIRE_THROWS(lattice.add_sublattice("A", {0, 0, 0}));

        lattice.add_sublattice("B", {0, 0, 0}, 1.0);
        REQUIRE(lattice.has_onsite_energy());

        lattice.add_alias("B2", "B", {1, 0, 0});
        REQUIRE_THROWS(lattice.add_alias("B3", "bad_name", {2, 0, 0}));
        REQUIRE_THROWS(lattice.add_alias("B4", "B4", {2, 0, 0}));

        while (lattice.nsub() != std::numeric_limits<sub_id>::max() + 1) {
            lattice.add_sublattice(std::to_string(lattice.nsub()), {0, 0, 0});
        }
        REQUIRE_THROWS(lattice.add_sublattice("overflow", {0, 0, 0}));
    }

    SECTION("Add multi-orbital sublattice") {
        REQUIRE_FALSE(lattice.has_multiple_orbitals());
        lattice.add_sublattice("A", {0, 0, 0}, VectorXd::Constant(2, 0.0).eval());
        REQUIRE(lattice.has_multiple_orbitals());

        auto const not_square = MatrixXcd::Zero(2, 3).eval();
        REQUIRE_THROWS(lattice.add_sublattice("not_square", {0, 0, 0}, not_square));

        auto complex_diagonal = MatrixXcd::Zero(2, 2).eval();
        complex_diagonal(0, 0) = std::complex<double>{0.0, 1.0};
        REQUIRE_THROWS(lattice.add_sublattice("complex_diagonal", {0, 0, 0}, complex_diagonal));

        auto not_hermitian = MatrixXcd::Zero(2, 2).eval();
        not_hermitian(0, 1) = std::complex<double>{0.0, 1.0};
        not_hermitian(1, 0) = std::complex<double>{0.0, 1.0};
        REQUIRE_THROWS(lattice.add_sublattice("not_hermitian", {0, 0, 0}, not_hermitian));

        auto not_symmetric = MatrixXcd::Zero(2, 2).eval();
        not_symmetric(0, 1) = std::complex<double>{1.0, 0.0};
        not_symmetric(1, 0) = std::complex<double>{2.0, 0.0};
        REQUIRE_THROWS(lattice.add_sublattice("not_symmetric", {0, 0, 0}, not_symmetric));

        auto upper_triangular = MatrixXcd::Zero(2, 2).eval();
        upper_triangular(0, 1) = std::complex<double>{0.0, 1.0};
        REQUIRE_NOTHROW(lattice.add_sublattice("upper_triangular", {0, 0, 0}, upper_triangular));
    }

    SECTION("Register hoppings") {
        REQUIRE_THROWS(lattice.register_hopping_energy("", 0.0));

        lattice.register_hopping_energy("t1", 1.0);
        REQUIRE_FALSE(lattice.has_complex_hoppings());
        REQUIRE_THROWS(lattice.register_hopping_energy("t1", 1.0));

        lattice.register_hopping_energy("t2", std::complex<double>{0, 1.0});
        REQUIRE(lattice.has_complex_hoppings());

        while (lattice.get_hoppings().size() != std::numeric_limits<hop_id>::max() + 1) {
            auto e = static_cast<double>(lattice.get_hoppings().size());
            lattice.register_hopping_energy(std::to_string(e), e);
        }
        REQUIRE_THROWS(lattice.register_hopping_energy("overflow", 1.0));
    }

    SECTION("Add scalar hoppings") {
        lattice.add_sublattice("A", {0, 0, 0});
        lattice.add_sublattice("B", {0, 0, 0});
        lattice.register_hopping_energy("t1", 1.0);

        REQUIRE_THROWS(lattice.add_hopping({0, 0, 0}, "A", "A", "t1"));
        REQUIRE_THROWS(lattice.add_hopping({0, 0, 0}, "bad_name", "A", "t1"));
        REQUIRE_THROWS(lattice.add_hopping({0, 0, 0}, "A", "A", "bad_name"));

        lattice.add_hopping({1, 0, 0}, "A", "A", "t1");
        REQUIRE_THROWS(lattice.add_hopping({1, 0, 0}, "A", "A", "t1"));
        REQUIRE(lattice.max_hoppings() == 2);

        lattice.add_hopping({1, 0, 0}, "A", "B", "t1");
        REQUIRE(lattice.max_hoppings() == 3);
        lattice.add_hopping({1, 0, 0}, "B", "B", "t1");
        REQUIRE(lattice.max_hoppings() == 3);

        lattice.add_hopping({1, 1, 0}, "A", "A", 2.0);
        REQUIRE(lattice.get_hoppings().size() == 2);
        lattice.add_hopping({1, 1, 0}, "A", "B", 2.0);
        REQUIRE(lattice.get_hoppings().size() == 2);
    }

    SECTION("Add matrix hoppings") {
        lattice.add_sublattice("A", {0, 0, 0}, VectorXd::Constant(2, 0.0).eval());
        lattice.add_sublattice("B", {0, 0, 0}, VectorXd::Constant(2, 0.0).eval());
        lattice.add_sublattice("C", {0, 0, 0}, VectorXd::Constant(3, 0.0).eval());
        lattice.register_hopping_energy("t22", MatrixXcd::Constant(2, 2, 1.0));
        lattice.register_hopping_energy("t23", MatrixXcd::Constant(2, 3, 1.0));
        lattice.register_hopping_energy("t32", MatrixXcd::Constant(3, 2, 1.0));

        REQUIRE(lattice.max_hoppings() == 2);
        lattice.add_hopping({0, 0, 0}, "A", "B", "t22");
        REQUIRE(lattice.max_hoppings() == 3);
        lattice.add_hopping({1, 0, 0}, "A", "A", "t22");
        REQUIRE(lattice.max_hoppings() == 7);
        lattice.add_hopping({0, 0, 0}, "A", "C", "t23");
        REQUIRE(lattice.max_hoppings() == 10);
        lattice.add_hopping({1, 0, 0}, "C", "A", "t32");
        REQUIRE(lattice.max_hoppings() == 13);

        REQUIRE_THROWS(lattice.add_hopping({0, 0, 0}, "A", "A", "t22"));
        REQUIRE_THROWS(lattice.add_hopping({0, 0, 0}, "B", "C", "t22"));
        REQUIRE_THROWS(lattice.add_hopping({0, 0, 0}, "C", "B", "t23"));
    }

    SECTION("Calculate position") {
        lattice.add_sublattice("A", {0, 0, 0.5});
        REQUIRE(lattice.calc_position({1, 2, 0}, "A").isApprox(Cartesian(1, 2, 0.5)));
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
        REQUIRE(copy.get_min_neighbors() == 3);
    }
}

TEST_CASE("Lattice translate coordinates") {
    auto const lattice = Lattice({1, 0, 0}, {1, 1, 0});

    REQUIRE(lattice.translate_coordinates({1, 0, 0}).isApprox(Vector3f(1, 0, 0)));
    REQUIRE(lattice.translate_coordinates({1.5, 0.5, 0}).isApprox(Vector3f(1, 0.5, 0)));
    REQUIRE(lattice.translate_coordinates({0, 0, 1}).isApprox(Vector3f(0, 0, 0)));
}
