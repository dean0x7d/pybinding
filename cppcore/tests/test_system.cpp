#include <catch.hpp>

#include "fixtures.hpp"
using namespace cpb;

TEST_CASE("CompressedSublattices") {
    auto inject = [](CompressedSublattices& cs, idx_t size, SiteID id, idx_t norb) {
        for (auto i = 0; i < size; ++i) {
            cs.add(id, norb);
        }
    };

    constexpr auto size = 30;
    auto cs = CompressedSublattices();
    auto ds = VectorX<storage_idx_t>(size);

    inject(cs, 10, SiteID{1}, 1);
    ds.segment(0, 10).setConstant(1);

    inject(cs, 15, SiteID{0}, 2);
    ds.segment(10, 15).setConstant(0);

    inject(cs, 2, SiteID{2}, 2);
    ds.segment(25, 2).setConstant(2);

    inject(cs, 3, SiteID{4}, 3);
    ds.segment(27, 3).setConstant(4);

    REQUIRE(cs.decompressed().matrix() == ds);

    REQUIRE_NOTHROW(cs.verify(size));
    REQUIRE(cs.alias_ids().size() == 4);
    REQUIRE(cs.decompressed_size() == size);

    REQUIRE(cs.start_index(1) == 0);
    REQUIRE(cs.start_index(2) == 10);
    REQUIRE(cs.start_index(3) == 27);
    REQUIRE_THROWS_WITH(cs.start_index(4), Catch::Contains("invalid num_orbitals"));
}

TEST_CASE("HoppingBlocks") {
    auto model = Model(lattice::square(), Primitive(6, 4));
    auto const& hb = model.system()->hopping_blocks;

    auto const neighbor_counts = hb.count_neighbors();
    REQUIRE(neighbor_counts.sum() == 2 * 4 + 3 * 12 + 4 * 8);
}

TEST_CASE("to_hamiltonian_indices") {
    auto vec = [](std::initializer_list<storage_idx_t> const& init) -> VectorXi {
        auto v = VectorXi(static_cast<idx_t>(init.size()));
        std::copy(init.begin(), init.end(), v.data());
        return v;
    };

    SECTION("single-orbital") {
        auto const model = Model(lattice::square(), Primitive(3, 3));
        auto const& system = *model.system();

        REQUIRE(system.num_sites() == 9);
        REQUIRE(system.hamiltonian_size() == 9);
        REQUIRE(system.hamiltonian_nnz() == 33);
        REQUIRE(system.to_hamiltonian_indices(0).matrix() == vec({0}));
        REQUIRE(system.to_hamiltonian_indices(4).matrix() == vec({4}));
        REQUIRE(system.to_hamiltonian_indices(8).matrix() == vec({8}));
    }

    SECTION("multi-orbital") {
        auto const model = Model(lattice::square_multiorbital(), Primitive(1, 2));
        auto const& system = *model.system();

        REQUIRE(system.num_sites() == 8);
        REQUIRE(system.hamiltonian_size() == 16);
        REQUIRE(system.hamiltonian_nnz() == 96);
        REQUIRE(system.to_hamiltonian_indices(0).matrix() == vec({0}));
        REQUIRE(system.to_hamiltonian_indices(1).matrix() == vec({1}));
        REQUIRE(system.to_hamiltonian_indices(2).matrix() == vec({2, 3}));
        REQUIRE(system.to_hamiltonian_indices(3).matrix() == vec({4, 5}));
        REQUIRE(system.to_hamiltonian_indices(4).matrix() == vec({6, 7}));
        REQUIRE(system.to_hamiltonian_indices(5).matrix() == vec({8, 9}));
        REQUIRE(system.to_hamiltonian_indices(6).matrix() == vec({10, 11, 12}));
        REQUIRE(system.to_hamiltonian_indices(7).matrix() == vec({13, 14, 15}));
    }
}

TEST_CASE("complex_valued_hoppings") {
    SECTION("single-orbital-complex") {
        using constant::i1;
        auto const lattice = lattice::hexagonal_complex();
        // distance from A to three neighbor sites
        auto const d1 = Cartesian{0.0f, -1.0f / 3 * sqrt(3.0f), 0.0f};
        auto const d2 = Cartesian{d1 + Cartesian{ 0.5f, 0.5f * sqrt(3.0f), 0.0f}};
        auto const d3 = Cartesian{d1 + Cartesian{-0.5f, 0.5f * sqrt(3.0f), 0.0f}};
        // hoppings
        auto const t1 = -i1;
        auto const t2 = 2.0f * i1;
        auto const t3 = 3.0f * i1;

        auto model = Model(lattice, TranslationalSymmetry(1, 1));
        using constant::pi;
        // set the wavevector to K point
        auto const k_vector = Cartesian{2 * pi, 0, 0};
        model.set_wave_vector(k_vector);

        auto const& system = *model.system();
        auto const& matrix = ham::get_reference<std::complex<float>>(model.hamiltonian());

        auto const expected_hopping = t1 * exp(i1 * k_vector.dot(d1)) +
                                      t2 * exp(i1 * k_vector.dot(d2)) +
                                      t3 * exp(i1 * k_vector.dot(d3));

        REQUIRE(system.num_sites() == 2);
        REQUIRE(system.hamiltonian_size() == 2);
        REQUIRE(system.hamiltonian_nnz() == 4);
        REQUIRE(num::approx_equal(matrix.coeff(0, 1).real(),  expected_hopping.real()));
        REQUIRE(num::approx_equal(matrix.coeff(0, 1).imag(),  expected_hopping.imag()));
        REQUIRE(num::approx_equal(matrix.coeff(1, 0).real(),  expected_hopping.real()));
        REQUIRE(num::approx_equal(matrix.coeff(1, 0).imag(), -expected_hopping.imag()));
    }

    SECTION("multi-orbital-complex") {
        auto const model = Model(lattice::checkerboard_multiorbital(),
                                 TranslationalSymmetry(1, 1),
                                 field::force_double_precision());
        auto const& system = *model.system();
        auto const& matrix = ham::get_reference<std::complex<double>>(model.hamiltonian());

        constexpr auto i1 = num::get_complex_t<double>{constant::i1};
        auto expected_hopping = MatrixXcd(2, 2);
        expected_hopping << 2.0 + 2.0 * i1,
                            3.0 + 3.0 * i1,
                            4.0 + 4.0 * i1,
                            5.0 + 5.0 * i1;

        REQUIRE(system.num_sites() == 2);
        REQUIRE(system.hamiltonian_size() == 4);
        REQUIRE(system.hamiltonian_nnz() == 16);
        REQUIRE(matrix.block(0, 0, 2, 2).isApprox((-matrix.block(2, 2, 2, 2)).eval()));
        REQUIRE(matrix.block(0, 2, 2, 2).isApprox(4.0 * expected_hopping));
    }
}

TEST_CASE("sublattice_range") {
    auto const model = Model(lattice::square_multiorbital(), shape::rectangle(1, 2));
    auto const& system = *model.system();

    auto const ra = system.sublattice_range("A");
    REQUIRE(ra.start == 2);
    REQUIRE(ra.end   == 4);

    auto const rb = system.sublattice_range("B");
    REQUIRE(rb.start == 0);
    REQUIRE(rb.end   == 2);

    auto const rc = system.sublattice_range("C");
    REQUIRE(rc.start == 4);
    REQUIRE(rc.end   == 6);

    auto const rd = system.sublattice_range("D");
    REQUIRE(rd.start == 6);
    REQUIRE(rd.end   == 8);
}

TEST_CASE("expanded_positions") {
    auto const model = Model(lattice::square_multiorbital());
    auto const& sys = *model.system();
    auto const& pos = sys.positions;
    auto const& ep = sys.expanded_positions();

    REQUIRE(pos.size() == 4);
    REQUIRE(ep.size() == 8);

    REQUIRE(ep[0] == pos[0]);
    REQUIRE(ep[1] == pos[0]);

    REQUIRE(ep[2] == pos[1]);

    REQUIRE(ep[3] == pos[2]);
    REQUIRE(ep[4] == pos[2]);

    REQUIRE(ep[5] == pos[3]);
    REQUIRE(ep[6] == pos[3]);
    REQUIRE(ep[7] == pos[3]);
}
