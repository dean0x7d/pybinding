#include <catch.hpp>

#include "fixtures.hpp"
using namespace cpb;

TEST_CASE("CompressedSublattices") {
    auto inject = [](CompressedSublattices& cs, idx_t size, SubAliasID id, idx_t norb) {
        for (auto i = 0; i < size; ++i) {
            cs.add(id, norb);
        }
    };

    constexpr auto size = 30;
    auto cs = CompressedSublattices();
    auto ds = VectorX<storage_idx_t>(size);

    inject(cs, 10, SubAliasID{1}, 1);
    ds.segment(0, 10).setConstant(1);

    inject(cs, 15, SubAliasID{0}, 2);
    ds.segment(10, 15).setConstant(0);

    inject(cs, 2, SubAliasID{2}, 2);
    ds.segment(25, 2).setConstant(2);

    inject(cs, 3, SubAliasID{4}, 3);
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
