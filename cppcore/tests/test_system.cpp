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

TEST_CASE("to_hamiltonian_index") {
    SECTION("single-orbital") {
        auto const model = Model(lattice::square(), shape::rectangle(3, 3));
        auto const& system = *model.system();

        REQUIRE(system.num_sites() == 9);
        REQUIRE(system.hamiltonian_size() == 9);
        REQUIRE(system.to_hamiltonian_index(0) == 0);
        REQUIRE(system.to_hamiltonian_index(4) == 4);
        REQUIRE(system.to_hamiltonian_index(8) == 8);
    }

    SECTION("multi-orbital") {
        auto const model = Model(lattice::square_multiorbital(), shape::rectangle(1, 2));
        auto const& system = *model.system();

        REQUIRE(system.num_sites() == 8);
        REQUIRE(system.hamiltonian_size() == 16);
        REQUIRE(system.to_hamiltonian_index(0) == 0);
        REQUIRE(system.to_hamiltonian_index(2) == 2);
        REQUIRE(system.to_hamiltonian_index(3) == 4);
        REQUIRE(system.to_hamiltonian_index(4) == 6);
        REQUIRE(system.to_hamiltonian_index(5) == 8);
        REQUIRE(system.to_hamiltonian_index(6) == 10);
        REQUIRE(system.to_hamiltonian_index(7) == 13);
    }
}
