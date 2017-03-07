#include <catch.hpp>

#include "fixtures.hpp"
using namespace cpb;

TEST_CASE("CompressedSublattices") {
    auto inject = [](CompressedSublattices& cs, idx_t size, sub_id id, idx_t norb) {
        for (auto i = 0; i < size; ++i) {
            cs.add(id, norb);
        }
    };

    constexpr auto size = 30;
    auto cs = CompressedSublattices();
    auto ds = VectorX<sub_id>(size);

    inject(cs, 10, 1, 1);
    ds.segment(0, 10).setConstant(1);

    inject(cs, 15, 0, 2);
    ds.segment(10, 15).setConstant(0);

    inject(cs, 2, 2, 2);
    ds.segment(25, 2).setConstant(2);

    inject(cs, 3, 4, 3);
    ds.segment(27, 3).setConstant(4);

    REQUIRE(cs.decompress().matrix() == ds);

    REQUIRE_NOTHROW(cs.verify(size));
    REQUIRE(cs.alias_ids().size() == 4);
    REQUIRE(cs.decompressed_size() == size);

    REQUIRE(cs.start_index(1) == 0);
    REQUIRE(cs.start_index(2) == 10);
    REQUIRE(cs.start_index(3) == 27);
    REQUIRE_THROWS_WITH(cs.start_index(4), Catch::Contains("invalid num_orbitals"));
}
