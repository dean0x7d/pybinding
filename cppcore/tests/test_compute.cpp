#include <catch.hpp>

#include "compute/lanczos.hpp"
#include "fixtures.hpp"
using namespace tbm;

TEST_CASE("Lanczos", "[lanczos]") {
    auto const lattice = graphene::monolayer();
    auto model = Model(lattice);
    model.set_primitive({5, 5});
    model.set_symmetry({1, 1});

    using H = HamiltonianT<std::complex<float>> const;
    auto const hamiltonian = std::dynamic_pointer_cast<H>(model.hamiltonian());
    REQUIRE(hamiltonian);

    auto loop_counters = std::vector<int>(3);
    for (auto& count : loop_counters) {
        auto const bounds = compute::minmax_eigenvalues(hamiltonian->get_matrix(), 1e-3f);
        auto const expected = abs(3 * graphene::t);
        REQUIRE(bounds.max == Approx(expected));
        REQUIRE(bounds.min == Approx(-expected));
        count = bounds.loops;
    }

    auto const all_equal = std::all_of(loop_counters.begin(), loop_counters.end(),
                                       [&](int c) { return c == loop_counters.front(); });
    REQUIRE(all_equal);
}
