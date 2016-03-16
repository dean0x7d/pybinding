#include <catch.hpp>

#include "Model.hpp"
#include "compute/lanczos.hpp"
using namespace tbm;

namespace graphene {
    static constexpr auto a = 0.24595f; // [nm] unit cell length
    static constexpr auto a_cc = 0.142f; // [nm] carbon-carbon distance
    static constexpr auto t = -2.8f; // [eV] nearest neighbor hopping

    Lattice monolayer() {
        auto lattice = Lattice({a, 0, 0}, {a/2, a/2 * sqrt(3.0f), 0});

        auto const A = lattice.add_sublattice("A", {0, -a_cc/2, 0});
        auto const B = lattice.add_sublattice("B", {0,  a_cc/2, 0});
        auto const t0 = lattice.register_hopping_energy("t", t);

        lattice.add_registered_hopping({0,  0, 0}, A, B, t0);
        lattice.add_registered_hopping({1, -1, 0}, A, B, t0);
        lattice.add_registered_hopping({0, -1, 0}, A, B, t0);

        return lattice;
    }
} // namespace graphene

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
