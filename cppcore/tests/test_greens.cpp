#include <catch.hpp>

#include "fixtures.hpp"
#include "greens/KPM.hpp"
using namespace tbm;

TEST_CASE("OptimizedHamiltonian reordering", "[kpm]") {
    auto const lattice = graphene::monolayer();
    auto model = Model(lattice);
    model.set_shape(shape::rectangle(0.6f, 0.8f));
    auto const num_sites = model.system()->num_sites();

    using scalat_t = float;
    using H = HamiltonianT<scalat_t> const;
    auto const hamiltonian = std::dynamic_pointer_cast<H>(model.hamiltonian());
    REQUIRE(hamiltonian);
    auto const& matrix = hamiltonian->get_matrix();

    auto scale = kpm::Scale<scalat_t>();
    scale.compute(matrix, KPMConfig{}.lanczos_precision);

    auto size_indices = [](kpm::OptimizedHamiltonian<scalat_t> const& oh, int num_moments) {
        auto v = std::vector<int>(num_moments);
        for (auto n = 0; n < num_moments; ++n) {
            v[n] = oh.optimized_size_index(n, num_moments);
        }
        return v;
    };

    SECTION("Diagonal") {
        auto oh = kpm::OptimizedHamiltonian<scalat_t>();
        auto const i = num_sites / 2;
        oh.create(matrix, {i, i}, scale, /*use_reordering*/true);

        REQUIRE(oh.optimized_idx.i == 0);
        REQUIRE(oh.optimized_idx.j == 0);
        REQUIRE(oh.optimized_sizes[0] == 1);
        REQUIRE(oh.optimized_sizes.back() == num_sites);
        REQUIRE(oh.size_index_offset == 0);

        auto const expected6 = std::vector<int>{0, 1, 2, 2, 1, 0};
        REQUIRE(size_indices(oh, 6) == expected6);
        auto const expected9 = std::vector<int>{0, 1, 2, 3, 4, 3, 2, 1, 0};
        REQUIRE(size_indices(oh, 9) == expected9);
        auto const expected12 = std::vector<int>{0, 1, 2, 3, 4, 4, 4, 4, 3, 2, 1, 0};
        REQUIRE(size_indices(oh, 12) == expected12);
    }

    SECTION("Off-diagonal") {
        auto oh = kpm::OptimizedHamiltonian<scalat_t>();
        auto const i = num_sites / 2;
        auto const j = num_sites / 4;
        oh.create(matrix, {i, j}, scale, /*use_reordering*/true);

        REQUIRE(oh.optimized_idx.i != oh.optimized_idx.j);
        REQUIRE(oh.optimized_sizes[0] == 1);
        REQUIRE(oh.optimized_sizes.back() == num_sites);
        REQUIRE(oh.size_index_offset > 0);

        auto const expected6 = std::vector<int>{0, 1, 2, 3, 3, 3};
        REQUIRE(size_indices(oh, 6) == expected6);
        auto const expected9 = std::vector<int>{0, 1, 2, 3, 4, 4, 4, 4, 3};
        REQUIRE(size_indices(oh, 9) == expected9);
        auto const expected12 = std::vector<int>{0, 1, 2, 3, 4, 5, 6, 6, 6, 5, 4, 3};
        REQUIRE(size_indices(oh, 12) == expected12);
    }
}


TEST_CASE("KPM optimization levels", "[kpm]") {
    auto const lattice = graphene::monolayer();
    auto model = Model(lattice);
    model.set_shape(shape::rectangle(0.6f, 0.8f));
    auto const num_sites = model.system()->num_sites();

    auto calc_greens = [&](int i, int j, int opt_level) {
        auto config = KPMConfig{};
        config.optimization_level = opt_level;
        auto kpm = Greens<KPM>(model, config);
        auto const g = kpm.calc_greens(i, j, ArrayXd::LinSpaced(10, -0.3, 0.3), 0.02);
        return ArrayXcf{g.cast<std::complex<float>>()};
    };

    SECTION("Diagonal") {
        auto const i = num_sites / 2;
        auto const g0 = calc_greens(i, i, /*opt_level*/0);
        auto const g1 = calc_greens(i, i, /*opt_level*/1);
        auto const g2 = calc_greens(i, i, /*opt_level*/2);

        REQUIRE(g0.isApprox(g1));
        REQUIRE(g0.isApprox(g2));
    }

    SECTION("Off-diagonal") {
        auto const i = num_sites / 2;
        auto const j = num_sites / 4;
        auto const g0 = calc_greens(i, j, /*opt_level*/0);
        auto const g1 = calc_greens(i, j, /*opt_level*/1);
        auto const g2 = calc_greens(i, j, /*opt_level*/2);

        REQUIRE(g0.isApprox(g1));
        REQUIRE(g0.isApprox(g2));
    }
}
