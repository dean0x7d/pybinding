#include <catch.hpp>

#include "fixtures.hpp"
#include "greens/KPM.hpp"
using namespace tbm;

Model make_test_model(bool is_double = false, bool is_complex = false) {
    auto model = Model(graphene::monolayer());
    model.set_shape(shape::rectangle(0.6f, 0.8f));
    model.add_onsite_modifier(field::constant_potential(1));
    if (is_double) {
        model.add_hopping_modifier(field::force_double_precision());
    }
    if (is_complex) {
        model.add_hopping_modifier(field::constant_magnetic_field(1e4));
    }
    return model;
}

TEST_CASE("OptimizedHamiltonian reordering", "[kpm]") {
    auto const model = make_test_model();
    auto const num_sites = model.system()->num_sites();

    using scalat_t = float;
    auto const matrix = ham::get_reference<scalat_t>(model.hamiltonian());
    auto bounds = kpm::Bounds<scalat_t>(&matrix, KPMConfig{}.lanczos_precision);

    auto size_indices = [](kpm::OptimizedHamiltonian<scalat_t> const& oh, int num_moments) {
        auto v = std::vector<int>(num_moments);
        for (auto n = 0; n < num_moments; ++n) {
            v[n] = oh.sizes().index(n, num_moments);
        }
        return v;
    };

    SECTION("Diagonal") {
        auto oh = kpm::OptimizedHamiltonian<scalat_t>(&matrix);
        auto const i = num_sites / 2;
        oh.optimize_for({i, i}, bounds.scaling_factors());

        REQUIRE(oh.idx().row == 0);
        REQUIRE(oh.idx().cols[0] == 0);
        REQUIRE(oh.sizes().get_data().front() == 1);
        REQUIRE(oh.sizes().get_data().back() == num_sites);
        REQUIRE(oh.sizes().get_offset() == 0);

        auto const expected6 = std::vector<int>{0, 1, 2, 2, 1, 0};
        REQUIRE(size_indices(oh, 6) == expected6);
        auto const expected9 = std::vector<int>{0, 1, 2, 3, 4, 3, 2, 1, 0};
        REQUIRE(size_indices(oh, 9) == expected9);
        auto const expected12 = std::vector<int>{0, 1, 2, 3, 4, 4, 4, 4, 3, 2, 1, 0};
        REQUIRE(size_indices(oh, 12) == expected12);
    }

    SECTION("Off-diagonal") {
        auto oh = kpm::OptimizedHamiltonian<scalat_t>(&matrix);
        auto const i = num_sites / 4;
        auto const j = num_sites / 2;
        oh.optimize_for({i, std::vector<int>{j, j+1, j+2}}, bounds.scaling_factors());

        REQUIRE(oh.idx().row != oh.idx().cols[0]);
        REQUIRE(oh.sizes().get_data().front() == 1);
        REQUIRE(oh.sizes().get_data().back() == num_sites);
        REQUIRE(oh.sizes().get_offset() > 0);

        auto const expected6 = std::vector<int>{0, 1, 2, 3, 3, 3};
        REQUIRE(size_indices(oh, 6) == expected6);
        auto const expected9 = std::vector<int>{0, 1, 2, 3, 4, 4, 4, 4, 3};
        REQUIRE(size_indices(oh, 9) == expected9);
        auto const expected12 = std::vector<int>{0, 1, 2, 3, 4, 5, 6, 6, 6, 5, 4, 3};
        REQUIRE(size_indices(oh, 12) == expected12);
    }
}

TEST_CASE("KPM strategy", "[kpm]") {
    for (auto is_double_precision : {false, true}) {
        for (auto is_complex : {false, true}) {
            INFO("double: " << is_double_precision << ", complex: " << is_complex);
            auto const model = make_test_model(is_double_precision, is_complex);

            auto const num_sites = model.system()->num_sites();
            auto const i = num_sites / 2;
            auto const j = num_sites / 4;
            auto const energy_range = ArrayXd::LinSpaced(10, -0.3, 0.3);
            auto const broadening = 0.8;
            auto const cols = std::vector<int>{i, j, j+1, j+2};
            auto precision = Eigen::NumTraits<float>::dummy_precision();

            struct Result { ArrayXcd g_ii, g_ij; };
            auto results = std::vector<Result>();

            for (auto opt_level = 0; opt_level <= 3; ++opt_level) {
                INFO("opt_level: " << opt_level);
                auto config = KPMConfig{};
                config.optimization_level = opt_level;
                auto kpm = make_greens_strategy<KPM>(model.hamiltonian(), config);

                auto const gs = kpm->calc_vector(i, cols, energy_range, broadening);
                REQUIRE(gs.size() == cols.size());
                REQUIRE_FALSE(gs[0].isApprox(gs[1], precision));
                REQUIRE_FALSE(gs[1].isApprox(gs[2], precision));

                kpm->change_hamiltonian(model.hamiltonian());
                auto const g_ii = kpm->calc(i, i, energy_range, broadening);
                REQUIRE(g_ii.isApprox(gs[0], precision));

                auto const g_ij = kpm->calc(i, j, energy_range, broadening);
                REQUIRE(g_ij.isApprox(gs[1], precision));

                if (!is_complex) {
                    auto const g_ji = kpm->calc(j, i, energy_range, broadening);
                    REQUIRE(g_ij.isApprox(g_ji, precision));
                }

                results.push_back({g_ii, g_ij});
            }

            REQUIRE(results[0].g_ii.isApprox(results[1].g_ii, precision));
            REQUIRE(results[0].g_ii.isApprox(results[2].g_ii, precision));

            REQUIRE(results[0].g_ij.isApprox(results[1].g_ij, precision));
            REQUIRE(results[0].g_ij.isApprox(results[2].g_ij, precision));
        }
    }
}

TEST_CASE("KPM optimization levels", "[kpm]") {
    for (auto is_double_precision : {false, true}) {
        for (auto is_complex : {false, true}) {
            INFO("double: " << is_double_precision << ", complex: " << is_complex);
            auto const model = make_test_model(is_double_precision, is_complex);

            auto const num_sites = model.system()->num_sites();
            auto calc_greens = [&](int i, int j, int opt_level) {
                auto config = KPMConfig{};
                config.optimization_level = opt_level;
                auto kpm = make_greens_strategy<KPM>(model.hamiltonian(), config);
                auto const g = kpm->calc(i, j, ArrayXd::LinSpaced(10, -0.3, 0.3), 0.8);
                return ArrayXcf{g.cast<std::complex<float>>()};
            };

            SECTION("Diagonal") {
                auto const i = num_sites / 2;
                auto const g0 = calc_greens(i, i, /*opt_level*/0);
                auto const g1 = calc_greens(i, i, /*opt_level*/1);
                auto const g2 = calc_greens(i, i, /*opt_level*/2);
                auto const g3 = calc_greens(i, i, /*opt_level*/3);

                REQUIRE(g0.isApprox(g1));
                REQUIRE(g0.isApprox(g2));
                REQUIRE(g0.isApprox(g3));
            }

            SECTION("Off-diagonal") {
                auto const i = num_sites / 2;
                auto const j = num_sites / 4;
                auto const g0 = calc_greens(i, j, /*opt_level*/0);
                auto const g1 = calc_greens(i, j, /*opt_level*/1);
                auto const g2 = calc_greens(i, j, /*opt_level*/2);
                auto const g3 = calc_greens(i, j, /*opt_level*/3);

                REQUIRE(g0.isApprox(g1));
                REQUIRE(g0.isApprox(g2));
                REQUIRE(g0.isApprox(g3));
            }
        }
    }
}
