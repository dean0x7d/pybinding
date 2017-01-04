#include <catch.hpp>

#include "fixtures.hpp"
#include "KPM.hpp"
using namespace cpb;

Model make_test_model(bool is_double = false, bool is_complex = false) {
    auto model = Model(graphene::monolayer(), shape::rectangle(0.6f, 0.8f),
                       field::constant_potential(1));
    if (is_double) {
        model.add(field::force_double_precision());
    }
    if (is_complex) {
        model.add(field::constant_magnetic_field(1e4));
    }
    return model;
}

TEST_CASE("OptimizedHamiltonian reordering", "[kpm]") {
    auto const model = make_test_model();
    auto const num_sites = model.system()->num_sites();

    using scalat_t = float;
    auto const matrix = ham::get_reference<scalat_t>(model.hamiltonian());
    auto bounds = kpm::Bounds<scalat_t>(&matrix, kpm::Config{}.lanczos_precision);

    auto size_indices = [](kpm::OptimizedHamiltonian<scalat_t> const& oh, int num_moments) {
        auto v = std::vector<int>(num_moments);
        for (auto n = 0; n < num_moments; ++n) {
            v[n] = oh.map().index(n, num_moments);
        }
        return v;
    };

    SECTION("Diagonal") {
        auto oh = kpm::OptimizedHamiltonian<scalat_t>(&matrix, kpm::MatrixFormat::CSR, true);
        auto const i = model.system()->find_nearest({0, 0.07f, 0}, "B");
        oh.optimize_for({i, i}, bounds.scaling_factors());

        REQUIRE(oh.idx().row == 0);
        REQUIRE(oh.idx().cols[0] == 0);
        REQUIRE(oh.map().get_data().front() == 1);
        REQUIRE(oh.map().get_data().back() == num_sites);
        REQUIRE(oh.map().get_offset() == 0);

        auto const expected6 = std::vector<int>{0, 1, 2, 2, 1, 0};
        REQUIRE(size_indices(oh, 6) == expected6);
        auto const expected9 = std::vector<int>{0, 1, 2, 3, 4, 3, 2, 1, 0};
        REQUIRE(size_indices(oh, 9) == expected9);
        auto const expected12 = std::vector<int>{0, 1, 2, 3, 4, 4, 4, 4, 3, 2, 1, 0};
        REQUIRE(size_indices(oh, 12) == expected12);
    }

    SECTION("Off-diagonal") {
        auto oh = kpm::OptimizedHamiltonian<scalat_t>(&matrix, kpm::MatrixFormat::CSR, true);
        auto const i = model.system()->find_nearest({0, 0.35f, 0}, "A");
        auto const j1 = model.system()->find_nearest({0, 0.07f, 0}, "B");
        auto const j2 = model.system()->find_nearest({0.12f, 0.14f, 0}, "A");
        auto const j3 = model.system()->find_nearest({0.12f, 0.28f, 0}, "B");
        oh.optimize_for({i, std::vector<int>{j1, j2, j3}}, bounds.scaling_factors());

        REQUIRE(oh.idx().row != oh.idx().cols[0]);
        REQUIRE(oh.map().get_data().front() == 1);
        REQUIRE(oh.map().get_data().back() == num_sites);
        REQUIRE(oh.map().get_offset() > 0);

        auto const expected6 = std::vector<int>{0, 1, 2, 3, 3, 3};
        REQUIRE(size_indices(oh, 6) == expected6);
        auto const expected9 = std::vector<int>{0, 1, 2, 3, 4, 4, 4, 4, 3};
        REQUIRE(size_indices(oh, 9) == expected9);
        auto const expected12 = std::vector<int>{0, 1, 2, 3, 4, 5, 6, 6, 6, 5, 4, 3};
        REQUIRE(size_indices(oh, 12) == expected12);
    }
}

struct TestGreensResult {
    ArrayXcd g_ii, g_ij;
};

template<template<class> class Strategy>
std::vector<TestGreensResult> test_kpm_strategy(std::vector<kpm::Config> const& configs) {
    constexpr auto pi = double{constant::pi};
    auto results = std::vector<TestGreensResult>();

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
            auto const precision = Eigen::NumTraits<float>::dummy_precision();

            auto unoptimized_result = TestGreensResult{};
            for (auto opt_level = 0; opt_level < configs.size(); ++opt_level) {
                INFO("opt_level: " << opt_level);
                auto strategy = make_kpm_strategy<Strategy>(model.hamiltonian(),
                                                            configs[opt_level]);

                auto const gs = strategy->greens_vector(i, cols, energy_range, broadening);
                REQUIRE(gs.size() == cols.size());
                REQUIRE_FALSE(gs[0].isApprox(gs[1], precision));
                REQUIRE_FALSE(gs[1].isApprox(gs[2], precision));

                strategy->change_hamiltonian(model.hamiltonian());
                auto const g_ii = strategy->greens(i, i, energy_range, broadening);
                REQUIRE(g_ii.isApprox(gs[0], precision));

                auto const g_ij = strategy->greens(i, j, energy_range, broadening);
                REQUIRE(g_ij.isApprox(gs[1], precision));

                if (!is_complex) {
                    auto const g_ji = strategy->greens(j, i, energy_range, broadening);
                    REQUIRE(g_ij.isApprox(g_ji, precision));
                }

                auto const ldos = strategy->ldos(i, energy_range, broadening);
                REQUIRE(ldos.isApprox(-1/pi * g_ii.imag(), precision));

                if (opt_level == 0) {
                    unoptimized_result = {g_ii, g_ij};
                } else {
                    REQUIRE(g_ii.isApprox(unoptimized_result.g_ii, precision));
                    REQUIRE(g_ij.isApprox(unoptimized_result.g_ij, precision));
                }
            } // for opt_level

            results.push_back(unoptimized_result);
        } // for is_complex
    } // for is_double_precision

    return results;
}

TEST_CASE("KPM strategy", "[kpm]") {
    auto make_config = [](kpm::MatrixFormat matrix_format, bool optimal_size, bool interleaved) {
        auto config = kpm::Config{};
        config.matrix_format = matrix_format;
        config.algorithm.optimal_size = optimal_size;
        config.algorithm.interleaved = interleaved;
        return config;
    };

#ifndef CPB_USE_CUDA
    test_kpm_strategy<kpm::DefaultStrategy>({
        make_config(kpm::MatrixFormat::CSR, false, false),
        make_config(kpm::MatrixFormat::CSR, true,  false),
        make_config(kpm::MatrixFormat::CSR, false,  true),
        make_config(kpm::MatrixFormat::CSR, true,  true),
        make_config(kpm::MatrixFormat::ELL, false, false),
        make_config(kpm::MatrixFormat::ELL, true,  false),
        make_config(kpm::MatrixFormat::ELL, false,  true),
        make_config(kpm::MatrixFormat::ELL, true,  true),
    });
#else
    auto const cpu_results = test_kpm_strategy<kpm::DefaultStrategy>({
        make_config(kpm::MatrixFormat::ELL, true,  true)
    });
    auto const cuda_results = test_kpm_strategy<kpm::CudaStrategy>({
        make_config(kpm::MatrixFormat::ELL, false, false),
        make_config(kpm::MatrixFormat::ELL, true,  false)
    });
    auto const precision = Eigen::NumTraits<float>::dummy_precision();
    for (auto i = 0u; i < cuda_results.size(); ++i) {
        REQUIRE(cuda_results[i].g_ii.isApprox(cpu_results[0].g_ii, precision));
        REQUIRE(cuda_results[i].g_ij.isApprox(cpu_results[0].g_ij, precision));
    }
#endif // CPB_USE_CUDA
}
