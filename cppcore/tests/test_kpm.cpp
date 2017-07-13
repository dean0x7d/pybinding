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
    auto bounds = kpm::Bounds(model.hamiltonian(), kpm::Config{}.lanczos_precision);

    auto size_indices = [](kpm::OptimizedHamiltonian const& oh, int num_moments) {
        auto v = std::vector<idx_t>(num_moments);
        for (auto n = 0; n < num_moments; ++n) {
            v[n] = oh.map().index(n, num_moments);
        }
        return v;
    };
    auto equals = [](std::vector<idx_t> const& v) { return Catch::Equals(v); };

    SECTION("Diagonal single") {
        auto oh = kpm::OptimizedHamiltonian(model.hamiltonian(), kpm::MatrixFormat::CSR, true);
        auto const i = model.system()->find_nearest({0, 0.07f, 0}, "B");
        oh.optimize_for({i, i}, bounds.scaling_factors());

        REQUIRE(oh.idx().src[0] == 0);
        REQUIRE(oh.idx().dest[0] == 0);
        REQUIRE(oh.idx().is_diagonal());
        REQUIRE(oh.map().get_data().front() == 1);
        REQUIRE(oh.map().get_data().back() == num_sites);
        REQUIRE(oh.map().get_data().size() == 5);
        REQUIRE(oh.map().get_src_offset() == 0);
        REQUIRE(oh.map().get_dest_offset() == 0);

        REQUIRE_THAT(size_indices(oh,  6), equals({0, 1, 2, 2, 1, 0}));
        REQUIRE_THAT(size_indices(oh,  9), equals({0, 1, 2, 3, 4, 3, 2, 1, 0}));
        REQUIRE_THAT(size_indices(oh, 12), equals({0, 1, 2, 3, 4, 4, 4, 4, 3, 2, 1, 0}));
    }

    SECTION("Diagonal multi 1") {
        auto oh = kpm::OptimizedHamiltonian(model.hamiltonian(), kpm::MatrixFormat::CSR, true);
        auto const i1 = model.system()->find_nearest({0, -0.07f, 0}, "A");
        auto const i2 = model.system()->find_nearest({0,  0.07f, 0}, "B");
        REQUIRE(i1 != i2);

        auto const idx = std::vector<idx_t>{i1, i2};
        oh.optimize_for({idx, idx}, bounds.scaling_factors());

        REQUIRE(oh.idx().src[0] == 0);
        REQUIRE(oh.idx().src[1] == 3);
        REQUIRE(oh.idx().dest[0] == 0);
        REQUIRE(oh.idx().dest[1] == 3);
        REQUIRE(oh.idx().is_diagonal());
        REQUIRE(oh.map().get_data().front() == 1);
        REQUIRE(oh.map().get_data().back() == num_sites);
        REQUIRE(oh.map().get_data().size() == 5);
        REQUIRE(oh.map().get_src_offset() == 1);
        REQUIRE(oh.map().get_dest_offset() == 1);

        REQUIRE_THAT(size_indices(oh,  6), equals({1, 2, 3, 3, 2, 1}));
        REQUIRE_THAT(size_indices(oh,  9), equals({1, 2, 3, 4, 4, 4, 3, 2, 1}));
        REQUIRE_THAT(size_indices(oh, 12), equals({1, 2, 3, 4, 4, 4, 4, 4, 4, 3, 2, 1}));
    }

    SECTION("Diagonal multi 2") {
        auto oh = kpm::OptimizedHamiltonian(model.hamiltonian(), kpm::MatrixFormat::CSR, true);
        auto const i1 = model.system()->find_nearest({0,  0.07f, 0}, "B");
        auto const i2 = model.system()->find_nearest({0, -0.07f, 0}, "A");
        auto const i3 = model.system()->find_nearest({0,  0.35f, 0}, "A");
        auto const idx = std::vector<idx_t>{i1, i2, i3};
        oh.optimize_for({idx, idx}, bounds.scaling_factors());

        REQUIRE(oh.idx().src[0] == 0);
        REQUIRE(oh.idx().src[1] == 1);
        REQUIRE(oh.idx().src[2] == 15);
        REQUIRE(oh.idx().dest[0] == 0);
        REQUIRE(oh.idx().dest[1] == 1);
        REQUIRE(oh.idx().dest[2] == 15);
        REQUIRE(oh.idx().is_diagonal());
        REQUIRE(oh.map().get_data().front() == 1);
        REQUIRE(oh.map().get_data().back() == num_sites);
        REQUIRE(oh.map().get_data().size() == 5);
        REQUIRE(oh.map().get_src_offset() == 3);
        REQUIRE(oh.map().get_dest_offset() == 3);

        REQUIRE_THAT(size_indices(oh,  6), equals({3, 4, 4, 4, 4, 3}));
        REQUIRE_THAT(size_indices(oh,  9), equals({3, 4, 4, 4, 4, 4, 4, 4, 3}));
        REQUIRE_THAT(size_indices(oh, 12), equals({3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3}));
    }

    SECTION("Off-diagonal single") {
        auto oh = kpm::OptimizedHamiltonian(model.hamiltonian(), kpm::MatrixFormat::CSR, true);
        auto const i = model.system()->find_nearest({0, 0.35f, 0}, "A");
        auto const j = model.system()->find_nearest({0, 0.07f, 0}, "B");
        oh.optimize_for({i, j}, bounds.scaling_factors());

        REQUIRE(oh.idx().src[0] == 0);
        REQUIRE(oh.idx().dest[0] == 8);
        REQUIRE(oh.idx().is_diagonal() == false);
        REQUIRE(oh.map().get_data().front() == 1);
        REQUIRE(oh.map().get_data().back() == num_sites);
        REQUIRE(oh.map().get_data().size() == 8);
        REQUIRE(oh.map().get_src_offset() == 0);
        REQUIRE(oh.map().get_dest_offset() == 3);

        REQUIRE_THAT(size_indices(oh,  6), equals({0, 1, 2, 3, 4, 3}));
        REQUIRE_THAT(size_indices(oh,  9), equals({0, 1, 2, 3, 4, 5, 5, 4, 3}));
        REQUIRE_THAT(size_indices(oh, 12), equals({0, 1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 3}));
        REQUIRE_THAT(size_indices(oh, 14), equals({0, 1, 2, 3, 4, 5, 6, 7, 7, 7, 6, 5, 4, 3}));
    }

    SECTION("Off-diagonal multi 1") {
        auto oh = kpm::OptimizedHamiltonian(model.hamiltonian(), kpm::MatrixFormat::CSR, true);
        auto const i = model.system()->find_nearest({0, 0.35f, 0}, "A");
        auto const j1 = model.system()->find_nearest({0, 0.07f, 0}, "B");
        auto const j2 = model.system()->find_nearest({0.12f, 0.14f, 0}, "A");
        auto const j3 = model.system()->find_nearest({0.12f, 0.28f, 0}, "B");
        oh.optimize_for({i, std::vector<idx_t>{j1, j2, j3}}, bounds.scaling_factors());

        REQUIRE(oh.idx().src[0] == 0);
        REQUIRE(oh.idx().dest[0] == 8);
        REQUIRE(oh.idx().dest[1] == 5);
        REQUIRE(oh.idx().dest[2] == 2);
        REQUIRE(oh.map().get_data().front() == 1);
        REQUIRE(oh.map().get_data().back() == num_sites);
        REQUIRE(oh.map().get_data().size() == 8);
        REQUIRE(oh.map().get_src_offset() == 0);
        REQUIRE(oh.map().get_dest_offset() == 3);

        REQUIRE_THAT(size_indices(oh,  6), equals({0, 1, 2, 3, 4, 3}));
        REQUIRE_THAT(size_indices(oh,  9), equals({0, 1, 2, 3, 4, 5, 5, 4, 3}));
        REQUIRE_THAT(size_indices(oh, 12), equals({0, 1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 3}));
    }

    SECTION("Off-diagonal multi 2") {
        auto oh = kpm::OptimizedHamiltonian(model.hamiltonian(), kpm::MatrixFormat::CSR, true);
        auto const i1 = model.system()->find_nearest({0,  0.35f, 0}, "A");
        auto const i2 = model.system()->find_nearest({0, -0.35f, 0}, "B");
        auto const idx1 = std::vector<idx_t>{i1, i2};
        auto const j1 = model.system()->find_nearest({ 0.12f, 0.28f, 0}, "B");
        auto const j2 = model.system()->find_nearest({-0.12f, 0.28f, 0}, "B");
        auto const idx2 = std::vector<idx_t>{j1, j2};
        oh.optimize_for({idx1, idx2}, bounds.scaling_factors());

        REQUIRE(oh.idx().src[0] == 0);
        REQUIRE(oh.idx().src[1] == 18);
        REQUIRE(oh.idx().dest[0] == 2);
        REQUIRE(oh.idx().dest[1] == 1);
        REQUIRE(oh.idx().is_diagonal() == false);
        REQUIRE(oh.map().get_data().front() == 1);
        REQUIRE(oh.map().get_data().back() == num_sites);
        REQUIRE(oh.map().get_data().size() == 8);
        REQUIRE(oh.map().get_src_offset() == 7);
        REQUIRE(oh.map().get_dest_offset() == 1);

        REQUIRE_THAT(size_indices(oh,  6), equals({6, 5, 4, 3, 2, 1}));
        REQUIRE_THAT(size_indices(oh,  9), equals({7, 7, 7, 6, 5, 4, 3, 2, 1}));
        REQUIRE_THAT(size_indices(oh, 12), equals({7, 7, 7, 7, 7, 7, 6, 5, 4, 3, 2, 1}));
    }
}

TEST_CASE("OptimizedHamiltonian scaling") {
    auto model = Model(graphene::monolayer(), shape::rectangle(0.6f, 0.8f));
    auto oh = kpm::OptimizedHamiltonian(model.hamiltonian(), kpm::MatrixFormat::CSR, true);
    auto bounds = kpm::Bounds(-12, 10); // ensures `scale.b != 0`
    oh.optimize_for({0, 0}, bounds.scaling_factors());
    auto scaled = oh.matrix().get<SparseMatrixX<float>>();

    // Because `scale.b != 0` the scaled Hamiltonian matrix should get
    // a non-zero diagonal even if the original matrix didn't have one.
    REQUIRE(scaled.nonZeros() == model.hamiltonian().non_zeros() + model.hamiltonian().rows());
}

struct TestGreensResult {
    ArrayXcd g_ii, g_ij;

    TestGreensResult() = default;
};

struct TestDosResult {
    ArrayXd dos1, dos2, dos3, dos20;

    TestDosResult() = default;
};

std::vector<TestGreensResult> test_kpm_core(kpm::Compute const& compute,
                                            std::vector<kpm::Config> const& configs) {
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
            auto const cols = std::vector<idx_t>{i, j, j+1, j+2};
            auto const precision = Eigen::NumTraits<float>::dummy_precision();

            auto unoptimized_greens = TestGreensResult();
            auto unoptimized_dos = TestDosResult();

            for (auto opt_level = size_t{0}; opt_level < configs.size(); ++opt_level) {
                INFO("opt_level: " << opt_level);
                auto core = kpm::Core(model.hamiltonian(), compute, configs[opt_level]);

                auto const gs = core.greens_vector(i, cols, energy_range, broadening);
                REQUIRE(gs.size() == cols.size());
                REQUIRE_FALSE(gs[0].isApprox(gs[1], precision));
                REQUIRE_FALSE(gs[1].isApprox(gs[2], precision));

                core.set_hamiltonian(model.hamiltonian());
                auto const g_ii = core.greens(i, i, energy_range, broadening);
                REQUIRE(g_ii.isApprox(gs[0], precision));

                auto const g_ij = core.greens(i, j, energy_range, broadening);
                REQUIRE(g_ij.isApprox(gs[1], precision));

                if (!is_complex) {
                    auto const g_ji = core.greens(j, i, energy_range, broadening);
                    REQUIRE(g_ij.isApprox(g_ji, precision));
                }

                auto const ldos0 = core.ldos({i}, energy_range, broadening);
                auto const ldos1 = core.ldos({j}, energy_range, broadening);
                REQUIRE(ldos0.isApprox(-1/pi * g_ii.imag(), precision));
                REQUIRE_FALSE(ldos0.isApprox(ldos1, precision));

                auto const ldos2 = core.ldos({i, j, i, j, i, j, i, j, i, j},
                                             energy_range, broadening);
                REQUIRE(ldos2.cols() == 10);
                for (auto n = 0; n < 10; n += 2) {
                    REQUIRE(ldos0.isApprox(ldos2.col(n + 0), precision));
                    REQUIRE(ldos1.isApprox(ldos2.col(n + 1), precision));
                }

                if (opt_level == 0) {
                    unoptimized_greens = {g_ii, g_ij};
                } else {
                    REQUIRE(g_ii.isApprox(unoptimized_greens.g_ii, precision));
                    REQUIRE(g_ij.isApprox(unoptimized_greens.g_ij, precision));
                }

                auto const dos1 = core.dos(energy_range, broadening, 1);
                auto const dos2 = core.dos(energy_range, broadening, 2);
                auto const dos3 = core.dos(energy_range, broadening, 3);
                auto const dos20 = core.dos(energy_range, broadening, 20);
                REQUIRE_FALSE(dos1.isApprox(dos2, precision));
                REQUIRE_FALSE(dos2.isApprox(dos3, precision));
                REQUIRE_FALSE(dos3.isApprox(dos20, precision));
                REQUIRE_FALSE(dos20.isApprox(dos1, precision));

                if (opt_level == 0) {
                    unoptimized_dos = {dos1, dos2, dos3, dos20};
                } else {
                    REQUIRE(dos1.isApprox(unoptimized_dos.dos1, precision));
                    REQUIRE(dos2.isApprox(unoptimized_dos.dos2, precision));
                    REQUIRE(dos3.isApprox(unoptimized_dos.dos3, precision));
                    REQUIRE(dos20.isApprox(unoptimized_dos.dos20, precision));
                }
            } // for opt_level

            results.push_back(unoptimized_greens);
        } // for is_complex
    } // for is_double_precision

    return results;
}

TEST_CASE("KPM core", "[kpm]") {
    auto make_config = [](kpm::MatrixFormat matrix_format, bool optimal_size, bool interleaved) {
        auto config = kpm::Config{};
        config.matrix_format = matrix_format;
        config.algorithm.optimal_size = optimal_size;
        config.algorithm.interleaved = interleaved;
        return config;
    };

#ifndef CPB_USE_CUDA
    test_kpm_core(kpm::DefaultCompute(), {
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
