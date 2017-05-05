#include "kpm/Core.hpp"

#include "kpm/reconstruct.hpp"

namespace cpb { namespace kpm {

namespace {
    Bounds reset_bounds(Hamiltonian const& h, Config const& config) {
        if (config.min_energy == config.max_energy) {
            return {h, config.lanczos_precision}; // will be automatically computed
        } else {
            return {config.min_energy, config.max_energy}; // user-defined bounds
        }
    }
} // anonymous namespace

Core::Core(Hamiltonian const& h, Compute const& compute, Config const& config)
    : hamiltonian(h), compute(compute), config(config), bounds(reset_bounds(h, config)),
      optimized_hamiltonian(h, config.matrix_format, config.algorithm.reorder()) {
    if (config.min_energy > config.max_energy) {
        throw std::invalid_argument("KPM: Invalid energy range specified (min > max).");
    }
}

void Core::set_hamiltonian(Hamiltonian const& h) {
    hamiltonian = h;
    optimized_hamiltonian = {h, config.matrix_format, config.algorithm.reorder()};
    bounds = reset_bounds(h, config);
}

std::string Core::report(bool shortform) const {
    return bounds.report(shortform) + stats.report(shortform) + (shortform ? "|" : "Total time:");
}

ArrayXcd Core::moments(idx_t num_moments, VectorXcd const& alpha, VectorXcd const& beta,
                       SparseMatrixXcd const& op) {
    auto specialized_algorithm = config.algorithm;
    specialized_algorithm.optimal_size = false; // not applicable for this calculation

    optimized_hamiltonian.optimize_for({0, 0}, bounds.scaling_factors());
    stats.reset(num_moments, optimized_hamiltonian, specialized_algorithm);

    auto const starter = constant_starter(optimized_hamiltonian, alpha);

    if (beta.size() == 0 && op.size() == 0) {
        auto moments = DiagonalMoments(round_num_moments(num_moments));
        timed_compute(&moments, starter, specialized_algorithm);
        apply_damping(moments, config.kernel);
        return extract_data(moments, num_moments);
    } else {
        auto moments = GenericMoments(round_num_moments(num_moments), alpha, beta, op);
        timed_compute(&moments, starter, specialized_algorithm);
        apply_damping(moments, config.kernel);
        return extract_data(moments, num_moments);
    }
}

ArrayXXdCM Core::ldos(std::vector<idx_t> const& idx, ArrayXd const& energy, double broadening) {
    auto const scale = bounds.scaling_factors();
    auto const num_moments = config.kernel.required_num_moments(broadening / scale.a);
    auto const num_indices = static_cast<idx_t>(idx.size());

    optimized_hamiltonian.optimize_for({idx, idx}, scale);
    stats.reset(num_moments, optimized_hamiltonian, config.algorithm, num_indices);

    auto const scalar_tag = optimized_hamiltonian.scalar_tag();
    auto const batch_size = compute->batch_size(scalar_tag);
    auto all_moments = MomentConcatenator(num_moments, num_indices, scalar_tag);

    if (num_indices <= 2) {
        auto starter = unit_starter(optimized_hamiltonian);
        auto moments = DiagonalMoments(num_moments);

        for (auto n = idx_t{0}; n < num_indices; ++n) {
            timed_compute(&moments, starter, config.algorithm);
            all_moments.add(moments.data);
        }
    } else {
        auto starter = unit_starter(optimized_hamiltonian, batch_size);
        auto moments = BatchDiagonalMoments(num_moments, batch_size);

        for (auto n = idx_t{0}; n < num_indices; n += batch_size) {
            timed_compute(&moments, starter, config.algorithm);
            all_moments.add(moments.data);
        }
    }

    apply_damping(all_moments, config.kernel);
    return reconstruct<SpectralDensity>(all_moments, energy, scale);
}

ArrayXd Core::dos(ArrayXd const& energy, double broadening, idx_t num_random) {
    auto const scale = bounds.scaling_factors();
    auto const num_moments = config.kernel.required_num_moments(broadening / scale.a);

    auto specialized_algorithm = config.algorithm;
    specialized_algorithm.optimal_size = false; // not applicable for this calculation

    optimized_hamiltonian.optimize_for({0, 0}, scale);
    stats.reset(num_moments, optimized_hamiltonian, specialized_algorithm, num_random);

    auto const batch_size = compute->batch_size(optimized_hamiltonian.scalar_tag());
    auto total_mu = MomentAccumulator(num_moments, num_random, batch_size);

    if (num_random <= 2) {
        auto starter = random_starter(optimized_hamiltonian);
        auto moments = DiagonalMoments(num_moments);

        for (auto n = idx_t{0}; n < num_random; ++n) {
            timed_compute(&moments, starter, specialized_algorithm);
            total_mu.add(moments.data);
        }
    } else {
        auto starter = random_starter(optimized_hamiltonian, batch_size);
        auto moments = BatchDiagonalMoments(num_moments, batch_size);

        for (auto n = idx_t{0}; n < num_random; n += batch_size) {
            timed_compute(&moments, starter, specialized_algorithm);
            total_mu.add(moments.data);
        }
    }

    apply_damping(total_mu, config.kernel);
    return reconstruct<SpectralDensity>(total_mu, energy, scale);
}

ArrayXcd Core::greens(idx_t row, idx_t col, ArrayXd const& energy, double broadening) {
    return std::move(greens_vector(row, {col}, energy, broadening).front());
}

std::vector<ArrayXcd> Core::greens_vector(idx_t row, std::vector<idx_t> const& cols,
                                          ArrayXd const& energy, double broadening) {
    assert(!cols.empty());
    auto const scale = bounds.scaling_factors();
    auto const num_moments = config.kernel.required_num_moments(broadening / scale.a);

    auto& oh = optimized_hamiltonian;
    oh.optimize_for({row, cols}, scale);
    stats.reset(num_moments, oh, config.algorithm);

    if (oh.idx().is_diagonal()) {
        auto moments = DiagonalMoments(num_moments);
        timed_compute(&moments, unit_starter(oh), config.algorithm);
        apply_damping(moments, config.kernel);
        return {reconstruct<GreensFunction>(moments, energy, scale)};
    } else {
        auto moments_vector = MultiUnitMoments(num_moments, oh.idx());
        timed_compute(&moments_vector, unit_starter(oh), config.algorithm);
        apply_damping(moments_vector, config.kernel);
        return reconstruct<GreensFunction>(moments_vector, energy, scale);
    }
}

ArrayXcd Core::conductivity(ArrayXf const& left_coords, ArrayXf const& right_coords,
                            ArrayXd const& chemical_potential, double broadening,
                            double temperature, idx_t num_random, idx_t num_points) {
    auto const scale = bounds.scaling_factors();
    auto const num_moments = config.kernel.required_num_moments(broadening / scale.a);

    auto specialized_algorithm = config.algorithm;
    specialized_algorithm.optimal_size = false; // not applicable for this calculation

    optimized_hamiltonian.optimize_for({0, 0}, scale);
    stats.reset(num_moments, optimized_hamiltonian, specialized_algorithm, num_random);

    // On the left, the velocity operator is only applied to the starter
    auto starter_l = random_starter(optimized_hamiltonian, velocity(hamiltonian, left_coords));
    auto moments_l = DenseMatrixMoments(num_moments);

    // On the right, the operator is applied at collection time to each vector
    auto starter_r = random_starter(optimized_hamiltonian);
    auto moments_r = DenseMatrixMoments(num_moments, velocity(hamiltonian, right_coords));

    auto total_mu = MomentMultiplication(num_moments, optimized_hamiltonian.scalar_tag());
    for (auto j = 0; j < num_random; ++j) {
        timed_compute(&moments_l, starter_l, specialized_algorithm);
        timed_compute(&moments_r, starter_r, specialized_algorithm);
        total_mu.matrix_mul_add(moments_l, moments_r);
    }
    total_mu.normalize(num_random);

    apply_damping(total_mu, config.kernel);
    return reconstruct<KuboBastin>(total_mu, chemical_potential, bounds.linspaced(num_points),
                                   temperature, scale);
}

void Core::timed_compute(MomentsRef m, Starter const& starter, AlgorithmConfig const& ac) {
    stats.moments_timer.tic();
    compute->moments(std::move(m), starter, ac, optimized_hamiltonian);
    stats.moments_timer.toc_accumulate();
}

}} // namespace cpb::kpm
