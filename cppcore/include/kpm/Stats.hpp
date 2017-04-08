#pragma once
#include "kpm/Config.hpp"
#include "kpm/OptimizedHamiltonian.hpp"

#include "utils/Chrono.hpp"
#include "support/format.hpp"

namespace cpb { namespace kpm {

inline std::string format_report(std::string msg, Chrono const& time, bool shortform) {
    auto const fmt_str = shortform ? "{:s} [{}] " : "- {:<80s} | {}\n";
    return fmt::format(fmt_str, msg, time);
}

/**
 Stats of the KPM calculation
 */
struct Stats {
    idx_t num_moments;
    bool uses_full_system;

    size_t nnz; ///< original number of processed non-zero matrix elements (over all iterations)
    size_t opt_nnz; ///< same as above, but with optimizations applied (if any)
    size_t vec; ///< number of elements in a single KPM vector times the number of moments
    size_t opt_vec; ///< same as above, but with optimizations applied (if any)
    double multiplier = 1; ///< account for any repeated calculations

    size_t matrix_memory; ///< memory used by the Hamiltonian matrix
    size_t vector_memory; ///< memory used by a single KPM vector

    Chrono hamiltonian_timer;
    Chrono moments_timer;

    template<class scalar_t>
    void reset(idx_t num_moments, OptimizedHamiltonian<scalar_t> const& oh,
               AlgorithmConfig const& ac, idx_t multiplier = 1) {
        this->num_moments = num_moments;
        uses_full_system = oh.map().uses_full_system(num_moments);

        nnz = oh.num_nonzeros(num_moments, /*optimal_size*/false);
        opt_nnz = oh.num_nonzeros(num_moments, ac.optimal_size);
        vec = oh.num_vec_elements(num_moments, /*optimal_size*/false);
        opt_vec = oh.num_vec_elements(num_moments, ac.optimal_size);
        this->multiplier = static_cast<double>(multiplier);

        matrix_memory = oh.matrix_memory();
        vector_memory = oh.vector_memory();

        hamiltonian_timer = oh.timer;
        moments_timer = {};
    }

    /// Non-zero elements per second
    double eps() const {
        return multiplier * static_cast<double>(opt_nnz) / moments_timer.elapsed_seconds();
    }

    /// Approximate number of executed mul + add operations per second
    double ops(bool is_diagonal, bool non_unit_vector) const;

    std::string report(bool shortform) const;
};

}} // namespace cpb::kpm
