#pragma once
#include "utils/Chrono.hpp"
#include "detail/config.hpp"
#include "support/format.hpp"

namespace cpb { namespace kpm {

inline std::string format_report(std::string msg, Chrono const& time, bool shortform) {
    auto const fmt_str = shortform ? "{:s} [{}] " : "- {:<80s} | {}\n";
    return fmt::format(fmt_str, msg, time);
}

struct AlgorithmConfig;
class OptimizedHamiltonian;

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

    void reset(idx_t num_moments, OptimizedHamiltonian const& oh,
               AlgorithmConfig const& ac, idx_t multiplier = 1);

    /// Non-zero elements per second
    double eps() const;

    /// Approximate number of executed mul + add operations per second
    double ops(bool is_diagonal, bool non_unit_vector) const;

    std::string report(bool shortform) const;
};

}} // namespace cpb::kpm
