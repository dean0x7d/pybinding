#pragma once
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
    int num_moments = 0;
    size_t num_operations = 0; ///< approximate number of executed mul + add operations
    size_t matrix_memory = 0; ///< memory used by the Hamiltonian matrix
    size_t vector_memory = 0; ///< memory used by a single KPM vector
    Chrono moments_timer;

    Stats() = default;
    Stats(int num_moments, size_t num_operations, size_t matrix_memory, size_t vector_memory)
        : num_moments(num_moments), num_operations(num_operations),
          matrix_memory(matrix_memory), vector_memory(vector_memory) {}

    /// Operations per second
    double ops() const { return num_operations / moments_timer.elapsed_seconds(); }

    std::string report(bool shortform) const {
        auto const fmt_str = shortform ? "{} @ {}ops"
                                       : "KPM calculated {} moments at {} operations per second";
        auto const msg = fmt::format(fmt_str,
                                     fmt::with_suffix(num_moments),
                                     fmt::with_suffix(ops()));
        return format_report(msg, moments_timer, shortform);
    }
};

}} // namespace cpb::kpm
