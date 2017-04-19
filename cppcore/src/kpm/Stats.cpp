#include "kpm/Stats.hpp"

#include "kpm/Config.hpp"
#include "kpm/OptimizedHamiltonian.hpp"

namespace cpb { namespace kpm {

namespace {
    std::string hamiltonian_report(Stats const& s, bool shortform) {
        auto const nnz_diff = static_cast<double>(s.nnz - s.opt_nnz);
        auto const percent_removed = 100.0 * nnz_diff / static_cast<double>(s.nnz);
        auto const not_efficient = s.uses_full_system ? "" : "*";
        auto const fmt_str = shortform ? "{:.0f}%{}"
                                       : "The reordering optimization was able to "
                                         "remove {:.0f}%{} of the workload";
        auto const msg = fmt::format(fmt_str, percent_removed, not_efficient);
        return format_report(msg, s.hamiltonian_timer, shortform);
    }

    std::string moments_report(Stats const& s, bool shortform) {
        auto const fmt_str = shortform ? "{} @ {}eps"
                                       : "KPM calculated {} moments "
                                         "at {} non-zero elements per second";
        auto const msg = fmt::format(fmt_str,
                                     fmt::with_suffix(s.num_moments),
                                     fmt::with_suffix(s.eps()));
        return format_report(msg, s.moments_timer, shortform);
    }
}

void Stats::reset(idx_t num_moments, OptimizedHamiltonian const& oh,
                  AlgorithmConfig const& ac, idx_t multiplier) {
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

double Stats::eps() const {
    return multiplier * static_cast<double>(opt_nnz) / moments_timer.elapsed_seconds();
}

double Stats::ops(bool is_diagonal, bool non_unit_vector) const {
    auto operations = size_t{0};
    operations += nnz * 2; // 1 mul + 1 add per nnz
    operations += vec; // 1 sub

    if (is_diagonal) {
        operations += vec * 4; // 2 * (1 mul + 1 add) for the dot products
    } else if (non_unit_vector) {
        operations += vec *2; // 1 mul + 1 add for the single dot product
    }

    return multiplier * static_cast<double>(operations) / moments_timer.elapsed_seconds();
}

std::string Stats::report(bool shortform) const {
    return hamiltonian_report(*this, shortform) + moments_report(*this, shortform);
}

}} // namespace cpb::kpm
