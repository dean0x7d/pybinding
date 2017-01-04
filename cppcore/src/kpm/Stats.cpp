#include "kpm/Stats.hpp"

namespace cpb { namespace kpm {

namespace {
    std::string hamiltonian_report(Stats const& s, bool shortform) {
        auto const percent_removed = 100.0 * (s.nnz - s.opt_nnz) / static_cast<double>(s.nnz);
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

double Stats::ops(bool is_diagonal, bool non_unit_vector) const {
    auto operations = size_t{0};
    operations += nnz * 2; // 1 mul + 1 add per nnz
    operations += vec; // 1 sub

    if (is_diagonal) {
        operations += vec * 4; // 2 * (1 mul + 1 add) for the dot products
    } else if (non_unit_vector) {
        operations += vec *2; // 1 mul + 1 add for the single dot product
    }

    return multiplier * operations / moments_timer.elapsed_seconds();
}

std::string Stats::report(bool shortform) const {
    return hamiltonian_report(*this, shortform) + moments_report(*this, shortform);
}

}} // namespace cpb::kpm
