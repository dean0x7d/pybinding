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
    int last_num_moments = 0;
    Chrono moments_timer;

    Stats() = default;
    Stats(int num_moments) : last_num_moments(num_moments) {}

    std::string report(std::uint64_t operations, bool shortform) const {
        auto const moments_with_suffix = fmt::with_suffix(last_num_moments);
        auto const ops_with_suffix = fmt::with_suffix(operations / moments_timer.elapsed_seconds());
        auto const fmt_str = shortform ? "{} @ {}ops"
                                       : "KPM calculated {} moments at {} operations per second";
        auto const msg = fmt::format(fmt_str, moments_with_suffix, ops_with_suffix);
        return format_report(msg, moments_timer, shortform);
    }
};

}} // namespace cpb::kpm
