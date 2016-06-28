#pragma once
#include <format.h>

namespace fmt {

/**
 Convert number to string with SI suffix, e.g.: 14226 -> 14.2k, 5395984 -> 5.39M
 */
inline std::string with_suffix(double number) {
    struct Pair {
        double value;
        char const* suffix;
    };
    static constexpr Pair mapping[] = {{1e9, "G"}, {1e6, "M"}, {1e3, "k"}};

    auto const result = [&]{
        for (auto const& bucket : mapping) {
            if (number > 0.999 * bucket.value) {
                return Pair{number / bucket.value, bucket.suffix};
            }
        }
        return Pair{number, ""};
    }();

    return fmt::format("{:.3g}{}", result.value, result.suffix);
}

} // namespace fmt
