#include "kpm/Kernel.hpp"
#include "numeric/constant.hpp"

namespace cpb { namespace kpm {

namespace {
    // Moment calculations at higher optimization levels require specific rounding.
    // `n - 2` considers only moments in the main KPM loop. Divisible by 4 because
    // that is the strictest requirement imposed by `opt_size_and_interleaved`.
    inline int round_num_moments(int n) {
        while ((n - 2) % 4 != 0) { ++n; }
        return n;
    }
} // anonymous namespace

Kernel jackson_kernel() {
    using constant::pi;
    return {
        [=](int N) -> ArrayXd {
            auto const Np = N + 1;
            auto const ns = make_integer_range<double>(N);
            return ns.unaryExpr([&](double n) { // n is not an integer to get proper fp division
                return ((Np - n) * cos(pi * n / Np) + sin(pi * n / Np) / tan(pi / Np)) / Np;
            });
        },
        [=](double scaled_broadening) {
            auto const n = static_cast<int>(pi / scaled_broadening) + 1;
            return round_num_moments(n);
        }
    };
}

Kernel lorentz_kernel(double lambda) {
    if (lambda <= 0) { throw std::invalid_argument("Lorentz kernel: lambda must be positive."); }
    return {
        [=](int N) -> ArrayXd {
            auto const ns = make_integer_range<double>(N);
            return ns.unaryExpr([&](double n) { // n is not an integer to get proper fp division
                return std::sinh(lambda * (1 - n / N)) / std::sinh(lambda);
            });
        },
        [=](double scaled_broadening) {
            auto const n = static_cast<int>(lambda / scaled_broadening) + 1;
            return round_num_moments(n);
        }
    };
}

}} // namespace cpb::kpm
