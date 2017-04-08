#include "kpm/Kernel.hpp"
#include "numeric/constant.hpp"

namespace cpb { namespace kpm {

Kernel jackson_kernel() {
    return {
        [](idx_t num_moments) -> ArrayXd {
            auto const N = static_cast<double>(num_moments);
            auto const Np = N + 1;
            auto const ns = make_integer_range<double>(num_moments);
            constexpr auto pi = double{constant::pi};
            return ns.unaryExpr([&](double n) { // n is not an integer to get proper fp division
                return ((Np - n) * cos(pi * n / Np) + sin(pi * n / Np) / tan(pi / Np)) / Np;
            });
        },
        [](double scaled_broadening) {
            auto const n = static_cast<idx_t>(constant::pi / scaled_broadening) + 1;
            return round_num_moments(n);
        }
    };
}

Kernel lorentz_kernel(double lambda) {
    if (lambda <= 0) { throw std::invalid_argument("Lorentz kernel: lambda must be positive."); }
    return {
        [=](idx_t num_moments) -> ArrayXd {
            auto const N = static_cast<double>(num_moments);
            auto const ns = make_integer_range<double>(num_moments);
            return ns.unaryExpr([&](double n) { // n is not an integer to get proper fp division
                return std::sinh(lambda * (1 - n / N)) / std::sinh(lambda);
            });
        },
        [=](double scaled_broadening) {
            auto const n = static_cast<idx_t>(lambda / scaled_broadening) + 1;
            return round_num_moments(n);
        }
    };
}

Kernel dirichlet_kernel() {
    return {
        [](idx_t num_moments) -> ArrayXd { return ArrayXd::Ones(num_moments); },
        [](double scaled_broadening) {
            auto const n = static_cast<idx_t>(constant::pi / scaled_broadening) + 1;
            return round_num_moments(n);
        }
    };
}

}} // namespace cpb::kpm
