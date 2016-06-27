#include "greens/kpm/Bounds.hpp"
#include "greens/kpm/Stats.hpp"

#include "compute/lanczos.hpp"

namespace cpb { namespace kpm {

template<class scalar_t>
void Bounds<scalar_t>::compute_factors() {
    timer.tic();
    auto const lanczos = compute::minmax_eigenvalues(*matrix, precision_percent);
    min = lanczos.min;
    max = lanczos.max;
    lanczos_loops = lanczos.loops;
    factors = {min, max};
    timer.toc();
}

template<class scalar_t>
std::string Bounds<scalar_t>::report(bool shortform) const {
    auto const fmt_str = shortform ? "{:.2f}, {:.2f}, {}"
                                   : "Spectrum bounds found ({:.2f}, {:.2f} eV) "
                                     "using Lanczos procedure with {} loops";
    auto const msg = fmt::format(fmt_str, min, max, lanczos_loops);
    return format_report(msg, timer, shortform);
}

CPB_INSTANTIATE_TEMPLATE_CLASS(Bounds)

}} // namespace cpb::kpm
