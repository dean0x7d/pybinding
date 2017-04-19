#include "kpm/Bounds.hpp"
#include "kpm/Stats.hpp"

#include "compute/lanczos.hpp"

namespace cpb { namespace kpm {

namespace {

struct MinMaxEigenvalues {
    double precision_percent;

    template<class scalar_t>
    compute::LanczosBounds operator()(SparseMatrixRC<scalar_t> const& ph) const {
        return compute::minmax_eigenvalues(*ph, precision_percent);
    }
};

} // anonymous namespace

void Bounds::compute_bounds() {
    if (!hamiltonian || min != max) { return; }

    timer.tic();
    auto const lanczos = hamiltonian.get_variant().match(MinMaxEigenvalues{precision_percent});
    timer.toc();

    min = lanczos.min;
    max = lanczos.max;
    lanczos_loops = lanczos.loops;
}

std::string Bounds::report(bool shortform) const {
    auto const fmt_str = shortform ? "{:.2f}, {:.2f}, {}"
                                   : "Spectrum bounds found ({:.2f}, {:.2f} eV) "
                                     "using Lanczos procedure with {} loops";
    auto const msg = fmt::format(fmt_str, min, max, lanczos_loops);
    return format_report(msg, timer, shortform);
}

}} // namespace cpb::kpm
