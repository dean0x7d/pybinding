#pragma once
#include "kpm/OptimizedHamiltonian.hpp"

#include "numeric/dense.hpp"
#include "numeric/constant.hpp"
#include "numeric/random.hpp"
#include "compute/detail.hpp"

namespace cpb { namespace kpm {

/// Return the unit vector starter r0 for the KPM procedure (`oh` encodes the unit index)
template<class scalar_t>
VectorX<scalar_t> unit_starter(OptimizedHamiltonian const& oh, var::tag<scalar_t>) {
    auto r0 = VectorX<scalar_t>::Zero(oh.size()).eval();
    r0[oh.idx().row] = 1;
    return r0;
}

/// Return the starter vector r0 for the stochastic KPM procedure
template<class real_t>
VectorX<real_t> random_starter(OptimizedHamiltonian const& oh, std::mt19937& generator,
                               var::tag<real_t>) {
    auto r0 = num::make_random<VectorX<real_t>>(oh.size(), generator);
    oh.reorder(r0); // needed to maintain consistent results for all optimizations
    return transform<VectorX>(r0, [](real_t x) -> real_t { return (x < 0.5f) ? -1.f : 1.f; });
}

template<class real_t, class complex_t = std::complex<real_t>>
VectorX<complex_t> random_starter(OptimizedHamiltonian const& oh, std::mt19937& generator,
                                  var::tag<std::complex<real_t>>) {
    auto phase = num::make_random<ArrayX<real_t>>(oh.size(), generator);
    oh.reorder(phase);
    return exp(complex_t{2 * constant::pi * constant::i1} * phase);
}

/// Return the vector following the starter: r1 = h2 * r0 * 0.5
/// -> multiply by 0.5 because h2 was pre-multiplied by 2
template<class scalar_t>
VectorX<scalar_t> make_r1(SparseMatrixX<scalar_t> const& h2, VectorX<scalar_t> const& r0) {
    auto const size = h2.rows();
    auto const data = h2.valuePtr();
    auto const indices = h2.innerIndexPtr();
    auto const indptr = h2.outerIndexPtr();

    auto r1 = VectorX<scalar_t>(size);
    for (auto row = 0; row < size; ++row) {
        auto tmp = scalar_t{0};
        for (auto n = indptr[row]; n < indptr[row + 1]; ++n) {
            tmp += compute::detail::mul(data[n], r0[indices[n]]);
        }
        r1[row] = tmp * scalar_t{0.5};
    }
    return r1;
}

template<class scalar_t>
VectorX<scalar_t> make_r1(num::EllMatrix<scalar_t> const& h2, VectorX<scalar_t> const& r0) {
    auto const size = h2.rows();
    auto r1 = VectorX<scalar_t>::Zero(size).eval();
    for (auto n = 0; n < h2.nnz_per_row; ++n) {
        for (auto row = 0; row < size; ++row) {
            auto const a = h2.data(row, n);
            auto const b = r0[h2.indices(row, n)];
            r1[row] += compute::detail::mul(a, b) * scalar_t{0.5};
        }
    }
    return r1;
}

}} // namespace cpb::kpm
