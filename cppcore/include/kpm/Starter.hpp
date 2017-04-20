#pragma once
#include "kpm/OptimizedHamiltonian.hpp"

#include "numeric/dense.hpp"
#include "compute/detail.hpp"

namespace cpb { namespace kpm {

/// Produce the r0 starter vector for the KPM procedure
using Starter = std::function<var::Complex<VectorX> (var::scalar_tag)>;

/// Starter vector equal to the constant `alpha` (`oh` is needed for reordering)
Starter constant_starter(OptimizedHamiltonian const& oh, VectorXcd const& alpha);

/// Unit vector starter (`oh` encodes the unit index)
Starter unit_starter(OptimizedHamiltonian const& oh);

/// Starter vector for the stochastic KPM procedure (`oh` is needed for size and reordering)
Starter random_starter(OptimizedHamiltonian const& oh, VariantCSR const& op = {});

/// Construct a concrete scalar type r0 vector based on a `Starter`
template<class scalar_t>
VectorX<scalar_t> make_r0(Starter const& starter) {
    return starter(var::tag<scalar_t>{}).template get<VectorX<scalar_t>>();
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
