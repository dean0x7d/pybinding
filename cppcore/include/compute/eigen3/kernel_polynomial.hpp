#pragma once
#include "numeric/dense.hpp"
#include "numeric/sparse.hpp"

namespace tbm { namespace compute {

namespace detail {
    template<class real_t>
    inline real_t kpm_kernel_mul(real_t a, real_t b) {
        return a * b;
    }

    template<class real_t>
    inline std::complex<real_t> kpm_kernel_mul(std::complex<real_t> a, std::complex<real_t> b) {
        return {a.real() * b.real() - a.imag() * b.imag(),
                a.real() * b.imag() + a.imag() * b.real()};
    }
}

template<class scalar_t>
inline void kpm_kernel(int start, int end, SparseMatrixX<scalar_t> const& matrix,
                       VectorX<scalar_t> const& x, VectorX<scalar_t>& y) {
    const auto* const value = matrix.valuePtr();
    const auto* const row_start = matrix.outerIndexPtr();
    const auto* const column_index = matrix.innerIndexPtr();

    for (auto i = start; i < end; ++i) {
        auto r = scalar_t{0};
        for (auto j = row_start[i]; j < row_start[i + 1]; ++j) {
            r += detail::kpm_kernel_mul(value[j], x[column_index[j]]);
        }
        y[i] = r - y[i];
    }
}

}} // namespace tbm::compute
