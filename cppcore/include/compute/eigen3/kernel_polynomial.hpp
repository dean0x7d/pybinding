#pragma once
#include "support/dense.hpp"
#include "support/sparse.hpp"

namespace tbm { namespace compute {

namespace detail {
    template<class real_t>
    inline void kpm_kernel_sum(real_t& result, real_t const& a, real_t const& b) {
        result += a * b;
    }

    template<class real_t>
    inline void kpm_kernel_sum(std::complex<real_t>& result, std::complex<real_t> const& a,
                               std::complex<real_t> const& b) {
        auto const re = result.real() + a.real() * b.real() - a.imag() * b.imag();
        auto const im = result.imag() + a.real() * b.imag() + a.imag() * b.real();
        result.real(re);
        result.imag(im);
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
        for (auto j = row_start[i]; j < row_start[i + 1]; ++j)
            detail::kpm_kernel_sum(r, value[j], x[column_index[j]]);
        y[i] = r - y[i];
    }
}

}} // namespace tbm::compute
