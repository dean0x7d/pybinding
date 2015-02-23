#pragma once
#include "support/dense.hpp"
#include "support/sparse.hpp"

namespace compute {

template<typename scalar_t>
inline void kpm_kernel(const int size, const SparseMatrixX<scalar_t>& matrix,
                       const VectorX<scalar_t>& x, VectorX<scalar_t>& y)
{
    const auto* const value = matrix.valuePtr();
    const auto* const row_start = matrix.outerIndexPtr();
    const auto* const column_index = matrix.innerIndexPtr();

    for (int i = 0; i < size; ++i) {
        y[i] = -y[i];
        for (int j = row_start[i]; j < row_start[i + 1]; ++j)
            y[i] += value[j] * x[column_index[j]];
    }
}

template<typename scalar_t>
inline void kpm_kernel(const int size, const SparseMatrixX<std::complex<scalar_t>>& matrix,
                       const VectorX<std::complex<scalar_t>>& x, VectorX<std::complex<scalar_t>>& y)
{
    const auto* const value = matrix.valuePtr();
    const auto* const row_start = matrix.outerIndexPtr();
    const auto* const column_index = matrix.innerIndexPtr();

    for (int i = 0; i < size; ++i) {
        y[i] = -y[i];
        for (int j = row_start[i]; j < row_start[i + 1]; ++j) {
            auto& r = y[i];
            const auto a = value[j];
            const auto b = x[column_index[j]];

            r.real(r.real() + a.real() * b.real() - a.imag() * b.imag());
            r.imag(r.imag() + a.real() * b.imag() + a.imag() * b.real());
        }
    }
}

} // namespace compute
