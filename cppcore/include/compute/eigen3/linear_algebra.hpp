#pragma once
#include "numeric/dense.hpp"
#include "numeric/sparse.hpp"

namespace cpb { namespace compute {

template<class scalar_t>
inline scalar_t dot_product(const VectorX<scalar_t>& x, const VectorX<scalar_t>& y) {
    return x.dot(y);
}

template<class scalar_t>
inline void axpy(const scalar_t a, const VectorX<scalar_t>& x, VectorX<scalar_t>& y) {
    y += a*x;
}

template<class scalar_t>
inline void matrix_vector_mul(const SparseMatrixX<scalar_t>& matrix,
                              const VectorX<scalar_t>& x_vector, VectorX<scalar_t>& y_vector)
{
    y_vector = matrix * x_vector;
}

}} // namespace cpb::compute
