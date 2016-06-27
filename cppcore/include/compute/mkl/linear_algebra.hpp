#pragma once
#include "numeric/dense.hpp"
#include "numeric/sparse.hpp"
#include "compute/mkl/wrapper.hpp"

namespace cpb { namespace compute {

template<class real_t>
inline std::complex<real_t> dot_product(VectorX<std::complex<real_t>> const& x,
                                        VectorX<std::complex<real_t>> const& y) {
    std::complex<real_t> result;
    mkl::dotc<std::complex<real_t>>::call(x.size(), x.data(), 1, y.data(), 1, &result);
    return result;
}

// the MKL version of this function returns wrong results sometimes - removed for now
template<class real_t>
inline real_t dot_product(VectorX<real_t> const& x, VectorX<real_t> const& y) {
    return x.dot(y);
}

template<class real_t>
inline void axpy(std::complex<real_t> const a, VectorX<std::complex<real_t>> const& x,
                 VectorX<std::complex<real_t>>& y) {
    mkl::axpy<std::complex<real_t>>::call(x.size(), &a, x.data(), 1, y.data(), 1);
}

template<class real_t>
inline void axpy(real_t const a, VectorX<real_t> const& x, VectorX<real_t>& y) {
    mkl::axpy<real_t>::call(x.size(), a, x.data(), 1, y.data(), 1);
}

template<class scalar_t>
inline void matrix_vector_mul(SparseMatrixX<scalar_t> const& matrix,
                              VectorX<scalar_t> const& x_vector,
                              VectorX<scalar_t>& result_vector) {
    auto const transa = 'n';  // specifies normal (non-transposed) matrix-vector multiplication
    auto const size = static_cast<int>(x_vector.size());

    using mkl_scalar_t = mkl::type<scalar_t>;
    mkl::csrgemv<scalar_t>::call(
        &transa,
        &size,
        // input matrix
        reinterpret_cast<mkl_scalar_t const*>(matrix.valuePtr()),
        matrix.outerIndexPtr(),
        matrix.innerIndexPtr(),
        // input vector
        reinterpret_cast<mkl_scalar_t const*>(x_vector.data()),
        // output vector
        reinterpret_cast<mkl_scalar_t*>(result_vector.data())
    );
}

}} // namespace cpb::compute
