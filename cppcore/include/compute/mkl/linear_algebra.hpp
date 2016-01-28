#pragma once
#include "support/dense.hpp"
#include "support/sparse.hpp"

namespace tbm { namespace compute {
/**
 dot product
*/
template<class scalar_t>
inline scalar_t dot_product(const VectorX<scalar_t>& x, const VectorX<scalar_t>& y);

template<>
inline float dot_product<float>(const VectorXf& x, const VectorXf& y) {
    // the MKL version of this function returns wrong results sometimes - removed for now
    return x.dot(y);
}

template<>
inline std::complex<float> dot_product<std::complex<float>>(const VectorXcf& x, const VectorXcf& y) {
    std::complex<float> result;
    cblas_cdotc_sub(x.size(), x.data(), 1, y.data(), 1, &result);
    return result;
}

/**
 axpy: y = a*x + y
*/
template<class scalar_t>
inline void axpy(const scalar_t a, const VectorX<scalar_t>& x, VectorX<scalar_t>& y);

template<>
inline void axpy<float>(const float a, const VectorXf& x, VectorXf& y) {
    cblas_saxpy(x.size(), a, x.data(), 1, y.data(), 1);
}

template<>
inline void axpy<std::complex<float>>(const std::complex<float> a, const VectorXcf& x, VectorXcf& y) {
    cblas_caxpy(x.size(), &a, x.data(), 1, y.data(), 1);
}

/**
 csrgemv: compressed sparse row, general matrix vector multiplication
*/
template<class scalar_t>
inline void matrix_vector_mul(const SparseMatrixX<scalar_t>& matrix,
                              const VectorX<scalar_t>& x_vector, VectorX<scalar_t>& result_vector);

template<>
inline void matrix_vector_mul<float>(const SparseMatrixXf& matrix,
                                     const VectorXf& x_vector, VectorXf& result_vector)
{
    const char transa = 'n';  // specifies normal (non-transposed) matrix-vector multiplication
    const int size = x_vector.size();

    mkl_cspblas_scsrgemv(
        &transa,
        &size,
        // input matrix
        matrix.valuePtr(),
        matrix.outerIndexPtr(),
        matrix.innerIndexPtr(),
        // input vector
        x_vector.data(),
        // output vector
        result_vector.data()
    );
}

template<>
inline void matrix_vector_mul<std::complex<float>>(const SparseMatrixXcf& matrix,
                                                   const VectorXcf& x_vector, VectorXcf& result_vector)
{
    const char transa = 'n';  // specifies normal (non-transposed) matrix-vector multiplication
    const int size = x_vector.size();

    mkl_cspblas_ccsrgemv(
        &transa,
        &size,
        // input matrix - cast from complex<float>* to MKL_Complex8*
        reinterpret_cast<const MKL_Complex8*>(matrix.valuePtr()),
        matrix.outerIndexPtr(),
        matrix.innerIndexPtr(),
        // input vector
        reinterpret_cast<const MKL_Complex8*>(x_vector.data()),
        // output vector
        reinterpret_cast<MKL_Complex8*>(result_vector.data())
    );
}

}} // namespace tbm::compute
