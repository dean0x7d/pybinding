#pragma once
#include <mkl.h>
#include <complex>

namespace cpb { namespace mkl {

namespace detail {
    template<class scalar_t> struct typemap;
    template<> struct typemap<float> { using type = float; };
    template<> struct typemap<double> { using type = double; };
    template<> struct typemap<std::complex<float>> { using type = MKL_Complex8; };
    template<> struct typemap<std::complex<double>> { using type = MKL_Complex16; };
}

/// Get the corresponding MKL C API type from the C++ type `scalar_t`
template<class scalar_t>
using type = typename detail::typemap<scalar_t>::type;

/// Dot product
template<class scalar_t> struct dotc;
template<> struct dotc<std::complex<float>> { static constexpr auto call = cblas_cdotc_sub; };
template<> struct dotc<std::complex<double>> { static constexpr auto call = cblas_zdotc_sub; };

/// axpy: y = a*x + y
template<class scalar_t> struct axpy;
template<> struct axpy<float> { static constexpr auto call = cblas_saxpy; };
template<> struct axpy<double> { static constexpr auto call = cblas_daxpy; };
template<> struct axpy<std::complex<float>> { static constexpr auto call = cblas_caxpy; };
template<> struct axpy<std::complex<double>> { static constexpr auto call = cblas_zaxpy; };

/// Eigenvalue and eigenvectors of a real symmetry tridiagonal matrix
template<class scalar_t> struct stev;
template<> struct stev<float> { static constexpr auto call = LAPACKE_sstev; };
template<> struct stev<double> { static constexpr auto call = LAPACKE_dstev; };

/// CSR matrix vector multiplication
template<class scalar_t> struct csrmv;
template<> struct csrmv<float> { static constexpr auto call = mkl_scsrmv; };
template<> struct csrmv<double> { static constexpr auto call = mkl_dcsrmv; };
template<> struct csrmv<std::complex<float>> { static constexpr auto call = mkl_ccsrmv; };
template<> struct csrmv<std::complex<double>> { static constexpr auto call = mkl_zcsrmv; };

/// CSR general matrix vector multiplication
template<class scalar_t> struct csrgemv;
template<> struct csrgemv<float> { static constexpr auto call = mkl_cspblas_scsrgemv; };
template<> struct csrgemv<double> { static constexpr auto call = mkl_cspblas_dcsrgemv; };
template<> struct csrgemv<std::complex<float>> { static constexpr auto call = mkl_cspblas_ccsrgemv; };
template<> struct csrgemv<std::complex<double>> { static constexpr auto call = mkl_cspblas_zcsrgemv; };


template<class scalar_t> struct feast_hcsrev;
template<> struct feast_hcsrev<float> { static constexpr auto call = sfeast_scsrev; };
template<> struct feast_hcsrev<double> { static constexpr auto call = dfeast_scsrev; };
template<> struct feast_hcsrev<std::complex<float>> { static constexpr auto call = cfeast_hcsrev; };
template<> struct feast_hcsrev<std::complex<double>> { static constexpr auto call = zfeast_hcsrev; };

}} // namespace cpb::mkl
