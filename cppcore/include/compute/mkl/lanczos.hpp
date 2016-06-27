#pragma once
#include "numeric/dense.hpp"
#include "compute/mkl/wrapper.hpp"

namespace cpb { namespace compute {

template<class Derived, class scalar_t = typename Derived::Scalar>
inline ArrayX<scalar_t> tridiagonal_eigenvalues(DenseBase<Derived> const& alpha,
                                                DenseBase<Derived> const& beta) {
    ArrayX<scalar_t> eigenvalues = alpha;
    ArrayX<scalar_t> temp = beta;

    auto const error_id = mkl::stev<scalar_t>::call(
        LAPACK_COL_MAJOR, 'N', eigenvalues.size(), eigenvalues.data(),
        temp.data(), nullptr, eigenvalues.size()
    );
    if (error_id)
        throw std::runtime_error{"LAPACK stev() error: " + std::to_string(error_id)};

    return eigenvalues;
}

}} // namespace cpb::compute
