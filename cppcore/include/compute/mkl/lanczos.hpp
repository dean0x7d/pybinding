#pragma once
#include "support/dense.hpp"

namespace tbm { namespace compute {

template<class Derived, class scalar_t = typename Derived::Scalar>
inline ArrayX<scalar_t> tridiagonal_eigenvalues(const DenseBase<Derived>& alpha,
                                                const DenseBase<Derived>& beta)
{
    ArrayX<scalar_t> eigenvalues = alpha;
    ArrayX<scalar_t> temp = beta;

    auto error_id = LAPACKE_sstev(LAPACK_COL_MAJOR, 'N', eigenvalues.size(), eigenvalues.data(),
                                  temp.data(), nullptr, eigenvalues.size());
    if (error_id)
        throw std::runtime_error{"LAPACK stev() error: " + std::to_string(error_id)};

    return eigenvalues;
}

}} // namespace tbm::compute
