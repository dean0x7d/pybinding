#pragma once

#ifdef TBM_USE_MKL
# include "mkl/lanczos.hpp"
#else
# include "eigen3/lanczos.hpp"
#endif

#include "support/sparse.hpp"
#include "compute/linear_algebra.hpp"
#include <tuple>

namespace tbm { namespace compute {

template<class real_t>
struct LanczosBounds {
    real_t min; ///< the lowest eigenvalue
    real_t max; ///< the highest eigenvalue
    int loops;  ///< number of iterations needed to converge
};

/// Use the Lanczos algorithm to find the min and max eigenvalues at given precision (%)
template<class scalar_t, class real_t = num::get_real_t<scalar_t>>
LanczosBounds<real_t> minmax_eigenvalues(SparseMatrixX<scalar_t> const& matrix,
                                         real_t precision_percent) {
    auto precision = precision_percent / 100;
    const auto matrix_size = static_cast<int>(matrix.rows());

    VectorX<scalar_t> left = VectorX<scalar_t>::Zero(matrix_size);
    VectorX<scalar_t> right_previous = VectorX<scalar_t>::Zero(matrix_size);

    VectorX<scalar_t> right = VectorX<scalar_t>::Random(matrix_size);
    right.normalize();

    // Alpha and beta are the diagonals of the tridiagonal matrix.
    // The final size is not known ahead of time, but it will be small.
    std::vector<real_t> alpha; alpha.reserve(100);
    std::vector<real_t> beta; beta.reserve(100);

    // Energy values from the previous iteration. Used to test convergence.
    // Initial values as far away from expected as possible.
    real_t previous_min = std::numeric_limits<real_t>::max();
    real_t previous_max = std::numeric_limits<real_t>::lowest();

    constexpr auto loop_limit = 1000;
    // This may iterate up to matrix_size, but since only the extreme eigenvalues are required it
    // will converge very quickly. More than loop_limit iterations would suggest something is wrong.
    for (int i = 0; i < loop_limit; ++i) {
        // PART 1: Calculate tridiagonal matrix elements a and b
        // =====================================================
        // left = h_matrix * right
        // matrix-vector multiplication (the most compute intensive part of each iteration)
        compute::matrix_vector_mul(matrix, right, left);

        // vector dot product (left*right) -> we just need the real part
        real_t a = std::real(compute::dot_product(left, right));
        // get b from the previous iteration (or 0 if this is the first iteration)
        real_t b = !beta.empty() ? beta.back() : 0;

        // left -= a*right + b*right_previous;
        compute::axpy(scalar_t(-a), right, left);
        compute::axpy(scalar_t(-b), right_previous, left);
        b = left.norm();

        // right_previous gets the old value of right
        right_previous.swap(right);
        // right moves on
        right = (1/b) * left;

        // add a and b to tridiagonal matrix
        alpha.push_back(a);
        beta.push_back(b);

        // PART 2: Check if the largest magnitude eigenvalues have converged
        // =================================================================
        auto eigenvalues = compute::tridiagonal_eigenvalues(eigen_cast<ArrayX>(alpha),
                                                            eigen_cast<ArrayX>(beta));

        real_t min = eigenvalues.minCoeff();
        real_t max = eigenvalues.maxCoeff();

        using std::abs;
        bool is_converged_min = abs((previous_min - min) / min) < precision;
        bool is_converged_max = abs((previous_max - max) / max) < precision;

        previous_min = min;
        previous_max = max;

        if (is_converged_min && is_converged_max)
            return {min, max, i};
    };

    throw std::runtime_error{"Lanczos algorithm did not converge for the min/max eigenvalues."};
}

}} // namespace tbm::compute
