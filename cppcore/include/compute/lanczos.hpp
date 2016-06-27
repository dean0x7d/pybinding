#pragma once

#ifdef CPB_USE_MKL
# include "mkl/lanczos.hpp"
#else
# include "eigen3/lanczos.hpp"
#endif

#include "numeric/sparse.hpp"
#include "numeric/random.hpp"
#include "compute/linear_algebra.hpp"

namespace cpb { namespace compute {

template<class real_t>
struct LanczosBounds {
    real_t min; ///< the lowest eigenvalue
    real_t max; ///< the highest eigenvalue
    int loops;  ///< number of iterations needed to converge
};

/// Use the Lanczos algorithm to find the min and max eigenvalues at given precision (%)
template<class scalar_t, class real_t = num::get_real_t<scalar_t>>
LanczosBounds<real_t> minmax_eigenvalues(SparseMatrixX<scalar_t> const& matrix,
                                         double precision_percent) {
    auto const precision = static_cast<real_t>(precision_percent / 100);
    auto const matrix_size = static_cast<int>(matrix.rows());

    auto left = VectorX<scalar_t>{VectorX<scalar_t>::Zero(matrix_size)};
    auto right_previous = VectorX<scalar_t>{VectorX<scalar_t>::Zero(matrix_size)};

    auto right = num::make_random<VectorX<scalar_t>>(matrix_size);
    right.normalize();

    // Alpha and beta are the diagonals of the tridiagonal matrix.
    // The final size is not known ahead of time, but it will be small.
    auto alpha = std::vector<real_t>();
    alpha.reserve(100);
    auto beta = std::vector<real_t>();
    beta.reserve(100);

    // Energy values from the previous iteration. Used to test convergence.
    // Initial values as far away from expected as possible.
    auto previous_min = std::numeric_limits<real_t>::max();
    auto previous_max = std::numeric_limits<real_t>::lowest();

    constexpr auto loop_limit = 1000;
    // This may iterate up to matrix_size, but since only the extreme eigenvalues are required it
    // will converge very quickly. Exceeding `loop_limit` would suggest something is wrong.
    for (int i = 0; i < loop_limit; ++i) {
        // PART 1: Calculate tridiagonal matrix elements a and b
        // =====================================================
        // left = h_matrix * right
        // matrix-vector multiplication (the most compute intensive part of each iteration)
        compute::matrix_vector_mul(matrix, right, left);

        auto const a = std::real(compute::dot_product(left, right));
        auto const b_prev = !beta.empty() ? beta.back() : real_t{0};

        // left -= a*right + b_prev*right_previous;
        compute::axpy(scalar_t{-a}, right, left);
        compute::axpy(scalar_t{-b_prev}, right_previous, left);
        auto const b = left.norm();

        right_previous.swap(right);
        right = (1/b) * left;

        alpha.push_back(a);
        beta.push_back(b);

        // PART 2: Check if the largest magnitude eigenvalues have converged
        // =================================================================
        auto const eigenvalues = compute::tridiagonal_eigenvalues(eigen_cast<ArrayX>(alpha),
                                                                  eigen_cast<ArrayX>(beta));
        auto const min = eigenvalues.minCoeff();
        auto const max = eigenvalues.maxCoeff();
        auto const is_converged_min = abs((previous_min - min) / min) < precision;
        auto const is_converged_max = abs((previous_max - max) / max) < precision;

        if (is_converged_min && is_converged_max) {
            return {min, max, i};
        }

        previous_min = min;
        previous_max = max;
    };

    throw std::runtime_error{"Lanczos algorithm did not converge for the min/max eigenvalues."};
}

}} // namespace cpb::compute
