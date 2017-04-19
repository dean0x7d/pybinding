#pragma once

#ifdef CPB_USE_MKL
# include "mkl/lanczos.hpp"
#else
# include "eigen3/lanczos.hpp"
#endif

#include "numeric/sparse.hpp"
#include "numeric/random.hpp"
#include "compute/detail.hpp"
#include "support/simd.hpp"

namespace cpb { namespace compute {

/**
 Lanczos-specialized sparse matrix-vector multiplication + dot product

 Equivalent to:
   tmp = matrix * v1
   a = real(dot_product(tmp, v1))
   v0 = tmp - b_prev * v0
   return a
 */
template<class scalar_t, class real_t = num::get_real_t<scalar_t>> CPB_ALWAYS_INLINE
real_t lanczos_spmv(real_t b_prev, SparseMatrixX<scalar_t> const& matrix,
                    VectorX<scalar_t> const& v1, VectorX<scalar_t>& v0) {
    auto const size = matrix.rows();
    auto const data = matrix.valuePtr();
    auto const indices = matrix.innerIndexPtr();
    auto const indptr = matrix.outerIndexPtr();

    auto a = real_t{0};
    for (auto row = 0; row < size; ++row) {
        auto tmp = scalar_t{0};
        for (auto n = indptr[row]; n < indptr[row + 1]; ++n) {
            tmp += detail::mul(data[n], v1[indices[n]]);
        }
        v0[row] = tmp - b_prev * v0[row];
        a += detail::real_dot(tmp, v1[row]);
    }

    return a;
}

/**
  Lanczos-specialized a * x + y

  Equivalent to:
    v0 -= a * v1
    b = norm(v0)
    return b
 */
#if SIMDPP_USE_NULL // generic version
template<class scalar_t, class real_t = num::get_real_t<scalar_t>> CPB_ALWAYS_INLINE
real_t lanczos_axpy(real_t a, VectorX<scalar_t> const& v1, VectorX<scalar_t>& v0) {
    auto const size = v0.size();
    auto norm2 = real_t{0};
    for (auto i = 0; i < size; ++i) {
        auto const l = v0[i] - a * v1[i];
        norm2 += detail::square(l);
        v0[i] = l;
    }
    return std::sqrt(norm2);
}
#else // vectorized using SIMD intrinsics
template<class scalar_t, class real_t = num::get_real_t<scalar_t>> CPB_ALWAYS_INLINE
real_t lanczos_axpy(real_t a, VectorX<scalar_t> const& v1, VectorX<scalar_t>& v0) {
    using simd_register_t = simd::select_vector_t<scalar_t>;
    auto const loop = simd::split_loop(v0.data(), 0, v0.size());
    assert(loop.peel_end == 0); // all eigen vectors are properly aligned when starting from 0

    auto norm2_vec = simd::make_float<simd_register_t>(0);
    for (auto i = idx_t{0}; i < loop.vec_end; i += loop.step) {
        auto const r0 = simd::load<simd_register_t>(v0.data() + i);
        auto const r1 = simd::load<simd_register_t>(v1.data() + i);
        auto const tmp = simd_register_t{r0 - a * r1};
        norm2_vec = norm2_vec + tmp * tmp;
        simd::store(v0.data() + i, tmp);
    }

    auto norm2_remainder = real_t{0};
    for (auto i = loop.vec_end; i < loop.end; ++i) {
        auto const tmp = v0[i] - a * v1[i];
        norm2_remainder += detail::square(tmp);
        v0[i] = tmp;
    }

    return std::sqrt(simd::reduce_add(norm2_vec) + norm2_remainder);
}
#endif // SIMDPP_USE_NULL

struct LanczosBounds {
    double min; ///< the lowest eigenvalue
    double max; ///< the highest eigenvalue
    int loops;  ///< number of iterations needed to converge
};

/// Use the Lanczos algorithm to find the min and max eigenvalues at given precision (%)
template<class scalar_t>
LanczosBounds minmax_eigenvalues(SparseMatrixX<scalar_t> const& matrix, double precision_percent) {
    using real_t = num::get_real_t<scalar_t>;
    simd::scope_disable_denormals guard;

    auto v0 = VectorX<scalar_t>::Zero(matrix.rows()).eval();
    auto v1 = num::make_random<VectorX<scalar_t>>(matrix.rows());
    v1.normalize();

    // Alpha and beta are the diagonals of the tridiagonal matrix.
    // The final size is not known ahead of time, but it will be small.
    auto alpha = std::vector<real_t>(); alpha.reserve(100);
    auto beta = std::vector<real_t>(); beta.reserve(100);

    // Energy values from the previous iteration. Used to test convergence.
    // Initial values as far away from expected as possible.
    auto previous_min = std::numeric_limits<real_t>::max();
    auto previous_max = std::numeric_limits<real_t>::lowest();
    auto const precision = static_cast<real_t>(precision_percent / 100);

    constexpr auto loop_limit = 1000;
    // This may iterate up to matrix_size, but since only the extreme eigenvalues are required it
    // will converge very quickly. Exceeding `loop_limit` would suggest something is wrong.
    for (int i = 0; i < loop_limit; ++i) {
        // PART 1: Calculate tridiagonal matrix elements a and b
        // =====================================================
        auto const b_prev = !beta.empty() ? beta.back() : real_t{0};
        auto const a = lanczos_spmv(b_prev, matrix, v1, v0);
        auto const b = lanczos_axpy(a, v1, v0);

        v0 *= 1 / b;
        v0.swap(v1);

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
