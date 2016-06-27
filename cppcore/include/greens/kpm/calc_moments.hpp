#pragma once
#include "greens/kpm/Bounds.hpp"
#include "greens/kpm/Moments.hpp"

#include "compute/kernel_polynomial.hpp"

namespace cpb { namespace kpm {

/**
 Return the number of moments needed to compute Green's at the specified broadening
 */
template<class real_t>
int required_num_moments(Scale<real_t> scale, double lambda, double broadening) {
    auto const scaled_broadening = broadening / scale.a;
    auto num_moments = static_cast<int>(lambda / scaled_broadening) + 1;
    // Moment calculations at higher optimization levels require specific rounding.
    // `num_moments - 2` considers only moments in the main KPM loop. Divisible by 4
    // because that is the strictest requirement imposed by `calc_diag_moments2()`.
    while ((num_moments - 2) % 4 != 0) {
        ++num_moments;
    }
    return num_moments;
}

/**
 Return the KPM r0 vector with all zeros except for the source index
 */
template<class Matrix, class scalar_t = typename Matrix::Scalar>
VectorX<scalar_t> make_r0(Matrix const& h2, int i) {
    auto r0 = VectorX<scalar_t>::Zero(h2.rows()).eval();
    r0[i] = 1;
    return r0;
}

/**
 Return the KPM r1 vector which is equal to the Hamiltonian matrix column at the source index
 */
template<class scalar_t>
VectorX<scalar_t> make_r1(SparseMatrixX<scalar_t> const& h2, int i) {
    // -> r1 = h * r0; <- optimized thanks to `r0[i] = 1`
    // Note: h2.col(i) == h2.row(i).conjugate(), but the second is row-major friendly
    // multiply by 0.5 because H2 was pre-multiplied by 2
    return h2.row(i).conjugate() * scalar_t{0.5};
}

template<class scalar_t>
VectorX<scalar_t> make_r1(num::EllMatrix<scalar_t> const& h2, int i) {
    auto r1 = VectorX<scalar_t>::Zero(h2.rows()).eval();
    for (auto n = 0; n < h2.nnz_per_row; ++n) {
        auto const col = h2.indices(i, n);
        auto const value = h2.data(i, n);
        r1[col] = num::conjugate(value) * scalar_t{0.5};
    }
    return r1;
}

/**
 KPM moments -- reference implementation, no optimizations

 Calculates moments for multiple indices in the same row as specified by `idx`.
 Both diagonal and off-diagonal moments can be computed, but note that the
 diagonal version of this function is more efficient for that special case.
 */
template<class scalar_t>
MomentsMatrix<scalar_t> calc_moments0(SparseMatrixX<scalar_t> const& h2,
                                      Indices const& idx, int num_moments) {
    auto moment_matrix = MomentsMatrix<scalar_t>(num_moments, idx.cols);
    auto r0 = make_r0(h2, idx.row);
    auto r1 = make_r1(h2, idx.row);
    moment_matrix.collect_initial(r0, r1);

    auto const size = h2.rows();
    for (auto n = 2; n < num_moments; ++n) {
        // -> r0 = h2 * r1 - r0; <- compute kernel
        compute::kpm_kernel(0, size, h2, r1, r0);

        // r1 gets the primary result of this iteration
        // r0 gets the value old value r1 (it will be needed in the next iteration)
        r1.swap(r0);

        moment_matrix.collect(n, r1);
    }

    return moment_matrix;
}

/**
 Diagonal KPM moments -- reference implementation, no optimizations

 Calculates moments for a single matrix element (i, i) on the main diagonal.
 It's 1.5x to 2x times faster than the general version.
 */
template<class scalar_t>
ArrayX<scalar_t> calc_diag_moments0(SparseMatrixX<scalar_t> const& h2, int i, int num_moments) {
    auto moments = ArrayX<scalar_t>(num_moments);
    auto r0 = make_r0(h2, i);
    auto r1 = make_r1(h2, i);
    auto const m0 = moments[0] = r0[i] * scalar_t{0.5};
    auto const m1 = moments[1] = r1[i];

    assert(num_moments % 2 == 0);
    for (auto n = 2; n <= num_moments / 2; ++n) {
        compute::kpm_kernel(0, h2.rows(), h2, r1, r0);
        r1.swap(r0);
        moments[2 * (n - 1)] = scalar_t{2} * (r0.squaredNorm() - m0);
        moments[2 * (n - 1) + 1] = scalar_t{2} * r1.dot(r0) - m1;
    }

    return moments;
}

/**
 KPM moments -- with reordering optimization (optimal system size for each iteration)
 */
template<class Matrix, class scalar_t = typename Matrix::Scalar>
MomentsMatrix<scalar_t> calc_moments1(Matrix const& h2, Indices const& idx, int num_moments,
                                      OptimizedSizes const& sizes) {
    auto moment_matrix = MomentsMatrix<scalar_t>(num_moments, idx.cols);
    auto r0 = make_r0(h2, idx.row);
    auto r1 = make_r1(h2, idx.row);
    moment_matrix.collect_initial(r0, r1);

    for (auto n = 2; n < num_moments; ++n) {
        auto const optimized_size = sizes.optimal(n, num_moments);
        compute::kpm_kernel(0, optimized_size, h2, r1, r0); // r0 = matrix * r1 - r0
        r1.swap(r0);
        moment_matrix.collect(n, r1);
    }

    return moment_matrix;
}

/**
 Diagonal KPM moments -- with reordering optimization (optimal system size for each iteration)
 */
template<class Matrix, class scalar_t = typename Matrix::Scalar>
ArrayX<scalar_t> calc_diag_moments1(Matrix const& h2, int i, int num_moments,
                                    OptimizedSizes const& sizes) {
    auto moments = ArrayX<scalar_t>(num_moments);
    auto r0 = make_r0(h2, i);
    auto r1 = make_r1(h2, i);
    auto const m0 = moments[0] = r0[i] * scalar_t{0.5};
    auto const m1 = moments[1] = r1[i];

    assert(num_moments % 2 == 0);
    for (auto n = 2; n <= num_moments / 2; ++n) {
        auto const opt_size = sizes.optimal(n, num_moments);
        compute::kpm_kernel(0, opt_size, h2, r1, r0);
        r1.swap(r0);
        moments[2 * (n - 1)] = scalar_t{2} * (r0.head(opt_size).squaredNorm() - m0);
        moments[2 * (n - 1) + 1] = scalar_t{2} * r1.head(opt_size).dot(r0.head(opt_size)) - m1;
    }

    return moments;
}

/**
 KPM moments -- like previous plus bandwidth optimization (interleaved moments)
 */
template<class Matrix, class scalar_t = typename Matrix::Scalar>
MomentsMatrix<scalar_t> calc_moments2(Matrix const& h2, Indices const& idx, int num_moments,
                                      OptimizedSizes const& sizes) {
    auto moment_matrix = MomentsMatrix<scalar_t>(num_moments, idx.cols);
    auto r0 = make_r0(h2, idx.row);
    auto r1 = make_r1(h2, idx.row);
    moment_matrix.collect_initial(r0, r1);

    // Interleave moments `n` and `n + 1` for better data locality
    assert(num_moments % 2 == 0);
    for (auto n = 2; n < num_moments; n += 2) {
        auto p0 = 0;
        auto p1 = 0;

        auto const max_m = sizes.index(n, num_moments);
        for (auto m = 0; m <= max_m; ++m) {
            auto const p2 = sizes[m];
            compute::kpm_kernel(p1, p2, h2, r1, r0);
            compute::kpm_kernel(p0, p1, h2, r0, r1);

            p0 = p1;
            p1 = p2;
        }
        moment_matrix.collect(n, r0);

        auto const max_m2 = sizes.index(n + 1, num_moments);
        compute::kpm_kernel(p0, sizes[max_m2], h2, r0, r1);
        moment_matrix.collect(n + 1, r1);
    }

    return moment_matrix;
}

/**
 Diagonal KPM moments -- like previous plus bandwidth optimization (interleaved moments)
 */
template<class Matrix, class scalar_t = typename Matrix::Scalar>
ArrayX<scalar_t> calc_diag_moments2(Matrix const& h2, int i, int num_moments,
                                    OptimizedSizes const& sizes) {
    auto moments = ArrayX<scalar_t>(num_moments);
    auto r0 = make_r0(h2, i);
    auto r1 = make_r1(h2, i);
    auto const m0 = moments[0] = r0[i] * scalar_t{0.5};
    auto const m1 = moments[1] = r1[i];

    assert((num_moments - 2) % 4 == 0);
    for (auto n = 2; n <= num_moments / 2; n += 2) {
        auto m2 = scalar_t{0};
        auto m3 = scalar_t{0};
        auto m4 = scalar_t{0};
        auto m5 = scalar_t{0};

        auto const max1 = sizes.index(n, num_moments);
        for (auto k = 0, p0 = 0, p1 = 0; k <= max1; ++k) {
            auto const p2 = sizes[k];
            compute::kpm_diag_kernel(p1, p2, h2, r1, r0, m2, m3);
            compute::kpm_diag_kernel(p0, p1, h2, r0, r1, m4, m5);

            p0 = p1;
            p1 = p2;
        }
        moments[2 * (n - 1)] = scalar_t{2} * (m2 - m0);
        moments[2 * (n - 1) + 1] = scalar_t{2} * m3 - m1;

        auto const max2 = sizes.index(n + 1, num_moments);
        compute::kpm_diag_kernel(sizes[max1 - 1], sizes[max2], h2, r0, r1, m4, m5);
        moments[2 * n] = scalar_t{2} * (m4 - m0);
        moments[2 * n + 1] = scalar_t{2} * m5 - m1;
    }

    return moments;
}

}} // namespace cpb::kpm
