#pragma once
#include "Bounds.hpp"

#include "compute/kernel_polynomial.hpp"

namespace cpb { namespace kpm { namespace calc_moments {

/**
 Diagonal KPM implementation: the left and right vectors are identical
 */
namespace diagonal {

/**
 Reference implementation, no optimizations, no special requirements

 Calculates moments for a single matrix element (i, i) on the main diagonal.
 It's 1.5x to 2x times faster than the general (off-diagonal) version.
 */
template<class Moments, class Matrix>
void basic(Moments& moments, Matrix const& h2) {
    // The `Moments` class sets the initial vectors `r0` and `r1`. E.g. `r0` is
    // a unit vector for the expectation value variant or a random vector for
    // the stochastic trace evaluation variant.
    auto r0 = moments.r0(h2);
    auto r1 = moments.r1(h2, r0);
    moments.collect_initial(r0, r1);

    // The diagonal KPM algorithm computes 2 moments per iteration
    auto const num_moments = moments.size();
    assert(num_moments % 2 == 0);

    for (auto n = 2; n <= num_moments / 2; ++n) {
        // Any kind of pre- and post-processing, e.g. complex absorbing potential
        moments.pre_process(r0, r1);

        // Compute kernel: `r0 = h2 * r1 - r0` <- the most expensive part of the algorithm
        compute::kpm_kernel(0, h2.rows(), h2, r1, r0);

        // The pre- and post-processing are usually empty functions
        moments.post_process(r0, r1);

        // r1 gets the primary result of this iteration
        // r0 gets the value old value of r1 (it will be needed in the next iteration)
        r1.swap(r0);

        // Pick out the result of this iteration relevant for the given `Moments` class
        moments.collect(n, r0, r1);
    }
}

/**
 Optimal size optimization, requires a specially ordered matrix as input

 At each iteration, the computation is performed only for a subset of the total
 system which contains non-zero values. The speedup is about equal to the amount
 of removed work.
 */
template<class Moments, class Matrix>
void opt_size(Moments& moments, Matrix const& h2, OptimizedSizes const& sizes) {
    auto r0 = moments.r0(h2);
    auto r1 = moments.r1(h2, r0);
    moments.collect_initial(r0, r1);

    auto const num_moments = moments.size();
    assert(num_moments % 2 == 0);

    for (auto n = 2; n <= num_moments / 2; ++n) {
        // Only compute up to optimal size for each iteration
        auto const opt_size = sizes.optimal(n, num_moments);

        moments.pre_process(r0.head(opt_size), r1.head(opt_size));
        compute::kpm_kernel(0, opt_size, h2, r1, r0);
        moments.post_process(r0.head(opt_size), r1.head(opt_size));

        r1.swap(r0);
        moments.collect(n, r0.head(opt_size), r1.head(opt_size));
    }
}

/**
 Interleave two moment calculations, requires a specially ordered matrix as input

 The two concurrent operations share some of the same data, thus promoting cache
 usage and reducing main memory bandwidth.
 */
template<class Moments, class Matrix, class scalar_t = typename Matrix::Scalar>
void interleaved(Moments& moments, Matrix const& h2, OptimizedSizes const& sizes) {
    auto r0 = moments.r0(h2);
    auto r1 = moments.r1(h2, r0);
    moments.collect_initial(r0, r1);

    auto const num_moments = moments.size();
    assert((num_moments - 2) % 4 == 0);

    // Interleave moments `n` and `n + 1` for better data locality
    // Diagonal + interleaved computes 4 moments per iteration
    for (auto n = 2; n <= num_moments / 2; n += 2) {
        auto m2 = scalar_t{0}, m3 = scalar_t{0}, m4 = scalar_t{0}, m5 = scalar_t{0};

        auto const max = sizes.max_index();
        for (auto k = 0, p0 = 0, p1 = 0; k <= max; ++k) {
            auto const p2 = sizes[k];
            compute::kpm_diag_kernel(p1, p2, h2, r1, r0, m2, m3);
            compute::kpm_diag_kernel(p0, p1, h2, r0, r1, m4, m5);

            p0 = p1;
            p1 = p2;
        }
        compute::kpm_diag_kernel(sizes[max - 1], sizes[max], h2, r0, r1, m4, m5);

        moments.collect(n, m2, m3);
        moments.collect(n + 1, m4, m5);
    }
}

/**
 Optimal size + interleaved
 */
template<class Moments, class Matrix, class scalar_t = typename Matrix::Scalar>
void opt_size_and_interleaved(Moments& moments, Matrix const& h2, OptimizedSizes const& sizes) {
    auto r0 = moments.r0(h2);
    auto r1 = moments.r1(h2, r0);
    moments.collect_initial(r0, r1);

    auto const num_moments = moments.size();
    assert((num_moments - 2) % 4 == 0);

    for (auto n = 2; n <= num_moments / 2; n += 2) {
        auto m2 = scalar_t{0}, m3 = scalar_t{0}, m4 = scalar_t{0}, m5 = scalar_t{0};

        auto const max1 = sizes.index(n, num_moments);
        for (auto k = 0, p0 = 0, p1 = 0; k <= max1; ++k) {
            auto const p2 = sizes[k];
            compute::kpm_diag_kernel(p1, p2, h2, r1, r0, m2, m3);
            compute::kpm_diag_kernel(p0, p1, h2, r0, r1, m4, m5);

            p0 = p1;
            p1 = p2;
        }
        auto const max2 = sizes.index(n + 1, num_moments);
        compute::kpm_diag_kernel(sizes[max1 - 1], sizes[max2], h2, r0, r1, m4, m5);

        moments.collect(n, m2, m3);
        moments.collect(n + 1, m4, m5);
    }
}

} // namespace diagonal

/**
 Off-diagonal KPM implementation: different left and right vectors
 */
namespace off_diagonal {

/**
 Reference implementation, no optimizations, no special requirements

 Calculates moments for multiple indices in the same row -- the Moments class
 knows which final values to collect. Both diagonal and off-diagonal moments
 can be computed, but the diagonal version of this function is more efficient
 for that special case.
 */
template<class Moments, class Matrix>
void basic(Moments& moments, Matrix const& h2) {
    auto r0 = moments.r0(h2);
    auto r1 = moments.r1(h2, r0);
    moments.collect_initial(r0, r1);

    auto const num_moments = moments.size();
    for (auto n = 2; n < num_moments; ++n) {
        moments.pre_process(r0, r1);
        compute::kpm_kernel(0, h2.rows(), h2, r1, r0);
        moments.post_process(r0, r1);

        r1.swap(r0);
        moments.collect(n, r1);
    }
}

/**
 Optimal size optimization, requires a specially ordered matrix as input

 See the diagonal version of this function for more information.
 */
template<class Moments, class Matrix>
void opt_size(Moments& moments, Matrix const& h2, OptimizedSizes const& sizes) {
    auto r0 = moments.r0(h2);
    auto r1 = moments.r1(h2, r0);
    moments.collect_initial(r0, r1);

    auto const num_moments = moments.size();
    for (auto n = 2; n < num_moments; ++n) {
        auto const optimized_size = sizes.optimal(n, num_moments);

        moments.pre_process(r0, r1);
        compute::kpm_kernel(0, optimized_size, h2, r1, r0); // r0 = matrix * r1 - r0
        moments.post_process(r0, r1);

        r1.swap(r0);
        moments.collect(n, r1);
    }
}

/**
 Interleave two moment calculations, requires a specially ordered matrix as input

 See the diagonal version of this function for more information.
 */
template<class Moments, class Matrix, class scalar_t = typename Matrix::Scalar>
void interleaved(Moments& moments, Matrix const& h2, OptimizedSizes const& sizes) {
    auto r0 = moments.r0(h2);
    auto r1 = moments.r1(h2, r0);
    moments.collect_initial(r0, r1);

    auto const num_moments = moments.size();
    assert(num_moments % 2 == 0);

    // Interleave moments `n` and `n + 1` for better data locality
    for (auto n = 2; n < num_moments; n += 2) {
        auto const max = sizes.max_index();
        for (auto m = 0, p0 = 0, p1 = 0; m <= max; ++m) {
            auto const p2 = sizes[m];
            compute::kpm_kernel(p1, p2, h2, r1, r0);
            compute::kpm_kernel(p0, p1, h2, r0, r1);

            p0 = p1;
            p1 = p2;
        }
        compute::kpm_kernel(sizes[max - 1], sizes[max], h2, r0, r1);

        moments.collect(n, r0);
        moments.collect(n + 1, r1);
    }
}

/**
 Optimal size + interleaved
 */
template<class Moments, class Matrix, class scalar_t = typename Matrix::Scalar>
void opt_size_and_interleaved(Moments& moments, Matrix const& h2, OptimizedSizes const& sizes) {
    auto r0 = moments.r0(h2);
    auto r1 = moments.r1(h2, r0);
    moments.collect_initial(r0, r1);

    auto const num_moments = moments.size();
    assert(num_moments % 2 == 0);

    // Interleave moments `n` and `n + 1` for better data locality
    for (auto n = 2; n < num_moments; n += 2) {
        auto const max1 = sizes.index(n, num_moments);
        for (auto m = 0, p0 = 0, p1 = 0; m <= max1; ++m) {
            auto const p2 = sizes[m];
            compute::kpm_kernel(p1, p2, h2, r1, r0);
            compute::kpm_kernel(p0, p1, h2, r0, r1);

            p0 = p1;
            p1 = p2;
        }
        auto const max2 = sizes.index(n + 1, num_moments);
        compute::kpm_kernel(sizes[max1 - 1], sizes[max2], h2, r0, r1);

        moments.collect(n, r0);
        moments.collect(n + 1, r1);
    }
}

} // namespace off_diagonal

}}} // namespace cpb::kpm::calc_moments
