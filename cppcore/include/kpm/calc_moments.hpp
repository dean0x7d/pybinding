#pragma once

namespace cpb { namespace kpm { namespace calc_moments {

template<class...> struct void_type { using type = void; };
template<class... Ts> using void_t = typename void_type<Ts...>::type;

template<class Collector, class = void>
struct is_diagonal : std::false_type {};

template<class Collector>
struct is_diagonal<Collector, void_t<decltype(Collector::zero())>> : std::true_type {};

template<class Collector>
using requires_diagonal = typename std::enable_if<is_diagonal<Collector>::value, int>::type;

template<class Collector>
using requires_offdiagonal = typename std::enable_if<!is_diagonal<Collector>::value, int>::type;

/**********************************************************************\
 Diagonal KPM implementation: the left and right vectors are identical,
 i.e. `mu_n = <r|Tn(H)|r>` where `bra == ket == r`.
\**********************************************************************/

/**
 Reference implementation, no optimizations, no special requirements

 Calculates moments for a single matrix element (i, i) on the main diagonal.
 It's 1.5x to 2x times faster than the general (off-diagonal) version.

 The initial vectors `r0` and `r1` are given by the user. E.g. `r0` can be
 a unit vector for the expectation value variant or a random vector for the
 stochastic trace variant.
 */
template<class C, class Vector, class Matrix, requires_diagonal<C> = 1>
void basic(C& collect, Vector r0, Vector r1, Matrix const& h2) {
    auto const num_moments = collect.size();
    assert(num_moments % 2 == 0);

    static constexpr auto zero = C::zero();
    for (auto n = 2; n <= num_moments / 2; ++n) {
        auto m2 = zero, m3 = zero;
        compute::kpm_spmv_diagonal(0, h2.rows(), h2, r1, r0, m2, m3);
        collect(n, m2, m3);
        r1.swap(r0);
    }
}

/**
 Optimal size optimization, requires a specially ordered matrix as input

 At each iteration, the computation is performed only for a subset of the total
 system which contains non-zero values. The speedup is about equal to the amount
 of removed work.
 */
template<class C, class Vector, class Matrix, requires_diagonal<C> = 1>
void opt_size(C& collect, Vector r0, Vector r1, Matrix const& h2, SliceMap const& map) {
    auto const num_moments = collect.size();
    assert(num_moments % 2 == 0);

    static constexpr auto zero = C::zero();
    for (auto n = 2; n <= num_moments / 2; ++n) {
        auto m2 = zero, m3 = zero;
        compute::kpm_spmv_diagonal(0, map.optimal_size(n, num_moments), h2, r1, r0, m2, m3);
        collect(n, m2, m3);
        r1.swap(r0);
    }
}

/**
 Interleave two moment calculations, requires a specially ordered matrix as input

 The two concurrent operations share some of the same data, thus promoting cache
 usage and reducing main memory bandwidth.
 */
template<class C, class Vector, class Matrix, requires_diagonal<C> = 1>
void interleaved(C& collect, Vector r0, Vector r1, Matrix const& h2, SliceMap const& map) {
    auto const num_moments = collect.size();
    assert((num_moments - 2) % 4 == 0);

    // Interleave moments `n` and `n + 1` for better data locality
    // Diagonal + interleaved computes 4 moments per iteration
    static constexpr auto zero = C::zero();
    for (auto n = idx_t{2}; n <= num_moments / 2; n += 2) {
        auto m2 = zero, m3 = zero, m4 = zero, m5 = zero;

        auto const max = map.last_index();
        for (auto k = idx_t{0}, p0 = idx_t{0}, p1 = idx_t{0}; k <= max; ++k) {
            auto const p2 = map[k];
            compute::kpm_spmv_diagonal(p1, p2, h2, r1, r0, m2, m3);
            compute::kpm_spmv_diagonal(p0, p1, h2, r0, r1, m4, m5);

            p0 = p1;
            p1 = p2;
        }
        compute::kpm_spmv_diagonal(map[max - 1], map[max], h2, r0, r1, m4, m5);

        collect(n, m2, m3);
        collect(n + 1, m4, m5);
    }
}

/**
 Optimal size + interleaved
 */
template<class C, class Vector, class Matrix, requires_diagonal<C> = 1>
void opt_size_and_interleaved(C& collect, Vector r0, Vector r1, Matrix const& h2,
                              SliceMap const& map) {
    auto const num_moments = collect.size();
    assert((num_moments - 2) % 4 == 0);

    static constexpr auto zero = C::zero();
    for (auto n = idx_t{2}; n <= num_moments / 2; n += 2) {
        auto m2 = zero, m3 = zero, m4 = zero, m5 = zero;

        auto const max1 = map.index(n, num_moments);
        for (auto k = idx_t{0}, p0 = idx_t{0}, p1 = idx_t{0}; k <= max1; ++k) {
            auto const p2 = map[k];
            compute::kpm_spmv_diagonal(p1, p2, h2, r1, r0, m2, m3);
            compute::kpm_spmv_diagonal(p0, p1, h2, r0, r1, m4, m5);

            p0 = p1;
            p1 = p2;
        }
        auto const max2 = map.index(n + 1, num_moments);
        compute::kpm_spmv_diagonal(map[max1 - 1], map[max2], h2, r0, r1, m4, m5);

        collect(n, m2, m3);
        collect(n + 1, m4, m5);
    }
}

/******************************************************************\
 Off-diagonal KPM implementation: different left and right vectors,
 i.e. `mu_n = <l|Tn(H)|r>` where `l != r`. The `Moments` collector
 contains the left vector.
\******************************************************************/

/**
 Reference implementation, no optimizations, no special requirements

 Calculates moments for multiple indices in the same row -- the Moments class
 knows which final values to collect. Both diagonal and off-diagonal moments
 can be computed, but the diagonal version of this function is more efficient
 for that special case.
 */
template<class C, class Vector, class Matrix, requires_offdiagonal<C> = 1>
void basic(C& collect, Vector r0, Vector r1, Matrix const& h2) {
    auto const num_moments = collect.size();
    for (auto n = idx_t{2}; n < num_moments; ++n) {
        compute::kpm_spmv(0, h2.rows(), h2, r1, r0);

        r1.swap(r0);
        collect(n, r1);
    }
}

/**
 Optimal size optimization, requires a specially ordered matrix as input

 See the diagonal version of this function for more information.
 */
template<class C, class Vector, class Matrix, requires_offdiagonal<C> = 1>
void opt_size(C& collect, Vector r0, Vector r1, Matrix const& h2, SliceMap const& map) {
    auto const num_moments = collect.size();
    for (auto n = idx_t{2}; n < num_moments; ++n) {
        auto const opt_size = map.optimal_size(n, num_moments);

        compute::kpm_spmv(0, opt_size, h2, r1, r0); // r0 = matrix * r1 - r0

        r1.swap(r0);
        collect(n, r1);
    }
}

/**
 Interleave two moment calculations, requires a specially ordered matrix as input

 See the diagonal version of this function for more information.
 */
template<class C, class Vector, class Matrix, requires_offdiagonal<C> = 1>
void interleaved(C& collect, Vector r0, Vector r1, Matrix const& h2, SliceMap const& map) {
    auto const num_moments = collect.size();
    assert(num_moments % 2 == 0);

    // Interleave moments `n` and `n + 1` for better data locality
    for (auto n = idx_t{2}; n < num_moments; n += 2) {
        auto const max = map.last_index();
        for (auto k = idx_t{0}, p0 = idx_t{0}, p1 = idx_t{0}; k <= max; ++k) {
            auto const p2 = map[k];
            compute::kpm_spmv(p1, p2, h2, r1, r0);
            compute::kpm_spmv(p0, p1, h2, r0, r1);

            p0 = p1;
            p1 = p2;
        }
        compute::kpm_spmv(map[max - 1], map[max], h2, r0, r1);

        collect(n, r0);
        collect(n + 1, r1);
    }
}

/**
 Optimal size + interleaved
 */
template<class C, class Vector, class Matrix, requires_offdiagonal<C> = 1>
void opt_size_and_interleaved(C& collect, Vector r0, Vector r1, Matrix const& h2,
                              SliceMap const& map) {
    auto const num_moments = collect.size();
    assert(num_moments % 2 == 0);

    // Interleave moments `n` and `n + 1` for better data locality
    for (auto n = idx_t{2}; n < num_moments; n += 2) {
        auto const max1 = map.index(n, num_moments);
        for (auto k = idx_t{0}, p0 = idx_t{0}, p1 = idx_t{0}; k <= max1; ++k) {
            auto const p2 = map[k];
            compute::kpm_spmv(p1, p2, h2, r1, r0);
            compute::kpm_spmv(p0, p1, h2, r0, r1);

            p0 = p1;
            p1 = p2;
        }
        auto const max2 = map.index(n + 1, num_moments);
        compute::kpm_spmv(map[max1 - 1], map[max2], h2, r0, r1);

        collect(n, r0);
        collect(n + 1, r1);
    }
}

}}} // namespace cpb::kpm::calc_moments
