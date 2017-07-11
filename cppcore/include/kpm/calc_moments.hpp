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

/************************************************************************\
 Diagonal KPM implementation: the left and right vectors are identical,
 i.e. `mu_n = <r|Tn(H)|r>` where `bra == ket == r`. It's 1.5x to 2x times
 faster than the general (off-diagonal) version.
\************************************************************************/

/**
 Basic implementation with an optional size optimization (when applicable)

 When `opt_size == true`, the optimal size optimization is used. This requires
 a specially ordered matrix as input. This matrix is divided into slices which
 are mapped by `SliceMap`. At each iteration, the computation is performed only
 for a subset of the total system which contains non-zero values. The speedup
 is about equal to the amount of removed work.
 */
template<class Collector, class Vector, class Matrix, requires_diagonal<Collector> = 1>
void basic(Collector& collect, Vector r0, Vector r1, Matrix const& h2,
           SliceMap const& map, bool opt_size) {
    auto const num_moments = collect.size();
    assert(num_moments % 2 == 0);

    auto const zero = Collector::zero();
    for (auto n = 2; n <= num_moments / 2; ++n) {
        auto m2 = zero, m3 = zero;
        auto const size = opt_size ? map.optimal_size(n, num_moments) : h2.rows();

        compute::kpm_spmv_diagonal(0, size, h2, r1, r0, m2, m3);

        collect(n, m2, m3);
        r1.swap(r0);
    }
}

/**
 Optimized implementation: interleave two consecutive moment calculations

 Requires a specially ordered matrix as input.

 The two concurrent operations share some of the same data, thus promoting cache
 usage and reducing main memory bandwidth.
 */
template<class Collector, class Vector, class Matrix, requires_diagonal<Collector> = 1>
void interleaved(Collector& collect, Vector r0, Vector r1, Matrix const& h2,
                 SliceMap const& map, bool opt_size) {
    auto const num_moments = collect.size();
    assert((num_moments - 2) % 4 == 0);

    // Interleave moments `n` and `n + 1` for better data locality
    // Diagonal + interleaved computes 4 moments per iteration
    auto const zero = Collector::zero();
    for (auto n = idx_t{2}; n <= num_moments / 2; n += 2) {
        auto m2 = zero, m3 = zero, m4 = zero, m5 = zero;
        auto const max1 = opt_size ? map.index(n,     num_moments) : map.last_index();
        auto const max2 = opt_size ? map.index(n + 1, num_moments) : map.last_index();

        for (auto k = idx_t{0}, start0 = idx_t{0}, start1 = idx_t{0}; k <= max1; ++k) {
            auto const end0 = map[k];
            auto const end1 = (k == max1) ? map[max2] : start0;

            compute::kpm_spmv_diagonal(start0, end0, h2, r1, r0, m2, m3);
            compute::kpm_spmv_diagonal(start1, end1, h2, r0, r1, m4, m5);

            start1 = end1;
            start0 = end0;
        }

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
 Basic implementation with an optional size optimization (when applicable)

 See the diagonal version of this function for more information.
 */
template<class Collector, class Vector, class Matrix, requires_offdiagonal<Collector> = 1>
void basic(Collector& collect, Vector r0, Vector r1, Matrix const& h2,
           SliceMap const& map, bool opt_size) {
    auto const num_moments = collect.size();
    for (auto n = idx_t{2}; n < num_moments; ++n) {
        auto const size = opt_size ? map.optimal_size(n, num_moments) : h2.rows();

        compute::kpm_spmv(0, size, h2, r1, r0); // r0 = matrix * r1 - r0

        r1.swap(r0);
        collect(n, r1);
    }
}

/**
 Optimized implementation: interleave two consecutive moment calculations

 See the diagonal version of this function for more information.
 */
template<class C, class Vector, class Matrix, requires_offdiagonal<C> = 1>
void interleaved(C& collect, Vector r0, Vector r1, Matrix const& h2,
                 SliceMap const& map, bool opt_size) {
    auto const num_moments = collect.size();
    assert(num_moments % 2 == 0);

    // Interleave moments `n` and `n + 1` for better data locality
    for (auto n = idx_t{2}; n < num_moments; n += 2) {
        auto const max1 = opt_size ? map.index(n,     num_moments) : map.last_index();
        auto const max2 = opt_size ? map.index(n + 1, num_moments) : map.last_index();

        for (auto k = idx_t{0}, start0 = idx_t{0}, start1 = idx_t{0}; k <= max1; ++k) {
            auto const end0 = map[k];
            auto const end1 = (k == max1) ? map[max2] : start0;

            compute::kpm_spmv(start0, end0, h2, r1, r0);
            compute::kpm_spmv(start1, end1, h2, r0, r1);

            start1 = start0;
            start0 = end0;
        }

        collect(n, r0);
        collect(n + 1, r1);
    }
}

}}} // namespace cpb::kpm::calc_moments
