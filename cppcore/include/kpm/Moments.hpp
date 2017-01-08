#pragma once
#include "numeric/dense.hpp"
#include "detail/macros.hpp"

#include <vector>

namespace cpb { namespace kpm {

/**
 Collects the computed KPM moments for a single index on the diagonal
 */
template<class scalar_t>
class DiagonalMoments {
public:
    DiagonalMoments(int num_moments, int index) : moments(num_moments), index(index) {}

    int size() const { return static_cast<int>(moments.size()); }
    ArrayX<scalar_t>& get() { return moments; }

    /// Collect the first 2 moments which are computer outside the main KPM loop
    void collect_initial(VectorX<scalar_t> const& r0, VectorX<scalar_t> const& r1) {
        m0 = moments[0] = r0[index] * scalar_t{0.5};
        m1 = moments[1] = r1[index];
    }

    /// Collect moments `n` and `n + 1` from the result vectors. Expects `n >= 2`.
    template<class Vector>
    CPB_ALWAYS_INLINE void collect(int n, Vector const& r0, Vector const& r1) {
        collect(n, r0.squaredNorm(), r1.dot(r0));
    }

    CPB_ALWAYS_INLINE void collect(int n, scalar_t a, scalar_t b) {
        assert(n >= 2 && n <= size() / 2);
        moments[2 * (n - 1)] = scalar_t{2} * (a - m0);
        moments[2 * (n - 1) + 1] = scalar_t{2} * b - m1;
    }

    template<class V1, class V2> void pre_process(V1 const&, V2 const&) {}
    template<class V1, class V2> void post_process(V1 const&, V2 const&) {}

private:
    ArrayX<scalar_t> moments;
    int index;
    scalar_t m0;
    scalar_t m1;
};

/**
  Like `DiagonalMoments` but collects the computed moments for several indices
*/
template<class scalar_t>
class OffDiagonalMoments {
    using MomentsVector = ArrayX<scalar_t>;
    using Data = std::vector<MomentsVector>;

public:
    OffDiagonalMoments(int num_moments, Indices const& idx)
        : idx(idx), data(idx.cols.size()) {
        for (auto& moments : data) {
            moments.resize(num_moments);
        }
    }

    int size() const { return static_cast<int>(data[0].size()); }
    Data& get() { return data; }

    /// Collect the first 2 moments which are computer outside the main KPM loop
    void collect_initial(VectorX<scalar_t> const& r0, VectorX<scalar_t> const& r1) {
        using real_t = num::get_real_t<scalar_t>;

        for (auto i = 0; i < idx.cols.size(); ++i) {
            auto const col = idx.cols[i];
            data[i][0] = r0[col] * real_t{0.5}; // 0.5 is special for the moment zero
            data[i][1] = r1[col];
        }
    }

    /// Collect moment `n` from the result vector `r1`. Expects `n >= 2`.
    void collect(int n, VectorX<scalar_t> const& r1) {
        assert(n >= 2 && n < data[0].size());
        for (auto i = 0; i < idx.cols.size(); ++i) {
            auto const col = idx.cols[i];
            data[i][n] = r1[col];
        }
    }

    template<class V1, class V2> void pre_process(V1 const&, V2 const&) {}
    template<class V1, class V2> void post_process(V1 const&, V2 const&) {}

private:
    Indices idx;
    Data data;
};

}} // namespace cpb::kpm
