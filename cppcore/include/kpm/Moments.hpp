#pragma once
#include "kpm/OptimizedHamiltonian.hpp"

#include "numeric/dense.hpp"
#include "numeric/constant.hpp"

#include "detail/macros.hpp"

namespace cpb { namespace kpm {

/**
 Utility functions for expectation value moments
 */
namespace exval {

/// Return the KPM r0 vector with all zeros except for the source index
template<class Matrix, class scalar_t = typename Matrix::Scalar>
VectorX<scalar_t> make_r0(Matrix const& h2, int i) {
    auto r0 = VectorX<scalar_t>::Zero(h2.rows()).eval();
    r0[i] = 1;
    return r0;
}

/// Return the KPM r1 vector which is equal to the Hamiltonian matrix column at the source index
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

} // namespace exval

/**
 Sets the initial conditions for the diagonal expectation
 value routine and collects the computed KPM moments.
 */
template<class scalar_t>
class ExvalDiagonalMoments {
public:
    ExvalDiagonalMoments(int num_moments, int index) : moments(num_moments), index(index) {}

    int size() const { return static_cast<int>(moments.size()); }
    ArrayX<scalar_t>& get() { return moments; }

    /// Initial vector
    template<class Matrix>
    VectorX<scalar_t> r0(Matrix const& h2) const {
        return exval::make_r0(h2, index);
    }

    /// Next vector
    template<class Matrix, class Vector>
    VectorX<scalar_t> r1(Matrix const& h2, Vector const& /*r0*/) const {
        return exval::make_r1(h2, index);
    }

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
  Like `ExvalDiagonalMoments` but collects the computed moments for several indices.
*/
template<class scalar_t>
class ExvalOffDiagonalMoments {
public:
    ExvalOffDiagonalMoments(int num_moments, Indices const& idx)
        : idx(idx), data(idx.cols.size()) {
        for (auto& moments : data) {
            moments.resize(num_moments);
        }
    }

    int size() const { return static_cast<int>(data[0].size()); }
    std::vector<ArrayX<scalar_t>>& get() { return data; }

    /// Initial vector
    template<class Matrix>
    VectorX<scalar_t> r0(Matrix const& h2) const {
        return exval::make_r0(h2, idx.row);
    }

    /// Next vector
    template<class Matrix, class Vector>
    VectorX<scalar_t> r1(Matrix const& h2, Vector const&) const {
        return exval::make_r1(h2, idx.row);
    }

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
    std::vector<ArrayX<scalar_t>> data;
};

}} // namespace cpb::kpm
