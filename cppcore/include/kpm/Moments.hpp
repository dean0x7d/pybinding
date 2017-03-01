#pragma once
#include "kpm/OptimizedHamiltonian.hpp"

#include <vector>

namespace cpb { namespace kpm {

/**
 Collects the computed KPM moments for a single index on the diagonal
 */
template<class scalar_t>
class DiagonalMoments {
public:
    using VectorRef = Ref<VectorX<scalar_t>>;

    DiagonalMoments(idx_t num_moments) : moments(ArrayX<scalar_t>::Zero(num_moments)) {}
    virtual ~DiagonalMoments() = default;

    idx_t size() const { return moments.size(); }
    ArrayX<scalar_t>& get() { return moments; }

    /// Collect the first 2 moments which are computer outside the main KPM loop
    virtual void collect_initial(VectorRef r0, VectorRef r1);

    /// Collect moments `n` and `n + 1` from the result vectors. Expects `n >= 2`.
    virtual void collect(idx_t n, VectorRef r0, VectorRef r1);
    virtual void collect(idx_t n, scalar_t a, scalar_t b);

private:
    ArrayX<scalar_t> moments;
    scalar_t m0;
    scalar_t m1;
};

/**
  Like `DiagonalMoments` but collects the computed moments for several indices
*/
template<class scalar_t>
class OffDiagonalMoments {
    using VectorRef = Ref<VectorX<scalar_t>>;
    using MomentsVector = ArrayX<scalar_t>;
    using Data = std::vector<MomentsVector>;

public:
    OffDiagonalMoments(idx_t num_moments, Indices const& idx)
        : idx(idx), data(idx.cols.size()) {
        for (auto& moments : data) {
            moments.resize(num_moments);
        }
    }
    virtual ~OffDiagonalMoments() = default;

    idx_t size() const { return data[0].size(); }
    Data& get() { return data; }

    /// Collect the first 2 moments which are computer outside the main KPM loop
    virtual void collect_initial(VectorRef r0, VectorRef r1);

    /// Collect moment `n` from the result vector `r1`. Expects `n >= 2`.
    virtual void collect(idx_t n, VectorRef r1);

private:
    Indices idx;
    Data data;
};

CPB_EXTERN_TEMPLATE_CLASS(DiagonalMoments)
CPB_EXTERN_TEMPLATE_CLASS(OffDiagonalMoments)

}} // namespace cpb::kpm
