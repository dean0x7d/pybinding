#pragma once
#include "numeric/dense.hpp"
#include "numeric/sparse.hpp"

namespace tbm {

struct SparseURef {
    tbm::num::ArrayConstRef const values;
    tbm::num::ArrayConstRef const inner_indices;
    tbm::num::ArrayConstRef const outer_starts;
    int const rows, cols;

    template<class scalar_t>
    SparseURef(tbm::SparseMatrixX<scalar_t> const& v)
        : values(tbm::arrayref(v.valuePtr(), v.nonZeros())),
          inner_indices(tbm::arrayref(v.innerIndexPtr(), v.nonZeros())),
          outer_starts(tbm::arrayref(v.outerIndexPtr(), v.outerSize() + 1)),
          rows(v.rows()), cols(v.cols())
    {}
};

} // namespace tbm
