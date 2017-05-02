#pragma once
#include "numeric/dense.hpp"
#include "numeric/sparseref.hpp"

namespace cpb { namespace num {

/**
 ELLPACK format sparse matrix
 */
template<class scalar_t>
class EllMatrix {
    using DataArray = ColMajorArrayXX<scalar_t>;
    using IndexArray = ColMajorArrayXX<storage_idx_t>;
    static constexpr auto align_bytes = 32;

public:
    idx_t _rows, _cols;
    idx_t nnz_per_row;
    DataArray data;
    IndexArray indices;

public:
    using Scalar = scalar_t;
    using StorageIndex = storage_idx_t;

    EllMatrix() = default;
    EllMatrix(idx_t rows, idx_t cols, idx_t nnz_per_row)
        : _rows(rows), _cols(cols), nnz_per_row(nnz_per_row) {
        data.resize(aligned_size<scalar_t, align_bytes>(rows), nnz_per_row);
        indices.resize(aligned_size<storage_idx_t, align_bytes>(rows), nnz_per_row);
    }

    idx_t rows() const { return _rows; }
    idx_t cols() const { return _cols; }
    idx_t nonZeros() const { return _rows * nnz_per_row; }

    template<class F>
    void for_each(F lambda) const {
        for (auto n = 0; n < nnz_per_row; ++n) {
            for (auto row = 0; row < _rows; ++row) {
                lambda(row, indices(row, n), data(row, n));
            }
        }
    }

    template<class F>
    void for_slice(idx_t start, idx_t end, F lambda) const {
        for (auto n = 0; n < nnz_per_row; ++n) {
            for (auto row = start; row < end; ++row) {
                lambda(row, indices(row, n), data(row, n));
            }
        }
    }
};

/// Return an ELLPACK matrix reference
template<class scalar_t>
inline EllConstRef<scalar_t> ellref(EllMatrix<scalar_t> const& m) {
    return {m.rows(), m.cols(), m.nnz_per_row, static_cast<int>(m.data.rows()),
            m.data.data(), m.indices.data()};
}

/// Convert an Eigen CSR matrix to ELLPACK
template<class scalar_t>
num::EllMatrix<scalar_t> csr_to_ell(SparseMatrixX<scalar_t> const& csr) {
    auto ell = num::EllMatrix<scalar_t>(csr.rows(), csr.cols(),
                                        sparse::max_nnz_per_row(csr));
    auto const loop = sparse::make_loop(csr);
    for (auto row = 0; row < csr.rows(); ++row) {
        auto n = 0;
        loop.for_each_in_row(row, [&](storage_idx_t col, scalar_t value) {
            ell.data(row, n) = value;
            ell.indices(row, n) = col;
            ++n;
        });
        for (; n < ell.nnz_per_row; ++n) {
            ell.data(row, n) = scalar_t{0};
            ell.indices(row, n) = (row > 0) ? ell.indices(row - 1, n) : 0;
        }
    }
    return ell;
}

}} // namespace cpb::num
