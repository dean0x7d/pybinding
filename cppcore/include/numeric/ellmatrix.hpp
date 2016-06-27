#pragma once
#include "numeric/dense.hpp"
#include "numeric/sparseref.hpp"

namespace cpb { namespace num {

/**
 ELLPACK format sparse matrix
 */
template<class scalar_t, class index_t = int>
class EllMatrix {
    using DataArray = Eigen::Array<scalar_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
    using IndexArray = Eigen::Array<index_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
    static constexpr auto align_bytes = 32;

public:
    index_t _rows, _cols;
    index_t nnz_per_row;
    DataArray data;
    IndexArray indices;

public:
    using Scalar = scalar_t;
    using Index = index_t;

    EllMatrix() = default;
    EllMatrix(index_t rows, index_t cols, index_t nnz_per_row)
        : _rows(rows), _cols(cols), nnz_per_row(nnz_per_row) {
        auto const aligned_rows = aligned_size<scalar_t, align_bytes>(rows);
        data.resize(aligned_rows, nnz_per_row);
        indices.resize(aligned_rows, nnz_per_row);
    }

    Index rows() const { return _rows; }
    Index cols() const { return _cols; }

    template<class F>
    void for_each(F lambda) const {
        for (auto n = 0; n < nnz_per_row; ++n) {
            for (auto row = 0; row < _rows; ++row) {
                lambda(row, indices(row, n), data(row, n));
            }
        }
    }

    template<class F>
    void for_slice(index_t start, index_t end, F lambda) const {
        for (auto n = 0; n < nnz_per_row; ++n) {
            for (auto row = start; row < end; ++row) {
                lambda(row, indices(row, n), data(row, n));
            }
        }
    }
};

/**
 Return an ELLPACK matrix reference
 */
template<class scalar_t>
inline EllConstRef<scalar_t> ellref(EllMatrix<scalar_t> const& m) {
    return {m.rows(), m.cols(), m.nnz_per_row, static_cast<int>(m.data.rows()),
            m.data.data(), m.indices.data()};
}

}} // namespace cpb::num
