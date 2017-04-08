#pragma once
#include "detail/config.hpp"
#include "numeric/sparseref.hpp"

#include <Eigen/SparseCore>

namespace cpb {

template <class scalar_t>
using SparseMatrixX = Eigen::SparseMatrix<scalar_t, Eigen::RowMajor, storage_idx_t>;

using SparseMatrixXf = SparseMatrixX<float>;
using SparseMatrixXcf = SparseMatrixX<std::complex<float>>;
using SparseMatrixXd = SparseMatrixX<double>;
using SparseMatrixXcd = SparseMatrixX<std::complex<double>>;

using num::RealCsrConstRef;
using num::ComplexCsrConstRef;
using num::RealEllConstRef;
using num::ComplexEllConstRef;

/**
 Return a CSR matrix reference
 */
template<class scalar_t>
inline num::CsrConstRef<scalar_t> csrref(SparseMatrixX<scalar_t> const& m) {
    return {static_cast<int>(m.rows()), static_cast<int>(m.cols()), static_cast<int>(m.nonZeros()),
            m.valuePtr(), m.innerIndexPtr(), m.outerIndexPtr()};
};

template<class scalar_t>
class CompressedInserter {
public:
    CompressedInserter(SparseMatrixX<scalar_t>& mat, idx_t size)
        : matrix(mat) { matrix.reserve(size); }

    void start_row() {
        matrix.outerIndexPtr()[row++] = static_cast<storage_idx_t>(idx);
    }

    void start_row(idx_t row_index) {
        while (row <= row_index)
            matrix.outerIndexPtr()[row++] = static_cast<storage_idx_t>(idx);
    }

    void insert(idx_t column, scalar_t value) {
        auto start_idx = matrix.outerIndexPtr()[row-1];
        auto n = idx++;
        while (n > start_idx && matrix.innerIndexPtr()[n - 1] > column) {
            matrix.innerIndexPtr()[n] = matrix.innerIndexPtr()[n - 1];
            matrix.valuePtr()[n] = matrix.valuePtr()[n - 1];
            --n;
        }
        matrix.innerIndexPtr()[n] = static_cast<storage_idx_t>(column);
        matrix.valuePtr()[n] = value;
    }

    void compress() {
        // close outerIndexPtr
        start_row(matrix.outerSize());
        // trim valuePtr and innerIndexPtr
        matrix.resizeNonZeros(idx);
    }

private:
    idx_t idx = 0;
    idx_t row = 0;
    SparseMatrixX<scalar_t>& matrix;
};

template<class scalar_t>
inline CompressedInserter<scalar_t> compressed_inserter(SparseMatrixX<scalar_t>& mat, idx_t size) {
    return {mat, size};
}

template<class SparseMatrix, class Index>
inline auto sparse_row(const SparseMatrix& mat, Index outer_index)
    -> typename SparseMatrix::InnerIterator
{
    return {mat, outer_index};
}

namespace num {

template<class scalar_t>
SparseMatrixX<scalar_t> force_cast(SparseMatrixXcd const& m) { return m.cast<scalar_t>(); }
template<>
inline SparseMatrixX<double> force_cast<double>(SparseMatrixXcd const& m) { return m.real(); }
template<>
inline SparseMatrixX<float> force_cast<float>(SparseMatrixXcd const& m) {
    return m.real().cast<float>();
}

} // namespace num

namespace sparse {

/// SparseMatrix wrapper with several functions for efficient CSR matrix element access
template<class scalar_t>
class Loop {
public:
    Loop(SparseMatrixX<scalar_t> const& matrix)
        : outer_size(matrix.outerSize()), data(matrix.valuePtr()),
          indices(matrix.innerIndexPtr()), indptr(matrix.outerIndexPtr()) {}

    /// Visit each index and value of the sparse matrix:
    ///     lambda(idx_t outer, idx_t inner, scalar_t value)
    template<class F>
    void for_each(F lambda) const {
        for (auto outer = idx_t{0}; outer < outer_size; ++outer) {
            for (auto idx = indptr[outer]; idx < indptr[outer + 1]; ++idx) {
                lambda(outer, indices[idx], data[idx]);
            }
        }
    }

    /// Visit each index and value of the sparse matrix:
    ///     lambda(idx_t outer, idx_t inner, scalar_t value, idx_t buffer_position)
    /// After every 'buffer_size' iterations, the 'process_buffer' function is called:
    ///     process_buffer(idx_t start_outer, idx_t start_data, idx_t last_buffer_size)
    template<class F1, class F2>
    void buffered_for_each(idx_t buffer_size, F1 lambda, F2 process_buffer) const {
        auto n = idx_t{0};
        auto previous_outer = idx_t{0};
        auto previous_idx = static_cast<idx_t>(indptr[0]);

        for (auto outer = idx_t{0}; outer < outer_size; ++outer) {
            for (auto idx = indptr[outer]; idx < indptr[outer + 1]; ++idx, ++n) {
                if (n == buffer_size) {
                    process_buffer(previous_outer, previous_idx, buffer_size);
                    previous_outer = outer;
                    previous_idx = idx;
                    n = 0;
                }

                lambda(outer, indices[idx], data[idx], n);
            }
        }

        process_buffer(previous_outer, previous_idx, n);
    }

    /// Iterate over all elements in a single row (or column) at the 'outer' index:
    ///     lambda(idx_t inner, scalar_t value)
    template<class F>
    void for_each_in_row(idx_t outer, F lambda) const {
        for (auto idx = indptr[outer]; idx < indptr[outer + 1]; ++idx) {
            lambda(indices[idx], data[idx]);
        }
    }

    /// Start iteration from some position given by 'outer' and 'data' indices
    /// and loop for 'slice_size' iterations:
    ///     lambda(idx_t outer, idx_t inner, scalar_t value, idx_t current_iteration)
    template<class F>
    void slice_for_each(idx_t outer, idx_t idx, idx_t slice_size, F lambda) const {
        auto n = idx_t{0};
        for (; outer < outer_size; ++outer) {
            for (; idx < indptr[outer + 1]; ++idx, ++n) {
                if (n == slice_size)
                    return;

                lambda(outer, indices[idx], data[idx], n);
            }
        }
    }

private:
    idx_t const outer_size;
    scalar_t const* const data;
    storage_idx_t const* const indices;
    storage_idx_t const* const indptr;
};

template<class scalar_t>
inline Loop<scalar_t> make_loop(SparseMatrixX<scalar_t> const& m) { return {m}; }

/**
 Return the maximum number of non-zeros per row
 */
template<class scalar_t>
idx_t max_nnz_per_row(SparseMatrixX<scalar_t> const& m) {
    auto max = 0;
    for (auto i = 0; i < m.outerSize(); ++i) {
        auto const nnz = m.outerIndexPtr()[i + 1] - m.outerIndexPtr()[i];
        if (nnz > max) {
            max = nnz;
        }
    }
    return max;
}

}} // namespace cpb::sparse
