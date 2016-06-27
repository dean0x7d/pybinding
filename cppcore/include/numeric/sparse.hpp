#pragma once
#include "detail/config.hpp"
#include "numeric/sparseref.hpp"

#include <Eigen/SparseCore>

namespace cpb {

template <class scalar_t>
using SparseMatrixX = Eigen::SparseMatrix<scalar_t, Eigen::RowMajor, int>;

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
    CompressedInserter(SparseMatrixX<scalar_t>& mat, int size)
        : matrix(mat) { matrix.reserve(size); }

    void start_row() {
        matrix.outerIndexPtr()[row++] = idx;
    }

    void start_row(int row_index) {
        while (row <= row_index)
            matrix.outerIndexPtr()[row++] = idx;
    }

    void insert(int column, scalar_t value) {
        auto start_idx = matrix.outerIndexPtr()[row-1];
        auto n = idx++;
        while (n > start_idx && matrix.innerIndexPtr()[n - 1] > column) {
            matrix.innerIndexPtr()[n] = matrix.innerIndexPtr()[n - 1];
            matrix.valuePtr()[n] = matrix.valuePtr()[n - 1];
            --n;
        }
        matrix.innerIndexPtr()[n] = column;
        matrix.valuePtr()[n] = value;
    }

    void compress() {
        // close outerIndexPtr
        start_row(matrix.outerSize());
        // trim valuePtr and innerIndexPtr
        matrix.resizeNonZeros(idx);
    }

private:
    int idx = 0;
    int row = 0;
    SparseMatrixX<scalar_t>& matrix;
};

template<class scalar_t>
inline CompressedInserter<scalar_t> compressed_inserter(SparseMatrixX<scalar_t>& mat, int size) {
    return {mat, size};
}

template<class SparseMatrix, class Index>
inline auto sparse_row(const SparseMatrix& mat, Index outer_index)
    -> typename SparseMatrix::InnerIterator
{
    return {mat, outer_index};
}

namespace sparse {

/// SparseMatrix wrapper with several functions for efficient CSR matrix element access
template<class scalar_t>
class Loop {
    using index_t = typename SparseMatrixX<scalar_t>::Index;

public:
    Loop(SparseMatrixX<scalar_t> const& matrix)
        : outer_size(matrix.outerSize()), data(matrix.valuePtr()),
          indices(matrix.innerIndexPtr()), indptr(matrix.outerIndexPtr()) {}

    /// Visit each index and value of the sparse matrix:
    ///     lambda(index_t outer, index_t inner, scalar_t value)
    template<class F>
    void for_each(F lambda) const {
        for (index_t outer = 0; outer < outer_size; ++outer) {
            for (index_t idx = indptr[outer]; idx < indptr[outer + 1]; ++idx) {
                lambda(outer, indices[idx], data[idx]);
            }
        }
    }

    /// Visit each index and value of the sparse matrix:
    ///     lambda(index_t outer, index_t inner, scalar_t value, int buffer_position)
    /// After every 'buffer_size' iterations, the 'process_buffer' function is called:
    ///     process_buffer(index_t start_outer, index_t start_data, int last_buffer_size)
    template<class F1, class F2>
    void buffered_for_each(int buffer_size, F1 lambda, F2 process_buffer) const {
        int n = 0;
        index_t previous_outer = 0;
        index_t previous_idx = indptr[0];

        for (index_t outer = 0; outer < outer_size; ++outer) {
            for (index_t idx = indptr[outer]; idx < indptr[outer + 1]; ++idx, ++n) {
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
    ///     lambda(index_t inner, scalar_t value)
    template<class F>
    void for_each_in_row(index_t outer, F lambda) const {
        for (auto idx = indptr[outer]; idx < indptr[outer + 1]; ++idx) {
            lambda(indices[idx], data[idx]);
        }
    }

    /// Start iteration from some position given by 'outer' and 'data' indices
    /// and loop for 'slice_size' iterations:
    ///     lambda(index_t outer, index_t inner, scalar_t value, int current_iteration)
    template<class F>
    void slice_for_each(index_t outer, index_t idx, int slice_size, F lambda) const {
        auto n = 0;
        for (; outer < outer_size; ++outer) {
            for (; idx < indptr[outer + 1]; ++idx, ++n) {
                if (n == slice_size)
                    return;

                lambda(outer, indices[idx], data[idx], n);
            }
        }
    }

private:
    index_t const outer_size;
    scalar_t const* const data;
    index_t const* const indices;
    index_t const* const indptr;
};

template<class scalar_t>
inline Loop<scalar_t> make_loop(SparseMatrixX<scalar_t> const& m) { return {m}; }

/**
 Return the maximum number of non-zeros per row
 */
template<class scalar_t>
int max_nnz_per_row(SparseMatrixX<scalar_t> const& m) {
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
