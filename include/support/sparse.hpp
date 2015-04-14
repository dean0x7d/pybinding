#pragma once
#include "support/config.hpp"
#include <Eigen/SparseCore>

template <typename scalar_t>
using SparseMatrixX = Eigen::SparseMatrix<scalar_t, Eigen::RowMajor, int>;

using SparseMatrixXf = SparseMatrixX<float>;
using SparseMatrixXcf = SparseMatrixX<std::complex<float>>;
using SparseMatrixXd = SparseMatrixX<double>;
using SparseMatrixXcd = SparseMatrixX<std::complex<double>>;

template<typename scalar_t>
class CompressedInserter {
public:
    CompressedInserter(SparseMatrixX<scalar_t>& mat, int size)
        : matrix(mat) { matrix.reserve(size); }

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

template<typename scalar_t>
inline CompressedInserter<scalar_t> compressed_inserter(SparseMatrixX<scalar_t>& mat, int size) {
    return {mat, size};
}

template<class SparseMatrix, class Index>
inline auto sparse_row(const SparseMatrix& mat, Index outer_index)
    -> typename SparseMatrix::InnerIterator
{
    return {mat, outer_index};
}
