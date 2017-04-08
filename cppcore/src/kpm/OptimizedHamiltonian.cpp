#include "kpm/OptimizedHamiltonian.hpp"

namespace cpb { namespace kpm {

SliceMap::SliceMap(std::vector<storage_idx_t> indices, Indices const& optimized_idx)
    : data(std::move(indices)) {
    assert(optimized_idx.cols.size() != 0);
    auto const max_index = *std::max_element(begin(optimized_idx.cols), end(optimized_idx.cols));
    auto const it = std::find_if(data.begin(), data.end(),
                                 [&](storage_idx_t index) { return index > max_index; });
    assert(it != data.end());
    offset = static_cast<idx_t>(it - data.begin());
}

template<class scalar_t>
void OptimizedHamiltonian<scalar_t>::optimize_for(Indices const& idx, Scale<real_t> scale) {
    if (original_idx == idx) {
        return; // already optimized for this idx
    }

    timer.tic();
    if (is_reordered) {
        create_reordered(idx, scale);
    } else {
        create_scaled(idx, scale);
    }

    if (matrix_format == MatrixFormat::ELL) {
        auto const& csr = optimized_matrix.template get<SparseMatrixX<scalar_t>>();
        optimized_matrix = convert_to_ellpack(csr);
    }
    timer.toc();

    original_idx = idx;
}

template<class scalar_t>
void OptimizedHamiltonian<scalar_t>::reorder(SparseMatrixX<scalar_t>& matrix) const {
    if (reorder_map.empty()) { return; }

    auto reordered_matrix = SparseMatrixX<scalar_t>(matrix.rows(), matrix.cols());
    auto const reserve_per_row = static_cast<int>(sparse::max_nnz_per_row(matrix));
    reordered_matrix.reserve(ArrayXi::Constant(matrix.rows(), reserve_per_row));

    sparse::make_loop(matrix).for_each([&](idx_t row, idx_t col, scalar_t value) {
        reordered_matrix.insert(reorder_map[row], reorder_map[col]) = value;
    });

    reordered_matrix.makeCompressed();
    matrix.swap(reordered_matrix);
}

template<class scalar_t>
void OptimizedHamiltonian<scalar_t>::create_scaled(Indices const& idx, Scale<real_t> scale) {
    optimized_idx = idx;

    auto const& h = *original_matrix;
    auto h2 = SparseMatrixX<scalar_t>();
    if (scale.b == 0) { // just scale, no b offset
        h2 = h * (2 / scale.a);
    } else { // scale and offset
        auto I = SparseMatrixX<scalar_t>{h.rows(), h.cols()};
        I.setIdentity();
        h2 = (h - I * scale.b) * (2 / scale.a);
    }
    h2.makeCompressed();
    optimized_matrix = h2.markAsRValue();
}

template<class scalar_t>
void OptimizedHamiltonian<scalar_t>::create_reordered(Indices const& idx, Scale<real_t> scale) {
    auto const& h = *original_matrix;
    auto const system_size = h.rows();
    auto const inverted_a = real_t{2 / scale.a};

    auto h2 = SparseMatrixX<scalar_t>(system_size, system_size);
    // Reserve the same nnz per row as the original + 1 in case the scaling adds diagonal elements
    h2.reserve(VectorX<idx_t>::Constant(system_size, sparse::max_nnz_per_row(h) + 1));

    // Note: The following "queue" and "map" use vectors instead of other container types because
    //       they serve a very simple purpose. Using preallocated vectors results in better
    //       performance (this is not an assumption, it has been tested).

    // The index queue will contain the indices that need to be checked next
    auto index_queue = std::vector<storage_idx_t>();
    index_queue.reserve(system_size);
    index_queue.push_back(idx.row); // starting from the given index

    // Map from original matrix indices to reordered matrix indices
    reorder_map = std::vector<storage_idx_t>(system_size, -1); // reset all to invalid state
    // The point of the reordering is to have the target become index number 0
    reorder_map[idx.row] = 0;

    // As the reordered matrix is filled, the slice border indices are recorded
    auto slice_border_indices = std::vector<storage_idx_t>();
    slice_border_indices.push_back(1);

    // Fill the reordered matrix row by row
    auto const h_view = sparse::make_loop(h);
    for (auto h2_row = 0; h2_row < system_size; ++h2_row) {
        auto diagonal_inserted = false;

        // Loop over elements in the row of the original matrix
        // corresponding to the h2_row of the reordered matrix
        auto const row = index_queue[h2_row];
        h_view.for_each_in_row(row, [&](storage_idx_t col, scalar_t value) {
            // A diagonal element may need to be inserted into the reordered matrix
            // even if the original matrix doesn't have an element on the main diagonal
            if (scale.b != 0 && !diagonal_inserted && col > row) {
                h2.insert(h2_row, h2_row) = -scale.b * inverted_a;
                diagonal_inserted = true;
            }

            // This may be a new index, map it
            if (reorder_map[col] < 0) {
                reorder_map[col] = static_cast<storage_idx_t>(index_queue.size());
                index_queue.push_back(col);
            }

            // Get the reordered column index
            auto const h2_col = reorder_map[col];

            // Calculate the new value that will be inserted into the scaled/reordered matrix
            auto h2_value = value * inverted_a;
            if (row == col) { // diagonal elements
                h2_value -= scale.b * inverted_a;
                diagonal_inserted = true;
            }

            h2.insert(h2_row, h2_col) = h2_value;
        });

        // Reached the end of a slice
        if (h2_row == slice_border_indices.back() - 1) {
            slice_border_indices.push_back(static_cast<storage_idx_t>(index_queue.size()));
        }
    }
    h2.makeCompressed();
    optimized_matrix = h2.markAsRValue();

    slice_border_indices.pop_back(); // the last element is a duplicate of the second to last
    slice_border_indices.shrink_to_fit();

    optimized_idx = reorder_indices(idx, reorder_map);
    slice_map = {std::move(slice_border_indices), optimized_idx};
}

template<class scalar_t>
num::EllMatrix<scalar_t>
OptimizedHamiltonian<scalar_t>::convert_to_ellpack(SparseMatrixX<scalar_t> const& h2_csr) {
    auto h2_ell = num::EllMatrix<scalar_t>(h2_csr.rows(), h2_csr.cols(),
                                           sparse::max_nnz_per_row(h2_csr));
    auto const h2_csr_loop = sparse::make_loop(h2_csr);
    for (auto row = 0; row < h2_csr.rows(); ++row) {
        auto n = 0;
        h2_csr_loop.for_each_in_row(row, [&](storage_idx_t col, scalar_t value) {
            h2_ell.data(row, n) = value;
            h2_ell.indices(row, n) = col;
            ++n;
        });
        for (; n < h2_ell.nnz_per_row; ++n) {
            h2_ell.data(row, n) = scalar_t{0};
            h2_ell.indices(row, n) = (row > 0) ? h2_ell.indices(row - 1, n) : 0;
        }
    }
    return h2_ell;
}

template<class scalar_t>
Indices OptimizedHamiltonian<scalar_t>::reorder_indices(Indices const& original_idx,
                                                        std::vector<storage_idx_t> const& map) {
    auto const size = original_idx.cols.size();
    ArrayX<storage_idx_t> cols(size);
    for (auto i = 0; i < size; ++i) {
        cols[i] = map[original_idx.cols[i]];
    }
    // original_idx.row is always reordered to 0, that's the whole purpose of the optimization
    return {0, std::move(cols)};
}

namespace {
    /// Return the number of non-zeros present up to `rows`
    struct NonZeros {
        idx_t rows;

        template<class scalar_t>
        size_t operator()(SparseMatrixX<scalar_t> const& csr) {
            return static_cast<size_t>(csr.outerIndexPtr()[rows]);
        }

        template<class scalar_t>
        size_t operator()(num::EllMatrix<scalar_t> const& ell) {
            return static_cast<size_t>(rows * ell.nnz_per_row);
        }
    };
}

template<class scalar_t>
size_t OptimizedHamiltonian<scalar_t>::num_nonzeros(idx_t num_moments, bool optimal_size) const {
    auto result = size_t{0};
    if (!optimal_size) {
        result = num_moments * var::apply_visitor(NonZeros{size()}, optimized_matrix);
    } else {
        for (auto n = 0; n < num_moments; ++n) {
            auto const opt_size = slice_map.optimal_size(n, num_moments);
            auto const num_nonzeros = var::apply_visitor(NonZeros{opt_size}, optimized_matrix);
            result += num_nonzeros;
        }
    }
    if (optimized_idx.is_diagonal()) {
        result /= 2;
    }
    return result;
}

template<class scalar_t>
size_t OptimizedHamiltonian<scalar_t>::num_vec_elements(idx_t num_moments,
                                                        bool optimal_size) const {
    auto result = size_t{0};
    if (!optimal_size) {
        result = num_moments * size();
    } else {
        for (auto n = 0; n < num_moments; ++n) {
            result += static_cast<size_t>(slice_map.optimal_size(n, num_moments));
        }
    }
    if (optimized_idx.is_diagonal()) {
        result /= 2;
    }
    return result;
}

namespace {
    /// Return the data size in bytes
    struct MatrixMemory {
        template<class scalar_t>
        size_t operator()(SparseMatrixX<scalar_t> const& csr) const {
            using index_t = typename SparseMatrixX<scalar_t>::StorageIndex;
            auto const nnz = static_cast<size_t>(csr.nonZeros());
            auto const row_starts = static_cast<size_t>(csr.rows() + 1);
            return nnz * sizeof(scalar_t) + nnz * sizeof(index_t) + row_starts * sizeof(index_t);
        }

        template<class scalar_t>
        size_t operator()(num::EllMatrix<scalar_t> const& ell) const {
            using index_t = typename num::EllMatrix<scalar_t>::StorageIndex;
            auto const nnz = static_cast<size_t>(ell.nonZeros());
            return nnz * sizeof(scalar_t) + nnz * sizeof(index_t);
        }
    };
}

template<class scalar_t>
size_t OptimizedHamiltonian<scalar_t>::matrix_memory() const {
    return var::apply_visitor(MatrixMemory{}, optimized_matrix);
}

template<class scalar_t>
size_t OptimizedHamiltonian<scalar_t>::vector_memory() const {
    return size() * sizeof(scalar_t);
}

CPB_INSTANTIATE_TEMPLATE_CLASS(OptimizedHamiltonian)

}} // namespace cpb::kpm
