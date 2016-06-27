#include "greens/kpm/OptimizedHamiltonian.hpp"
#include "greens/kpm/Stats.hpp"

namespace cpb { namespace kpm {

OptimizedSizes::OptimizedSizes(std::vector<int> sizes, Indices const& idx) : data(sizes) {
    assert(idx.cols.size() != 0);
    auto const max_index = *std::max_element(begin(idx.cols), end(idx.cols));
    auto const it = std::find_if(data.begin(), data.end(),
                                 [&](int size) { return size > max_index; });
    assert(it != data.end());
    offset = static_cast<int>(it - data.begin());
}

template<class scalar_t>
void OptimizedHamiltonian<scalar_t>::optimize_for(Indices const& idx, Scale<real_t> scale) {
    if (original_idx == idx) {
        return; // already optimized for this idx
    }

    timer.tic();
    if (config.reorder == MatrixConfig::Reorder::ON) {
        create_reordered(idx, scale);
    } else {
        create_scaled(idx, scale);
    }

    if (config.format == MatrixConfig::Format::ELL) {
        optimized_matrix = convert_to_ellpack(csr());
    }
    timer.toc();

    original_idx = idx;
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
    h2.reserve(VectorXi::Constant(system_size, sparse::max_nnz_per_row(h) + 1));

    // Note: The following "queue" and "map" use vectors instead of other container types because
    //       they serve a very simple purpose. Using preallocated vectors results in better
    //       performance (this is not an assumption, it has been tested).

    // The index queue will contain the indices that need to be checked next
    auto index_queue = std::vector<int>();
    index_queue.reserve(system_size);
    index_queue.push_back(idx.row); // starting from the given index

    // Map from original matrix indices to reordered matrix indices
    auto reorder_map = ArrayXi{ArrayXi::Constant(system_size, -1)};
    // The point of the reordering is to have the target become index number 0
    reorder_map[idx.row] = 0;

    // As the reordered matrix is filled, the optimal size for the first few KPM steps is recorded
    auto sizes = std::vector<int>();
    sizes.push_back(1);

    // Fill the reordered matrix row by row
    auto const h_view = sparse::make_loop(h);
    for (auto h2_row = 0; h2_row < system_size; ++h2_row) {
        auto diagonal_inserted = false;

        // Loop over elements in the row of the original matrix
        // corresponding to the h2_row of the reordered matrix
        auto const row = index_queue[h2_row];
        h_view.for_each_in_row(row, [&](int col, scalar_t value) {
            // A diagonal element may need to be inserted into the reordered matrix
            // even if the original matrix doesn't have an element on the main diagonal
            if (scale.b != 0 && !diagonal_inserted && col > row) {
                h2.insert(h2_row, h2_row) = -scale.b * inverted_a;
                diagonal_inserted = true;
            }

            // This may be a new index, map it
            if (reorder_map[col] < 0) {
                reorder_map[col] = static_cast<int>(index_queue.size());
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

        // Store the system size for the next KPM iteration
        if (h2_row == sizes.back() - 1) {
            sizes.push_back(static_cast<int>(index_queue.size()));
        }
    }
    h2.makeCompressed();
    optimized_matrix = h2.markAsRValue();

    sizes.pop_back(); // the last element is a duplicate of the second to last
    sizes.shrink_to_fit();

    optimized_idx = reorder_indices(idx, reorder_map);
    optimized_sizes = {std::move(sizes), optimized_idx};
}

template<class scalar_t>
num::EllMatrix<scalar_t>
OptimizedHamiltonian<scalar_t>::convert_to_ellpack(SparseMatrixX<scalar_t> const& h2_csr) {
    auto h2_ell = num::EllMatrix<scalar_t>(h2_csr.rows(), h2_csr.cols(),
                                           sparse::max_nnz_per_row(h2_csr));
    auto const h2_csr_loop = sparse::make_loop(h2_csr);
    for (auto row = 0; row < h2_csr.rows(); ++row) {
        auto n = 0;
        h2_csr_loop.for_each_in_row(row, [&](int col, scalar_t value) {
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
                                                        ArrayXi const& reorder_map) {
    auto const size = original_idx.cols.size();
    ArrayXi cols(size);
    for (auto i = 0; i < size; ++i) {
        cols[i] = reorder_map[original_idx.cols[i]];
    }
    // original_idx.row is always reordered to 0, that's the whole purpose of the optimization
    return {0, cols};
}

namespace {
    /// Return the number of non-zeros present up to `rows`
    struct NonZeros {
        int rows;

        template<class scalar_t>
        std::uint64_t operator()(SparseMatrixX<scalar_t> const& csr) {
            return static_cast<std::uint64_t>(csr.outerIndexPtr()[rows]);
        }

        template<class scalar_t>
        std::uint64_t operator()(num::EllMatrix<scalar_t> const& ell) {
            return static_cast<std::uint64_t>((rows - 1) * ell.nnz_per_row);
        }
    };
}

template<class scalar_t>
std::uint64_t OptimizedHamiltonian<scalar_t>::optimized_area(int num_moments) const {
    auto area = std::uint64_t{0};
    for (auto n = 0; n < num_moments; ++n) {
        auto const rows = optimized_sizes.optimal(n, num_moments);
        auto const num_nonzeros = var::apply_visitor(NonZeros{rows}, optimized_matrix);
        area += num_nonzeros;
    }
    return area;
}

template<class scalar_t>
std::uint64_t OptimizedHamiltonian<scalar_t>::operations(int num_moments) const {
    auto ops = optimized_area(num_moments);
    if (optimized_idx.is_diagonal()) {
        ops /= 2;
        for (auto n = 0; n <= num_moments / 2; ++n) {
            ops += 2 * static_cast<std::uint64_t>(optimized_sizes.optimal(n, num_moments));
        }
    }
    return ops;
}

template<class scalar_t>
std::string OptimizedHamiltonian<scalar_t>::report(int num_moments, bool shortform) const {
    auto const removed_percent = [&]{
        auto const nnz = var::apply_visitor(NonZeros{original_matrix->rows()}, optimized_matrix);
        auto const full_area = static_cast<double>(nnz) * num_moments;
        return 100 * (full_area - optimized_area(num_moments)) / full_area;
    }();
    auto const not_efficient = optimized_sizes.uses_full_system(num_moments) ? "" : "*";
    auto const fmt_str = shortform ? "{:.0f}%{}"
                                   : "The reordering optimization was able to "
                             "remove {:.0f}%{} of the workload";
    auto const msg = fmt::format(fmt_str, removed_percent, not_efficient);
    return format_report(msg, timer, shortform);
}

CPB_INSTANTIATE_TEMPLATE_CLASS(OptimizedHamiltonian)

}} // namespace cpb::kpm
