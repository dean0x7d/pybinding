#include "greens/KPM.hpp"

#include "compute/kernel_polynomial.hpp"
#include "support/format.hpp"


namespace tbm {
using namespace kpm;

template<class scalar_t>
void Scale<scalar_t>::compute(SparseMatrixX<scalar_t> const& matrix, real_t lanczos_tolerance) {
    if (a != 0)
        return; // already computed

    if (bounds.min == bounds.max) {
        bounds = compute::minmax_eigenvalues(matrix, lanczos_tolerance);
    }

    a = 0.5f * (bounds.max - bounds.min) * (1 + scaling_tolerance);
    b = 0.5f * (bounds.max + bounds.min);

    // Round to zero if b is very small in order to make the the sparse matrix smaller
    if (std::abs(b / a) < 0.01f * scaling_tolerance)
        b = 0;
}


template<class scalar_t>
void OptimizedHamiltonian<scalar_t>::create(SparseMatrixX<scalar_t> const& H, Indices const& idx,
                                            Scale<scalar_t> scale, bool use_reordering) {
    if (original_idx == idx) {
        return; // already optimized for this idx
    }

    if (use_reordering) {
        create_reordered(H, idx, scale);
    } else {
        create_scaled(H, idx, scale);
    }
}

template<class scalar_t>
void OptimizedHamiltonian<scalar_t>::create_scaled(SparseMatrixX<scalar_t> const& H,
                                                   Indices const& idx, Scale<scalar_t> scale) {
    original_idx = idx;
    optimized_idx = idx;

    if (H2.rows() != 0)
        return; // already optimal

    if (scale.b == 0) {
        // just scale, no b offset
        H2 = H * (2 / scale.a);
    } else {
        // scale and offset
        auto I = SparseMatrixX<scalar_t>{H.rows(), H.cols()};
        I.setIdentity();
        H2 = (H - I * scale.b) * (2 / scale.a);
    }

    H2.makeCompressed();
}

template<class scalar_t>
void OptimizedHamiltonian<scalar_t>::create_reordered(SparseMatrixX<scalar_t> const& H,
                                                      Indices const& idx, Scale<scalar_t> scale) {
    auto const system_size = H.rows();
    auto const inverted_a = real_t{2 / scale.a};

    // Find the maximum number of non-zero elements in the original matrix
    auto const max_nonzeros = [&]{
        auto max = 1;
        for (auto i = 0; i < H.outerSize(); ++i) {
            auto const nnz = H.outerIndexPtr()[i+1] - H.outerIndexPtr()[i];
            if (nnz > max) {
                max = nnz;
            }
        }
        return max;
    }();

    // Allocate the new matrix
    H2.resize(system_size, system_size);
    H2.reserve(VectorXi::Constant(system_size, max_nonzeros + 1)); // +1 for padding

    // Note: The following "queue" and "map" use vectors instead of other container types because
    //       they serve a very simple purpose. Using preallocated vectors results in better
    //       performance (this is not an assumption, it has been tested).

    // The index queue will contain the indices that need to be checked next
    auto index_queue = std::vector<int>{};
    index_queue.reserve(system_size);
    index_queue.push_back(idx.row); // starting from the given index

    // Map from original matrix indices to reordered matrix indices
    auto reorder_map = std::vector<int>(static_cast<std::size_t>(system_size), -1);
    // The point of the reordering is to have the target become index number 0
    reorder_map[idx.row] = 0;

    // As the reordered matrix is filled, the optimal size for the first few KPM steps is recorded
    optimized_sizes.clear();
    optimized_sizes.push_back(1);

    // Fill the reordered matrix row by row
    auto const H_view = sparse::make_loop(H);
    for (auto h2_row = 0; h2_row < system_size; ++h2_row) {
        auto diagonal_inserted = false;

        // Loop over elements in the row of the original matrix
        // corresponding to the h2_row of the reordered matrix
        auto const row = index_queue[h2_row];
        H_view.for_each_in_row(row, [&](int col, scalar_t value) {
            // A diagonal element may need to be inserted into the reordered matrix
            // even if the original matrix doesn't have an element on the main diagonal
            if (scale.b != 0 && !diagonal_inserted && col > row) {
                H2.insert(h2_row, h2_row) = -scale.b * inverted_a;
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

            H2.insert(h2_row, h2_col) = h2_value;
        });

        // Store the system size for the next KPM iteration
        if (h2_row == optimized_sizes.back() - 1) {
            optimized_sizes.push_back(static_cast<int>(index_queue.size()));
        }
    }

    optimized_sizes.pop_back(); // the last element is a duplicate of the second to last
    optimized_sizes.shrink_to_fit();
    H2.makeCompressed();

    original_idx = idx;
    optimized_idx = reorder_indices(original_idx, reorder_map);
    size_index_offset = compute_index_offset(optimized_idx, optimized_sizes);
}

template<class scalar_t>
Indices OptimizedHamiltonian<scalar_t>::reorder_indices(Indices const& original_idx,
                                                        std::vector<int> const& reorder_map) {
    auto const size = original_idx.cols.size();
    ArrayXi cols(size);
    for (auto i = 0; i < size; ++i) {
        cols[i] = reorder_map[original_idx.cols[i]];
    }
    // original_idx.row is always reordered to 0, that the whole purpose of the optimization
    return {0, cols};
}

template<class scalar_t>
int OptimizedHamiltonian<scalar_t>::compute_index_offset(Indices const& optimized_idx,
                                                         std::vector<int> const& optimized_sizes) {
    auto const max_index = *std::max_element(begin(optimized_idx.cols),
                                             end(optimized_idx.cols));
    auto const size = static_cast<int>(optimized_sizes.size());
    for (auto i = 0; i < size; ++i) {
        if (optimized_sizes[i] > max_index) {
            return i;
        }
    }
    return size - 1;
}

template<class scalar_t>
double OptimizedHamiltonian<scalar_t>::optimized_area(int num_moments) const {
    auto area = .0;
    for (auto n = 0; n < num_moments; ++n) {
        auto const max_row = optimized_size(n, num_moments);
        auto const num_nonzeros = H2.outerIndexPtr()[max_row];
        area += num_nonzeros;
    }

    return area;
}


template<class scalar_t>
KPM<scalar_t>::KPM(SparseMatrixRC<scalar_t> hamiltonian, Config const& config)
    : hamiltonian(std::move(hamiltonian)), config(config) {
    if (config.min_energy > config.max_energy) {
        throw std::invalid_argument("KPM: Invalid energy range specified (min > max).");
    }
    if (config.lambda <= 0) {
        throw std::invalid_argument("KPM: Lambda must be positive.");
    }
    if (config.min_energy != config.max_energy) {
        scale = {config.min_energy, config.max_energy};
    }
}

template<class scalar_t>
bool KPM<scalar_t>::change_hamiltonian(Hamiltonian const& h) {
    if (!ham::is<scalar_t>(h)) {
        return false;
    }

    hamiltonian = ham::get_shared_ptr<scalar_t>(h);
    optimized_hamiltonian = {};
    if (config.min_energy == config.max_energy) {
        scale = {}; // will be automatically computed
    } else {
        scale = {config.min_energy, config.max_energy}; // user-defined bounds
    }

    return true;
}

template<class scalar_t>
ArrayXcd KPM<scalar_t>::calc(int i, int j, ArrayXd const& energy, double broadening) {
    return std::move(calc_vector(i, {j}, energy, broadening).front());
}

template<class scalar_t>
std::vector<ArrayXcd> KPM<scalar_t>::calc_vector(int row, std::vector<int> const& cols,
                                                 ArrayXd const& energy, double broadening) {
    assert(!cols.empty());
    stats = {};
    auto timer = Chrono{};

    // Determine the scaling parameters of the Hamiltonian (fast)
    timer.tic();
    scale.compute(*hamiltonian, config.lanczos_precision);
    stats.lanczos(scale.bounds, timer.toc());

    auto const num_moments = [&] {
        auto const scaled_broadening = broadening / scale.a;
        return static_cast<int>(config.lambda / scaled_broadening) + 1;
    }();

    // Scale and optimize Hamiltonian (fast)
    timer.tic();
    optimized_hamiltonian.create(*hamiltonian, {row, cols}, scale,
                                 /*use_reordering*/config.optimization_level >= 1);
    stats.reordering(optimized_hamiltonian, num_moments, timer.toc());

    // Calculate KPM moments (slow)
    timer.tic();
    auto moments = [&] {
        if (config.optimization_level < 2) {
            return calc_moments(optimized_hamiltonian, num_moments);
        } else {
            return calc_moments2(optimized_hamiltonian, num_moments);
        }
    }();
    moments.apply_lorentz_kernel(config.lambda);
    stats.kpm(optimized_hamiltonian, num_moments, timer.toc());

    // Final Green's function (fast)
    timer.tic();
    auto const scaled_energy = (energy.template cast<real_t>() - scale.b) / scale.a;
    auto greens = moments.calc_greens(scaled_energy);
    stats.greens(timer.toc());

    return greens;
}

template<class scalar_t>
std::string KPM<scalar_t>::report(bool is_shortform) const {
    return stats.report(is_shortform);
}

template<class scalar_t>
MomentMatrix<scalar_t> KPM<scalar_t>::calc_moments(OptimizedHamiltonian<scalar_t> const& oh,
                                                   int num_moments) {
    auto const source_idx = oh.optimized_idx.row;
    VectorX<scalar_t> r0 = VectorX<scalar_t>::Zero(oh.H2.rows());
    r0[source_idx] = 1; // all zeros except for the source index
    
    // -> left[dest_idx] = 1; <- optimized out
    // This vector is constant (and simple) so we don't actually need it. Its dot product
    // with the r0 vector may be simplified -> see the last line of the loop below.

    // -> r1 = H * r0; <- optimized thanks to `r0[source_idx] = 1`
    // Note: H2.col(j) == H2.row(j).conjugate(), but the second is row-major friendly
    VectorX<scalar_t> r1 = oh.H2.row(source_idx).conjugate();
    r1 *= real_t{0.5}; // because H == 0.5*H2

    auto moments = MomentMatrix<scalar_t>(num_moments, oh.optimized_idx.cols);
    moments.collect_initial(r0, r1);

    for (auto n = 2; n < num_moments; ++n) {
        auto const optimized_size = oh.optimized_size(n, num_moments);

        // -> r0 = H2*r1 - r0; <- optimized compute kernel
        compute::kpm_kernel(0, optimized_size, oh.H2, r1, r0);

        // r1 gets the primary result of this iteration
        // r0 gets the value old value r1 (it will be needed in the next iteration)
        r1.swap(r0);

        // -> moments[n] = left.dot(r1); <- optimized thanks to constant `left[i] = 1`
        moments.collect(n, r1);
    }

    return moments;
}

template<class scalar_t>
MomentMatrix<scalar_t> KPM<scalar_t>::calc_moments2(OptimizedHamiltonian<scalar_t> const& oh,
                                                    int num_moments) {
    auto const source_idx = oh.optimized_idx.row;
    VectorX<scalar_t> r0 = VectorX<scalar_t>::Zero(oh.H2.rows());
    r0[source_idx] = 1;

    VectorX<scalar_t> r1 = oh.H2.row(source_idx).conjugate();
    r1 *= real_t{0.5};

    auto moments = MomentMatrix<scalar_t>(num_moments, oh.optimized_idx.cols);
    moments.collect_initial(r0, r1);

    // Interleave moments `n` and `n + 1` for better data locality
    for (auto n = 2; n < num_moments; n += 2) {
        auto p0 = 0;
        auto p1 = 0;

        auto const max_m = oh.optimized_size_index(n, num_moments);
        for (auto m = 0; m <= max_m; ++m) {
            auto const p2 = oh.optimized_sizes[m];
            compute::kpm_kernel(p1, p2, oh.H2, r1, r0);
            compute::kpm_kernel(p0, p1, oh.H2, r0, r1);

            p0 = p1;
            p1 = p2;
        }
        moments.collect(n, r0);

        if (n + 1 < num_moments) {
            auto const max_m2 = oh.optimized_size_index(n + 1, num_moments);
            compute::kpm_kernel(p0, oh.optimized_sizes[max_m2], oh.H2, r0, r1);
            moments.collect(n + 1, r1);
        }
    }

    return moments;
}

std::string Stats::report(bool shortform) const {
    if (shortform)
        return short_report + "|";
    else
        return long_report + "Total time:";
}

template<class real_t>
void Stats::lanczos(compute::LanczosBounds<real_t> const& bounds, Chrono const& time) {
    append(fmt::format("{:.2f}, {:.2f}, {}", bounds.min, bounds.max, bounds.loops),
           fmt::format("Spectrum bounds found ({:.2f}, {:.2f} eV) "
                       "using Lanczos procedure with {} loops",
                       bounds.min, bounds.max, bounds.loops),
           time);
}

template<class scalar_t>
void Stats::reordering(OptimizedHamiltonian<scalar_t> const& oh, int num_moments, Chrono const& time) {
    auto const nnz = static_cast<double>(oh.H2.nonZeros());
    auto const full_area = nnz * num_moments;
    auto const optimized_area = oh.optimized_area(num_moments);
    auto const removed_percent = 100 * (full_area - optimized_area) / full_area;

    bool const used_full_system = static_cast<int>(oh.optimized_sizes.size()) < num_moments / 2;
    auto const not_efficient = !used_full_system ? "*" : "";

    append(fmt::format("{:.0f}%{}", removed_percent, not_efficient),
           fmt::format("The reordering optimization was able to remove {:.0f}%{} of the workload",
                       removed_percent, not_efficient),
           time);
}

template<class scalar_t>
void Stats::kpm(OptimizedHamiltonian<scalar_t> const& oh, int num_moments, Chrono const& time) {
    auto const moments_with_suffix = fmt::with_suffix(num_moments);
    auto const operations_per_second = oh.optimized_area(num_moments) / time.elapsed_seconds();
    auto const ops_with_suffix = fmt::with_suffix(operations_per_second);

    append(fmt::format("{} @ {}ops", moments_with_suffix, ops_with_suffix),
           fmt::format("KPM calculated {} moments at {} operations per second",
                       moments_with_suffix, ops_with_suffix),
           time);
}

void Stats::greens(Chrono const& time) {
    auto str = std::string{"Green's function calculated"};
    long_report += fmt::format(long_line, str, time);
}

void Stats::append(std::string short_str, std::string long_str, Chrono const& time) {
    short_report += fmt::format(short_line, short_str, time);
    long_report += fmt::format(long_line, long_str, time);
}

TBM_INSTANTIATE_TEMPLATE_CLASS(KPM)
TBM_INSTANTIATE_TEMPLATE_CLASS(kpm::Scale)
TBM_INSTANTIATE_TEMPLATE_CLASS(kpm::OptimizedHamiltonian)
TBM_INSTANTIATE_TEMPLATE_CLASS(kpm::MomentMatrix)

} // namespace tbm
