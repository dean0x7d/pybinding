#include "greens/KPM.hpp"

#include "compute/kernel_polynomial.hpp"
#include "support/format.hpp"

namespace tbm { namespace kpm {

namespace {
    inline std::string make_report(std::string msg, Chrono const& time, bool shortform) {
        auto const fmt_str = shortform ? "{:s} [{}] " : "- {:<80s} | {}\n";
        return fmt::format(fmt_str, msg, time);
    }
}

template<class scalar_t>
void Bounds<scalar_t>::compute_factors() {
    timer.tic();
    bounds = compute::minmax_eigenvalues(*matrix, precision_percent);
    factors = {bounds.min, bounds.max};
    timer.toc();
}

template<class scalar_t>
std::string Bounds<scalar_t>::report(bool shortform) const {
    auto const fmt_str = shortform ? "{:.2f}, {:.2f}, {}"
                                   : "Spectrum bounds found ({:.2f}, {:.2f} eV) "
                                     "using Lanczos procedure with {} loops";
    auto const msg = fmt::format(fmt_str, bounds.min, bounds.max, bounds.loops);
    return make_report(msg, timer, shortform);
}

OptimizedSizes::OptimizedSizes(std::vector<int> sizes, Indices idx) : data(std::move(sizes)) {
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
    if (use_reordering) {
        create_reordered(idx, scale);
    } else {
        create_scaled(idx, scale);
    }
    timer.toc();

    original_idx = idx;
}

template<class scalar_t>
void OptimizedHamiltonian<scalar_t>::create_scaled(Indices const& idx, Scale<real_t> scale) {
    optimized_idx = idx;

    auto const& H = *original_matrix;
    if (scale.b == 0) { // just scale, no b offset
        optimized_matrix = H * (2 / scale.a);
    } else { // scale and offset
        auto I = SparseMatrixX<scalar_t>{H.rows(), H.cols()};
        I.setIdentity();
        optimized_matrix = (H - I * scale.b) * (2 / scale.a);
    }
    optimized_matrix.makeCompressed();
}

template<class scalar_t>
void OptimizedHamiltonian<scalar_t>::create_reordered(Indices const& idx, Scale<real_t> scale) {
    auto const& H = *original_matrix;
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
    optimized_matrix.resize(system_size, system_size);
    optimized_matrix.reserve(VectorXi::Constant(system_size, max_nonzeros + 1)); // +1 for padding

    // Note: The following "queue" and "map" use vectors instead of other container types because
    //       they serve a very simple purpose. Using preallocated vectors results in better
    //       performance (this is not an assumption, it has been tested).

    // The index queue will contain the indices that need to be checked next
    auto index_queue = std::vector<int>();
    index_queue.reserve(system_size);
    index_queue.push_back(idx.row); // starting from the given index

    // Map from original matrix indices to reordered matrix indices
    auto reorder_map = std::vector<int>(static_cast<std::size_t>(system_size), -1);
    // The point of the reordering is to have the target become index number 0
    reorder_map[idx.row] = 0;

    // As the reordered matrix is filled, the optimal size for the first few KPM steps is recorded
    auto sizes = std::vector<int>();
    sizes.push_back(1);

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
                optimized_matrix.insert(h2_row, h2_row) = -scale.b * inverted_a;
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

            optimized_matrix.insert(h2_row, h2_col) = h2_value;
        });

        // Store the system size for the next KPM iteration
        if (h2_row == sizes.back() - 1) {
            sizes.push_back(static_cast<int>(index_queue.size()));
        }
    }
    optimized_matrix.makeCompressed();

    sizes.pop_back(); // the last element is a duplicate of the second to last
    sizes.shrink_to_fit();

    optimized_idx = reorder_indices(idx, reorder_map);
    optimized_sizes = {std::move(sizes), optimized_idx};
}

template<class scalar_t>
Indices OptimizedHamiltonian<scalar_t>::reorder_indices(Indices const& original_idx,
                                                        std::vector<int> const& reorder_map) {
    auto const size = original_idx.cols.size();
    ArrayXi cols(size);
    for (auto i = 0; i < size; ++i) {
        cols[i] = reorder_map[original_idx.cols[i]];
    }
    // original_idx.row is always reordered to 0, that's the whole purpose of the optimization
    return {0, cols};
}

template<class scalar_t>
std::uint64_t OptimizedHamiltonian<scalar_t>::optimized_area(int num_moments) const {
    auto area = std::uint64_t{0};
    for (auto n = 0; n < num_moments; ++n) {
        auto const max_row = optimized_sizes.optimal(n, num_moments);
        auto const num_nonzeros = optimized_matrix.outerIndexPtr()[max_row];
        area += num_nonzeros;
    }
    return area;
}

template<class scalar_t>
std::string OptimizedHamiltonian<scalar_t>::report(int num_moments, bool shortform) const {
    auto const removed_percent = [&]{
        auto const nnz = static_cast<double>(optimized_matrix.nonZeros());
        auto const full_area = nnz * num_moments;
        return 100 * (full_area - optimized_area(num_moments)) / full_area;
    }();
    auto const not_efficient = optimized_sizes.uses_full_system(num_moments) ? "" : "*";
    auto const fmt_str = shortform ? "{:.0f}%{}"
                                   : "The reordering optimization was able to "
                                     "remove {:.0f}%{} of the workload";
    auto const msg = fmt::format(fmt_str, removed_percent, not_efficient);
    return make_report(msg, timer, shortform);
}

template<class scalar_t>
std::string Stats::report(Bounds<scalar_t> const& bounds, OptimizedHamiltonian<scalar_t> const& oh,
                          bool shortform) const {
    auto const moments_with_suffix = fmt::with_suffix(last_num_moments);
    auto const ops_with_suffix = [&]{
        auto const operations = oh.optimized_area(last_num_moments);
        auto const operations_per_second = operations / moments_timer.elapsed_seconds();
        return fmt::with_suffix(operations_per_second);
    }();

    auto const moments_report = [&]{
        auto const fmt_str = shortform ? "{} @ {}ops"
                                       : "KPM calculated {} moments at {} operations per second";
        auto const msg = fmt::format(fmt_str, moments_with_suffix, ops_with_suffix);
        return make_report(msg, moments_timer, shortform);
    }();

    return bounds.report(shortform) + oh.report(last_num_moments, shortform) + moments_report;
}

template<class scalar_t>
MomentsMatrix<scalar_t> calc_moments0(SparseMatrixX<scalar_t> const& h2,
                                     Indices const& idx, int num_moments) {
    auto moment_matrix = MomentsMatrix<scalar_t>(num_moments, idx.cols);
    auto r0 = make_r0(h2, idx.row);
    auto r1 = make_r1(h2, idx.row);
    moment_matrix.collect_initial(r0, r1);

    auto const size = h2.rows();
    for (auto n = 2; n < num_moments; ++n) {
        // -> r0 = h2 * r1 - r0; <- optimized compute kernel
        compute::kpm_kernel(0, size, h2, r1, r0);

        // r1 gets the primary result of this iteration
        // r0 gets the value old value r1 (it will be needed in the next iteration)
        r1.swap(r0);

        // -> moments[n] = left.dot(r1); <- optimized thanks to constant `left[i] = 1`
        moment_matrix.collect(n, r1);
    }

    return moment_matrix;
}

template<class scalar_t>
MomentsMatrix<scalar_t> calc_moments1(OptimizedHamiltonian<scalar_t> const& oh, int num_moments) {
    auto const& idx = oh.idx();
    auto const& h2 = oh.matrix();

    auto moment_matrix = MomentsMatrix<scalar_t>(num_moments, idx.cols);
    auto r0 = make_r0(h2, idx.row);
    auto r1 = make_r1(h2, idx.row);
    moment_matrix.collect_initial(r0, r1);

    for (auto n = 2; n < num_moments; ++n) {
        auto const optimized_size = oh.sizes().optimal(n, num_moments);
        compute::kpm_kernel(0, optimized_size, h2, r1, r0); // r0 = matrix * r1 - r0
        r1.swap(r0);
        moment_matrix.collect(n, r1);
    }

    return moment_matrix;
}

template<class scalar_t>
MomentsMatrix<scalar_t> calc_moments2(OptimizedHamiltonian<scalar_t> const& oh, int num_moments) {
    auto const& idx = oh.idx();
    auto const& h2 = oh.matrix();
    auto const& sizes = oh.sizes();

    auto moment_matrix = MomentsMatrix<scalar_t>(num_moments, idx.cols);
    auto r0 = make_r0(h2, idx.row);
    auto r1 = make_r1(h2, idx.row);
    moment_matrix.collect_initial(r0, r1);

    // Interleave moments `n` and `n + 1` for better data locality
    assert(num_moments % 2 == 0);
    for (auto n = 2; n < num_moments; n += 2) {
        auto p0 = 0;
        auto p1 = 0;

        auto const max_m = sizes.index(n, num_moments);
        for (auto m = 0; m <= max_m; ++m) {
            auto const p2 = sizes[m];
            compute::kpm_kernel(p1, p2, h2, r1, r0);
            compute::kpm_kernel(p0, p1, h2, r0, r1);

            p0 = p1;
            p1 = p2;
        }
        moment_matrix.collect(n, r0);

        auto const max_m2 = sizes.index(n + 1, num_moments);
        compute::kpm_kernel(p0, sizes[max_m2], h2, r0, r1);
        moment_matrix.collect(n + 1, r1);
    }

    return moment_matrix;
}

} // namespace kpm

namespace {
    template<class scalar_t>
    kpm::Bounds<scalar_t> reset_bounds(SparseMatrixX<scalar_t> const* hamiltonian,
                                       KPMConfig const& config) {
        if (config.min_energy == config.max_energy) {
            return {hamiltonian, config.lanczos_precision}; // will be automatically computed
        } else {
            return {config.min_energy, config.max_energy}; // user-defined bounds
        }
    }
}

template<class scalar_t>
KPM<scalar_t>::KPM(SparseMatrixRC<scalar_t> h, Config const& config)
    : hamiltonian(std::move(h)), config(config), bounds(reset_bounds(hamiltonian.get(), config)),
      optimized_hamiltonian(hamiltonian.get(), config.optimization_level) {
    if (config.min_energy > config.max_energy) {
        throw std::invalid_argument("KPM: Invalid energy range specified (min > max).");
    }
    if (config.lambda <= 0) {
        throw std::invalid_argument("KPM: Lambda must be positive.");
    }
}


template<class scalar_t>
bool KPM<scalar_t>::change_hamiltonian(Hamiltonian const& h) {
    if (!ham::is<scalar_t>(h)) {
        return false;
    }

    hamiltonian = ham::get_shared_ptr<scalar_t>(h);
    optimized_hamiltonian = {hamiltonian.get(), config.optimization_level};
    bounds = reset_bounds(hamiltonian.get(), config);

    return true;
}

template<class scalar_t>
ArrayXcd KPM<scalar_t>::calc(int row, int col, ArrayXd const& energy, double broadening) {
    return std::move(calc_vector(row, {col}, energy, broadening).front());
}

template<class scalar_t>
std::vector<ArrayXcd> KPM<scalar_t>::calc_vector(int row, std::vector<int> const& cols,
                                                 ArrayXd const& energy, double broadening) {
    assert(!cols.empty());
    auto const moment_matrix = calc_moments_matrix({row, cols}, broadening);
    auto const scale = bounds.scaling_factors();
    auto const scaled_energy = (energy.template cast<real_t>() - scale.b) / scale.a;
    return moment_matrix.calc_greens(scaled_energy);
}

template<class scalar_t>
std::string KPM<scalar_t>::report(bool shortform) const {
    return stats.report(bounds, optimized_hamiltonian, shortform)
           + (shortform ? "|" : "Total time:");
}

template<class scalar_t>
int KPM<scalar_t>::required_num_moments(double broadening, kpm::Scale<real_t> scale) {
    auto const scaled_broadening = broadening / scale.a;
    auto num_moments = static_cast<int>(config.lambda / scaled_broadening) + 1;
    // Moment calculations at higher optimization levels require specific rounding.
    // `num_moments - 2` considers only moments in the main KPM loop. Divisible by 4
    // because that is the strictest requirement imposed by `calc_diag_moments2()`.
    while ((num_moments - 2) % 4 != 0) {
        ++num_moments;
    }
    return num_moments;
}

template<class scalar_t>
kpm::MomentsMatrix<scalar_t> KPM<scalar_t>::calc_moments_matrix(kpm::Indices const& idx,
                                                                double broadening) {
    auto const scale = bounds.scaling_factors();
    optimized_hamiltonian.optimize_for(idx, scale);

    auto const num_moments = required_num_moments(broadening, scale);
    stats = {num_moments};
    stats.moments_timer.tic();
    auto moment_matrix = [&] {
        switch (config.optimization_level) {
            case 0: return kpm::calc_moments0(optimized_hamiltonian.matrix(), idx, num_moments);
            case 1: return kpm::calc_moments1(optimized_hamiltonian, num_moments);
            default: return kpm::calc_moments2(optimized_hamiltonian, num_moments);
        }
    }();
    moment_matrix.apply_lorentz_kernel(config.lambda);
    stats.moments_timer.toc();

    return moment_matrix;
}

TBM_INSTANTIATE_TEMPLATE_CLASS(KPM)
TBM_INSTANTIATE_TEMPLATE_CLASS(kpm::Bounds)
TBM_INSTANTIATE_TEMPLATE_CLASS(kpm::OptimizedHamiltonian)
TBM_INSTANTIATE_TEMPLATE_CLASS(kpm::MomentsMatrix)

} // namespace tbm
