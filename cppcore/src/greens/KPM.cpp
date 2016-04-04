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
    if (opt_level <= 0) {
        create_scaled(idx, scale);
    } else if (opt_level <= 2) {
        create_reordered(idx, scale);
    } else {
        create_ellpack(idx, scale);
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
    auto reorder_map = std::vector<int>(static_cast<std::size_t>(system_size), -1);
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
void OptimizedHamiltonian<scalar_t>::create_ellpack(Indices const& idx, Scale<real_t> scale) {
    create_reordered(idx, scale);
    auto const& h2_csr = csr();
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
    optimized_matrix = std::move(h2_ell);
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

namespace {
    /// Return the number of non-zeros present up to `rows`
    struct NonZeros {
        int rows;

        template<class scalar_t>
        int operator()(SparseMatrixX<scalar_t> const& csr) {
            return csr.outerIndexPtr()[rows];
        }

        template<class scalar_t>
        int operator()(num::EllMatrix<scalar_t> const& ell) {
            return (rows - 1) * ell.nnz_per_row;
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
    return make_report(msg, timer, shortform);
}

template<class scalar_t>
std::string Stats::report(Bounds<scalar_t> const& bounds, OptimizedHamiltonian<scalar_t> const& oh,
                          bool shortform) const {
    auto const moments_with_suffix = fmt::with_suffix(last_num_moments);
    auto const ops_with_suffix = [&]{
        auto const operations = [&]{
            auto ops = oh.optimized_area(last_num_moments);
            if (oh.idx().is_diagonal()) {
                ops /= 2;
                for (auto n = 0; n <= last_num_moments / 2; ++n) {
                    ops += 2 * oh.sizes().optimal(n, last_num_moments);
                }
            }
            return ops;
        }();

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
ArrayX<scalar_t> calc_diag_moments0(SparseMatrixX<scalar_t> const& h2, int i, int num_moments) {
    auto moments = ArrayX<scalar_t>(num_moments);
    auto r0 = make_r0(h2, i);
    auto r1 = make_r1(h2, i);
    auto const m0 = moments[0] = r0[i] * scalar_t{0.5};
    auto const m1 = moments[1] = r1[i];

    assert(num_moments % 2 == 0);
    for (auto n = 2; n <= num_moments / 2; ++n) {
        compute::kpm_kernel(0, h2.rows(), h2, r1, r0);
        r1.swap(r0);
        moments[2 * (n - 1)] = scalar_t{2} * (r0.squaredNorm() - m0);
        moments[2 * (n - 1) + 1] = scalar_t{2} * r1.dot(r0) - m1;
    }

    return moments;
}

template<class Matrix, class scalar_t>
MomentsMatrix<scalar_t> calc_moments1(Matrix const& h2, Indices const& idx, int num_moments,
                                      OptimizedSizes const& sizes) {
    auto moment_matrix = MomentsMatrix<scalar_t>(num_moments, idx.cols);
    auto r0 = make_r0(h2, idx.row);
    auto r1 = make_r1(h2, idx.row);
    moment_matrix.collect_initial(r0, r1);

    for (auto n = 2; n < num_moments; ++n) {
        auto const optimized_size = sizes.optimal(n, num_moments);
        compute::kpm_kernel(0, optimized_size, h2, r1, r0); // r0 = matrix * r1 - r0
        r1.swap(r0);
        moment_matrix.collect(n, r1);
    }

    return moment_matrix;
}

template<class Matrix, class scalar_t>
ArrayX<scalar_t> calc_diag_moments1(Matrix const& h2, int i, int num_moments,
                                    OptimizedSizes const& sizes) {
    auto moments = ArrayX<scalar_t>(num_moments);
    auto r0 = make_r0(h2, i);
    auto r1 = make_r1(h2, i);
    auto const m0 = moments[0] = r0[i] * scalar_t{0.5};
    auto const m1 = moments[1] = r1[i];

    assert(num_moments % 2 == 0);
    for (auto n = 2; n <= num_moments / 2; ++n) {
        auto const opt_size = sizes.optimal(n, num_moments);
        compute::kpm_kernel(0, opt_size, h2, r1, r0);
        r1.swap(r0);
        moments[2 * (n - 1)] = scalar_t{2} * (r0.head(opt_size).squaredNorm() - m0);
        moments[2 * (n - 1) + 1] = scalar_t{2} * r1.head(opt_size).dot(r0.head(opt_size)) - m1;
    }

    return moments;
}

template<class Matrix, class scalar_t>
MomentsMatrix<scalar_t> calc_moments2(Matrix const& h2, Indices const& idx, int num_moments,
                                      OptimizedSizes const& sizes) {
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

template<class Matrix, class scalar_t>
ArrayX<scalar_t> calc_diag_moments2(Matrix const& h2, int i, int num_moments,
                                    OptimizedSizes const& sizes) {
    auto moments = ArrayX<scalar_t>(num_moments);
    auto r0 = make_r0(h2, i);
    auto r1 = make_r1(h2, i);
    auto const m0 = moments[0] = r0[i] * scalar_t{0.5};
    auto const m1 = moments[1] = r1[i];

    assert((num_moments - 2) % 4 == 0);
    for (auto n = 2; n <= num_moments / 2; n += 2) {
        auto m2 = scalar_t{0};
        auto m3 = scalar_t{0};
        auto m4 = scalar_t{0};
        auto m5 = scalar_t{0};

        auto const max1 = sizes.index(n, num_moments);
        auto const max2 = sizes.index(n + 1, num_moments);

        for (auto k = 0, p0 = 0, p1 = 0; k <= max1; ++k) {
            auto const p2 = sizes[k];
            auto const d21 = p2 - p1;
            auto const d10 = p1 - p0;

            compute::kpm_kernel(p1, p2, h2, r1, r0);
            m2 += r1.segment(p1, d21).squaredNorm();
            m3 += r0.segment(p1, d21).dot(r1.segment(p1, d21));
            m4 += r0.segment(p1, d21).squaredNorm();

            compute::kpm_kernel(p0, p1, h2, r0, r1);
            m5 += r1.segment(p0, d10).dot(r0.segment(p0, d10));

            p0 = p1;
            p1 = p2;
        }
        moments[2 * (n - 1)] = scalar_t{2} * (m2 - m0);
        moments[2 * (n - 1) + 1] = scalar_t{2} * m3 - m1;

        auto const p0 = sizes[max1 - 1];
        auto const p1 = sizes[max2];
        auto const d10 = p1 - p0;

        compute::kpm_kernel(p0, p1, h2, r0, r1);
        m5 += r1.segment(p0, d10).dot(r0.segment(p0, d10));

        moments[2 * n] = scalar_t{2} * (m4 - m0);
        moments[2 * n + 1] = scalar_t{2} * m5 - m1;
    }

    return moments;
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
    if (row == col) {
        auto const moments = calc_moments_diag(row, broadening);
        auto const scaled_energy = bounds.scale_energy(energy.template cast<real_t>());
        auto const greens = kpm::detail::calculate_greens(scaled_energy, moments);
        return greens.template cast<std::complex<double>>();
    } else {
        return std::move(calc_vector(row, {col}, energy, broadening).front());
    }
}

template<class scalar_t>
std::vector<ArrayXcd> KPM<scalar_t>::calc_vector(int row, std::vector<int> const& cols,
                                                 ArrayXd const& energy, double broadening) {
    assert(!cols.empty());
    auto const moment_matrix = calc_moments_matrix({row, cols}, broadening);
    auto const scaled_energy = bounds.scale_energy(energy.template cast<real_t>());
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
        auto const& oh = optimized_hamiltonian;
        switch (config.optimization_level) {
            case 0: return kpm::calc_moments0(oh.csr(), idx, num_moments);
            case 1: return kpm::calc_moments1(oh.csr(), oh.idx(), num_moments, oh.sizes());
            case 2: return kpm::calc_moments2(oh.csr(), oh.idx(), num_moments, oh.sizes());
            default: return kpm::calc_moments2(oh.ell(), oh.idx(), num_moments, oh.sizes());
        }
    }();
    moment_matrix.apply_lorentz_kernel(config.lambda);
    stats.moments_timer.toc();

    return moment_matrix;
}

template<class scalar_t>
ArrayX<scalar_t> KPM<scalar_t>::calc_moments_diag(int i, double broadening) {
    auto const scale = bounds.scaling_factors();
    optimized_hamiltonian.optimize_for({i, i}, scale);

    auto const num_moments = required_num_moments(broadening, scale);
    stats = {num_moments};
    stats.moments_timer.tic();
    auto moments = [&] {
        auto const& oh = optimized_hamiltonian;
        auto const opt_i = oh.idx().row;
        switch (config.optimization_level) {
            case 0: return kpm::calc_diag_moments0(oh.csr(), i, num_moments);
            case 1: return kpm::calc_diag_moments1(oh.csr(), opt_i, num_moments, oh.sizes());
            case 2: return kpm::calc_diag_moments2(oh.csr(), opt_i, num_moments, oh.sizes());
            default: return kpm::calc_diag_moments2(oh.ell(), opt_i, num_moments, oh.sizes());
        }
    }();
    kpm::detail::apply_lorentz_kernel(moments, config.lambda);
    stats.moments_timer.toc();

    return moments;
}

TBM_INSTANTIATE_TEMPLATE_CLASS(KPM)
TBM_INSTANTIATE_TEMPLATE_CLASS(kpm::Bounds)
TBM_INSTANTIATE_TEMPLATE_CLASS(kpm::OptimizedHamiltonian)
TBM_INSTANTIATE_TEMPLATE_CLASS(kpm::MomentsMatrix)

} // namespace tbm
