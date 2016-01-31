#include "greens/KPM.hpp"

#include "compute/kernel_polynomial.hpp"
#include "support/format.hpp"
#include "support/physics.hpp"

using namespace tbm;
using namespace tbm::kpm;


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
void OptimizedHamiltonian<scalar_t>::create(SparseMatrixX<scalar_t> const& H,
                                            IndexPair idx, Scale<scalar_t> scale,
                                            bool use_reordering) {
    if (original_idx.i == idx.i && original_idx.j == idx.j)
        return; // already optimized for this idx

    if (use_reordering)
        create_reordered(H, idx, scale);
    else
        create_scaled(H, idx, scale);
}

template<class scalar_t>
void OptimizedHamiltonian<scalar_t>::create_scaled(SparseMatrixX<scalar_t> const& H,
                                                   IndexPair idx, Scale<scalar_t> scale) {
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
                                                      IndexPair idx, Scale<scalar_t> scale) {
    auto const target_index = idx.j; // this one will be relocated to position 0
    auto const system_size = H.rows();
    auto const inverted_a = real_t{2 / scale.a};

    // Find the maximum number of non-zero elements in the original matrix
    auto max_nonzeros = 1;
    for (int i = 0; i < H.outerSize(); ++i) {
        auto const nonzeros = H.outerIndexPtr()[i+1] - H.outerIndexPtr()[i];
        if (nonzeros > max_nonzeros)
            max_nonzeros = nonzeros;
    }

    // Allocate the new matrix
    H2.resize(system_size, system_size);
    H2.reserve(VectorXi::Constant(system_size, max_nonzeros + 1)); // +1 for padding

    // Note: The following "queue" and "map" use vectors instead of other container types because
    //       they serve a very simple purpose. Using preallocated vectors results in better
    //       performance (this is not an assumption, it has been tested).

    // The index queue will contain the indices that need to be checked next
    auto index_queue = std::vector<int>{};
    index_queue.reserve(system_size);
    index_queue.push_back(target_index); // starting from the given index

    // Map from original matrix indices to reordered matrix indices
    auto reorder_map = std::vector<int>(static_cast<std::size_t>(system_size), -1);
    // The point of the reordering is to have the target become index number 0
    reorder_map[target_index] = 0;

    // As the reordered matrix is filled, the optimal size for the first few KPM steps is recorded
    optimized_sizes.push_back(0);
    optimized_sizes.push_back(1);

    // Fill the reordered matrix row by row
    auto H_view = sparse::make_loop(H);
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
        if (h2_row == optimized_sizes.back() - 1)
            optimized_sizes.push_back(static_cast<int>(index_queue.size()));
    }

    optimized_sizes.pop_back(); // the last two elements are duplicates
    optimized_sizes.shrink_to_fit();
    H2.makeCompressed();

    // idx.j is always reordered so that reorder_map[idx.j] == 0
    optimized_idx = {reorder_map[idx.i], 0};
    original_idx = idx;
}

template<class scalar_t>
double OptimizedHamiltonian<scalar_t>::optimized_area(int num_moments) const {
    auto area = .0;
    for (auto n = 0; n < num_moments; ++n) {
        auto const max_row = get_optimized_size(n, num_moments);
        auto const num_nonzeros = H2.outerIndexPtr()[max_row];
        area += num_nonzeros;
    }

    return area;
}


template<class scalar_t>
KPM<scalar_t>::KPM(Config const& config) : config(config) {
    if (config.min_energy > config.max_energy)
        throw std::invalid_argument{"KPM: Invalid energy range specified (min > max)."};
    if (config.lambda <= 0)
        throw std::invalid_argument{"KPM: Lambda must be positive."};
}

template<class scalar_t>
void KPM<scalar_t>::hamiltonian_changed() {
    optimized_hamiltonian = {};

    if (config.min_energy == config.max_energy) {
        scale = {}; // will be automatically computed
    } else {
        scale = {config.min_energy, config.max_energy}; // user-defined bounds
    }
}

template<class scalar_t>
ArrayXcf KPM<scalar_t>::calculate(int i, int j, ArrayXf energy, float broadening) {
    stats = {};
    auto timer = Chrono{};

    // Determine the scaling parameters of the Hamiltonian (fast)
    timer.tic();
    scale.compute(hamiltonian->get_matrix(), config.lanczos_precision);
    stats.lanczos(scale.bounds, timer.toc());

    // Scale parameters
    energy = (energy - scale.b) / scale.a;
    broadening = broadening / scale.a;
    auto const num_moments = static_cast<int>(config.lambda / broadening) + 1;

    // Scale and optimize Hamiltonian (fast)
    timer.tic();
    optimized_hamiltonian.create(hamiltonian->get_matrix(), {i, j}, scale,
                                 /*use_reordering*/config.optimization_level >= 1);
    stats.reordering(optimized_hamiltonian, num_moments, timer.toc());

    // Calculate KPM moments (slow)
    timer.tic();
    auto moments = [&] {
        if (config.optimization_level < 2)
            return calculate_moments(optimized_hamiltonian, num_moments);
        else
            return calculate_moments2(optimized_hamiltonian, num_moments);
    }();
    apply_lorentz_kernel(moments, config.lambda);
    stats.kpm(optimized_hamiltonian, num_moments, timer.toc());

    // Final Green's function (fast)
    timer.tic();
    auto greens = calculate_greens(energy, moments);
    stats.greens(timer.toc());

    return greens;
}

template<class scalar_t>
std::string KPM<scalar_t>::report(bool is_shortform) const {
    return stats.report(is_shortform);
}

template<class scalar_t>
ArrayX<scalar_t> KPM<scalar_t>::calculate_moments(OptimizedHamiltonian<scalar_t> const& oh,
                                                  int num_moments) {
    auto const i = oh.optimized_idx.i;
    auto const j = oh.optimized_idx.j;

    VectorX<scalar_t> r0 = VectorX<scalar_t>::Zero(oh.H2.rows());
    r0[j] = 1; // all zeros except for position `j`
    
    // -> left[i] = 1; <- optimized out
    // This vector is constant (and simple) so we don't actually need it. Its dot product
    // with the r0 vector may be simplified -> see the last line of the loop below.

    // -> r1 = H * r0; <- optimized thanks to `r0[j] = 1`
    // Note: H2.col(j) == H2.row(j).conjugate(), but the second is row-major friendly
    VectorX<scalar_t> r1 = oh.H2.row(j).conjugate();
    r1 *= real_t{0.5}; // because H == 0.5*H2

    ArrayX<scalar_t> moments{num_moments};
    moments[0] = r0[i] * real_t{0.5}; // 0.5 is special for moments[0] (not related to H2)
    moments[1] = r1[i];

    for (auto n = 2; n < num_moments; ++n) {
        auto const optimized_size = oh.get_optimized_size(n, num_moments);

        // -> r0 = H2*r1 - r0; <- optimized compute kernel
        compute::kpm_kernel(0, optimized_size, oh.H2, r1, r0);

        // r1 gets the primary result of this iteration
        // r0 gets the value old value r1 (it will be needed in the next iteration)
        r1.swap(r0);

        // -> moments[n] = left.dot(r1); <- optimized thanks to constant `left[i] = 1`
        moments[n] = r1[i];
    }

    return moments;
}

template<class scalar_t>
ArrayX<scalar_t> KPM<scalar_t>::calculate_moments2(OptimizedHamiltonian<scalar_t> const& oh,
                                                   int num_moments) {
    auto const i = oh.optimized_idx.i;
    auto const j = oh.optimized_idx.j;

    VectorX<scalar_t> r0 = VectorX<scalar_t>::Zero(oh.H2.rows());
    r0[j] = 1;

    VectorX<scalar_t> r1 = oh.H2.row(j).conjugate();
    r1 *= real_t{0.5};

    ArrayX<scalar_t> moments{num_moments};
    moments[0] = r0[i] * real_t{0.5};
    moments[1] = r1[i];

    // Interleave moments `n` and `n + 1` for better data locality
    for (auto n = 2; n < num_moments; n += 2) {
        auto const max_m1 = oh.get_optimized_size_index(n, num_moments);
        auto const max_m2 = oh.get_optimized_size_index(n + 1, num_moments);

        auto p0 = 0;
        auto p1 = 0;
        for (auto m = 1; m <= max_m1; ++m) {
            auto const p2 = oh.optimized_sizes[m];
            compute::kpm_kernel(p1, p2, oh.H2, r1, r0);
            compute::kpm_kernel(p0, p1, oh.H2, r0, r1);

            p0 = p1;
            p1 = p2;
        }
        // Tail end in case `max_m2 >= max_m1`
        compute::kpm_kernel(p0, oh.optimized_sizes[max_m2], oh.H2, r0, r1);

        moments[n] = r0[i];
        if (n + 1 < num_moments)
            moments[n + 1] = r1[i];
    }

    return moments;
}

template<class scalar_t>
void KPM<scalar_t>::apply_lorentz_kernel(ArrayX<scalar_t>& moments, float lambda) {
    auto const N = moments.size();

    auto lorentz_kernel = [=](real_t n) { // n is real_t to get proper fp division
        using std::sinh;
        return sinh(lambda * (1 - n / N)) / sinh(lambda);
    };

    for (int n = 0; n < N; ++n)
        moments[n] *= lorentz_kernel(static_cast<real_t>(n));
}

template<class scalar_t>
auto KPM<scalar_t>::calculate_greens(ArrayX<real_t> const& energy, ArrayX<scalar_t> const& moments)
                                     -> ArrayX<complex_t> {
    // Note that this integer array has real type values
    ArrayX<real_t> ns{moments.size()};
    for (auto n = 0; n < ns.size(); ++n)
        ns[n] = static_cast<real_t>(n);

    ArrayX<complex_t> greens{energy.size()};

    // G = -2*i / sqrt(1 - E^2) * sum( moments * exp(-i*ns*acos(E)) )
    transform(energy, greens, [&](const real_t& E) {
        using physics::i1; using std::acos;
        return -real_t{2}*i1 / sqrt(1 - E*E) * sum( moments * exp(-i1 * ns * acos(E)) );
    });

    return greens;
}

std::string Stats::report(bool shortform) const {
    if (shortform)
        return short_report + "|";
    else
        return long_report + "Total time:";
}

template<class real_t>
void Stats::lanczos(compute::LanczosBounds<real_t> const& bounds, Chrono const& time) {
    append(fmt::format("{min_energy:.2f}, {max_energy:.2f}, {loops}",
                       bounds.min, bounds.max, bounds.loops),
           fmt::format("Spectrum bounds found ({min_energy:.2f}, {max_energy:.2f} eV) "
                       "using Lanczos procedure with {loops} loops",
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

    append(fmt::format("{removed_percent:.0f}%{not_efficient}",
                       removed_percent, not_efficient),
           fmt::format("The reordering optimization was able to remove "
                       "{removed_percent:.0f}%{not_efficient} of the workload",
                       removed_percent, not_efficient),
           time);
}

template<class scalar_t>
void Stats::kpm(OptimizedHamiltonian<scalar_t> const& oh, int num_moments, Chrono const& time) {
    auto const moments_with_suffix = fmt::with_suffix(num_moments);
    auto const operations_per_second = oh.optimized_area(num_moments) / time.seconds();
    auto const ops_with_suffix = fmt::with_suffix(operations_per_second);

    append(fmt::format("{num_moments} @ {ops}ops", moments_with_suffix, ops_with_suffix),
           fmt::format("KPM calculated {num_moments} moments at {ops} operations per second",
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


template class tbm::KPM<float>;
template class tbm::KPM<std::complex<float>>;
//template class tbm::KPM<double>;
//template class tbm::KPM<std::complex<double>>;
