#include "greens/KPM.hpp"
#include "hamiltonian/Hamiltonian.hpp"

#include "compute/lanczos.hpp"
#include "compute/kernel_polynomial.hpp"
#include "support/format.hpp"
#include "support/physics.hpp"

using namespace tbm;
using namespace tbm::kpm;


template<class scalar_t>
void Bounds<scalar_t>::compute(SparseMatrixX<scalar_t> const& H, real_t lanczos_tolerance) {
    if (a != 0)
        return; // already computed

    std::tie(min_energy, max_energy, lanczos_loops)
        = compute::minmax_eigenvalues(H, lanczos_tolerance);

    set(min_energy, max_energy);
}

template<class scalar_t>
void Bounds<scalar_t>::set(real_t min_, real_t max_) {
    min_energy = min_;
    max_energy = max_;

    a = 0.5f * (max_energy - min_energy) * (1 + scaling_tolerance);
    b = 0.5f * (max_energy + min_energy);

    // Round to zero if b is very small in order to make the the sparse matrix smaller
    if (std::abs(b / a) < 0.01f * scaling_tolerance)
        b = 0;
}


template<class scalar_t>
void OptimizedHamiltonian<scalar_t>::create(SparseMatrixX<scalar_t> const& H,
                                            IndexPair idx, Bounds<scalar_t> bounds,
                                            bool use_reordering) {
    if (original_idx.i == idx.i && original_idx.j == idx.j)
        return; // already optimized for this idx

    if (use_reordering)
        create_reordered(H, idx, bounds);
    else
        create_scaled(H, idx, bounds);
}

template<class scalar_t>
void OptimizedHamiltonian<scalar_t>::create_scaled(SparseMatrixX<scalar_t> const& H,
                                                   IndexPair idx, Bounds<scalar_t> bounds) {
    original_idx = idx;
    optimized_idx = idx;

    if (H2.rows() != 0)
        return; // already optimal

    if (bounds.b == 0) {
        // just scale, no b offset
        H2 = H * (2 / bounds.a);
    } else {
        // scale and offset
        auto I = SparseMatrixX<scalar_t>{H.rows(), H.cols()};
        I.setIdentity();
        H2 = (H - I * bounds.b) * (2 / bounds.a);
    }

    H2.makeCompressed();
}

template<class scalar_t>
void OptimizedHamiltonian<scalar_t>::create_reordered(SparseMatrixX<scalar_t> const& H,
                                                      IndexPair idx, Bounds<scalar_t> bounds) {
    auto const target_index = idx.j; // this one will be relocated to position 0
    auto const system_size = H.rows();
    auto const inverted_a = real_t{2 / bounds.a};

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
    auto reorder_map = std::vector<int>(system_size, -1);
    // The point of the reordering is to have the target become index number 0
    reorder_map[target_index] = 0;

    // As the reordered matrix is filled, the optimal size for the first few KPM steps is recorded
    optimized_sizes.push_back(1); // initial size is always 1

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
            if (bounds.b != 0 && !diagonal_inserted && col > row) {
                H2.insert(h2_row, h2_row) = -bounds.b * inverted_a;
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
                h2_value -= bounds.b * inverted_a;
                diagonal_inserted = true;
            }

            H2.insert(h2_row, h2_col) = h2_value;
        });

        // Store the system size for the next KPM iteration
        if (h2_row == optimized_sizes.back() - 1)
            optimized_sizes.push_back(static_cast<int>(index_queue.size()));
    }

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
ArrayXcf Strategy<scalar_t>::calculate(int i, int j, ArrayXf energy, float broadening) {
    stats = {};
    auto timer = Chrono{};

    // Determine the energy bounds of the Hamiltonian (fast)
    timer.tic();
    if (config.min_energy == config.max_energy) // automatically computed
        bounds.compute(hamiltonian->get_matrix(), config.lanczos_precision);
    else // manually set user-defined bounds
        bounds.set(config.min_energy, config.max_energy);
    stats.lanczos(bounds.min_energy, bounds.max_energy, bounds.lanczos_loops, timer.toc());

    // Scale parameters
    energy = (energy - bounds.b) / bounds.a;
    broadening = broadening / bounds.a;
    auto const num_moments = static_cast<int>(config.lambda / broadening) + 1;

    // Scale and optimize Hamiltonian (fast)
    timer.tic();
    optimized_hamiltonian.create(hamiltonian->get_matrix(), {i, j}, bounds, config.use_reordering);
    stats.reordering(optimized_hamiltonian, num_moments, timer.toc());

    // Calculate KPM moments (slow)
    timer.tic();
    auto moments = calculate_moments(optimized_hamiltonian, num_moments);
    apply_lorentz_kernel(moments, config.lambda);
    stats.kpm(optimized_hamiltonian, num_moments, timer.toc());

    // Final Green's function (fast)
    timer.tic();
    auto greens = calculate_greens(energy, moments);
    stats.greens(timer.toc());

    return greens;
}

template<class scalar_t>
ArrayX<scalar_t> Strategy<scalar_t>::calculate_moments(OptimizedHamiltonian<scalar_t> const& oh,
                                                       int num_moments) {
    auto const i = oh.optimized_idx.i;
    auto const j = oh.optimized_idx.j;

    VectorX<scalar_t> right = VectorX<scalar_t>::Zero(oh.H2.rows());
    right[j] = 1; // all zeros except for position `j`
    
    // -> left[i] = 1; <- optimized out
    // This vector is constant (and simple) so we don't actually need it. Its dot product
    // with the right vector may be simplified -> see the last line of the loop below.

    // -> right_next = H * right; <- optimized thanks to `right[j] = 1`
    // Note: H2.col(j) == H2.row(j).conjugate(), but the second is row-major friendly
    VectorX<scalar_t> right_next = oh.H2.row(j).conjugate();
    right_next *= real_t{0.5}; // because H == 0.5*H2

    ArrayX<scalar_t> moments{num_moments};
    moments[0] = right[i] * real_t{0.5}; // 0.5 is special for moments[0] (not related to H2)
    moments[1] = right_next[i];
    
    // Each iteration does: right_next = 2*H*right_next - right
    for (auto n = 2; n < num_moments; ++n) {
        auto const optimized_size = oh.get_optimized_size(n, num_moments);

        // -> right = H2*right_next - right; <- optimized compute kernel
        compute::kpm_kernel(optimized_size, oh.H2, right_next, right);

        // right_next gets the primary result of this iteration
        // right gets the value old value right_next (it will be needed in the next iteration)
        right_next.swap(right);

        // -> moments[n] = left.dot(right_next); <- optimized thanks to constant `left[i] = 1`
        moments[n] = right_next[i];
    }

    return moments;
}

template<class scalar_t>
void Strategy<scalar_t>::apply_lorentz_kernel(ArrayX<scalar_t>& moments, float lambda) {
    auto const N = moments.size();

    auto lorentz_kernel = [=](real_t n) { // n is real_t to get proper fp division
        using std::sinh;
        return sinh(lambda * (1 - n / N)) / sinh(lambda);
    };

    for (int n = 0; n < N; ++n)
        moments[n] *= lorentz_kernel(static_cast<real_t>(n));
}

template<class scalar_t>
auto Strategy<scalar_t>::calculate_greens(const ArrayX<real_t>& energy,
                                          const ArrayX<scalar_t>& moments)
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

template<class scalar_t>
void Strategy<scalar_t>::hamiltonian_changed() {
    bounds = {};
    optimized_hamiltonian = {};
}

template<class scalar_t>
std::string Strategy<scalar_t>::report(bool is_shortform) const {
    return stats.report(is_shortform);
}


std::string Stats::report(bool shortform) const {
    if (shortform)
        return short_report + "|";
    else
        return long_report + "Total time:";
}

void Stats::lanczos(double min_energy, double max_energy, int loops, Chrono const& time) {
    append(fmt::format("L: {min_energy:.2f}, {max_energy:.2f}, {loops}",
                       min_energy, max_energy, loops),
           fmt::format("Spectrum bounds found ({min_energy:.2f}, {max_energy:.2f} eV) "
                       "using Lanczos procedure with {loops} loops",
                       min_energy, max_energy, loops),
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

    append(fmt::format("R: {removed_percent:.0f}%{not_efficient}",
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

    append(fmt::format("K: {num_moments} @ {ops}ops", moments_with_suffix, ops_with_suffix),
           fmt::format("KPM calculated {num_moments} moments at {ops} operations per second",
                       moments_with_suffix, ops_with_suffix),
           time);
}

void Stats::greens(Chrono const& time) {
    append("G:", "Green's function calculated", time);
}

void Stats::append(std::string short_str, std::string long_str, Chrono const& time) {
    short_report += fmt::format(short_line, short_str, time);
    long_report += fmt::format(long_line, long_str, time);
}

namespace {
    template<class scalar_t>
    std::unique_ptr<GreensStrategy> try_create_for(std::shared_ptr<const Hamiltonian> const& ham,
                                                   kpm::Config const& config) {
        auto cast_ham = std::dynamic_pointer_cast<HamiltonianT<scalar_t> const>(ham);
        if (!cast_ham)
            return nullptr;

        auto kpm_strategy = cpp14::make_unique<kpm::Strategy<scalar_t>>(config);
        kpm_strategy->set_hamiltonian(cast_ham);

        return std::move(kpm_strategy);
    }
}

std::unique_ptr<GreensStrategy>
KPM::create_strategy_for(const std::shared_ptr<const Hamiltonian>& hamiltonian) const
{
    std::unique_ptr<GreensStrategy> new_greens;
    
    if (!new_greens) new_greens = try_create_for<float>(hamiltonian, config);
    if (!new_greens) new_greens = try_create_for<std::complex<float>>(hamiltonian, config);
//    if (!new_greens) new_greens = try_create_for<double>(hamiltonian, config);
//    if (!new_greens) new_greens = try_create_for<std::complex<double>>(hamiltonian, config);
    if (!new_greens)
        throw std::runtime_error{"KPM: unknown Hamiltonian type."};
    
    return new_greens;
}
