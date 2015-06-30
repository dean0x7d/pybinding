#include "greens/KPM.hpp"
#include "Model.hpp"
#include "hamiltonian/Hamiltonian.hpp"
using namespace tbm;

#include "compute/lanczos.hpp"
#include "compute/kernel_polynomial.hpp"
#include "support/format.hpp"
#include "support/physics.hpp"


template<typename scalar_t>
auto KPMStrategy<scalar_t>::scaling_params() const -> std::tuple<real_t, real_t>
{
    real_t a = (config.max_energy - config.min_energy) / 2;
    real_t b = (config.max_energy + config.min_energy) / 2;
    a *= 1 + config.scaling_tolerance;

    // Round to zero if b is very small in order to make the the sparse matrix smaller
    if (std::abs(b / a) < 0.01 * config.scaling_tolerance)
        b = 0;

    return std::make_tuple(a, b);
}

template<typename scalar_t>
void KPMStrategy<scalar_t>::scale_hamiltonian()
{
    const auto& h_matrix = hamiltonian->get_matrix();
    real_t a, b; std::tie(a, b) = scaling_params();

    // Scale the Hamiltonian matrix [(H - b)/a] and save as a member variable.
    // The matrix is also multiplied by 2 at this point -> this is to improve performance.
    // We'll have a lot of "x = 2*H*x - y" operations, so it's beneficial to multiply only once,
    // but note that we'll need to divide by 2 when we need the original value (very rarely).

    if (b == 0) { // just scale, no b offset
        h2_matrix = h_matrix * (2 / a);
    }
    else { // scale and offset
        SparseMatrix I{h_matrix.rows(), h_matrix.cols()};
        I.setIdentity();
        h2_matrix = (h_matrix - I*b) * (2 / a);
    }
    h2_matrix.makeCompressed();
}

template<typename scalar_t>
int KPMStrategy<scalar_t>::scale_and_reorder_hamiltonian(int target_index, int translate_index)
{
    // The original *unscaled* Hamiltonian
    const auto& h_matrix = hamiltonian->get_matrix();
    const auto system_size = h_matrix.rows();

    // We're reordering the original Hamiltonian so we'll also need to scale it for KPM
    real_t a, b; std::tie(a, b) = scaling_params();
    const auto inverted_a = real_t{2 / a};

    // Find the maximum number of non-zero elements in the original matrix
    auto max_nonzeros = 1;
    for (int i = 0; i < h_matrix.outerSize(); ++i) {
        const auto nonzeros = h_matrix.outerIndexPtr()[i+1] - h_matrix.outerIndexPtr()[i];
        if (nonzeros > max_nonzeros)
            max_nonzeros = nonzeros;
    }
    
    // Allocate the new matrix
    h2_matrix.resize(system_size, system_size);
    h2_matrix.reserve(VectorXi::Constant(system_size, max_nonzeros + 1)); // +1 for padding
    
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
    reordered_steps.clear();
    reordered_steps.push_back(1); // ...every journey starts with a single step

    // Fill the reordered matrix row by row
    for (int h2_row = 0; h2_row < system_size; ++h2_row) {
        auto diagonal_inserted = false;

        // Loop over elements in the row of the original matrix
        // corresponding to the h2_row of the reordered matrix
        for (auto it = sparse_row(h_matrix, index_queue[h2_row]); it; ++it) {
            // A diagonal element may need to be inserted into the reordered matrix
            // even if the original matrix doesn't have an element on the main diagonal
            if (b != 0 && !diagonal_inserted && it.col() > it.row()) {
                h2_matrix.insert(h2_row, h2_row) = -b * inverted_a;
                diagonal_inserted = true;
            }

            // This may be a new index, map it
            if (reorder_map[it.col()] == -1) {
                reorder_map[it.col()] = index_queue.size();
                index_queue.push_back(it.col());
            }

            // Get the reordered column index
            const auto h2_col = reorder_map[it.col()];

            // Calculate the new value that will be inserted into the scaled/reordered matrix
            auto h2_value = it.value() * inverted_a;
            if (it.row() == it.col()) { // add offset to diagonal elements
                h2_value -= b* inverted_a;
                diagonal_inserted = true;
            }
            
            h2_matrix.insert(h2_row, h2_col) = h2_value;
        } // end row loop

        // Store the system size for the next KPM iteration
        if (h2_row == reordered_steps.back() - 1)
            reordered_steps.push_back(index_queue.size());
    } // end main loop

    h2_matrix.makeCompressed();
    return reorder_map[translate_index];
};

template<typename scalar_t>
ArrayXcf KPMStrategy<scalar_t>::calculate(int i, int j, ArrayXd energy_, float broadening) {
    ArrayX<real_t> energy = energy_.cast<real_t>();

    if (config.min_energy == config.max_energy) {
        lanczos_timer.tic();
        // Determine energy bounds with a quick Lanczos procedure
        std::tie(config.min_energy, config.max_energy, lanczos_loops) =
            compute::minmax_eigenvalues(hamiltonian->get_matrix(), config.lanczos_precision);
        lanczos_timer.toc();
    }

    // Scaling the Hamiltonian between -1 and 1 is required by KPM.
    // The matrix reordering is an optional optimization.
    if (config.use_reordering) {
        reordering_timer.tic();
        // The reordered value of index i is returned by the function
        i = scale_and_reorder_hamiltonian(j, i);
        // ...and j becomes 0 which is the whole point of the reordering
        j = 0;
        reordering_timer.toc();
    }
    else if (h2_matrix.rows() == 0) {
        scale_hamiltonian();
    }

    // Scale the parameters to match the Hamiltonian
    real_t a, b; std::tie(a, b) = scaling_params();
    energy = (energy - b) / a;
    broadening = broadening / a; // just scale, no offset for broadening

    num_moments = static_cast<int>(std::ceil(config.lambda / broadening));
    num_moments = (num_moments > 0) ? num_moments : 2;

    moments_timer.tic();
    // The most compute intensive part
    ArrayX<scalar_t> moments = calculate_moments(i, j);
    moments_timer.toc();

    greens_timer.tic();
    // Very fast calculation
    ArrayX<complex_t> greens = calculate_greens(energy, moments);
    greens_timer.toc();

    return greens;
}

template<typename scalar_t>
ArrayX<scalar_t> KPMStrategy<scalar_t>::calculate_moments(int i, int j) const
{
    auto system_size = h2_matrix.rows();
    const auto reordered_steps_size = static_cast<int>(reordered_steps.size());
    
    // Allocate right vector to zeros
    VectorX<scalar_t> right = VectorX<scalar_t>::Zero(system_size);
    // ...and set only position *j* to 1
    right[j] = 1;
    
    // -> left[i] = 1;
    // The left vector is set to 1 only at *i*, but because this vector is constant (and simple),
    // we don't actually need it. Its dot product with the right vector may be simplified:
    // from: moments[n] = left.dot(right)
    // to:   moments[n] = right[i]

    // -> right_next = H * right
    // Simplified because only right[j] = 1
    // Also: h2_matrix.col(j) == h2_matrix.row(j).conjugate(), but the second is row-major friendly
    VectorX<scalar_t> right_next = h2_matrix.row(j).conjugate();
    right_next *= real_t{0.5}; // because: H == 0.5 * h2_matrix

    // The first two moments are easy
    ArrayX<scalar_t> moments{num_moments};
    moments[0] = right[i] * real_t{0.5}; // 0.5 is special for moments[0] (not related to h2_matrix)
    moments[1] = right_next[i];
    
    // Calculate moments[n >= 2], each iteration does: right_next = 2*H*right_next - right
    for (int n = 2; n < num_moments; ++n) {
        // We may use the optimized (smaller) system size
        if (reordered_steps_size > 0) {
            const auto reverse_n = num_moments - n; // the last reverse_n will be 1 (intentional)

            // For the initial iterations, system_size grows from a small number to h2_matrix.rows()
            if (n < reverse_n && n < reordered_steps_size)
                system_size = reordered_steps[n];
            // For the final iterations, it shrinks from h2_matrix.rows() to a small number
            else if (reverse_n < reordered_steps_size)
                system_size = reordered_steps[reverse_n];
            // For everything in between: system_size == h2_matrix.rows()
        }

        // -> right = h2_matrix*right_next - right  (note that h2_matrix == 2*H)
        compute::kpm_kernel(system_size, h2_matrix, right_next, right);

        // right_next gets the primary result of this iteration
        right_next.swap(right);
        // right gets the value old value right_next (it will be needed in the next iteration)

        // -> moments[n] = left.dot(right_next)
        // Simplified because only left[i] = 1
        moments[n] = right_next[i];
    }

    // The Lorentz kernel as a function of n (n is declared real_t to get proper fp division)
    auto lorentz_kernel = [=](real_t n) {
        using std::sinh;
        return sinh(config.lambda * (1 - n / num_moments)) / sinh(config.lambda);
    };

    for (int n = 0; n < num_moments; ++n)
        moments[n] *= lorentz_kernel(n);

    return moments;
};

template<typename scalar_t>
auto KPMStrategy<scalar_t>::calculate_greens(const ArrayX<real_t>& energy,
                                     const ArrayX<scalar_t>& moments)
-> ArrayX<complex_t>
{
    // Note that this integer array has real type values
    ArrayX<real_t> ns{moments.size()};
    for (int n = 0; n < ns.size(); ++n)
        ns[n] = n;

    // Green's is a function of energy
    ArrayX<complex_t> greens{energy.size()};
    // G = -2*i / sqrt(1 - E^2) * sum( moments * exp(-i*ns*acos(E)) )
    transform(energy, greens, [&](const real_t& E) {
        using physics::i1; using std::acos;
        return -real_t{2}*i1 / sqrt(1 - E*E) * sum( moments * exp(-i1 * ns * acos(E)) );
    });

    return greens;
}

template<typename scalar_t>
void KPMStrategy<scalar_t>::hamiltonian_changed()
{
    h2_matrix.resize(0, 0);
}

template<typename scalar_t>
std::string KPMStrategy<scalar_t>::report(bool is_shortform) const
{
    std::string report;
    std::string line_fmt, lanczos_fmt, reorder_fmt, kpm_fmt, greens_fmt;

    if (is_shortform) {
        line_fmt = "{message:s} ({time}) ";

        lanczos_fmt = "L: {min_energy:.2f}, {max_energy:.2f}, {loops}";
        reorder_fmt = "R: {reordered_steps}, {removed_percent:.0f}%";
        kpm_fmt = "K: {num_moments}";
        greens_fmt = "G:";
    }
    else {
        line_fmt = "- {message:-80s} | {time}\n";

        lanczos_fmt = "Spectrum bounds found ({min_energy:.2f}, {max_energy:.2f} eV) "
            "using Lanczos procedure with {loops} loops";
        reorder_fmt = "Reordering optimization applied to {reordered_steps} moments "
            "and removed {removed_percent:.0f}% of the workload";
        kpm_fmt = "Kernel Polynomial Method calculated {num_moments} moments";
        greens_fmt = "Green's function calculated";
    }

    using fmt::format;
    auto append_report = [&] (std::string message, Chrono time) {
        report += format(line_fmt, message, time);
    };

    if (lanczos_loops > 0) {
        append_report(
            format(lanczos_fmt, config.min_energy, config.max_energy, lanczos_loops),
            lanczos_timer
        );
    }

    if (config.use_reordering) {
        const int reordered_steps_size = reordered_steps.size();
        bool used_full_system = reordered_steps_size < num_moments / 2;
        int limit = !used_full_system ? num_moments / 2 : reordered_steps_size;

        double removed_steps = 0;
        for (int i = 5; i < limit; i++)
            removed_steps += h2_matrix.rows() - reordered_steps[i];
        removed_steps *= 2; // steps are removed at the start *and* end

        // Percent of total steps
        removed_steps /= h2_matrix.rows() * static_cast<double>(num_moments);
        removed_steps *= 100;

        std::string reordered_steps_str = fmt::with_suffix(reordered_steps_size - 3);
        if (!used_full_system)
            reordered_steps_str += '*';

        append_report(format(reorder_fmt, reordered_steps_str, removed_steps), reordering_timer);
    }

    append_report(format(kpm_fmt, fmt::with_suffix(num_moments)), moments_timer);
    append_report(greens_fmt, greens_timer);

    report += is_shortform ? "|" : "Total time:";
    return report;
}


namespace {
    template<class scalar_t>
    std::unique_ptr<GreensStrategy> try_create_for(const std::shared_ptr<const Hamiltonian>& ham,
                                                   const KPMConfig& config) {
        auto cast_ham = std::dynamic_pointer_cast<const HamiltonianT<scalar_t>>(ham);
        if (!cast_ham)
            return nullptr;

        auto kpm = cpp14::make_unique<KPMStrategy<scalar_t>>(config);
        kpm->set_hamiltonian(cast_ham);

        return std::move(kpm);
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

ArrayXcf KPM::calc_greens(int i, int j, ArrayXd energy, float broadening) {
    auto size = model->hamiltonian()->rows();
    if (i < 0 || i > size || j < 0 || j > size)
        throw std::logic_error{"KPM::calc_greens(i,j): invalid value for i or j."};

    // time the calculation
    calculation_timer.tic();
    auto greens_function = strategy->calculate(i, j, energy, broadening);
    calculation_timer.toc();

    return greens_function;
}

ArrayXf KPM::calc_ldos(ArrayXd energy, float broadening, Cartesian position, sub_id sublattice) {
    auto i = model->system()->find_nearest(position, sublattice);
    auto greens_function = calc_greens(i, i, energy, broadening);

    using physics::pi;
    return -1/pi * greens_function.imag();
}
