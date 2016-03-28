#pragma once
#include "greens/Greens.hpp"
#include "numeric/sparse.hpp"
#include "compute/lanczos.hpp"
#include "detail/macros.hpp"

namespace tbm { namespace kpm {

/**
 The KPM scaling factors `a` and `b`
*/
template<class real_t>
struct Scale {
    static constexpr auto tolerance = 0.01f; ///< needed because the energy bounds are not precise

    real_t a = 0;
    real_t b = 0;

    Scale() = default;
    Scale(real_t min_energy, real_t max_energy)
        : a(0.5f * (max_energy - min_energy) * (1 + tolerance)),
          b(0.5f * (max_energy + min_energy)) {
        if (std::abs(b / a) < 0.01f * tolerance) {
            b = 0; // rounding to zero saves space in the sparse matrix
        }
    }

    explicit operator bool() { return a != 0; }
};

/**
 Min and max eigenvalues of the Hamiltonian

 The bounds can be determined automatically using the Lanczos procedure,
 or set manually by the user. Also computes the KPM scaling factors a and b.
*/
template<class scalar_t>
class Bounds {
    using real_t = num::get_real_t<scalar_t>;

    Scale<real_t> factors;
    compute::LanczosBounds<real_t> bounds = {0, 0, 0};

    SparseMatrixX<scalar_t> const* matrix;
    real_t precision_percent;
    Chrono timer;

public:
    Bounds(SparseMatrixX<scalar_t> const* matrix, real_t precision_percent)
        : matrix(matrix), precision_percent(precision_percent) {}
    /// Set the energy bounds manually, therefore skipping the Lanczos computation
    Bounds(real_t min_energy, real_t max_energy)
        : factors(min_energy, max_energy), bounds{min_energy, max_energy, 0} {}

    /// The KPM scaling factors a and b
    Scale<real_t> scaling_factors() {
        if (!factors) {
            compute_factors();
        }
        return factors;
    }

    std::string report(bool shortform = false) const;

private:
    /// Compute the scaling factors using the Lanczos procedure
    void compute_factors();
};

/**
 Indices of the Green's function matrix that will be computed

 A single KPM calculation will compute an entire `row` of the Green's matrix,
 however only some column indices are required to be saved, as indicated by `cols`.
 */
struct Indices {
    int row = -1;
    ArrayXi cols;

    Indices() = default;
    Indices(int row, int col) : row(row), cols(1) { cols[0] = col; }
    Indices(int row, ArrayXi const& cols) : row(row), cols(cols) {}
    Indices(int row, std::vector<int> const& cols) : row(row), cols(eigen_cast<ArrayX>(cols)) {}

    friend bool operator==(Indices const& l, Indices const& r) {
        return l.row == r.row && all_of(l.cols == r.cols);
    }
};

/**
 Optimal matrix sizes needed for KPM moment calculation. See `OptimizedHamiltonian`.
 */
class OptimizedSizes {
    std::vector<int> data; ///< optimal matrix sizes for the first few KPM iterations
    int offset = 0; ///< needed to correctly compute off-diagonal elements (i != j)

public:
    OptimizedSizes(int system_size) : data{system_size} {}
    OptimizedSizes(std::vector<int> sizes, Indices idx);

    /// Return an index into `data`, indicating the optimal system size for
    /// the calculation of KPM moment number `n` out of total `num_moments`
    int index(int n, int num_moments) const {
        assert(n < num_moments);

        auto const max_index = std::min(
            static_cast<int>(data.size()) - 1,
            num_moments / 2
        );

        if (n < max_index) {
            return n; // size grows in the beginning
        } else { // constant in the middle and shrinking near the end as reverse `n`
            return std::min(max_index, num_moments - 1 - n + offset);
        }
    }

    /// Return the optimal system size for KPM moment number `n` out of total `num_moments`
    int optimal(int n, int num_moments) const {
        return data[index(n, num_moments)];
    }

    /// Would calculating this number of moments ever do a full matrix-vector multiplication?
    bool uses_full_system(int num_moments) const {
        return static_cast<int>(data.size()) < num_moments / 2;
    }

    int operator[](int i) const { return data[i]; }

    std::vector<int> const& get_data() const { return data; }
    int get_offset() const { return offset; }
};

/**
 Stores a scaled Hamiltonian `(H - b)/a` which limits it to (-1, 1) boundaries required for KPM.
 In addition, two optimisations are applied:

 1) The matrix is multiplied by 2. This benefits most calculations (e.g. `y = 2*H*x - y`),
    because the 2x multiplication is done only once, but it will need to be divided by 2
    when the original element values are needed (very rarely).

 2) Reorder the elements so that target indices are placed at the start of the matrix.
    This produces the `optimized_sizes` vector which may be used to reduce calculation
    time by skipping sparse matrix-vector multiplication of zero values.
 */
template<class scalar_t>
class OptimizedHamiltonian {
    using real_t = num::get_real_t<scalar_t>;

    SparseMatrixX<scalar_t> optimized_matrix; ///< reordered for faster compute
    Indices optimized_idx; ///< reordered target indices in the optimized matrix
    OptimizedSizes optimized_sizes; ///< optimal matrix sizes for each KPM iteration

    SparseMatrixX<scalar_t> const* original_matrix;
    Indices original_idx; ///< original target indices for which the optimization was done

    bool use_reordering;
    Chrono timer;

public:
    OptimizedHamiltonian(SparseMatrixX<scalar_t> const* m, int opt_level = 1)
        : optimized_sizes(m->rows()), original_matrix(m), use_reordering(opt_level >= 1) {}

    /// Create the optimized Hamiltonian targeting specific indices and scale factors
    void optimize_for(Indices const& idx, Scale<real_t> scale);

    Indices const& idx() const { return optimized_idx; }
    OptimizedSizes const& sizes() const { return optimized_sizes; }
    SparseMatrixX<scalar_t> const& matrix() const { return optimized_matrix; }

    /// The unoptimized compute area is matrix.nonZeros() * num_moments
    std::uint64_t optimized_area(int num_moments) const;

    std::string report(int num_moments, bool shortform = false) const;

private:
    /// Just scale the Hamiltonian: H2 = (H - I*b) * (2/a)
    void create_scaled(Indices const& idx, Scale<real_t> scale);
    /// Scale and reorder the Hamiltonian so that idx is at the start of the optimized matrix
    void create_reordered(Indices const& idx, Scale<real_t> scale);
    /// Get optimized indices which map to the given originals
    static Indices reorder_indices(Indices const& original_idx,
                                   std::vector<int> const& reorder_map);
};

namespace detail {
    /// Put the kernel in *Kernel* polynomial method
    template<class scalar_t, class real_t>
    void apply_lorentz_kernel(ArrayX<scalar_t>& moments, real_t lambda) {
        auto const N = moments.size();

        auto lorentz_kernel = [=](real_t n) { // n is real_t to get proper fp division
            using std::sinh;
            return sinh(lambda * (1 - n / N)) / sinh(lambda);
        };

        for (auto n = 0; n < N; ++n) {
            moments[n] *= lorentz_kernel(static_cast<real_t>(n));
        }
    }

    /// Calculate the final Green's function for `scaled_energy` using the KPM `moments`
    template<class scalar_t, class real_t, class complex_t = num::get_complex_t<scalar_t>>
    ArrayX<complex_t> calculate_greens(ArrayX<real_t> const& scaled_energy,
                                       ArrayX<scalar_t> const& moments) {
        // Note that this integer array has real type values
        auto ns = ArrayX<real_t>(moments.size());
        for (auto n = 0; n < ns.size(); ++n) {
            ns[n] = static_cast<real_t>(n);
        }

        // G = -2*i / sqrt(1 - E^2) * sum( moments * exp(-i*ns*acos(E)) )
        auto greens = ArrayX<complex_t>(scaled_energy.size());
        transform(scaled_energy, greens, [&](real_t E) {
            using std::acos;
            using constant::i1;
            auto const norm = -real_t{2} * complex_t{i1} / sqrt(1 - E*E);
            return norm * sum(moments * exp(-complex_t{i1} * ns * acos(E)));
        });

        return greens;
    }
} // namespace detail

/**
 Stores KPM moments (size `num_moments`) computed for each index (size of `indices`)
 */
template<class scalar_t>
class MomentsMatrix {
    using real_t = num::get_real_t<scalar_t>;

    ArrayXi indices;
    std::vector<ArrayX<scalar_t>> data;

public:
    MomentsMatrix(int num_moments, ArrayXi const& indices)
        : indices(indices), data(indices.size()) {
        for (auto& moments : data) {
            moments.resize(num_moments);
        }
    }

    /// Collect the first 2 moments which are computer outside the main KPM loop
    void collect_initial(VectorX<scalar_t> const& r0, VectorX<scalar_t> const& r1) {
        for (auto i = 0; i < indices.size(); ++i) {
            auto const idx = indices[i];
            data[i][0] = r0[idx] * real_t{0.5}; // 0.5 is special for the moment zero
            data[i][1] = r1[idx];
        }
    }

    /// Collect moment `n` from result vector `r` for each index. Expects `n >= 2`.
    void collect(int n, VectorX<scalar_t> const& r) {
        assert(n >= 2 && n < data[0].size());
        for (auto i = 0; i < indices.size(); ++i) {
            auto const idx = indices[i];
            data[i][n] = r[idx];
        }
    }

    /// Put the kernel in *Kernel* polynomial method
    void apply_lorentz_kernel(real_t lambda) {
        for (auto& moments : data) {
            detail::apply_lorentz_kernel(moments, lambda);
        }
    }

    /// Calculate the final Green's function at all indices for `scaled_energy`
    std::vector<ArrayXcd> calc_greens(ArrayX<real_t> const& scaled_energy) const {
        auto greens = std::vector<ArrayXcd>();
        greens.reserve(indices.size());
        for (auto const& moments : data) {
            auto const g = detail::calculate_greens(scaled_energy, moments);
            greens.push_back(g.template cast<std::complex<double>>());
        }
        return greens;
    }
};

/**
 Stats of the KPM calculation
 */
struct Stats {
    int last_num_moments = 0;
    Chrono moments_timer;

    Stats() = default;
    Stats(int num_moments) : last_num_moments(num_moments) {}

    template<class scalar_t>
    std::string report(Bounds<scalar_t> const& bounds, OptimizedHamiltonian<scalar_t> const& oh,
                       bool shortform) const;
};

/// Return the KPM r0 vector with all zeros except for the source index
template<class scalar_t>
VectorX<scalar_t> make_r0(SparseMatrixX<scalar_t> const& h2, int i) {
    auto r0 = VectorX<scalar_t>();
    r0.setZero(h2.rows());
    r0[i] = 1;
    return r0;
}

/// Return the KPM r1 vector which is equal to the Hamiltonian matrix column at the source index
template<class scalar_t>
VectorX<scalar_t> make_r1(SparseMatrixX<scalar_t> const& h2, int i) {
    // -> r1 = h * r0; <- optimized thanks to `r0[i] = 1`
    // Note: h2.col(i) == h2.row(i).conjugate(), but the second is row-major friendly
    // multiply by 0.5 because H2 was pre-multiplied by 2
    return h2.row(i).conjugate() * scalar_t{0.5};
}

/// Calculate KPM moments -- reference implementation, no optimizations
template<class scalar_t>
MomentsMatrix<scalar_t> calc_moments0(SparseMatrixX<scalar_t> const& h2,
                                     Indices const& idx, int num_moments);

/// Calculate KPM moments -- with reordering optimization (optimal system size for each iteration)
template<class scalar_t>
MomentsMatrix<scalar_t> calc_moments1(OptimizedHamiltonian<scalar_t> const& oh, int num_moments);

/// Calculate KPM moments -- like previous plus bandwidth optimization (interleaved moments)
template<class scalar_t>
MomentsMatrix<scalar_t> calc_moments2(OptimizedHamiltonian<scalar_t> const& oh, int num_moments);

} // namespace kpm


struct KPMConfig {
    float lambda = 4.0f; ///< controls the accuracy of the kernel polynomial method
    float min_energy = 0.0f; ///< lowest eigenvalue of the Hamiltonian
    float max_energy = 0.0f; ///< highest eigenvalue of the Hamiltonian

    int optimization_level = 2; ///< 0 to 2, higher levels apply more complex optimizations
    float lanczos_precision = 0.002f; ///< how precise should the min/max energy estimation be
};

/**
 Kernel polynomial method for calculating Green's function
 */
template<class scalar_t>
class KPM : public GreensStrategy {
    using real_t = num::get_real_t<scalar_t>;
    using complex_t = num::get_complex_t<scalar_t>;

    SparseMatrixRC<scalar_t> hamiltonian;
    KPMConfig config;

    kpm::Bounds<scalar_t> bounds;
    kpm::OptimizedHamiltonian<scalar_t> optimized_hamiltonian;
    kpm::Stats stats;

public:
    using Config = KPMConfig;
    explicit KPM(SparseMatrixRC<scalar_t> hamiltonian, Config const& config = {});

    bool change_hamiltonian(Hamiltonian const& h) override;
    ArrayXcd calc(int row, int col, ArrayXd const& energy, double broadening) override;
    std::vector<ArrayXcd> calc_vector(int row, std::vector<int> const& cols,
                                      ArrayXd const& energy, double broadening) override;
    std::string report(bool shortform) const override;

private:
    /// Return the number of moments needed to compute Green's at the specified broadening
    int required_num_moments(double broadening, kpm::Scale<real_t> scale);
    /// Return KPM moments for several indices
    kpm::MomentsMatrix<scalar_t> calc_moments_matrix(kpm::Indices const& idx, double broadening);
};

TBM_EXTERN_TEMPLATE_CLASS(KPM)
TBM_EXTERN_TEMPLATE_CLASS(kpm::Bounds)
TBM_EXTERN_TEMPLATE_CLASS(kpm::OptimizedHamiltonian)
TBM_EXTERN_TEMPLATE_CLASS(kpm::MomentsMatrix)

} // namespace tbm
