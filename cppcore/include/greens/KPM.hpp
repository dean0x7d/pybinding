#pragma once
#include "greens/Greens.hpp"
#include "numeric/sparse.hpp"
#include "compute/lanczos.hpp"
#include "detail/macros.hpp"

namespace tbm { namespace kpm {

/**
 Computes and stores the KPM scaling parameters `a` and `b` based on the energy
 bounds (min and max eigenvalue) of the Hamiltonian. The bounds are determined
 automatically with the Lanczos procedure, or set manually by the user.

 Note: `compute` must be called before `a` and `b` are used. This is slightly awkward
 but necessary because the computation is relatively expensive and should not be done
 at construction time.
 */
template<class scalar_t>
class Scale {
    using real_t = num::get_real_t<scalar_t>;
    static constexpr auto scaling_tolerance = 0.01f; ///< the eigenvalue bounds are not precise

public:
    Scale() = default;
    /// Set the energy bounds manually, therefore skipping the Lanczos procedure at `compute` time
    Scale(real_t min_energy, real_t max_energy) : bounds{min_energy, max_energy, 0} {}

    // Compute the scaling params of the Hamiltonian `matrix` using the Lanczos procedure
    void compute(SparseMatrixX<scalar_t> const& matrix, real_t lanczos_tolerance);

public:
    real_t a = 0;
    real_t b = 0;
    compute::LanczosBounds<real_t> bounds = {0, 0, 0};
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
 Stores a scaled Hamiltonian `(H - b)/a` which limits it to (-1, 1) boundaries required for KPM.
 In addition, two optimisations are applied:

 1) The matrix is multiplied by 2. This benefits most calculations (e.g. `x = 2*H*x - y`),
    because the 2x multiplication is done only once, but it will need to be divided by 2
    when the original element values are needed (very rarely).

 2) Reorder the elements so that the index pair (i, j) is at the start of the matrix.
    This produces the `optimized_sizes` vector which may be used to reduce calculation
    time by skipping sparse matrix-vector multiplication of zero values.
 */
template<class scalar_t>
class OptimizedHamiltonian {
    using real_t = num::get_real_t<scalar_t>;

public:
    /// Create the optimized Hamiltonian from `H` targeting index pair `idx`
    void create(SparseMatrixX<scalar_t> const& H, Indices const& idx,
                Scale<scalar_t> scale, bool use_reordering);

    /// Return an index into `optimized_sizes`, indicating the optimal system size
    /// for the calculation of KPM moment number `n` out of total `num_moments`
    int optimized_size_index(int n, int num_moments) const {
        assert(n < num_moments);
        assert(!optimized_sizes.empty());

        auto const max_index = std::min(
            static_cast<int>(optimized_sizes.size()) - 1,
            num_moments / 2
        );

        if (n < max_index) {
            return n; // size grows in the beginning
        } else { // constant in the middle and shrinking near the end as reverse `n`
            return std::min(max_index, num_moments - 1 - n + size_index_offset);
        }
    }

    /// Return the optimized system size for KPM moment number `n` out of total `num_moments`
    int optimized_size(int n, int num_moments) const {
        if (!optimized_sizes.empty()) {
            return optimized_sizes[optimized_size_index(n, num_moments)];
        } else {
            return H2.rows();
        }
    }

    /// The unoptimized compute area is H2.nonZeros() * num_moments
    double optimized_area(int num_moments) const;

public:
    SparseMatrixX<scalar_t> H2; ///< the optimized matrix
    Indices original_idx; ///< indices from the original `H` matrix
    Indices optimized_idx; ///< reordered indices in the `H2` matrix
    std::vector<int> optimized_sizes; ///< optimal matrix size "steps" for the KPM calculation
    int size_index_offset = 0; ///< needed to correctly compute off-diagonal elements (i != j)

private:
    /// Fill H2 with scaled Hamiltonian: H2 = (H - I*b) * (2/a)
    void create_scaled(SparseMatrixX<scalar_t> const& H, Indices const& idx,
                       Scale<scalar_t> scale);
    /// Scale and reorder the Hamiltonian so that idx is at the start of the H2 matrix
    void create_reordered(SparseMatrixX<scalar_t> const& H, Indices const& idx,
                          Scale<scalar_t> scale);
    static Indices reorder_indices(Indices const& original_idx,
                                   std::vector<int> const& reorder_map);
    static int compute_index_offset(Indices const& optimized_idx,
                                    std::vector<int> const& optimized_sizes);
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

        for (auto n = 0u; n < N; ++n) {
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
class MomentMatrix {
    using real_t = num::get_real_t<scalar_t>;

    ArrayXi indices;
    std::vector<ArrayX<scalar_t>> data;

public:
    MomentMatrix(int num_moments, ArrayXi const& indices)
        : indices(indices), data(indices.size()) {
        for (auto& moments : data) {
            moments.resize(num_moments);
        }
    }

    /// Collect the first 2 moments which are computer outside the main KPM loop
    void collect_initial(VectorX<scalar_t> const& r0, VectorX<scalar_t> const& r1) {
        for (auto i = 0u; i < indices.size(); ++i) {
            auto const idx = indices[i];
            data[i][0] = r0[idx] * real_t{0.5}; // 0.5 is special for the moment zero
            data[i][1] = r1[idx];
        }
    }

    /// Collect moment `n` from result vector `r` for each index. Expects `n >= 2`.
    void collect(int n, VectorX<scalar_t> const& r) {
        assert(n >= 2 && n < data[0].size());
        for (auto i = 0u; i < indices.size(); ++i) {
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
class Stats {
public:
    std::string report(bool shortform) const;

    template<class real_t>
    void lanczos(compute::LanczosBounds<real_t> const& bounds, Chrono const& time);
    template<class scalar_t>
    void reordering(OptimizedHamiltonian<scalar_t> const& oh, int num_moments, Chrono const& time);
    template<class scalar_t>
    void kpm(OptimizedHamiltonian<scalar_t> const& oh, int num_moments, Chrono const& time);
    void greens(Chrono const& time);

private:
    void append(std::string short_str, std::string long_str, Chrono const& time);

private:
    char const* short_line = "{:s} [{}] ";
    char const* long_line = "- {:<80s} | {}\n";
    std::string short_report;
    std::string long_report;
};

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

public:
    using Config = KPMConfig;
    explicit KPM(SparseMatrixRC<scalar_t> hamiltonian, Config const& config = {});

    bool change_hamiltonian(Hamiltonian const& h) override;
    ArrayXcd calc(int i, int j, ArrayXd const& energy, double broadening) override;
    std::vector<ArrayXcd> calc_vector(int row, std::vector<int> const& cols,
                                      ArrayXd const& energy, double broadening) override;
    std::string report(bool shortform) const override;

private:
    /// Calculate the KPM Green's function moments
    static kpm::MomentMatrix<scalar_t> calc_moments(kpm::OptimizedHamiltonian<scalar_t> const& oh,
                                                    int num_moments);
    /// Optimized `calc_moments`: lower memory bandwidth requirements
    static kpm::MomentMatrix<scalar_t> calc_moments2(kpm::OptimizedHamiltonian<scalar_t> const& oh,
                                                     int num_moments);

private:
    SparseMatrixRC<scalar_t> hamiltonian;
    Config const config;

    kpm::Scale<scalar_t> scale;
    kpm::OptimizedHamiltonian<scalar_t> optimized_hamiltonian;
    kpm::Stats stats;
};

TBM_EXTERN_TEMPLATE_CLASS(KPM)
TBM_EXTERN_TEMPLATE_CLASS(kpm::Scale)
TBM_EXTERN_TEMPLATE_CLASS(kpm::OptimizedHamiltonian)
TBM_EXTERN_TEMPLATE_CLASS(kpm::MomentMatrix)

} // namespace tbm
