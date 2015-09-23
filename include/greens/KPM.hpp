#pragma once
#ifdef _MSC_VER
# pragma warning(disable: 4579) // in-class static constexpr cannot be used at runtime
#endif

#include "greens/Greens.hpp"
#include "support/sparse.hpp"

namespace tbm { namespace kpm {

struct Config {
    float lambda = 4.0f; ///< controls the accuracy of the kernel polynomial method
    float min_energy = 0.0f; ///< lowest eigenvalue of the Hamiltonian
    float max_energy = 0.0f; ///< highest eigenvalue of the Hamiltonian

    int optimization_level = 2; ///< 0 to 2, higher levels apply more complex optimizations
    float lanczos_precision = 0.002f; ///< how precise should the min/max energy estimation be
};

/**
 Computes and stores the energy bounds (min and max energy)
 of the Hamiltonian and KPM scaling parameters `a` and `b`
 */
template<class scalar_t>
class Bounds {
    using real_t = num::get_real_t<scalar_t>;
    static constexpr auto scaling_tolerance = 0.01f; ///< the eigenvalue bounds are not precise

public:
    // Compute the bounds of Hamiltonian `H` using the Lanczos procedure
    void compute(SparseMatrixX<scalar_t> const& H, real_t lanczos_tolerance);
    // Set the bounds manually
    void set(real_t min_energy, real_t max_energy);

public:
    real_t a = 0;
    real_t b = 0;

    int lanczos_loops = 0;
    real_t min_energy, max_energy;
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
    struct IndexPair { int i, j; };

public:
    /// Create the optimized Hamiltonian from `H` targeting index pair `idx`
    void create(SparseMatrixX<scalar_t> const& H, IndexPair idx,
                Bounds<scalar_t> bounds, bool use_reordering);

    /// Return an index into `optimized_sizes`, indicating the optimal system size
    /// for the calculation of KPM moment number `n` out of total `num_moments`
    int get_optimized_size_index(int n, int num_moments) const {
        assert(!optimized_sizes.empty());

        auto const max_size = std::min(
            static_cast<int>(optimized_sizes.size()) - 2,
            num_moments / 2
        );

        if (n < max_size)
            return n + 1; // size grows in the beginning
        else if (n > num_moments - max_size - 1)
            return num_moments - n; // reverse `n + 1` -> shrinking near the end
        else
            return max_size + 1; // constant in the middle
    }

    /// Return the optimized system size for KPM moment number `n` out of total `num_moments`
    int get_optimized_size(int n, int num_moments) const {
        if (!optimized_sizes.empty())
            return optimized_sizes[get_optimized_size_index(n, num_moments)];
        else
            return H2.rows();
    }

    /// The unoptimized compute area is H2.nonZeros() * num_moments
    double optimized_area(int num_moments) const;

public:
    SparseMatrixX<scalar_t> H2; ///< the optimized matrix
    IndexPair original_idx = {-1, -1}; ///< indices from the original `H` matrix
    IndexPair optimized_idx = {-1, -1}; ///< reordered indices in the `H2` matrix
    std::vector<int> optimized_sizes; ///< optimal matrix size "steps" for the KPM calculation

private:
    /// Fill H2 with scaled Hamiltonian: H2 = (H - I*b) * (2/a)
    void create_scaled(SparseMatrixX<scalar_t> const& H, IndexPair idx, Bounds<scalar_t> b);
    /// Scale and reorder the Hamiltonian so that idx is at the start of the H2 matrix
    void create_reordered(SparseMatrixX<scalar_t> const& H, IndexPair idx, Bounds<scalar_t> b);
};

/**
 Stats of the KPM calculation
 */
class Stats {
public:
    std::string report(bool shortform) const;

    void lanczos(double min_energy, double max_energy, int loops, Chrono const& time);
    template<class scalar_t>
    void reordering(OptimizedHamiltonian<scalar_t> const& oh, int num_moments, Chrono const& time);
    template<class scalar_t>
    void kpm(OptimizedHamiltonian<scalar_t> const& oh, int num_moments, Chrono const& time);
    void greens(Chrono const& time);

private:
    void append(std::string short_str, std::string long_str, Chrono const& time);

private:
    char const* short_line = "{message:s} ({time}) ";
    char const* long_line = "- {message:-80s} | {time}\n";
    std::string short_report;
    std::string long_report;
};

/**
 Kernel polynomial method for calculating Green's function
 */
template<class scalar_t>
class Strategy : public GreensStrategyT<scalar_t> {
    using real_t = num::get_real_t<scalar_t>;
    using complex_t = num::get_complex_t<scalar_t>;

public:
    explicit Strategy(const Config& config) : config(config) {}

protected: // required implementation
    virtual void hamiltonian_changed() override;
    virtual ArrayXcf calculate(int i, int j, ArrayXf energy, float broadening) override;
    virtual std::string report(bool shortform) const override;
    
private:
    /// Calculate the KPM Green's function moments
    static ArrayX<scalar_t> calculate_moments(OptimizedHamiltonian<scalar_t> const& oh,
                                              int num_moments);
    /// Optimized `calculate_moments`: lower memory bandwidth requirements
    static ArrayX<scalar_t> calculate_moments2(OptimizedHamiltonian<scalar_t> const& oh,
                                               int num_moments);
    /// Put the kernel in *Kernel* polynomial method
    static void apply_lorentz_kernel(ArrayX<scalar_t>& moments, float lambda);
    /// Calculate the final Green's function for `energy` using the KPM `moments`
    static ArrayX<complex_t> calculate_greens(ArrayX<real_t> const& energy,
                                              ArrayX<scalar_t> const& moments);

private:
    Config const config;
    Bounds<scalar_t> bounds;
    OptimizedHamiltonian<scalar_t> optimized_hamiltonian;

    Stats stats;

protected: // declare used inherited members (template class requirement)
    using GreensStrategyT<scalar_t>::hamiltonian;
};

} // namespace tbm::kpm

/**
 Concrete Greens for creating KPM objects.
 */
class KPM : public Greens {
public: // construction and configuration
    using Config = kpm::Config;
    static constexpr auto defaults = Config{};

    KPM(Model const& model,
        float lambda = defaults.lambda,
        std::pair<float, float> energy_range = {defaults.min_energy, defaults.max_energy},
        int optimization_level = defaults.optimization_level,
        float lanczos_precision = defaults.lanczos_precision)
    {
        if (energy_range.first > energy_range.second)
            throw std::invalid_argument{"KPM: Invalid energy range specified (min > max)."};
        if (lambda <= 0)
            throw std::invalid_argument{"KPM: Lambda must be positive."};

        config.lambda = lambda;
        config.min_energy = energy_range.first;
        config.max_energy = energy_range.second;
        config.optimization_level = optimization_level;
        config.lanczos_precision = lanczos_precision;

        set_model(model);
    }

protected: // required implementation
    virtual std::unique_ptr<GreensStrategy>
        create_strategy_for(const std::shared_ptr<const Hamiltonian>&) const override;

private:
    Config config = {};
};
    
} // namespace tbm
