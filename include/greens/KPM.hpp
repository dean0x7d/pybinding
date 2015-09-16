#pragma once
#ifdef _MSC_VER
# pragma warning(disable: 4579) // in-class static constexpr cannot be used at runtime
#endif

#include "greens/Greens.hpp"
#include "system/Lattice.hpp"
#include "support/sparse.hpp"

namespace tbm {

struct KPMConfig {
    float lambda = 4.0f; ///< controls the accuracy of the kernel polynomial method
    float min_energy = 0.0f; ///< lowest eigenvalue of the Hamiltonian
    float max_energy = 0.0f; ///< highest eigenvalue of the Hamiltonian

    bool use_reordering = true; ///< Hamiltonian reordering optimization
    float lanczos_precision = 0.002f; ///< how precise should the min/max energy estimation be
    float scaling_tolerance = 0.01f; ///< the eigenvalue bounds are not precise
};

/**
 Kernel polynomial method for calculating Green's function.
 */
template<typename scalar_t>
class KPMStrategy : public GreensStrategyT<scalar_t> {
    using real_t = num::get_real_t<scalar_t>;
    using complex_t = num::get_complex_t<scalar_t>;
    using SparseMatrix = SparseMatrixX<scalar_t>;
    
public:
    explicit KPMStrategy(const KPMConfig& config) : config(config) {}

protected: // required implementation
    virtual void hamiltonian_changed() override;
    virtual ArrayXcf calculate(int i, int j, ArrayXd energy, float broadening) override;
    virtual std::string report(bool shortform) const override;
    
private:
    /// @return (a, b) scaling parameters calculated from min_energy and max_energy
    std::tuple<real_t, real_t> scaling_params() const;
    
    /// Fill h2_matrix with scaled Hamiltonian: h2 = (H - I*b) * (2/a)
    void scale_hamiltonian();

    /**
     Like scale_hamiltonian, but also reorder the elements so that the target_index is at the start,
     and all neighbours directly following. This produces the reordered_steps vector which may be
     used to reduce calculation time by skipping sparse matrix-vector multiplication of zero values.

     @return New value of the index passed as translate_index.
     */
    int scale_and_reorder_hamiltonian(int target_index, int translate_index = 0);

    /// calculate the KPM Green's function moments
    ArrayX<scalar_t> calculate_moments(int i, int j) const;

    static ArrayX<complex_t> calculate_greens(const ArrayX<real_t>& energy,
                                              const ArrayX<scalar_t>& moments);

private:
    KPMConfig config;
    SparseMatrix h2_matrix; ///< scaled Hamiltonian matrix

    // work variables
    int num_moments; ///< number of moments to use for the Green's function calculation
    std::vector<int> reordered_steps; ///< optimal matrix size "steps" for the KPM calculation

    // reporting
    int lanczos_loops = 0; ///< how many loops did it take for the Lanczos calculation to converge
    Chrono lanczos_timer, reordering_timer, moments_timer, greens_timer;

protected: // declare used inherited members (template class requirement)
    using GreensStrategyT<scalar_t>::hamiltonian;
};

/**
 Concrete Greens for creating KPM objects.
 */
class KPM : public Greens {
public: // construction and configuration
    static constexpr auto defaults = KPMConfig{};

    KPM(Model const& model,
        float lambda = defaults.lambda,
        std::pair<float, float> energy_range = {defaults.min_energy, defaults.max_energy})
    {
        if (energy_range.first > energy_range.second)
            throw std::invalid_argument{"KPM: Invalid energy range specified (min > max)."};
        if (lambda <= 0)
            throw std::invalid_argument{"KPM: Lambda must be positive."};

        set_model(model);

        config.lambda = lambda;
        config.min_energy = energy_range.first;
        config.max_energy = energy_range.second;
    }

    /// Set the advanced options of KPM
    void advanced(bool use_reordering = defaults.use_reordering,
                  float lanczos_precision = defaults.lanczos_precision,
                  float scaling_tolerance = defaults.scaling_tolerance)
    {
        config.use_reordering = use_reordering;
        config.lanczos_precision = lanczos_precision;
        config.scaling_tolerance = scaling_tolerance;
    }

protected: // required implementation
    virtual std::unique_ptr<GreensStrategy>
        create_strategy_for(const std::shared_ptr<const Hamiltonian>&) const override;

private:
    KPMConfig config = {};
};
    
} // namespace tbm
