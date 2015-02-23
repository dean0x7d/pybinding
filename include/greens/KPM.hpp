#pragma once
#include "greens/Greens.hpp"
#include "support/sparse.hpp"

namespace tbm {

/**
 Kernel polynomial method for calculating Green's function.
 */
template<typename scalar_t>
class KPM : public GreensT<scalar_t> {
    using real_t = num::get_real_t<scalar_t>;
    using complex_t = num::get_complex_t<scalar_t>;
    using SparseMatrix = SparseMatrixX<scalar_t>;
    
protected: // a factory must be used to create this
    KPM(real_t lambda, real_t min_energy, real_t max_energy);
    friend class KPMFactory;
    
protected: // required implementation
    virtual void hamiltonian_changed() override;
    virtual ArrayX<std::complex<float>> v_calculate(int i, int j, ArrayX<float> energy,
                                                   float broadening) override;
    virtual std::string v_report(bool shortform) const override;
    
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
    SparseMatrix h2_matrix; ///< scaled Hamiltonian matrix

    // input parameters
    const real_t lambda; ///< controls the accuracy of the kernel polynomial method
    real_t min_energy, max_energy; ///< extreme eigenvalues of the Hamiltonian
    bool use_reordering; ///< should the Hamiltonian reordering optimization be used
    real_t lanczos_precision; ///< how precise should the min/max energy estimation be
    real_t scaling_tolerance; ///< allow some tolerance because the energy bounds are not precise

    // work variables
    int num_moments; ///< number of moments to use for the Green's function calculation
    std::vector<int> reordered_steps; ///< optimal matrix size "steps" for the KPM calculation

    // reporting
    int lanczos_loops = 0; ///< how many loops did it take for the Lanczos calculation to converge
    Chrono lanczos_timer, reordering_timer, moments_timer, greens_timer;

protected: // declare used inherited members (template class requirement)
    using GreensT<scalar_t>::hamiltonian;
};


/**
 Concrete GreensFactory for creating KPM objects.
 */
class KPMFactory : public GreensFactory {
public:
    struct defaults {
        static constexpr double lambda = 4.0;
        static constexpr double min_energy = 0.0;
        static constexpr double max_energy = 0.0;
        
        static constexpr bool use_reordering = true;
        static constexpr double lanczos_precision = 0.002; // percent
        static constexpr double scaling_tolerance = 0.01;
    };

    KPMFactory(double lambda = defaults::lambda,
               double energy_min = defaults::min_energy,
               double energy_max = defaults::max_energy)
        : lambda{lambda}, energy_min{energy_min}, energy_max{energy_max}
    {}

    // required implementation
    virtual std::unique_ptr<Greens>
        create_for(const std::shared_ptr<const Hamiltonian>& h) const override;
    
    /// Set the advanced options of KPM
    void advanced(bool use_reordering = defaults::use_reordering,
                  double lanczos_precision = defaults::lanczos_precision,
                  double scaling_tolerance = defaults::scaling_tolerance);

private: // create template
    template<typename scalar_t>
    std::unique_ptr<Greens>
        try_create_for(const std::shared_ptr<const Hamiltonian>& hamiltonian) const;

private: // factory variables
    double lambda;
    double energy_min, energy_max;
    
    bool use_reordering = defaults::use_reordering;
    double lanczos_precision = defaults::lanczos_precision;
    double scaling_tolerance = defaults::scaling_tolerance;
};
    
} // namespace tbm
