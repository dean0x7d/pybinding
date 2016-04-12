#pragma once
#include "greens/Greens.hpp"
#include "greens/kpm/Bounds.hpp"
#include "greens/kpm/Moments.hpp"
#include "greens/kpm/OptimizedHamiltonian.hpp"
#include "greens/kpm/Stats.hpp"

#include "detail/macros.hpp"

namespace tbm {

struct KPMConfig {
    float lambda = 4.0f; ///< controls the accuracy of the kernel polynomial method
    float min_energy = 0.0f; ///< lowest eigenvalue of the Hamiltonian
    float max_energy = 0.0f; ///< highest eigenvalue of the Hamiltonian

    int optimization_level = 3; ///< 0 to 3, higher levels apply more complex optimizations
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
    /// Return KPM moments for several indices
    kpm::MomentsMatrix<scalar_t> calc_moments_matrix(kpm::Indices const& idx, double broadening);
    /// Return KPM moments for a diagonal element (faster algorithm than off-diagonal)
    ArrayX<scalar_t> calc_moments_diag(int i, double broadening);
};

TBM_EXTERN_TEMPLATE_CLASS(KPM)

} // namespace tbm
