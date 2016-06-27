#pragma once
#include "greens/Greens.hpp"
#include "greens/kpm/Bounds.hpp"
#include "greens/kpm/Moments.hpp"
#include "greens/kpm/OptimizedHamiltonian.hpp"
#include "greens/kpm/Stats.hpp"

#include "detail/macros.hpp"

namespace cpb { namespace kpm {

struct KPMConfig {
    float lambda = 4.0f; ///< controls the accuracy of the kernel polynomial method
    float min_energy = 0.0f; ///< lowest eigenvalue of the Hamiltonian
    float max_energy = 0.0f; ///< highest eigenvalue of the Hamiltonian

    int opt_level = 3; ///< 0 to 3, higher levels apply more complex optimizations
    float lanczos_precision = 0.002f; ///< how precise should the min/max energy estimation be
};

/**
 Default CPU implementation for computing KPM moments, see `Strategy`
 */
struct DefaultCalcMoments;

/**
 Kernel polynomial method for calculating Green's function

 The `Impl` template parameter provides functions which compute raw KPM moments.
 See `DefaultCalcMoments` for example. An implementation should be declared as follows:

 struct Impl {
     /// Return the `OptimizedHamiltonian` matrix configuration for the given optimization level
     static MatrixConfig matrix_config(int opt_level);

     /// Return KPM moments for several indices (the ones the Hamiltonian is optimized for)
     template<class scalar_t>
     static MomentsMatrix<scalar_t> moments_vector(OptimizedHamiltonian<scalar_t> const& oh,
                                                   int num_moments, int opt_level);
     /// Return KPM moments for a diagonal element (faster algorithm than off-diagonal)
     template<class scalar_t>
     static ArrayX<scalar_t> moments_diag(OptimizedHamiltonian<scalar_t> const& oh,
                                          int num_moments, int opt_level);
 };
 */
template<class scalar_t, class Impl = DefaultCalcMoments>
class Strategy : public GreensStrategy {
    using real_t = num::get_real_t<scalar_t>;
    using complex_t = num::get_complex_t<scalar_t>;

    SparseMatrixRC<scalar_t> hamiltonian;
    KPMConfig config;

    Bounds<scalar_t> bounds;
    OptimizedHamiltonian<scalar_t> optimized_hamiltonian;
    Stats stats;

public:
    using Config = KPMConfig;
    explicit Strategy(SparseMatrixRC<scalar_t> hamiltonian, Config const& config = {});

    bool change_hamiltonian(Hamiltonian const& h) override;
    ArrayXcd calc(int row, int col, ArrayXd const& energy, double broadening) override;
    std::vector<ArrayXcd> calc_vector(int row, std::vector<int> const& cols,
                                      ArrayXd const& energy, double broadening) override;
    std::string report(bool shortform) const override;
};

CPB_EXTERN_TEMPLATE_CLASS_VARGS(Strategy, DefaultCalcMoments)
} // namespace kpm

using kpm::KPMConfig;

template<class scalar_t>
using KPM = kpm::Strategy<scalar_t, kpm::DefaultCalcMoments>;

#ifdef CPB_USE_CUDA
namespace kpm {
/**
 Cuda GPU implementation for computing KPM moments
 */
struct CudaCalcMoments;

CPB_EXTERN_TEMPLATE_CLASS_VARGS(Strategy, CudaCalcMoments)
} // namespace kpm

template<class scalar_t>
using KPMcuda = kpm::Strategy<scalar_t, kpm::CudaCalcMoments>;
#endif // CPB_USE_CUDA

} // namespace cpb
