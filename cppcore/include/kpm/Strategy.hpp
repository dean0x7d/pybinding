#pragma once
#include "Model.hpp"
#include "hamiltonian/Hamiltonian.hpp"

#include "kpm/Kernel.hpp"
#include "kpm/Bounds.hpp"
#include "kpm/Moments.hpp"
#include "kpm/OptimizedHamiltonian.hpp"
#include "kpm/Stats.hpp"

#include "utils/Chrono.hpp"
#include "detail/strategy.hpp"

namespace cpb { namespace kpm {

/**
 KPM configuration struct with defaults
 */
struct Config {
    float min_energy = 0.0f; ///< lowest eigenvalue of the Hamiltonian
    float max_energy = 0.0f; ///< highest eigenvalue of the Hamiltonian
    Kernel kernel = lorentz_kernel(4.0f); ///< produces the damping coefficients

    int opt_level = 3; ///< 0 to 3, higher levels apply more complex optimizations
    float lanczos_precision = 0.002f; ///< how precise should the min/max energy estimation be
};

/**
 Abstract base which defines the interface for a KPM strategy

 A strategy does the actual work of computing KPM moments and other processing
 operation to produce final results like LDOS, DOS, Green's function, etc.

 Different derived classes all implement the same inferface but are optimized
 for specific scalar types and hardware (CPU, GPU).
 */
class Strategy {
public:
    virtual ~Strategy() = default;

    /// Returns false if the given Hamiltonian is the wrong type for this GreensStrategy
    virtual bool change_hamiltonian(Hamiltonian const& h) = 0;

    /// Return the LDOS at the given Hamiltonian index for the energy range and broadening
    virtual ArrayXd ldos(int index, ArrayXd const& energy, double broadening) = 0;
    /// Return the Green's function matrix element (row, col) for the given energy range
    virtual ArrayXcd greens(int row, int col, ArrayXd const& energy, double broadening) = 0;
    /// Return multiple Green's matrix elements for a single `row` and multiple `cols`
    virtual std::vector<ArrayXcd> greens_vector(int row, std::vector<int> const& cols,
                                                ArrayXd const& energy, double broadening) = 0;

    /// Get some information about what happened during the last calculation
    virtual std::string report(bool shortform = false) const = 0;
    virtual Stats const& get_stats() const = 0;
};

/**
 Concrete KPM strategy templated on scalar type and implementation (CPU or GPU)

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
template<class scalar_t, class Impl>
class StrategyTemplate final : public Strategy {
    using real_t = num::get_real_t<scalar_t>;
    using complex_t = num::get_complex_t<scalar_t>;

public:
    using Config = kpm::Config;
    explicit StrategyTemplate(SparseMatrixRC<scalar_t> hamiltonian, Config const& config = {});

    bool change_hamiltonian(Hamiltonian const& h) final;

    ArrayXd ldos(int index, ArrayXd const& energy, double broadening) final;
    ArrayXcd greens(int row, int col, ArrayXd const& energy, double broadening) final;
    std::vector<ArrayXcd> greens_vector(int row, std::vector<int> const& cols,
                                        ArrayXd const& energy, double broadening) final;

    std::string report(bool shortform) const final;
    Stats const& get_stats() const final { return stats; }

private:
    SparseMatrixRC<scalar_t> hamiltonian;
    Config config;

    Bounds<scalar_t> bounds;
    OptimizedHamiltonian<scalar_t> optimized_hamiltonian;
    Stats stats;
};

/**
 Default CPU implementation for computing KPM moments, see `Strategy`
 */
struct DefaultCalcMoments;
CPB_EXTERN_TEMPLATE_CLASS_VARGS(StrategyTemplate, DefaultCalcMoments)

template<class scalar_t>
using DefaultStrategy = StrategyTemplate<scalar_t, DefaultCalcMoments>;

#ifdef CPB_USE_CUDA
/**
 Cuda GPU implementation for computing KPM moments
 */
struct CudaCalcMoments;
CPB_EXTERN_TEMPLATE_CLASS_VARGS(StrategyTemplate, CudaCalcMoments)

template<class scalar_t>
using CudaStrategy = StrategyTemplate<scalar_t, CudaCalcMoments>;
#endif // CPB_USE_CUDA

} // namespace kpm

/**
 Return a strategy with the scalar type matching the given Hamiltonian
 */
template<template<class> class Strategy, class Config = typename Strategy<float>::Config>
std::unique_ptr<kpm::Strategy> make_kpm_strategy(Hamiltonian const& h, Config const& c = {}) {
    return detail::MakeStrategy<kpm::Strategy, Strategy>(c)(h);
}

} // namespace cpb
