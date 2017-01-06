#pragma once
#include "Model.hpp"
#include "hamiltonian/Hamiltonian.hpp"

#include "kpm/Bounds.hpp"
#include "kpm/Config.hpp"
#include "kpm/Kernel.hpp"
#include "kpm/OptimizedHamiltonian.hpp"
#include "kpm/Stats.hpp"

#include "utils/Chrono.hpp"
#include "detail/strategy.hpp"

namespace cpb { namespace kpm {

/**
 Abstract base which defines the interface for a KPM strategy

 A strategy does the actual work of computing KPM moments and other processing
 operation to produce final results like LDOS, DOS, Green's function, etc.

 Different derived classes all implement the same interface but are optimized
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
 Concrete KPM strategy templated on scalar type and compute implementation (CPU or GPU)

 The `Compute` template parameter provides functions which compute raw KPM moments.
 See `DefaultCompute` for the reference implementation.
 */
template<class scalar_t, class Compute>
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
struct DefaultCompute;
CPB_EXTERN_TEMPLATE_CLASS_VARGS(StrategyTemplate, DefaultCompute)

template<class scalar_t>
using DefaultStrategy = StrategyTemplate<scalar_t, DefaultCompute>;

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
