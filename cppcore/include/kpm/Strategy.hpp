#pragma once
#include "Model.hpp"
#include "hamiltonian/Hamiltonian.hpp"

#include "kpm/Bounds.hpp"
#include "kpm/Config.hpp"
#include "kpm/Kernel.hpp"
#include "kpm/OptimizedHamiltonian.hpp"
#include "kpm/Moments.hpp"
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
    virtual ArrayXd ldos(idx_t index, ArrayXd const& energy, double broadening) = 0;
    /// Return the Green's function matrix element (row, col) for the given energy range
    virtual ArrayXcd greens(idx_t row, idx_t col, ArrayXd const& energy, double broadening) = 0;
    /// Return multiple Green's matrix elements for a single `row` and multiple `cols`
    virtual std::vector<ArrayXcd> greens_vector(idx_t row, std::vector<idx_t> const& cols,
                                                ArrayXd const& energy, double broadening) = 0;
    /// Return the total DOS for the given energy range and broadening
    virtual ArrayXd dos(ArrayXd const& energy, double broadening, idx_t num_random) = 0;
    /// Bastin's DC conductivity in the directions defined by the `left` and `right` coordinates
    virtual ArrayXcd conductivity(ArrayXf const& left_coords, ArrayXf const& right_coords,
                                  ArrayXd const& chemical_potential, double broadening,
                                  double temperature, idx_t num_random, idx_t num_points) = 0;

    /// Get some information about what happened during the last calculation
    virtual std::string report(bool shortform = false) const = 0;
    virtual Stats const& get_stats() const = 0;
};

/**
 Concrete KPM strategy templated on scalar type and compute implementation (CPU or GPU)

 The `Compute` template parameter provides functions which compute raw KPM moments.
 See `DefaultCompute` for the reference implementation.
 */
template<class scalar_t>
class StrategyTemplate : public Strategy {
    using real_t = num::get_real_t<scalar_t>;
    using complex_t = num::get_complex_t<scalar_t>;

public:
    using Config = kpm::Config;
    explicit StrategyTemplate(SparseMatrixRC<scalar_t> hamiltonian, Config const& config = {});

    bool change_hamiltonian(Hamiltonian const& h) final;

    ArrayXd ldos(idx_t index, ArrayXd const& energy, double broadening) final;
    ArrayXcd greens(idx_t row, idx_t col, ArrayXd const& energy, double broadening) final;
    std::vector<ArrayXcd> greens_vector(idx_t row, std::vector<idx_t> const& cols,
                                        ArrayXd const& energy, double broadening) final;
    ArrayXd dos(ArrayXd const& energy, double broadening, idx_t num_random) final;
	ArrayXcd conductivity(ArrayXf const& left_coords, ArrayXf const& right_coords,
                          ArrayXd const& chemical_potential, double broadening,
                          double temperature, idx_t num_random, idx_t num_points) final;

    std::string report(bool shortform) const final;
    Stats const& get_stats() const final { return stats; }

protected:
    /// KPM moment computation which must be implemented by derived classes
    virtual void compute(DiagonalMoments<scalar_t>&, VectorX<scalar_t>&& r0,
                         AlgorithmConfig const&, OptimizedHamiltonian<scalar_t> const&) const = 0;
    virtual void compute(OffDiagonalMoments<scalar_t>&, VectorX<scalar_t>&& r0,
                         AlgorithmConfig const&, OptimizedHamiltonian<scalar_t> const&) const = 0;

private:
    void compute(DiagonalMoments<scalar_t>&, VectorX<scalar_t>&& r0, AlgorithmConfig const&);
    void compute(OffDiagonalMoments<scalar_t>&, VectorX<scalar_t>&& r0, AlgorithmConfig const&);

private:
    SparseMatrixRC<scalar_t> hamiltonian;
    Config config;

    Bounds<scalar_t> bounds;
    OptimizedHamiltonian<scalar_t> optimized_hamiltonian;
    Stats stats;
};

CPB_EXTERN_TEMPLATE_CLASS(StrategyTemplate)

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
