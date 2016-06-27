#pragma once
#include "Model.hpp"
#include "system/Lattice.hpp"
#include "hamiltonian/Hamiltonian.hpp"

#include "utils/Chrono.hpp"
#include "numeric/dense.hpp"
#include "detail/strategy.hpp"

#include <memory>

namespace cpb {

/**
 Abstract base class for an eigensolver
 */
class SolverStrategy {
public:
    virtual ~SolverStrategy() = default;

    /// Returns false if the given Hamiltonian is the wrong type for this SolverStrategy
    virtual bool change_hamiltonian(Hamiltonian const& h) = 0;
    virtual void solve() = 0;
    virtual std::string report(bool shortform) const = 0;

    virtual RealArrayConstRef eigenvalues() const = 0;
    virtual ComplexArrayConstRef eigenvectors() const = 0;
};

/**
 Main solver interface

 Internally it uses a SolverStrategy with the scalar of the given Hamiltonian.
 */
class BaseSolver {
public:
    void solve();
    void clear() { is_solved = false; }
    std::string report(bool shortform) const;

    void set_model(Model const&);
    Model const& get_model() const { return model; }
    std::shared_ptr<System const> system() const { return model.system(); }

    RealArrayConstRef eigenvalues();
    ComplexArrayConstRef eigenvectors();

    ArrayXd calc_dos(ArrayXf energies, float broadening);
    ArrayXd calc_spatial_ldos(float energy, float broadening);

protected:
    using MakeStrategy = std::function<std::unique_ptr<SolverStrategy>(Hamiltonian const&)>;
    BaseSolver(Model const& model, MakeStrategy const& make_strategy);

private:
    Model model;
    MakeStrategy make_strategy;
    std::unique_ptr<SolverStrategy> strategy;

    bool is_solved = false;
    mutable Chrono calculation_timer; ///< last calculation time
};

template<template<class> class Strategy>
class Solver : public BaseSolver {
    using Config = typename Strategy<float>::Config;
    using MakeStrategy = detail::MakeStrategy<SolverStrategy, Strategy>;

public:
    explicit Solver(Model const& model, Config const& config = {})
        : BaseSolver(model, MakeStrategy(config)) {}
};

} // namespace cpb
