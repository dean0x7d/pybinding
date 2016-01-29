#pragma once
#include "Model.hpp"
#include "system/Lattice.hpp"
#include "hamiltonian/Hamiltonian.hpp"

#include "detail/strategy.hpp"

#include "utils/Chrono.hpp"
#include "utils/Log.hpp"

#include "support/dense.hpp"

#include <memory>

namespace tbm {

/**
 Abstract base class for an eigensolver
 */
class SolverStrategy {
public:
    virtual ~SolverStrategy() = default;

    /// Returns false if the given Hamiltonian is the wrong type for this SolverStrategy
    virtual bool set_hamiltonian(const std::shared_ptr<const Hamiltonian>&) = 0;
    
    virtual void solve() = 0;
    virtual RealArrayRef eigenvalues() const = 0;
    virtual ComplexArrayRef eigenvectors() const = 0;

    virtual std::string report(bool shortform) const = 0;
};


/**
 Abstract base with scalar type specialization
 */
template<class scalar_t>
class SolverStrategyT : public SolverStrategy {
    using real_t = num::get_real_t<scalar_t>;

public:
    virtual ~SolverStrategyT() { Log::d("SolverStrategy<" + num::scalar_name<scalar_t>() + ">()"); }
    
    virtual bool set_hamiltonian(const std::shared_ptr<const Hamiltonian>& ham) final {
        // check if it's compatible
        if (auto cast_ham = std::dynamic_pointer_cast<const HamiltonianT<scalar_t>>(ham)) {
            if (hamiltonian != cast_ham) {
                hamiltonian = cast_ham;
                hamiltonian_changed();
            }
            return true;
        }
        // failed -> wrong scalar_type
        return false;
    }

public:
    virtual RealArrayRef eigenvalues() const override { return arrayref(_eigenvalues); }
    virtual ComplexArrayRef eigenvectors() const override { return arrayref(_eigenvectors); }

protected:
    /// possible post-processing that may be defined by derived classes
    virtual void hamiltonian_changed() {};
    
protected:
    ArrayX<real_t> _eigenvalues;
    ArrayXX<scalar_t> _eigenvectors;
    std::shared_ptr<const HamiltonianT<scalar_t>> hamiltonian; ///< the Hamiltonian to solve
};


/**
 The factory will produce the SolverStrategy class with the right scalar type.
 This is an abstract base factory. The derived factories will need to
 implement the create_for(hamiltonian) method.
 */
class BaseSolver {
    using MakeStrategy = std::function<std::unique_ptr<SolverStrategy>(Model const&)>;

public:
    void solve();
    void clear() { is_solved = false; }
    std::string report(bool shortform) const;

    void set_model(Model const&);
    Model const& get_model() const { return model; }
    std::shared_ptr<System const> system() const { return model.system(); }

    RealArrayRef eigenvalues();
    ComplexArrayRef eigenvectors();

    ArrayXd calc_dos(ArrayXf energies, float broadening);
    ArrayXd calc_spatial_ldos(float energy, float broadening);

protected:
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

} // namespace tbm
