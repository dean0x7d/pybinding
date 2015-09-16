#pragma once
#include "Model.hpp"
#include "system/Lattice.hpp"

#include "utils/Chrono.hpp"
#include "utils/Log.hpp"

#include "support/dense.hpp"
#include "support/uref.hpp"

#include <memory>

namespace tbm {

class System;
class Hamiltonian;
template <typename scalar_t> class HamiltonianT;

/**
 Abstract base class for an eigensolver
 */
class SolverStrategy {
public:
    virtual ~SolverStrategy() = default;

    /// Returns false if the given Hamiltonian is the wrong type for this SolverStrategy
    virtual bool set_hamiltonian(const std::shared_ptr<const Hamiltonian>&) = 0;
    
    virtual void solve() = 0;
    virtual DenseURef eigenvalues() const = 0;
    virtual DenseURef eigenvectors() const = 0;

    virtual std::string report(bool shortform) const = 0;
};


/**
 Abstract base with scalar type specialization
 */
template<typename scalar_t>
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
    virtual DenseURef eigenvalues() const override { return _eigenvalues; }
    virtual DenseURef eigenvectors() const override { return _eigenvectors; }

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
class Solver {
public:
    virtual ~Solver() = default;

public:
    void solve();
    void clear() { is_solved = false; }
    std::string report(bool shortform) const;

    void set_model(Model const&);
    Model const& get_model() const { return model; }
    std::shared_ptr<System const> system() const { return model.system(); }

    DenseURef eigenvalues();
    DenseURef eigenvectors();

    ArrayXd calc_dos(ArrayXf energies, float broadening);
    ArrayXd calc_ldos(float energy, float broadening, sub_id sublattice = -1);

protected:
    /// Create a new SolverStrategy object for this Hamiltonian
    virtual std::unique_ptr<SolverStrategy>
        create_strategy_for(const std::shared_ptr<const Hamiltonian>&) const = 0;

protected:
    bool is_solved = false;
    Model model;
    std::unique_ptr<SolverStrategy> strategy;
    Chrono calculation_timer; ///< last calculation time
};

} // namespace tbm
