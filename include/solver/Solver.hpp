#pragma once
#include "support/dense.hpp"
#include "support/uref.hpp"
#include "utils/Chrono.hpp"
#include "utils/Log.hpp"
#include <memory>

namespace tbm {

class Result;
class Hamiltonian;
template <typename scalar_t> class HamiltonianT;

/**
 Abstract base class for an eigensolver
 */
class Solver {
public:
    virtual ~Solver() = default;

    /// Try to set the Hamiltonian - returns false if the given H is the wrong type for this Solver
    virtual bool set_hamiltonian(const std::shared_ptr<const Hamiltonian>& hamiltonian) = 0;
    
    /// Diagonalize the given Hamiltonian
    void solve();
    
    /// Get some information
    std::string report(bool shortform) const;
    /// Accept a Result visitor
    virtual void accept(Result& result);
    /// Clear data - reset to unsolved state
    virtual void clear() = 0;
    
public: // get functions
    virtual DenseURef eigenvalues() const = 0;
    virtual DenseURef eigenvectors() const = 0;

protected: // derived classes implement these
    virtual void v_solve() = 0;
    virtual std::string v_report(bool /*shortform*/) const { return ""; }

protected:
    bool is_solved = false;

private:
    Chrono solve_timer;
};


/**
 Abstract base with scalar type specialization
 */
template<typename scalar_t>
class SolverT : public Solver {
    using real_t = num::get_real_t<scalar_t>;
public:
    virtual ~SolverT() { Log::d("~Solver<" + num::scalar_name<scalar_t>() + ">()"); }
    
    virtual bool set_hamiltonian(const std::shared_ptr<const Hamiltonian>& ham) final {
        // check if it's compatible
        if (auto cast_ham = std::dynamic_pointer_cast<const HamiltonianT<scalar_t>>(ham)) {
            if (hamiltonian != cast_ham) {
                clear();
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
 The factory will produce the Solver class with the right scalar type.
 This is an abstract base factory. The derived factories will need to
 implement the create_for(hamiltonian) method.
 */
class SolverFactory {
public:
    virtual ~SolverFactory() = default;

    /// Create a new Solver object for this Hamiltonian
    virtual std::unique_ptr<Solver>
        create_for(const std::shared_ptr<const Hamiltonian>& hamiltonian) const = 0;
};

} // namespace tbm
