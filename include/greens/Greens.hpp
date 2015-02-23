#pragma once
#include "support/dense.hpp"
#include "result/Result.hpp"
#include "utils/Chrono.hpp"
#include "utils/Log.hpp"
#include <memory>

namespace tbm {

class Hamiltonian;
template <typename scalar_t> class HamiltonianT;

/**
 Abstract base class for Green's function.
 */
class Greens {
public:
    virtual ~Greens() = default;

    /// Try to set the Hamiltonian - returns false if the given H is the wrong type for this Greens
    virtual bool set_hamiltonian(const std::shared_ptr<const Hamiltonian>& hamiltonian) = 0;

    /// Return the Green's function at (i,j) for the given energy range
    virtual ArrayXcf calculate(int i, int j, ArrayXf energy, float broadening) = 0;

    /// Get some information about what happened during the last calculation
    std::string report(bool shortform) const {
        return v_report(shortform) + " " + calculation_timer.str();
    }
    
    /// Accept a Result visitor
    virtual void accept(Result& results) { results.visit(this); }
    
protected:
    virtual std::string v_report(bool /*shortform*/) const { return ""; };
    
protected:
    Chrono calculation_timer; ///< last calculation time
};


/**
 Abstract base with type specialization.
 */
template<typename scalar_t>
class GreensT : public Greens {
public:
    virtual ~GreensT() { Log::d("~Greens<" + num::scalar_name<scalar_t>() + ">()"); }

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

    virtual ArrayXcf calculate(int i, int j, ArrayXf energy, float broadening) final {
        if (!hamiltonian)
            throw std::logic_error{"Greens::calculate(): Hamiltonian is not set."};

        auto size = hamiltonian->get_matrix().rows();
        if (i < 0 || i > size || j < 0 || j > size)
            throw std::logic_error{"Greens::calculate(i,j): invalid value for i or j."};

        // time the calculation
        calculation_timer.tic();
        // call vCalculate() from derived class
        auto g = v_calculate(i, j, energy, broadening);
        calculation_timer.toc();

        return g;
    }

protected:
    virtual ArrayXcf v_calculate(int i, int j, ArrayXf energy, float broadening) = 0;
    /// post-processing that may be defined by derived classes
    virtual void hamiltonian_changed() {};

protected:
    std::shared_ptr<const HamiltonianT<scalar_t>> hamiltonian; ///< pointer to the Hamiltonian to solve
};


/**
 The factory will produce the Greens class with the right scalar type.
 This is an abstract base factory. The derived factories will need to 
 implement the create_for(hamiltonian) method.
 */
class GreensFactory {
public:
    virtual ~GreensFactory() { Log::d("~GreensFactory()"); };

    /// Create a new Greens object for the given Hamiltonian
    virtual std::unique_ptr<Greens>
        create_for(const std::shared_ptr<const Hamiltonian>& hamiltonian) const = 0;
};

} // namespace tbm
