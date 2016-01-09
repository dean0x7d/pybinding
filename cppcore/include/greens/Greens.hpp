#pragma once
#include "Model.hpp"
#include "hamiltonian/Hamiltonian.hpp"

#include "detail/strategy.hpp"

#include "utils/Chrono.hpp"
#include "utils/Log.hpp"

#include "support/dense.hpp"
#include "support/thread.hpp"

namespace tbm {

/**
 Abstract base class for Green's function strategy.
 */
class GreensStrategy {
public:
    virtual ~GreensStrategy() = default;

    /// Try to set the Hamiltonian
    /// @return false if the given Hamiltonian is the wrong scalar type for this GreensStrategy
    virtual bool set_hamiltonian(const std::shared_ptr<const Hamiltonian>& hamiltonian) = 0;
    /// Return the Green's function at (i,j) for the given energy range
    virtual ArrayXcf calculate(int i, int j, ArrayXf energy, float broadening) = 0;
    /// Get some information about what happened during the last calculation
    virtual std::string report(bool shortform) const = 0;
};


/**
 Abstract base with type specialization.
 */
template<class scalar_t>
class GreensStrategyT : public GreensStrategy {
public:
    virtual ~GreensStrategyT() { Log::d("~GreensStrategy<" + num::scalar_name<scalar_t>() + ">()"); }

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

protected:
    /// post-processing that may be defined by derived classes
    virtual void hamiltonian_changed() {};

protected:
    std::shared_ptr<const HamiltonianT<scalar_t>> hamiltonian; ///< the Hamiltonian to solve
};

/**
 Green's function calculation interface.
 Internally it uses a GreensStrategy with the scalar of the given Hamiltonian.
 */
class BaseGreens {
    using MakeStrategy = std::function<std::unique_ptr<GreensStrategy>(Model const&)>;

public:
    void set_model(Model const&);
    Model const& get_model() const { return model; }
    std::shared_ptr<System const> system() const { return model.system(); }

    ArrayXcf calc_greens(int i, int j, ArrayXf energy, float broadening) const;
    ArrayXf calc_ldos(ArrayXf energy, float broadening,
                      Cartesian position, sub_id sublattice = -1) const;
    Deferred<ArrayXf> deferred_ldos(ArrayXf energy, float broadening,
                                    Cartesian position, sub_id sublattice = -1) const;

    /// Get some information about what happened during the last calculation
    std::string report(bool shortform) const;

protected:
    BaseGreens(Model const& model, MakeStrategy const& make_strategy);

private:
    Model model;
    MakeStrategy make_strategy;
    std::unique_ptr<GreensStrategy> strategy;
    mutable Chrono calculation_timer; ///< last calculation time
};

template<template<class> class Strategy>
class Greens : public BaseGreens {
    using Config = typename Strategy<float>::Config;
    using MakeStrategy = detail::MakeStrategy<GreensStrategy, Strategy>;

public:
    explicit Greens(Model const& model, Config const& config = {})
        : BaseGreens(model, MakeStrategy(config)) {}
};

} // namespace tbm
