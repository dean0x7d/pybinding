#pragma once
#include "Model.hpp"
#include "hamiltonian/Hamiltonian.hpp"

#include "utils/Chrono.hpp"
#include "numeric/dense.hpp"
#include "detail/strategy.hpp"
#include "detail/thread.hpp"

namespace cpb {

/**
 Abstract base class for Green's function strategy.
 */
class GreensStrategy {
public:
    virtual ~GreensStrategy() = default;

    /// Returns false if the given Hamiltonian is the wrong type for this GreensStrategy
    virtual bool change_hamiltonian(Hamiltonian const& h) = 0;
    /// Return the Green's function matrix element (row, col) for the given energy range
    virtual ArrayXcd calc(int row, int col, ArrayXd const& energy, double broadening) = 0;
    /// Return multiple Green's matrix elements for a single `row` and multiple `cols`
    virtual std::vector<ArrayXcd> calc_vector(int row, std::vector<int> const& cols,
                                              ArrayXd const& energy, double broadening) = 0;
    /// Get some information about what happened during the last calculation
    virtual std::string report(bool shortform = false) const = 0;
};

/**
 Green's function calculation interface

 Internally it uses a GreensStrategy with the scalar of the given Hamiltonian.
 */
class BaseGreens {
public:
    void set_model(Model const&);
    Model const& get_model() const { return model; }
    std::shared_ptr<System const> system() const { return model.system(); }

    ArrayXcd calc_greens(int row, int col, ArrayXd const& energy, double broadening) const;
    std::vector<ArrayXcd> calc_greens_vector(int row, std::vector<int> const& cols,
                                             ArrayXd const& energy, double broadening) const;

    ArrayXd calc_ldos(ArrayXd const& energy, double broadening,
                      Cartesian position, sub_id sublattice = -1) const;
    Deferred<ArrayXd> deferred_ldos(ArrayXd const& energy, double broadening,
                                    Cartesian position, sub_id sublattice = -1) const;

    /// Get some information about what happened during the last calculation
    std::string report(bool shortform) const;

protected:
    using MakeStrategy = std::function<std::unique_ptr<GreensStrategy>(Hamiltonian const&)>;
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

/**
 Return a strategy with the scalar type matching the given Hamiltonian
 */
template<template<class> class Strategy, class Config = typename Strategy<float>::Config>
std::unique_ptr<GreensStrategy> make_greens_strategy(Hamiltonian const& h, Config const& c = {}) {
    return detail::MakeStrategy<GreensStrategy, Strategy>(c)(h);
}

} // namespace cpb
