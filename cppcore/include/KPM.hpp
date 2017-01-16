#pragma once
#include "kpm/Strategy.hpp"

namespace cpb {

/**
 Kernel Polynomial Method calculation interface

 Internally it uses a kpm::Strategy with the scalar of the given Hamiltonian.
 Don't create it directly -- use the `make_kpm<Strategy>()` helper function.
 */
class KPM {
    using MakeStrategy = std::function<std::unique_ptr<kpm::Strategy>(Hamiltonian const&)>;

public:
    KPM(Model const& model, MakeStrategy const& make_strategy);

    void set_model(Model const&);
    Model const& get_model() const { return model; }
    std::shared_ptr<System const> system() const { return model.system(); }

    ArrayXcd calc_greens(int row, int col, ArrayXd const& energy, double broadening) const;
    std::vector<ArrayXcd> calc_greens_vector(idx_t row, std::vector<idx_t> const& cols,
                                             ArrayXd const& energy, double broadening) const;

    ArrayXd calc_ldos(ArrayXd const& energy, double broadening,
                      Cartesian position, std::string const& sublattice = "") const;

    ArrayXd calc_dos(ArrayXd const& energy, double broadening) const;

    /// Get some information about what happened during the last calculation
    std::string report(bool shortform) const;
    kpm::Stats const& get_stats() const;

private:
    Model model;
    MakeStrategy make_strategy;
    std::unique_ptr<kpm::Strategy> strategy;
    mutable Chrono calculation_timer; ///< last calculation time
};

/**
 Helper function for creating a KPM object

 For example::

     // Use default CPU strategy
     auto kpm = make_kpm(model);
     // Use any other strategy
     auto kpm_cuda = make_kpm<CudaStrategy>(model);
 */
template<template<class> class Strategy = kpm::DefaultStrategy,
         class Config = typename Strategy<float>::Config>
KPM make_kpm(Model const& model, Config const& config = {}) {
    return {model, detail::MakeStrategy<kpm::Strategy, Strategy>(config)};
}

} // namespace cpb
