#pragma once
#include "kpm/default/Compute.hpp"

namespace cpb {

/**
 Kernel Polynomial Method calculation interface
 */
class KPM {
public:
    KPM(Model const& model, kpm::Compute const& compute = kpm::DefaultCompute{},
        kpm::Config const& config = {});

    void set_model(Model const&);
    Model const& get_model() const { return model; }
    kpm::Core& get_core() { return core; }

    /// Get some information about what happened during the last calculation
    std::string report(bool shortform) const;

    ArrayXcd moments(idx_t num_moments, VectorXcd const& alpha, VectorXcd const& beta = {},
                     SparseMatrixXcd const& op = {}) const;

    /// LDOS at the given position and sublattice for the energy range and broadening
    ArrayXXdCM calc_ldos(ArrayXd const& energy, double broadening, Cartesian position,
                         string_view sublattice = "", bool reduce = true) const;
    /// LDOS for multiple positions determined by the given shape
    ArrayXXdCM calc_spatial_ldos(ArrayXd const& energy, double broadening, Shape const& shape,
                                 string_view sublattice = "") const;

    /// DOS for the given energy range and broadening
    ArrayXd calc_dos(ArrayXd const& energy, double broadening, idx_t num_random) const;

    /// Green's function matrix element (row, col) for the given energy range
    ArrayXcd calc_greens(idx_t row, idx_t col, ArrayXd const& energy, double broadening) const;
    /// Multiple Green's matrix elements for a single `row` and multiple `cols`
    std::vector<ArrayXcd> calc_greens_vector(idx_t row, std::vector<idx_t> const& cols,
                                             ArrayXd const& energy, double broadening) const;

    /// Kubo-Bastin conductivity in `direction` ("xx", "xy", etc.)
    ArrayXd calc_conductivity(ArrayXd const& chemical_potential, double broadening,
                              double temperature, string_view direction, idx_t num_random,
                              idx_t num_points) const;

private:
    Model model;
    mutable kpm::Core core;
    mutable Chrono calculation_timer; ///< last calculation time
};

} // namespace cpb
