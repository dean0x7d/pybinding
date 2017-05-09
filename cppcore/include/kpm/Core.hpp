#pragma once
#include "Model.hpp"
#include "hamiltonian/Hamiltonian.hpp"

#include "kpm/Bounds.hpp"
#include "kpm/Config.hpp"
#include "kpm/OptimizedHamiltonian.hpp"
#include "kpm/Starter.hpp"
#include "kpm/Moments.hpp"
#include "kpm/Stats.hpp"

#include "utils/Chrono.hpp"

namespace cpb { namespace kpm {

/**
 Does the actual work of computing KPM moments

 Different derived classes are optimized for specific hardware (CPU, GPU).
 */
class Compute {
public:
    class Interface {
    public:
        virtual ~Interface() = default;

        virtual void moments(MomentsRef, Starter const&, AlgorithmConfig const&,
                             OptimizedHamiltonian const&) const = 0;
    };

    template<class T>
    Compute(T x) : ptr(std::make_shared<T>(std::move(x))) {}

    Interface const* operator->() const { return ptr.get(); }

private:
    std::shared_ptr<Interface const> ptr;
};

/**
 Low-level KPM implementation

 No Model information (sublattice, positions, etc.), just the Hamiltonian matrix and indices.
 */
class Core {
public:
    explicit Core(Hamiltonian const& h, Compute const& compute, Config const& config = {});

    void set_hamiltonian(Hamiltonian const& h) ;
    Config const& get_config() const { return config; }
    Stats const& get_stats() const { return stats; }

    /// The KPM scaling factors `a` and `b`
    Scale<> scaling_factors() { return bounds.scaling_factors(); }

    /// Information about what happened during the last calculation
    std::string report(bool shortform = false) const;

    /// Return KPM moments in the form `mu_n = <beta|op Tn(H)|alpha>`
    ArrayXcd moments(idx_t num_moments, VectorXcd const& alpha, VectorXcd const& beta,
                     SparseMatrixXcd const& op);

    /// LDOS at the given Hamiltonian indices for the energy range and broadening
    ArrayXXdCM ldos(std::vector<idx_t> const& idx, ArrayXd const& energy, double broadening);
    /// DOS for the given energy range and broadening
    ArrayXd dos(ArrayXd const& energy, double broadening, idx_t num_random);

    /// Green's function matrix element (row, col) for the given energy range
    ArrayXcd greens(idx_t row, idx_t col, ArrayXd const& energy, double broadening);
    /// Multiple Green's matrix elements for a single `row` and multiple `cols`
    std::vector<ArrayXcd> greens_vector(idx_t row, std::vector<idx_t> const& cols,
                                        ArrayXd const& energy, double broadening);

    /// Kubo-Bastin conductivity in the directions defined by the `left` and `right` coordinates
    ArrayXcd conductivity(ArrayXf const& left_coords, ArrayXf const& right_coords,
                          ArrayXd const& chemical_potential, double broadening,
                          double temperature, idx_t num_random, idx_t num_points);

private:
    void timed_compute(MomentsRef, Starter const&, AlgorithmConfig const&);

private:
    Hamiltonian hamiltonian;
    Compute compute;
    Config config;
    Stats stats;

    Bounds bounds;
    OptimizedHamiltonian optimized_hamiltonian;
};

}} // namespace cpb::kpm
