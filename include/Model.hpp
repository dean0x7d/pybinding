#pragma once
#include "system/System.hpp"
#include "system/Lattice.hpp"
#include "system/Shape.hpp"
#include "system/Symmetry.hpp"
#include "system/SystemModifiers.hpp"
#include "hamiltonian/HamiltonianModifiers.hpp"
#include "solver/Solver.hpp"
#include "result/Result.hpp"
#include <string>

namespace tbm {

class Hamiltonian;

class Model {
public: // set parameters
    /// (Required) Set the crystal lattice specification
    void set_lattice(const std::shared_ptr<Lattice>& lattice);
    /// (Required) Set the shape of the system
    void set_shape(const std::shared_ptr<Shape>& lattice);
    /// (Optional)
    void set_symmetry(const std::shared_ptr<Symmetry>& symmetry);

    /// (Optional)
    void add_site_state_modifier(const std::shared_ptr<SiteStateModifier>& m);
    void add_position_modifier(const std::shared_ptr<PositionModifier>& m);
    void add_onsite_modifier(const std::shared_ptr<OnsiteModifier>& m);
    void add_hopping_modifier(const std::shared_ptr<HoppingModifier>& m);

    /// Set the matrix eigensolver
    void set_solver(const std::shared_ptr<SolverFactory>& solver_factory);

    /// (Required for periodic systems)
    void set_wave_vector(const Cartesian& k);

public: // get parameters
    std::shared_ptr<const Lattice> lattice() const { return _lattice; }
    std::shared_ptr<const Shape> shape() const { return _shape; }
    std::shared_ptr<const Symmetry> symmetry() const { return _symmetry; }
    
public: // get results
    std::shared_ptr<const System> system() const;
    std::shared_ptr<const Hamiltonian> hamiltonian() const;
    std::shared_ptr<Solver> solver() const;

    /// Accept a Result object that will calculate something.
    void calculate(Result& result);

public: // get information
    /// Report of the last build operation: system and Hamiltonian
    std::string build_report();
    /// Report of the last compute operation: eigensolver and/or Green's function
    std::string compute_report(bool shortform = false);

public:
    void clear_symmetry() { _symmetry.reset(); }
    void clear_system_modifiers() { system_modifiers.clear(); }
    void clear_hamiltonian_modifiers() { hamiltonian_modifiers.clear(); }
    void clear_all_modifiers() { clear_system_modifiers(); clear_hamiltonian_modifiers(); }
    void clear_solver() { solver_factory.reset(); _solver.reset(); }

private:
    std::shared_ptr<const Lattice> _lattice; ///< crystal lattice specification
    mutable std::shared_ptr<const Shape> _shape; ///< defines the shape of the system
    std::shared_ptr<const Symmetry> _symmetry;

    SystemModifiers system_modifiers;
    HamiltonianModifiers hamiltonian_modifiers;

    Cartesian wave_vector = Cartesian::Zero();

    mutable std::shared_ptr<const System> _system; ///< holds system data: atom coordinates and hoppings
    mutable std::shared_ptr<const Hamiltonian> _hamiltonian; ///< the Hamiltonian matrix

    std::shared_ptr<const SolverFactory> solver_factory;
    mutable std::shared_ptr<Solver> _solver; ///< eigensolver
};

} // namespace tbm
