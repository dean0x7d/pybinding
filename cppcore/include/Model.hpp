#pragma once
#include "system/System.hpp"
#include "system/Lattice.hpp"
#include "system/Shape.hpp"
#include "system/Symmetry.hpp"
#include "system/SystemModifiers.hpp"
#include "hamiltonian/HamiltonianModifiers.hpp"
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
    void add_site_state_modifier(SiteStateModifier const& m);
    void add_position_modifier(PositionModifier const& m);
    void add_onsite_modifier(OnsiteModifier const& m);
    void add_hopping_modifier(HoppingModifier const& m);

    /// (Required for periodic systems)
    void set_wave_vector(const Cartesian& k);

public: // get parameters
    std::shared_ptr<const Lattice> lattice() const { return _lattice; }
    std::shared_ptr<const Shape> shape() const { return _shape; }
    std::shared_ptr<const Symmetry> symmetry() const { return _symmetry; }

    std::vector<SiteStateModifier> state_modifiers() const { return system_modifiers.state; }
    std::vector<PositionModifier> position_modifiers() const { return system_modifiers.position; }
    std::vector<OnsiteModifier> onsite_modifiers() const { return hamiltonian_modifiers.onsite; }
    std::vector<HoppingModifier> hopping_modifiers() const { return hamiltonian_modifiers.hopping; }

public: // get results
    std::shared_ptr<const System> system() const;
    std::shared_ptr<const Hamiltonian> hamiltonian() const;

public: // get information
    /// Report of the last build operation: system and Hamiltonian
    std::string report();

public:
    void clear_symmetry() { _symmetry.reset(); }
    void clear_system_modifiers() { system_modifiers.clear(); }
    void clear_hamiltonian_modifiers() { hamiltonian_modifiers.clear(); }
    void clear_all_modifiers() { clear_system_modifiers(); clear_hamiltonian_modifiers(); }

private:
    std::shared_ptr<const Lattice> _lattice; ///< crystal lattice specification
    mutable std::shared_ptr<const Shape> _shape; ///< defines the shape of the system
    std::shared_ptr<const Symmetry> _symmetry;

    SystemModifiers system_modifiers;
    HamiltonianModifiers hamiltonian_modifiers;

    Cartesian wave_vector = Cartesian::Zero();

    mutable std::shared_ptr<const System> _system; ///< holds system data: atom coordinates and hoppings
    mutable std::shared_ptr<const Hamiltonian> _hamiltonian; ///< the Hamiltonian matrix

    mutable std::string build_report;
};

} // namespace tbm
