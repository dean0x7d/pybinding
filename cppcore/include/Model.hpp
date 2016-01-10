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
public:
    Model(Lattice const& lattice) : lattice(lattice) {}

public: // set parameters
    void set_primitive(Primitive primitive);
    void set_shape(Shape const& shape);
    void set_symmetry(Symmetry const& symmetry);

    void add_site_state_modifier(SiteStateModifier const& m);
    void add_position_modifier(PositionModifier const& m);
    void add_onsite_modifier(OnsiteModifier const& m);
    void add_hopping_modifier(HoppingModifier const& m);

    void set_wave_vector(const Cartesian& k);

public: // get parameters
    Lattice const& get_lattice() const { return lattice; }
    Primitive const& get_primitive() const { return primitive; }
    Shape const& get_shape() const { return shape; }
    Symmetry const& get_symmetry() const { return symmetry; }

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
    void clear_system_modifiers() { system_modifiers.clear(); }
    void clear_hamiltonian_modifiers() { hamiltonian_modifiers.clear(); }
    void clear_all_modifiers() { clear_system_modifiers(); clear_hamiltonian_modifiers(); }

private:
    Lattice lattice;
    Primitive primitive;
    Shape shape;
    Symmetry symmetry;

    SystemModifiers system_modifiers;
    HamiltonianModifiers hamiltonian_modifiers;

    Cartesian wave_vector = Cartesian::Zero();

    mutable std::shared_ptr<const System> _system; ///< holds system data: atom coordinates and hoppings
    mutable std::shared_ptr<const Hamiltonian> _hamiltonian; ///< the Hamiltonian matrix

    mutable std::string build_report;
};

} // namespace tbm
