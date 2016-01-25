#pragma once
#include "system/System.hpp"
#include "system/Lattice.hpp"
#include "system/Shape.hpp"
#include "system/Symmetry.hpp"
#include "system/SystemModifiers.hpp"
#include "system/Lead.hpp"
#include "hamiltonian/HamiltonianModifiers.hpp"

#include "utils/Chrono.hpp"

#include <string>
#include <vector>

namespace tbm {

class Hamiltonian;

class Model {
public:
    Model(Lattice const& lattice) : lattice(lattice) {}

public: // set parameters
    void set_primitive(Primitive primitive);
    void set_shape(Shape const& shape);
    void set_symmetry(Symmetry const& symmetry);

    void attach_lead(int direction, Shape const& shape);

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
    std::shared_ptr<System const> const& system() const;
    std::shared_ptr<Hamiltonian const> const& hamiltonian() const;

public: // get information
    /// Report of the last build operation: system and Hamiltonian
    std::string report();

public:
    void clear_system_modifiers() { system_modifiers.clear(); }
    void clear_hamiltonian_modifiers() { hamiltonian_modifiers.clear(); }
    void clear_all_modifiers() { clear_system_modifiers(); clear_hamiltonian_modifiers(); }

private:
    std::shared_ptr<System> make_system() const;
    std::shared_ptr<Hamiltonian> make_hamiltonian() const;

private:
    Lattice lattice;
    Primitive primitive;
    Shape shape;
    Symmetry symmetry;
    Cartesian wave_vector = Cartesian::Zero();

    Leads leads;

    SystemModifiers system_modifiers;
    HamiltonianModifiers hamiltonian_modifiers;

    mutable std::shared_ptr<System const> _system;
    mutable std::shared_ptr<Hamiltonian const> _hamiltonian;
    mutable Chrono system_build_time;
    mutable Chrono hamiltonian_build_time;
};

} // namespace tbm
