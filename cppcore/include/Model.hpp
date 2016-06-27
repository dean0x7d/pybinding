#pragma once
#include "system/System.hpp"
#include "system/Lattice.hpp"
#include "system/Shape.hpp"
#include "system/Symmetry.hpp"
#include "system/SystemModifiers.hpp"
#include "system/Generators.hpp"
#include "leads/Leads.hpp"
#include "hamiltonian/Hamiltonian.hpp"
#include "hamiltonian/HamiltonianModifiers.hpp"

#include "utils/Chrono.hpp"

#include <string>
#include <vector>

namespace cpb {

class Model {
public:
    Model(Lattice const& lattice) : lattice(lattice) {}

public: // set parameters
    void set_primitive(Primitive primitive);
    void set_shape(Shape const& shape);
    void set_symmetry(TranslationalSymmetry const& s);

    void attach_lead(int direction, Shape const& shape);

    void add_site_state_modifier(SiteStateModifier const& m);
    void add_position_modifier(PositionModifier const& m);
    void add_onsite_modifier(OnsiteModifier const& m);
    void add_hopping_modifier(HoppingModifier const& m);

    void add_hopping_family(HoppingGenerator const& g);

    void set_wave_vector(Cartesian const& k);

public:
    /// Uses double precision values in the Hamiltonian matrix?
    bool is_double() const;
    /// Uses complex values in the Hamiltonian matrix?
    bool is_complex() const;

public: // get parameters
    Lattice const& get_lattice() const { return lattice; }
    Primitive const& get_primitive() const { return primitive; }
    Shape const& get_shape() const { return shape; }
    TranslationalSymmetry const& get_symmetry() const { return symmetry; }

    std::vector<SiteStateModifier> state_modifiers() const { return system_modifiers.state; }
    std::vector<PositionModifier> position_modifiers() const { return system_modifiers.position; }
    std::vector<OnsiteModifier> onsite_modifiers() const { return hamiltonian_modifiers.onsite; }
    std::vector<HoppingModifier> hopping_modifiers() const { return hamiltonian_modifiers.hopping; }

public: // get results
    std::shared_ptr<System const> const& system() const;
    Hamiltonian const& hamiltonian() const;
    /// Return all leads
    Leads const& leads() const;
    /// Return lead at index
    Lead lead(size_t i) const { return leads()[i]; }

public: // get information
    /// Report of the last build operation: system and Hamiltonian
    std::string report();

public:
    void clear_system_modifiers() { system_modifiers.clear(); }
    void clear_hamiltonian_modifiers() { hamiltonian_modifiers.clear(); }
    void clear_all_modifiers() { clear_system_modifiers(); clear_hamiltonian_modifiers(); }

private:
    std::shared_ptr<System> make_system() const;
    Hamiltonian make_hamiltonian() const;

    /// Clear any existing structural data, implies clearing Hamiltonian
    void clear_structure();
    /// Clear Hamiltonian, but leave structural data untouched
    void clear_hamiltonian();

private:
    Lattice lattice;
    Primitive primitive;
    Shape shape;
    TranslationalSymmetry symmetry;
    Cartesian wave_vector = {0, 0, 0};

    SystemModifiers system_modifiers;
    HamiltonianModifiers hamiltonian_modifiers;
    HoppingGenerators hopping_generators;

    mutable std::shared_ptr<System const> _system;
    mutable Hamiltonian _hamiltonian;
    mutable Leads _leads;
    mutable Chrono system_build_time;
    mutable Chrono hamiltonian_build_time;
};

} // namespace cpb
