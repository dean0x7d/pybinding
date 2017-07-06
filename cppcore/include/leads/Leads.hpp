#pragma once
#include "leads/Spec.hpp"
#include "leads/Structure.hpp"
#include "leads/HamiltonianPair.hpp"

namespace cpb {

/**
 Full description of a single lead
 */
class Lead {
    leads::Spec specification;
    leads::Structure structure;
    leads::HamiltonianPair hamiltonian;

public:
    Lead(leads::Spec const& spec, leads::Structure const& ls, leads::HamiltonianPair const& lh)
        : specification(spec), structure(ls), hamiltonian(lh) {}

    leads::Spec const& spec() const { return specification; }
    std::vector<int> const& indices() const { return structure.indices; }
    System const& system() const { return structure.system; }
    Hamiltonian const& h0() const { return hamiltonian.h0; }
    Hamiltonian const& h1() const { return hamiltonian.h1; }
};

/**
 Container for all leads of a model
 */
class Leads {
    std::vector<leads::Spec> specs;
    std::vector<leads::Structure> structures;
    std::vector<leads::HamiltonianPair> hamiltonians;

public:
    /// The total number of leads
    int size() const { return static_cast<int>(specs.size()); }

    /// Description of lead number `i`
    Lead operator[](size_t i) const { return {specs.at(i), structures.at(i), hamiltonians.at(i)}; }

    /// Add a lead specified by `direction` and `shape`
    void add(int direction, Shape const& shape) { specs.emplace_back(direction, shape); }

    /// Modify the `foundation` so that all leads can be attached
    void create_attachment_area(Foundation& foundation) const;

    /// Create the structure of each lead
    void make_structure(Foundation const& foundation);

    /// Create a Hamiltonian pair for each lead
    void make_hamiltonian(Lattice const& lattice, HamiltonianModifiers const& modifiers,
                          bool is_double, bool is_complex);

    /// Clear any existing structural data, implies clearing Hamiltonian
    void clear_structure();
    /// Clear Hamiltonian, but leave structural data untouched
    void clear_hamiltonian();
};

} // namespace cpb
