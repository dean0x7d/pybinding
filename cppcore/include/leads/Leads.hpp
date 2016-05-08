#pragma once
#include "leads/Spec.hpp"
#include "leads/Structure.hpp"
#include "leads/HamiltonianPair.hpp"

namespace tbm {

/**
 Full description of a single lead
 */
class Lead {
    leads::Structure structure;
    leads::HamiltonianPair hamiltonian;

public:
    Lead(leads::Structure const& ls, leads::HamiltonianPair const& lh)
        : structure(ls), hamiltonian(lh) {}

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
    Lead operator[](int i) const { return {structures.at(i), hamiltonians.at(i)}; }

    /// Add a lead specified by `direction` and `shape`
    void add(int direction, Shape const& shape) { specs.emplace_back(direction, shape); }

    /// Modify the `foundation` so that all leads can be attached
    void create_attachment_area(Foundation& foundation) const;

    /// Create the structure of each lead
    void make_structure(Foundation const& foundation, HamiltonianIndices const& indices);

    /// Create a Hamiltonian pair for each lead
    void make_hamiltonian(HamiltonianModifiers const& modifiers, bool is_double, bool is_complex);
};

} // namespace tbm
