#pragma once
#include "support/dense.hpp"
#include <vector>

namespace tbm {

/// Sublattice ID data type
using sub_id = std::int8_t;

/**
 Hopping description
 */
struct Hopping {
    Index3D relative_index; ///< relative index between two unit cells - may be (0, 0, 0)
    sub_id to_sublattice; ///< destination sublattice ID
    float energy; ///< hopping energy
};

/**
 Sublattice description
 */
struct Sublattice {
    Cartesian offset; ///< position relative to the base lattice location
    float onsite; ///< onsite potential energy
    sub_id alias; ///< in case two sublattices at different positions need to have the same ID
    std::vector<Hopping> hoppings; ///< hoppings from this sublattice
};

/**
 Crystal lattice specification
 */
class Lattice {
public:
    /// Constructed with lattice vectors and a minimum number of neighbours
    Lattice(int min_neighbours = 1) : min_neighbours(min_neighbours) {}
    
    /// Primitive lattice vectors - at least one must be defined
    void add_vector(Cartesian primitive_vector);
    /// Creates a new sublattice at the given position offset and returns the sublattice ID
    sub_id create_sublattice(Cartesian offset, float onsite_potential = 0, sub_id alias = -1);
    /// Use the sublattice ID returned by CreateSublattice() in the from_/to_ sublattice fields 
    void add_hopping(Index3D const& relative_index, sub_id from_sublattice,
                     sub_id to_sublattice, float hopping_energy);
    
    /// Get the maximum possible number of hoppings from any site of this lattice
    int max_hoppings() const;

    // convenient way to access sublattices
    const Sublattice& operator[](int n) const {
        return sublattices[n];
    }
    
public:
    std::vector<Cartesian> vectors; ///< primitive vectors that define the lattice
    std::vector<Sublattice> sublattices; ///< all the atoms or orbitals that belong to the unit cell
    int min_neighbours = 1; ///< minimum number of neighbours required at each lattice site
    bool has_onsite_potential = false; ///< does at least one sublattice have an onsite potential
};

} // end namespace tbm
