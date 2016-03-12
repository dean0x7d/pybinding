#pragma once
#include "numeric/dense.hpp"
#include <vector>
#include <string>
#include <unordered_map>

namespace tbm {

/// Sublattice and hopping ID data types
using sub_id = std::int8_t;
using hop_id = std::int8_t;
using SubIdMap = std::unordered_map<std::string, sub_id>;
using HopIdMap = std::unordered_map<std::string, hop_id>;

/**
 Hopping description
 */
struct Hopping {
    Index3D relative_index; ///< relative index between two unit cells - may be (0, 0, 0)
    sub_id to_sublattice; ///< destination sublattice ID
    hop_id id; ///< hopping ID which points to the hopping energy -> Lattice::hopping_energies[id]
    bool is_conjugate; ///< true if this is an automatically added complex conjugate
};

/**
 Sublattice description
 */
struct Sublattice {
    Cartesian offset; ///< position relative to the base lattice location
    double onsite; ///< onsite energy
    sub_id alias; ///< in case two sublattices at different positions need to have the same ID
    std::vector<Hopping> hoppings; ///< hoppings from this sublattice

    void add_hopping(Index3D relative_index, sub_id to_sublattice, hop_id id, bool is_conjugate);
};

/**
 Helper class for passing sublattice information to modifier functions
 */
struct SubIdRef {
    ArrayX<sub_id> const& ids;
    SubIdMap const& name_map;
};

/**
 Helper class for passing hopping information to modifier functions
 */
struct HopIdRef {
    ArrayX<hop_id> const& ids;
    HopIdMap const& name_map;
};

/**
 Crystal lattice specification
 */
class Lattice {
public:
    Lattice(Cartesian v1, Cartesian v2 = {0, 0, 0}, Cartesian v3 = {0, 0, 0});

    /// Create a new sublattice and return it's ID
    sub_id add_sublattice(std::string const& name, Cartesian offset = {0, 0, 0},
                          double onsite_energy = .0, sub_id alias = -1);

    /// Connect sites via relative index/sublattices and return an ID for the given hopping energy
    hop_id add_hopping(Index3D relative_index, sub_id from_sublattice,
                       sub_id to_sublattice, std::complex<double> energy);

    /// Register just the energy and create an ID, but don't connect any sites
    hop_id register_hopping_energy(std::string const& name, std::complex<double> energy);

    /// Connect sites with already registered hopping ID/energy
    void add_registered_hopping(Index3D relative_index, sub_id from_sublattice,
                                sub_id to_sublattice, hop_id id);

    /// Get the maximum possible number of hoppings from any site of this lattice
    int max_hoppings() const;

    /// Convenient way to access sublattices
    const Sublattice& operator[](int n) const { return sublattices[n]; }

    /// Calculate the spatial position of a unit cell or a sublattice site if specified
    Cartesian calc_position(Index3D index, Cartesian origin = {0, 0, 0},
                            sub_id sublattice = -1) const;

public:
    std::vector<Cartesian> vectors; ///< primitive vectors that define the lattice
    std::vector<Sublattice> sublattices; ///< all the sites that belong to the primitive cell
    SubIdMap sub_name_map; ///< map from friendly sublattice name to numeric ID
    std::vector<std::complex<double>> hopping_energies; ///< unique energies indexed by hop_id
    HopIdMap hop_name_map; ///< map from friendly hopping name to numeric ID
    int min_neighbours = 1; ///< minimum number of neighbours required at each lattice site
    bool has_onsite_energy = false; ///< does at least one sublattice have non-zero onsite energy
    bool has_complex_hopping = false; ///< is at least one hopping complex
};

} // end namespace tbm
