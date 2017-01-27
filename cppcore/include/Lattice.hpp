#pragma once
#include "numeric/dense.hpp"
#include <vector>
#include <string>
#include <unordered_map>

namespace cpb {

// TODO: replace with proper string_view
using string_view = std::string const&;

/// Sublattice and hopping ID data types
using sub_id = std::int8_t;
using hop_id = std::int8_t;

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
    Cartesian position; ///< relative to lattice origin
    sub_id alias; ///< in case two sublattices at different positions need to have the same ID
    std::vector<Hopping> hoppings; ///< hoppings from this sublattice

    void add_hopping(Index3D relative_index, sub_id to_sublattice, hop_id id, bool is_conjugate);
};

/**
 Helper class for passing sublattice information to modifier functions
 */
struct SubIdRef {
    ArrayX<sub_id> const& ids;
    std::unordered_map<std::string, sub_id> const& name_map;
};

/**
 Helper class for passing hopping information to modifier functions
 */
struct HopIdRef {
    ArrayX<hop_id> const& ids;
    std::unordered_map<std::string, hop_id> const& name_map;
};

/**
 Crystal lattice specification
 */
class Lattice {
public:
    using Vectors = std::vector<Cartesian>;

    struct Sites {
        std::vector<Sublattice> structure; ///< positional data and aliases
        std::vector<MatrixXcd> energy; ///< onsite energy indexed by sub_id
        std::unordered_map<std::string, sub_id> id; ///< map from friendly name to numeric ID

        sub_id id_lookup(std::string const& name) const;
    };

    struct Hoppings {
        std::vector<Hopping> structure; ///< which sites are connected by hoppings
        std::vector<MatrixXcd> energy; ///< unique energies indexed by hop_id
        std::unordered_map<std::string, hop_id> id; ///< map from friendly name to numeric ID

        hop_id id_lookup(std::string const& name) const;
    };

public:
    Lattice(Cartesian a1, Cartesian a2 = {0, 0, 0}, Cartesian a3 = {0, 0, 0});
    Lattice(Vectors v, Sites s, Hoppings h)
        : vectors(std::move(v)), sites(std::move(s)), hoppings(std::move(h)) {}

    /// Create a new sublattice
    void add_sublattice(string_view name, Cartesian position, double onsite_energy = .0);
    void add_sublattice(string_view name, Cartesian position, VectorXd const& onsite_energy);
    void add_sublattice(string_view name, Cartesian position, MatrixXcd const& onsite_energy);

    /// Create a sublattice which an alias for an existing lattice at a different position
    void add_alias(string_view name, string_view original, Cartesian position);

    /// Associate a name with a hopping energy, but don't connect any sites
    void register_hopping_energy(string_view name, std::complex<double> energy);
    void register_hopping_energy(string_view name, MatrixXcd const& energy);

    /// Connect sites with an already registered hopping name/energy
    void add_hopping(Index3D relative_index, string_view from_sub, string_view to_sub,
                     string_view hopping);

    /// Connect sites with an anonymous hopping energy
    void add_hopping(Index3D relative_index, string_view from_sub, string_view to_sub,
                     std::complex<double> energy);
    void add_hopping(Index3D relative_index, string_view from_sub, string_view to_sub,
                     MatrixXcd const& energy);

public: // getters and setters
    /// The primitive vectors that define the lattice
    Vectors const& get_vectors() const { return vectors; }
    /// All the sites that belong to the primitive cell
    Sites const& get_sites() const { return sites; }
    /// Connections inside the unit cell or to a neighbor
    Hoppings const& get_hoppings() const { return hoppings; }

    /// The lattice offset value is added to the positions of all the sublattices
    Cartesian get_offset() const { return offset; }
    /// Set the lattice offset -- it must be within half the length of a primitive vector
    void set_offset(Cartesian position);
    /// Return a copy of this lattice with a different offset
    Lattice with_offset(Cartesian position) const;

    /// Any site which has less neighbors than this minimum will be considered as "dangling"
    int get_min_neighbors() const { return min_neighbors; }
    /// Set the minimum number of neighbors for any site
    void set_min_neighbors(int n) { min_neighbors = n; }
    /// Return a copy of this lattice with a different minimum neighbor count
    Lattice with_min_neighbors(int number) const;

public: // properties
    /// The dimensionality of the lattice
    int ndim() const { return static_cast<int>(vectors.size()); }
    /// Number of sublattices
    int nsub() const { return static_cast<int>(sites.structure.size()); }

    /// Get a single vector -- Expects: index < lattice.ndim()
    Cartesian vector(size_t index) const { return vectors[index]; }
    /// Onsite energy on the given sublattice
    double onsite_energy(sub_id id) const { return sites.energy[id].real()(0, 0); }
    /// Hopping energy for the specified ID
    std::complex<double> hopping_energy(hop_id id) const { return hoppings.energy[id](0, 0); }

    /// Get the maximum possible number of hoppings from any site of this lattice
    int max_hoppings() const;
    /// Does at least one sublattice have non-zero onsite energy?
    bool has_onsite_energy() const;
    /// Does at least one sublattice have multiple orbitals?
    bool has_multiple_orbitals() const;
    /// Is at least one hopping complex?
    bool has_complex_hoppings() const;

    /// Convenient way to access sublattices
    Sublattice const& operator[](size_t n) const { return sites.structure[n]; }

public: // utilities
    /// Calculate the spatial position of a unit cell or a sublattice site if specified
    Cartesian calc_position(Index3D index, std::string const& sublattice = "") const;

    /**
     Translate Cartesian `position` into lattice vector coordinates

     Return vector {n1, n2, n3} which satisfies `position = n1*a1 + n2*a2 + n3*a3`,
     where a1, a2 and a3 are lattice vectors. Note that the return vector is `float`.
     */
    Vector3f translate_coordinates(Cartesian position) const;

private:
    sub_id register_sublattice(string_view name);

private:
    Vectors vectors; ///< primitive vectors that define the lattice
    Sites sites; ///< all the sites that belong to the primitive cell
    Hoppings hoppings; ///< connections inside the unit cell or to a neighbor
    Cartesian offset = {0, 0, 0}; ///< global offset: sublattices are defined relative to this
    int min_neighbors = 1; ///< minimum number of neighbours required at each lattice site
};

} // end namespace cpb
