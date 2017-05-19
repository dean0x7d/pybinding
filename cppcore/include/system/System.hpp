#pragma once
#include "Lattice.hpp"
#include "system/CompressedSublattices.hpp"
#include "system/HoppingBlocks.hpp"
#include "system/Generators.hpp"

#include "numeric/dense.hpp"
#include "numeric/sparse.hpp"

#include <vector>
#include <memory>

namespace cpb {

class Foundation;
class FinalizedIndices;
class TranslationalSymmetry;

struct Range { idx_t start, end; };

/**
 Stores the positions, sublattice and hopping IDs for all lattice sites.
 */
struct System {
    struct Boundary;

    Lattice lattice;
    CartesianArray positions;
    CompressedSublattices compressed_sublattices;
    HoppingBlocks hopping_blocks;
    std::vector<Boundary> boundaries;

    System(Lattice const& lattice) : lattice(lattice) {}
    System(Foundation const& foundation, TranslationalSymmetry const& symmetry,
           HoppingGenerators const& hopping_generators);

    /// The total number of lattice sites i.e. unique positions. Note that a single site may
    /// consist of several orbitals/spins which means that the size of the Hamiltonian matrix
    /// must be >= to the number of sites. See `System::hamiltonian_size()`.
    idx_t num_sites() const { return positions.size(); }

    /// The square matrix size required to hold all the Hamiltonian terms after taking into
    /// account the number of orbitals/spins at each lattice site.
    idx_t hamiltonian_size() const;

    /// Total number of non-zero values which need to be reserved for a Hamiltonian.
    /// This function takes multi-orbital hopping terms into account.
    idx_t hamiltonian_nnz() const;

    /// Translate the given System site index into its corresponding Hamiltonian indices
    ArrayXi to_hamiltonian_indices(idx_t system_index) const;

    /// The [start, end) range (pair of system indices) of all sites of a sublattice
    Range sublattice_range(string_view sublattice) const;

    /// Find the index of the site nearest to the given position. Optional: filter by sublattice.
    idx_t find_nearest(Cartesian position, string_view sublattice_name = "") const;

    /// Expand `positions` to `hamiltonian_size` by replicating site positions for each orbital
    CartesianArray expanded_positions() const;
};

/**
 Stores sites that belong to a system boundary
 */
struct System::Boundary {
    HoppingBlocks hopping_blocks;
    Cartesian shift; ///< shift length (periodic boundary condition)
};

namespace detail {
    void populate_system(System& system, Foundation const& foundation);
    void populate_boundaries(System& system, Foundation const& foundation,
                             TranslationalSymmetry const& symmetry);
    void add_extra_hoppings(System& system, HoppingGenerator const& gen);
} // namespace detail

} // namespace cpb
