#pragma once
#include "system/Lattice.hpp"

#include "support/dense.hpp"
#include "support/sparse.hpp"

#include <vector>
#include <memory>

namespace tbm {

class Foundation;
class HamiltonianIndices;
class Symmetry;

/**
 Stores the positions, sublattice and hopping IDs for all lattice sites.
 */
struct System {
    struct Boundary;

    Lattice lattice;
    CartesianArray positions;
    ArrayX<sub_id> sublattices;
    SparseMatrixX<hop_id> hoppings;
    std::vector<Boundary> boundaries;

    System(Lattice const& lattice) : lattice(lattice) {}
    System(Foundation const& foundation, Symmetry const& symmetry);

    std::pair<Cartesian, Cartesian> get_position_pair(int i, int j) const {
        return {positions[i], positions[j]};
    }

    int num_sites() const { return positions.size(); }

    /// Find the index of the site nearest to the given position. Optional: filter by sublattice.
    int find_nearest(Cartesian position, sub_id sublattice = -1) const;
};

/**
 Stores sites that belong to a system boundary
 */
struct System::Boundary {
    System const& system;
    Lattice const& lattice;
    SparseMatrixX<hop_id> hoppings;
    Cartesian shift; ///< shift length (periodic boundary condition)

    Boundary(System const& system) : system(system), lattice(system.lattice) {}

    std::pair<Cartesian, Cartesian> get_position_pair(int i, int j) const {
        return {system.positions[i], system.positions[j] - shift};
    }
};

namespace detail {
    void populate_system(System& system, Foundation const& foundation,
                         HamiltonianIndices const& indices);
    void populate_boundaries(System& system, Foundation const& foundation,
                             HamiltonianIndices const& indices, Symmetry const& symmetry);
} // namespace detail

} // namespace tbm
