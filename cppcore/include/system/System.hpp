#pragma once
#include "system/Lattice.hpp"

#include "support/dense.hpp"
#include "support/sparse.hpp"

#include <vector>
#include <memory>

namespace tbm {

class Shape;
class Foundation;
class Symmetry;
class SystemModifiers;

/**
 Stores the positions, sublattice and hopping IDs for all lattice sites.
 */
class System {
public:
    /// Stores sites that belong to a boundary
    struct Boundary {
        Boundary(System const& system) : system(system), lattice(system.lattice) {}

        std::pair<Cartesian, Cartesian> get_position_pair(int i, int j) const {
            return {system.positions[i], system.positions[j] - shift};
        }

        System const& system;
        Lattice const& lattice;

        SparseMatrixX<hop_id> hoppings;
        Cartesian shift; ///< shift length (periodic boundary condition)
    };

    System(Lattice const& lattice) : lattice{lattice} {}

    /// Find the index of the site nearest to the given position. Optional: filter by sublattice.
    int find_nearest(Cartesian position, sub_id sublattice = -1) const;

    std::pair<Cartesian, Cartesian> get_position_pair(int i, int j) const {
        return {positions[i], positions[j]};
    }

    int num_sites() const { return positions.size(); }

    Lattice lattice;
    CartesianArray positions;
    ArrayX<sub_id> sublattices;
    SparseMatrixX<hop_id> hoppings;
    std::vector<Boundary> boundaries;
};

std::unique_ptr<System> build_system(Foundation&, SystemModifiers const&, Symmetry const&);
void populate_body(System&, Foundation&);
void populate_boundaries(System&, Foundation&, Symmetry const&);

} // namespace tbm
