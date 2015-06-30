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
 Stores the positions and base hoppings for all lattice sites.
 */
class System {
public:
    /// Stores sites that belong to a boundary
    struct Boundary {
        Boundary(System const& system) : system(system) {}

        std::pair<Cartesian, Cartesian> get_position_pair(int i, int j) const {
            return {system.positions[i], system.positions[j] - shift};
        }

        System const& system;
        SparseMatrixX<float> matrix;
        Cartesian shift; ///< shift length (periodic boundary condition)
        int max_elements_per_site;
    };

    /// Find the index of the site nearest to the given position. Optional: filter by sublattice.
    int find_nearest(Cartesian position, sub_id sublattice = -1) const;

    std::pair<Cartesian, Cartesian> get_position_pair(int i, int j) const {
        return {positions[i], positions[j]};
    }

    int num_sites() const { return positions.size(); }

    CartesianArray positions; ///< coordinates of all the lattice sites
    ArrayX<sub_id> sublattice; ///< sublattice indices of all the sites
    SparseMatrixX<float> matrix; ///< base hopping information
    std::vector<Boundary> boundaries; ///< boundary information
    int max_elements_per_site; ///< maximum number of Hamiltonian element at any site
};

std::unique_ptr<System> build_system(Lattice const&, Shape const&,
                                     SystemModifiers const&, Symmetry const*);
void populate_body(System&, Foundation&);
void populate_boundaries(System&, Foundation&, Symmetry const&);

} // namespace tbm
