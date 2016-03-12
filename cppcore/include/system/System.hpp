#pragma once
#include "system/Lattice.hpp"
#include "system/Lead.hpp"
#include "system/Generators.hpp"

#include "numeric/dense.hpp"
#include "numeric/sparse.hpp"

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
    struct Port;

    Lattice lattice;
    CartesianArray positions;
    ArrayX<sub_id> sublattices;
    SparseMatrixX<hop_id> hoppings;
    std::vector<Boundary> boundaries;
    std::vector<Port> ports;
    bool has_unbalanced_hoppings = false; ///< some sites have a lot more hopping than others

    System(Lattice const& lattice) : lattice(lattice) {}
    System(Foundation const& foundation, Symmetry const& symmetry, Leads const& leads,
           HoppingGenerators const& hopping_generators);

    int num_sites() const { return positions.size(); }

    /// Find the index of the site nearest to the given position. Optional: filter by sublattice.
    int find_nearest(Cartesian position, sub_id sublattice = -1) const;
};

/**
 Stores sites that belong to a system boundary
 */
struct System::Boundary {
    SparseMatrixX<hop_id> hoppings;
    Cartesian shift; ///< shift length (periodic boundary condition)
};

struct System::Port {
    Cartesian shift; ///< periodic shift between two lead unit cells
    std::vector<int> indices; ///< map from lead Hamiltonian indices to main system indices
    SparseMatrixX<hop_id> inner_hoppings; ///< hoppings within the lead
    SparseMatrixX<hop_id> outer_hoppings; ///< periodic hoppings

    Port() = default;
    Port(Foundation const& foundation, HamiltonianIndices const& indices, Lead const& lead);

    /// Return the lead index corresponding to the main system Hamiltonian index
    int lead_index(int system_index) const {
        auto const it = std::find(indices.begin(), indices.end(), system_index);
        if (it == indices.end()) {
            return -1;
        } else {
            return static_cast<int>(it - indices.begin());
        }
    }
};

namespace detail {
    void populate_system(System& system, Foundation const& foundation,
                         HamiltonianIndices const& indices);
    void populate_boundaries(System& system, Foundation const& foundation,
                             HamiltonianIndices const& indices, Symmetry const& symmetry);
    void add_extra_hoppings(System& system, HoppingGenerator const& gen);
} // namespace detail

/**
 Return the number of non-zeros in each row of the sparse matrix

 The input matrix must be compressed and triangular in the System::hoppings format.
 The result is used to reserve space for a Hamiltonian matrix.
 */
ArrayXi nonzeros_per_row(SparseMatrixX<hop_id> const& hoppings, bool has_onsite_energy = false);

} // namespace tbm
