#include "system/System.hpp"

#include "system/Shape.hpp"
#include "system/Foundation.hpp"
#include "system/Symmetry.hpp"

namespace tbm {

System::System(Foundation const& foundation, Symmetry const& symmetry)
    : lattice(foundation.get_lattice()) {
    auto const hamiltonian_indices = HamiltonianIndices(foundation);
    detail::populate_system(*this, foundation, hamiltonian_indices);
    if (symmetry)
        detail::populate_boundaries(*this, foundation, hamiltonian_indices, symmetry);

    if (num_sites() == 0)
        throw std::runtime_error{"Impossible system: built 0 lattice sites"};
}

int System::find_nearest(Cartesian target_position, sub_id target_sublattice) const {
    auto nearest_index = 0;
    auto min_distance = (positions[0] - target_position).norm();

    for (auto i = 1; i < num_sites(); ++i) {
        if (target_sublattice >= 0 && sublattices[i] != target_sublattice)
            continue; // only check the target sublattice (if any)

        auto const distance = (positions[i] - target_position).norm();
        if (distance < min_distance) {
            min_distance = distance;
            nearest_index = i;
        }
    }
    
    return nearest_index;
}

namespace detail {

void populate_system(System& system, Foundation const& foundation,
                     HamiltonianIndices const& hamiltonian_indices) {
    auto const size = hamiltonian_indices.size();
    system.positions.resize(size);
    system.sublattices.resize(size);
    system.hoppings.resize(size, size);

    auto const& lattice = foundation.get_lattice();
    auto const reserve_nonzeros = (lattice.max_hoppings() * size) / 2;
    auto matrix_view = compressed_inserter(system.hoppings, reserve_nonzeros);

    for (auto const& site : foundation) {
        auto const index = hamiltonian_indices[site];
        if (index < 0)
            continue; // invalid site

        system.positions[index] = site.get_position();
        system.sublattices[index] = lattice[site.get_sublattice()].alias;

        matrix_view.start_row(index);
        site.for_each_neighbour([&](Site neighbor, Hopping hopping) {
            auto const neighbor_index = hamiltonian_indices[neighbor];
            if (neighbor_index < 0)
                return; // invalid

            if (!hopping.is_conjugate) // only make half the matrix, other half is the conjugate
                matrix_view.insert(neighbor_index, hopping.id);
        });
    }
    matrix_view.compress();
}

void populate_boundaries(System& system, Foundation const& foundation,
                         HamiltonianIndices const& hamiltonian_indices,
                         Symmetry const& symmetry) {
    // a boundary is added first to prevent copying of Eigen::SparseMatrix
    // --> revise when Eigen types become movable

    auto const size = hamiltonian_indices.size();
    auto const& lattice = foundation.get_lattice();

    system.boundaries.emplace_back(system);
    for (const auto& translation : symmetry.translations(foundation)) {
        auto& boundary = system.boundaries.back();

        boundary.shift = translation.shift_lenght;
        boundary.hoppings.resize(size, size);

        // the reservation number is intentionally overestimated
        auto const reserve_nonzeros = [&]{
            auto nz = static_cast<int>(lattice.sublattices.size() * lattice.max_hoppings() / 2);
            for (auto i = 0; i < translation.boundary_slice.size(); ++i) {
                if (translation.boundary_slice[i].end < 0)
                    nz *= foundation.get_size()[i];
            }
            return nz;
        }();
        auto boundary_matrix_view = compressed_inserter(boundary.hoppings, reserve_nonzeros);

        for (auto const& site : foundation[translation.boundary_slice]) {
            if (!site.is_valid())
                continue;

            boundary_matrix_view.start_row(hamiltonian_indices[site]);

            auto const shifted_site = site.shifted(translation.shift_index);
            // the site is shifted to the opposite edge of the translation unit
            shifted_site.for_each_neighbour([&](Site neighbor, Hopping hopping) {
                auto const neighbor_index = hamiltonian_indices[neighbor];
                if (neighbor_index < 0)
                    return; // invalid

                boundary_matrix_view.insert(neighbor_index, hopping.id);
            });
        }
        boundary_matrix_view.compress();

        if (boundary.hoppings.nonZeros() > 0)
            system.boundaries.emplace_back(system);
    }
    system.boundaries.pop_back();
}

}} // namespace tbm::detail
