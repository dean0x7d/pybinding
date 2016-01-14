#include "system/System.hpp"

#include "system/Shape.hpp"
#include "system/Foundation.hpp"
#include "system/Symmetry.hpp"
#include "system/SystemModifiers.hpp"

#include "support/format.hpp"

namespace tbm {

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

std::unique_ptr<System> build_system(Foundation& foundation,
                                     SystemModifiers const& system_modifers,
                                     Symmetry const& symmetry) {
    auto system = cpp14::make_unique<System>(foundation.lattice);

    if (symmetry)
        symmetry.apply(foundation);

    if (!system_modifers.empty()) {
        ArrayX<sub_id> sublattices{foundation.num_sites};
        for (auto const& site : foundation) {
            sublattices[site.idx] = site.sublattice;
        }

        for (auto const& site_state_modifier : system_modifers.state) {
            site_state_modifier->apply(foundation.is_valid, foundation.positions, sublattices);
        }
        for (auto const& position_modifier : system_modifers.position) {
            position_modifier->apply(foundation.positions, sublattices);
        }
    }

    populate_body(*system, foundation);
    if (symmetry)
        populate_boundaries(*system, foundation, symmetry);

    if (system->num_sites() == 0) // sanity check
        throw std::runtime_error{"Sanity fail: the system was built with 0 lattice sites."};

    return system;
}

void populate_body(System& system, Foundation& foundation) {
    // count the number of valid sites and assign a Hamiltonian index to them
    auto const num_valid_sites = foundation.finalize();

    // allocate
    system.positions.resize(num_valid_sites);
    system.sublattices.resize(num_valid_sites);
    system.hoppings.resize(num_valid_sites, num_valid_sites);

    auto const reserve_nonzeros = (foundation.lattice.max_hoppings() * num_valid_sites) / 2;
    auto matrix_view = compressed_inserter(system.hoppings, reserve_nonzeros);

    // populate
    for (auto const& site : foundation) {
        auto const index = site.hamiltonian_index();
        if (index < 0)
            continue; // invalid site

        system.positions[index] = site.position();
        system.sublattices[index] = foundation.lattice[site.sublattice].alias;

        matrix_view.start_row(index);
        foundation.for_each_neighbour(site, [&](Site neighbour, Hopping const& hopping) {
            auto const neighbour_index = neighbour.hamiltonian_index();
            if (neighbour_index < 0)
                return; // invalid

            if (!hopping.is_conjugate) // only make half the matrix, other half is the conjugate
                matrix_view.insert(neighbour_index, hopping.id);
        });
    }
    matrix_view.compress();
}

void populate_boundaries(System& system, Foundation& foundation, Symmetry const& symmetry) {
    auto const num_valid_sites = foundation.finalize();

    // a boundary is added first to prevent copying of Eigen::SparseMatrix
    // --> revise when Eigen types become movable

    system.boundaries.emplace_back(system);
    for (const auto& translation : symmetry.translations(foundation)) {
        auto& boundary = system.boundaries.back();

        // preallocate data (overestimated)
        auto reserve_nonzeros = foundation.size_n * foundation.lattice.max_hoppings() / 2;
        for (int n = 0; n < translation.boundary.size(); ++n) {
            if (translation.boundary[n] < 0)
                reserve_nonzeros *= foundation.size[n];
        }

        boundary.shift = translation.shift_lenght;
        boundary.hoppings.resize(num_valid_sites, num_valid_sites);
        auto boundary_matrix_view = compressed_inserter(boundary.hoppings, reserve_nonzeros);

        // loop over all periodic boundary sites
        foundation.for_sites(translation.boundary, [&](Site site) {
            if (!site.is_valid())
                return;
            boundary_matrix_view.start_row(site.hamiltonian_index());

            // shift site by a translation unit
            auto shifted_site = site.shift(translation.shift_index);
            // and see if it has valid neighbours
            foundation.for_each_neighbour(shifted_site, [&](Site neighbour, Hopping const& hop) {
                auto const neighbour_index = neighbour.hamiltonian_index();
                if (neighbour_index < 0)
                    return; // invalid

                boundary_matrix_view.insert(neighbour_index, hop.id);
            });
        });
        boundary_matrix_view.compress();

        if (boundary.hoppings.nonZeros() > 0)
            system.boundaries.emplace_back(system);
    }
    system.boundaries.pop_back();
}

} // namespace tbm
