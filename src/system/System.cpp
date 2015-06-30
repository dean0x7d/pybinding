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
        if (target_sublattice >= 0 && sublattice[i] != target_sublattice)
            continue; // only check the target sublattice (if any)

        auto const distance = (positions[i] - target_position).norm();
        if (distance < min_distance) {
            min_distance = distance;
            nearest_index = i;
        }
    }
    
    return nearest_index;
}

std::unique_ptr<System> build_system(Lattice const& lattice, Shape const& shape,
                                     SystemModifiers const& system_modifers,
                                     Symmetry const* symmetry) {
    auto system = cpp14::make_unique<System>();

    auto foundation = Foundation{lattice, shape};
    foundation.cut_down_to(shape);
    if (symmetry)
        foundation.cut_down_to(*symmetry);

    for (auto const& site_state_modifier : system_modifers.state)
        site_state_modifier->apply(foundation.is_valid, foundation.positions);
    for (auto const position_modifier : system_modifers.position)
        position_modifier->apply(foundation.positions);

    populate_body(*system, foundation);
    if (symmetry)
        populate_boundaries(*system, foundation, *symmetry);

    if (system->num_sites() == 0) // sanity check
        throw std::runtime_error{"Sanity fail: the system was built with 0 lattice sites."};

    return system;
}

void populate_body(System& system, Foundation& foundation) {
    // count the number of valid sites and assign a Hamiltonian index to them
    auto const num_valid_sites = foundation.finalize();

    // allocate
    system.max_elements_per_site = foundation.lattice.max_hoppings()
                                   + foundation.lattice.has_onsite_potential;
    system.positions.resize(num_valid_sites);
    system.sublattice.resize(num_valid_sites);
    system.matrix.resize(num_valid_sites, num_valid_sites);

    auto const reserve_nonzeros = (foundation.lattice.max_hoppings() * num_valid_sites) / 2
                                  + foundation.lattice.has_onsite_potential * num_valid_sites;
    auto matrix_view = compressed_inserter(system.matrix, reserve_nonzeros);

    // populate
    foundation.for_each_site([&](Site site) {
        auto const index = site.hamiltonian_index();
        if (index < 0)
            return; // invalid site

        system.positions[index] = site.position();
        system.sublattice[index] = foundation.lattice[site.sublattice].alias;

        matrix_view.start_row(index);
        if (foundation.lattice.has_onsite_potential)
            matrix_view.insert(index, foundation.lattice[site.sublattice].onsite);

        foundation.for_each_neighbour(site, [&](Site neighbour, Hopping const& hopping) {
            auto const neighbour_index = neighbour.hamiltonian_index();
            // this also makes sure that the neighbour is valid, i.e. 'neighbour_index != -1'
            if (neighbour_index > index && hopping.energy != 0)
                matrix_view.insert(neighbour_index, hopping.energy);
        });
    });
    matrix_view.compress();
}

void populate_boundaries(System& system, Foundation& foundation, Symmetry const& symmetry) {
    auto const num_valid_sites = foundation.finalize();
    auto symmetry_area = symmetry.build_for(foundation);

    // a boundary is added first to prevent copying of Eigen::SparseMatrix
    // --> revise when Eigen types become movable

    system.boundaries.emplace_back(system);
    for (const auto& translation : symmetry_area.translations) {
        auto& boundary = system.boundaries.back();

        // preallocate data (overestimated)
        auto reserve_nonzeros = foundation.size_n * foundation.lattice.max_hoppings() / 2;
        for (int n = 0; n < translation.boundary.size(); ++n) {
            if (translation.boundary[n] < 0)
                reserve_nonzeros *= foundation.size[n];
        }

        boundary.shift = translation.shift_lenght;
        boundary.matrix.resize(num_valid_sites, num_valid_sites);
        auto boundary_matrix_view = compressed_inserter(boundary.matrix, reserve_nonzeros);

        // loop over all periodic boundary sites
        foundation.for_sites(translation.boundary, [&](Site site) {
            if (!site.is_valid())
                return;
            boundary_matrix_view.start_row(site.hamiltonian_index());

            // shift site by a translation unit
            auto shifted_site = site.shift(translation.shift_index);
            // and see if it has valid neighbours
            foundation.for_each_neighbour(shifted_site, [&](Site neighbour, Hopping const& hop) {
                if (neighbour.hamiltonian_index() < 0)
                    return; // invalid

                boundary_matrix_view.insert(neighbour.hamiltonian_index(), hop.energy);
            });
        });
        boundary_matrix_view.compress();

        boundary.max_elements_per_site = foundation.lattice.max_hoppings();
        if (boundary.matrix.nonZeros() > 0)
            system.boundaries.emplace_back(system);
    }
    system.boundaries.pop_back();
}

} // namespace tbm
