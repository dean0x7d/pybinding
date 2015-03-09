#include "system/System.hpp"
#include "system/Lattice.hpp"
#include "system/Shape.hpp"
#include "system/Foundation.hpp"
#include "system/Symmetry.hpp"
#include "system/SystemModifiers.hpp"
#include "utils/Chrono.hpp"
#include "support/format.hpp"
using namespace tbm;

System::System(const Lattice& lattice, const Shape& shape,
               const Symmetry* symmetry, const SystemModifiers& system_modifers)
{
    auto build_time = Chrono{};
    auto foundation = Foundation{lattice, shape, symmetry};

    for (const auto& site_state_modifier : system_modifers.state)
        site_state_modifier->apply(foundation.is_valid, foundation.positions);
    for (const auto& position_modifier : system_modifers.position)
        position_modifier->apply(foundation.positions);

    build_from(foundation);
    if (symmetry)
        build_boundaries_from(foundation, *symmetry);

    if (num_sites() == 0) // sanity check
        throw std::runtime_error{"Sanity fail: the system was built with 0 lattice sites."};

    report = fmt::format("Built system with {} lattice sites, {}",
                         fmt::with_suffix(num_sites()), build_time.toc());
}

int System::find_nearest(const Cartesian& target_position, short target_sublattice) const
{
    int nearest_index = 0;
    auto min_distance = (positions[0] - target_position).norm();

    // check the distance of every site from the target coordinates
    for (int i = 1; i < num_sites(); i++) {
        // only check the target sublattice (if any)
        if (target_sublattice >= 0 && sublattice[i] != target_sublattice)
            continue;
        
        const auto distance = (positions[i] - target_position).norm();
        if (distance < min_distance) {
            // new minimum has been found, save it
            min_distance = distance;
            nearest_index = i;
        }
    }
    
    return nearest_index;
}

void System::build_from(Foundation& foundation)
{
    // count the number of valid sites and assign a Hamiltonian index to them
    const auto num_valid_sites = foundation.finalize();

    // allocate base data
    max_elements_per_site = foundation.lattice.max_hoppings()
        + foundation.lattice.has_onsite_potential;
    positions.resize(num_valid_sites);
    sublattice.resize(num_valid_sites);
    matrix.resize(num_valid_sites, num_valid_sites);
    const auto reserve_nonzeros = (foundation.lattice.max_hoppings() * num_valid_sites) / 2
        + foundation.lattice.has_onsite_potential * num_valid_sites;
    auto matrix_view = compressed_inserter(matrix, reserve_nonzeros);
    
    // populate base data
    foundation.for_each_site([&](Site site) {
        const auto index = site.hamiltonian_index();
        if (index < 0)
            return; // invalid site

        positions[index] = site.position();
        sublattice[index] = foundation.lattice[site.sublattice].alias;

        matrix_view.start_row(index);
        if (foundation.lattice.has_onsite_potential)
            matrix_view.insert(index, foundation.lattice[site.sublattice].onsite);

        foundation.for_each_neighbour(site, [&](Site neighbour, const Hopping& hopping) {
            const auto neighbour_index = neighbour.hamiltonian_index();
            // this also makes sure that the neighbour is valid, i.e. 'neighbour_index != -1'
            if (neighbour_index > index && hopping.energy != 0)
                matrix_view.insert(neighbour_index, hopping.energy);
        });
    });
    matrix_view.compress();
}

void System::build_boundaries_from(Foundation& foundation, const Symmetry& symmetry)
{
    const auto num_valid_sites = foundation.finalize();
    auto symmetry_area = symmetry.build_for(foundation);

    boundaries.emplace_back(this);
    for (const auto& translation : symmetry_area.translations) {
        auto& boundary = boundaries.back();

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
            foundation.for_each_neighbour(shifted_site, [&](Site neighbour, const Hopping& hopping) {
                if (neighbour.hamiltonian_index() < 0)
                    return; // invalid

                boundary_matrix_view.insert(neighbour.hamiltonian_index(), hopping.energy);
            });
        });
        boundary_matrix_view.compress();

        boundary.max_elements_per_site = foundation.lattice.max_hoppings();
        if (boundary.matrix.nonZeros() > 0)
            boundaries.emplace_back(this);
    }
    boundaries.pop_back();
}
