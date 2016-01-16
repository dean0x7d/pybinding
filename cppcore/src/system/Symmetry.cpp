#include "system/Symmetry.hpp"
#include "system/Lattice.hpp"
#include "system/Foundation.hpp"
using namespace tbm;

bool SymmetryArea::contains(Index3D const& index) const {
    return all_of(left.array() <= index.array()) && all_of(index.array() <= right.array());
}

SymmetryArea Symmetry::area(Foundation const& foundation) const {
    auto const& lattice = foundation.get_lattice();
    auto const size = static_cast<Array3i>(foundation.get_size());

    SymmetryArea a;
    a.left.setZero();
    a.right = size - 1;
    a.middle.setZero();

    // see if we have periodicities in any of the lattice vector directions
    for (auto n = 0u; n < lattice.vectors.size(); ++n) {
        if (length[n] < 0)
            continue; // not periodic

        // number of lattice sites in one period length
        auto num_lattice_sites = static_cast<int>(std::round(
            length[n] / lattice.vectors[n].norm()
        ));
        if (num_lattice_sites == 0)
            num_lattice_sites = 1;

        // left and right borders of the periodic cell
        a.left[n] = (size[n] - num_lattice_sites) / 2;
        a.right[n] = a.left[n] + num_lattice_sites - 1;
        // length of the periodic cell
        a.middle[n] = num_lattice_sites;
    }

    return a;
}

std::vector<Translation> Symmetry::translations(Foundation const& foundation) const {
    auto const& lattice = foundation.get_lattice();
    auto const symmetry_area = area(foundation);

    std::vector<Translation> translations;
    auto add_translation = [&](Index3D direction) {
        if (direction == Index3D::Zero())
            return; // not a valid translation

        // check if the direction already exists
        for (auto const& t : translations) {
            if (t.direction == direction || t.direction == -direction)
                return;
        }

        auto boundary_slice = SliceIndex3D();
        for (auto n = 0u; n < lattice.vectors.size(); ++n) {
            if (direction[n] > 0)
                boundary_slice[n] = symmetry_area.left[n];
            else if (direction[n] < 0)
                boundary_slice[n] = symmetry_area.right[n];
        }

        auto const shift_index = direction.cwiseProduct(symmetry_area.middle);

        Cartesian shift_lenght = Cartesian::Zero();
        for (auto n = 0u; n < lattice.vectors.size(); ++n) {
            auto const shift = static_cast<float>(direction[n] * symmetry_area.middle[n]);
            shift_lenght += shift * lattice.vectors[n];
        }

        translations.push_back({direction, boundary_slice, shift_index, shift_lenght});
    };

    // add translations in the hopping directions
    for (auto const& sublattice : lattice.sublattices) {
        for (auto const& hopping : sublattice.hoppings) {
            // only take the directions which have a non-negative period length
            add_translation((length.array() >= 0).select(hopping.relative_index, 0));
        }
    }

    return translations;
}

void Symmetry::apply(Foundation& foundation) const {
    auto symmetry_area = area(foundation);

    for (auto& site : foundation) {
        site.set_valid(site.is_valid() && symmetry_area.contains(site.get_index()));
    }
}
