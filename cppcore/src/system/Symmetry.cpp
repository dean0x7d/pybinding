#include "system/Symmetry.hpp"
#include "system/Lattice.hpp"
#include "system/Foundation.hpp"
using namespace tbm;

SymmetrySpec Translational::build_for(const Foundation& foundation) const
{
    auto s = SymmetrySpec{};
    s.left.setZero();
    s.right = foundation.size.array() - 1;
    s.middle.setZero();

    // see if we have periodicities in any of the lattice vector directions
    for (std::size_t n = 0; n < foundation.lattice.vectors.size(); ++n) {
        if (length[n] < 0)
            continue; // not periodic

        // number of lattice sites in one period length
        auto num_lattice_sites = static_cast<int>(std::round(
            length[n] / foundation.lattice.vectors[n].norm()
        ));
        if (num_lattice_sites == 0)
            num_lattice_sites = 1;

        // left and right borders of the periodic cell
        s.left[n] = (foundation.size[n] - num_lattice_sites) / 2;
        s.right[n] = s.left[n] + num_lattice_sites - 1;
        // length of the periodic cell
        s.middle[n] = num_lattice_sites;
    }

    // add translations in the hopping directions
    for (const auto& sublattice : foundation.lattice.sublattices) {
        for (const auto& hopping : sublattice.hoppings) {
            // only take the directions which have a non-negative period length
            Index3D direction = (length.array() >= 0).select(hopping.relative_index, 0);
            s.add_translation(direction, foundation.lattice);
        }
    }

    return s;
}

void SymmetrySpec::add_translation(const Index3D& direction, const Lattice& lattice)
{
    if (direction == Index3D::Zero())
        return; // not a valid translation
    
    // check if the direction already exists
    for (const auto& translation : translations) {
        if (translation.direction == direction || translation.direction == -direction)
            return;
    }
    
    Translation translation;
    translation.direction = direction;

    // border site indices of this translation
    translation.boundary.setConstant(-1);
    for (std::size_t n = 0; n < lattice.vectors.size(); ++n) {
        if (direction[n] > 0)
            translation.boundary[n] = left[n];
        else if (direction[n] < 0)
            translation.boundary[n] = right[n];
    }

    translation.shift_lenght.setZero();
    for (std::size_t n = 0; n < lattice.vectors.size(); ++n) {
        auto const shift = static_cast<float>(direction[n] * middle[n]);
        translation.shift_lenght += shift * lattice.vectors[n];
    }

    // translation shift in number of lattice sites
    translation.shift_index = direction.cwiseProduct(middle);

    translations.push_back(translation);
}

bool SymmetrySpec::contains(const Index3D& index) const
{
    return all_of(left.array() <= index.array()) && all_of(index.array() <= right.array());
}
