#include "system/Symmetry.hpp"
#include "system/Foundation.hpp"

namespace cpb {

TranslationalSymmetry::TranslationalSymmetry(float a1, float a2, float a3)
    : length(a1, a2, a3), enabled_directions(a1 >= 0, a2 >= 0, a3 >= 0) {}

bool SymmetryArea::contains(Index3D const& index) const {
    return all_of(left.array() <= index.array()) && all_of(index.array() <= right.array());
}

SymmetryArea TranslationalSymmetry::area(Foundation const& foundation) const {
    auto const& lattice = foundation.get_lattice();
    auto const size = static_cast<Array3i>(foundation.get_spatial_size());

    SymmetryArea a;
    a.left.setZero();
    a.right = size - 1;
    a.middle.setZero();

    // see if we have periodicities in any of the lattice vector directions
    for (auto i = 0; i < lattice.ndim(); ++i) {
        if (!enabled_directions[i]) {
            continue;
        }

        // number of lattice sites in one period length
        auto const num_sites = [&]{
            auto const n = static_cast<int>(std::round(length[i] / lattice.vector(i).norm()));
            return (n > 0) ? n : 1;
        }();

        // left and right borders of the periodic cell
        a.left[i] = (size[i] - num_sites) / 2;
        a.right[i] = a.left[i] + num_sites - 1;
        // length of the periodic cell
        a.middle[i] = num_sites;
    }

    return a;
}

std::vector<Translation> TranslationalSymmetry::translations(Foundation const& foundation) const {
    auto const& lattice = foundation.get_lattice();
    auto const symmetry_area = area(foundation);

    std::vector<Translation> translations;
    auto add_translation = [&](Index3D direction) {
        if (direction == Index3D::Zero()) {
            return; // not a valid translation
        }

        // check if the direction already exists
        for (auto const& t : translations) {
            if (t.direction == direction) {
                return;
            }
        }

        auto boundary_slice = SliceIndex3D();
        for (auto n = 0, size = lattice.ndim(); n < size; ++n) {
            if (direction[n] > 0)
                boundary_slice[n] = symmetry_area.left[n];
            else if (direction[n] < 0)
                boundary_slice[n] = symmetry_area.right[n];
        }

        auto const shift_index = direction.cwiseProduct(symmetry_area.middle);

        auto shift_length = Cartesian{0, 0, 0};
        for (auto n = 0, size = lattice.ndim(); n < size; ++n) {
            auto const shift = static_cast<float>(direction[n] * symmetry_area.middle[n]);
            shift_length += shift * lattice.vector(n);
        }

        translations.push_back({direction, boundary_slice, shift_index, shift_length});
    };

    auto const masks = detail::make_masks(enabled_directions, lattice.ndim());
    for (auto const& sublattice : lattice.optimized_unit_cell()) {
        for (auto const& hopping : sublattice.hoppings) {
            for (auto const& mask : masks) {
                add_translation(hopping.relative_index.cwiseProduct(mask));
            }
        }
    }

    return translations;
}

void TranslationalSymmetry::apply(Foundation& foundation) const {
    auto symmetry_area = area(foundation);

    for (auto& site : foundation) {
        site.set_valid(site.is_valid() && symmetry_area.contains(site.get_spatial_idx()));
    }
}

namespace detail {

std::vector<Index3D> make_masks(Vector3b enabled_directions, int ndim) {
    auto const dirs = [&]{
        auto d = Index3D{enabled_directions.cast<int>()};
        for (auto i = ndim; i < d.size(); ++i) {
            d[i] = 0;
        }
        return d;
    }();

    auto masks = std::vector<Index3D>();
    for (auto i = 0; i <= dirs[0]; ++i) {
        for (auto j = 0; j <= dirs[1]; ++j) {
            for (auto k = 0; k <= dirs[2]; ++k) {
                masks.push_back({i, j, k});
            }
        }
    }
    return masks;
}

} // namespace detail
} // namespace cpb
