#include "system/Foundation.hpp"
#include "system/Shape.hpp"

namespace cpb { namespace detail {

std::pair<Index3D, Index3D> find_bounds(Shape const& shape, Lattice const& lattice) {
    Array3i lower_bound = Array3i::Constant(std::numeric_limits<int>::max());
    Array3i upper_bound = Array3i::Constant(std::numeric_limits<int>::min());
    for (auto const& point : shape.vertices) {
        // Translate Cartesian coordinates `point` into lattice vector coordinates `v`
        Array3i const v = lattice.translate_coordinates(point).cast<int>();
        lower_bound = (v < lower_bound).select(v, lower_bound);
        upper_bound = (v > upper_bound).select(v, upper_bound);
    }

    // Add +/- 1 padding to compensate for `cast<int>()` truncation
    auto const ndim = lattice.ndim();
    lower_bound.head(ndim) -= 1;
    upper_bound.head(ndim) += 1;

    return {lower_bound, upper_bound};
}

CartesianArray generate_positions(Cartesian origin, Index3D size, Lattice const& lattice) {
    // The nested loops look messy, but it's the fastest way to calculate all the positions
    // because the intermediate a, b, c positions are reused.
    auto const nsub = lattice.nsub();
    auto const num_sites = size.prod() * nsub;
    auto const unit_cell = lattice.optimized_unit_cell();

    auto positions = CartesianArray(num_sites);
    auto idx = 0;
    for (auto n = 0; n < nsub; ++n) {
        Cartesian ps = origin + unit_cell[n].position;
        for (auto c = 0; c < size[2]; ++c) {
            Cartesian pc = (c == 0) ? ps : ps + static_cast<float>(c) * lattice.vector(2);
            for (auto b = 0; b < size[1]; ++b) {
                Cartesian pb = (b == 0) ? pc : pc + static_cast<float>(b) * lattice.vector(1);
                for (auto a = 0; a < size[0]; ++a) {
                    Cartesian pa = pb + static_cast<float>(a) * lattice.vector(0);
                    positions[idx++] = pa;
                } // a
            } // b
        } // c
    } // n

    return positions;
}

ArrayXi count_neighbors(Foundation const& foundation) {
    ArrayXi neighbor_count(foundation.get_num_sites());

    auto const& unit_cell = foundation.get_optimized_unit_cell();
    auto const size = foundation.get_size().array();

    for (auto const& site : foundation) {
        auto const& sublattice = unit_cell[site.get_sublattice()];
        auto num_neighbors = static_cast<int>(sublattice.hoppings.size());

        // Reduce the neighbor count for sites on the edges
        for (auto const& hopping : sublattice.hoppings) {
            auto const index = (site.get_index() + hopping.relative_index).array();
            if (any_of(index < 0) || any_of(index >= size))
                num_neighbors -= 1;
        }

        neighbor_count[site.get_idx()] = num_neighbors;
    }

    return neighbor_count;
}

void clear_neighbors(Site& site, ArrayXi& neighbor_count, int min_neighbors) {
    if (neighbor_count[site.get_idx()] == 0)
        return;

    site.for_each_neighbour([&](Site neighbor, Hopping) {
        if (!neighbor.is_valid())
            return;

        auto const neighbor_idx = neighbor.get_idx();
        neighbor_count[neighbor_idx] -= 1;
        if (neighbor_count[neighbor_idx] < min_neighbors) {
            neighbor.set_valid(false);
            // recursive call... but it will not be very deep
            clear_neighbors(neighbor, neighbor_count, min_neighbors);
        }
    });

    neighbor_count[site.get_idx()] = 0;
}

ArrayX<sub_id> make_sublattice_ids(Foundation const& foundation) {
    ArrayX<sub_id> sublattice_ids(foundation.get_num_sites());
    for (auto const& site : foundation) {
        sublattice_ids[site.get_idx()] = static_cast<sub_id>(site.get_sublattice());
    }
    return sublattice_ids;
}

} // namespace detail

void remove_dangling(Foundation& foundation, int min_neighbors) {
    auto neighbor_count = detail::count_neighbors(foundation);
    for (auto& site : foundation) {
        if (!site.is_valid()) {
            detail::clear_neighbors(site, neighbor_count, min_neighbors);
        }
    }
}

Foundation::Foundation(Lattice const& lattice, Primitive const& primitive)
    : lattice(lattice),
      unit_cell(lattice.optimized_unit_cell()),
      bounds(-primitive.size.array() / 2, (primitive.size.array() - 1) / 2),
      size(primitive.size),
      nsub(lattice.nsub()),
      num_sites(size.prod() * nsub),
      positions(detail::generate_positions(lattice.calc_position(bounds.first), size, lattice)),
      is_valid(ArrayX<bool>::Constant(num_sites, true)) {}

Foundation::Foundation(Lattice const& lattice, Shape const& shape)
    : lattice(lattice),
      unit_cell(lattice.optimized_unit_cell()),
      bounds(detail::find_bounds(shape, lattice)),
      size((bounds.second - bounds.first) + Index3D::Ones()),
      nsub(lattice.nsub()),
      num_sites(size.prod() * nsub),
      positions(detail::generate_positions(lattice.calc_position(bounds.first), size, lattice)),
      is_valid(shape.contains(positions)) {
    remove_dangling(*this, lattice.get_min_neighbors());
}

FinalizedIndices::FinalizedIndices(Foundation const& foundation)
    : indices(ArrayXi::Constant(foundation.get_num_sites(), -1)),
      hopping_counts(ArrayXi::Zero(foundation.get_lattice().nhop())) {
    // Each sublattice block has the same initial number of sites (block_size),
    // but the number of final valid sites may differ.
    auto const nsub = foundation.get_num_sublattices();
    auto const block_size = foundation.get_size().prod();

    for (auto n = 0; n < nsub; ++n) {
        auto valid_sites_for_this_sublattice = 0;

        // Assign final indices to all valid sites
        auto const& is_valid = foundation.get_states();
        for (auto i = n * block_size; i < (n + 1) * block_size; ++i) {
            if (is_valid[i]) {
                indices[i] = total_valid_sites;
                ++total_valid_sites;
                ++valid_sites_for_this_sublattice;
            }
        }

        // Count the number of non-conjugate hoppings per family ID. This is
        // overestimated, i.e. it includes some invalid hoppings, but it's a
        // good quick estimate for memory reservation.
        auto const& unit_cell = foundation.get_optimized_unit_cell();
        for (auto const& hop : unit_cell[n].hoppings) {
            if (!hop.is_conjugate) {
                hopping_counts[hop.family_id] += valid_sites_for_this_sublattice;
            }
        }
    }
}

} // namespace cpb
