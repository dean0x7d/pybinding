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
    ArrayXi neighbor_count(foundation.size());

    auto const& unit_cell = foundation.get_optimized_unit_cell();
    auto const spatial_size = foundation.get_spatial_size().array();

    for (auto const& site : foundation) {
        auto const& sublattice = unit_cell[site.get_sub_idx()];
        auto num_neighbors = static_cast<storage_idx_t>(sublattice.hoppings.size());

        // Reduce the neighbor count for sites on the edges
        for (auto const& hopping : sublattice.hoppings) {
            auto const index = Array3i(site.get_spatial_idx() + hopping.relative_index);
            if ((index < 0).any() || (index >= spatial_size).any()) {
                num_neighbors -= 1;
            }
        }

        neighbor_count[site.get_flat_idx()] = num_neighbors;
    }

    return neighbor_count;
}

void clear_neighbors(Site& site, ArrayXi& neighbor_count, int min_neighbors) {
    if (neighbor_count[site.get_flat_idx()] == 0) { return; }

    site.for_each_neighbor([&](Site neighbor, Hopping) {
        if (!neighbor.is_valid()) { return; }

        auto const neighbor_idx = neighbor.get_flat_idx();
        neighbor_count[neighbor_idx] -= 1;
        if (neighbor_count[neighbor_idx] < min_neighbors) {
            neighbor.set_valid(false);
            // recursive call... but it will not be very deep
            clear_neighbors(neighbor, neighbor_count, min_neighbors);
        }
    });

    neighbor_count[site.get_flat_idx()] = 0;
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

FinalizedIndices::FinalizedIndices(ArrayXi i, ArrayXi h, idx_t n)
    : indices(std::move(i)), hopping_counts(std::move(h)), total_valid_sites(n) {}


Foundation::Foundation(Lattice const& lattice, Primitive const& primitive)
    : lattice(lattice),
      unit_cell(lattice.optimized_unit_cell()),
      bounds(-primitive.size.array() / 2, (primitive.size.array() - 1) / 2),
      spatial_size(primitive.size),
      sub_size(lattice.nsub()),
      positions(detail::generate_positions(lattice.calc_position(bounds.first), spatial_size, lattice)),
      is_valid(ArrayX<bool>::Constant(size(), true)) {}

Foundation::Foundation(Lattice const& lattice, Shape const& shape)
    : lattice(lattice),
      unit_cell(lattice.optimized_unit_cell()),
      bounds(detail::find_bounds(shape, lattice)),
      spatial_size((bounds.second - bounds.first) + Index3D::Ones()),
      sub_size(lattice.nsub()),
      positions(detail::generate_positions(lattice.calc_position(bounds.first), spatial_size, lattice)),
      is_valid(shape.contains(positions)) {
    remove_dangling(*this, lattice.get_min_neighbors());
}

FinalizedIndices const& Foundation::get_finalized_indices() const {
    if (finalized_indices) {
        return finalized_indices;
    }

    auto indices = ArrayXi::Constant(size(), -1).eval();
    auto hopping_counts = ArrayXi::Zero(lattice.nhop()).eval();
    auto total_valid_sites = storage_idx_t{0};

    // Each sublattice block has the same initial number of sites (block_size),
    // but the number of final valid sites may differ.
    auto const block_size = spatial_size.prod();

    for (auto n = 0; n < sub_size; ++n) {
        auto valid_sites_for_this_sublattice = 0;

        // Assign final indices to all valid sites
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
        for (auto const& hop : unit_cell[n].hoppings) {
            if (!hop.is_conjugate) {
                hopping_counts[hop.family_id.value()] += valid_sites_for_this_sublattice;
            }
        }
    }

    finalized_indices = {std::move(indices), std::move(hopping_counts), total_valid_sites};
    return finalized_indices;
}

} // namespace cpb
