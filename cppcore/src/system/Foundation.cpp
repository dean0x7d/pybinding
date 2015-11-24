#include "system/Foundation.hpp"
#include "system/Shape.hpp"
#include "system/Symmetry.hpp"
using namespace tbm;

Foundation::Foundation(Lattice const& lattice, Shape const& shape)
    : size(determine_size(lattice, shape)),
      size_n(static_cast<int>(lattice.sublattices.size())),
      lattice(lattice)
{
    num_sites = size.prod() * size_n;
    init_positions(shape.center());
    is_valid = shape.contains(positions);
    init_neighbor_count();

    if (shape.has_nice_edges)
        trim_edges();
}

Index3D Foundation::determine_size(Lattice const& lattice, Shape const& shape) {
    // TODO this function could be simpler
    Index3D size = Index3D::Constant(1);
    Cartesian vector_length = shape.length_for(lattice);

    // from: length [nanometers] in each lattice unit vector direction
    // to:   size - number of lattice sites
    for (auto i = 0u; i < lattice.vectors.size(); ++i) {
        if (shape.has_nice_edges) {
            // integer number of lattice vectors, plus one site (fencepost error otherwise)
            size[i] = static_cast<int>(
                std::ceil(vector_length[i] / lattice.vectors[i].norm()) + 1
            );
            // make sure it's an odd number, so that (size - 1) / 2 is an integer
            size[i] += !(size[i] % 2);
        } else {
            // primitive shape, just round
            size[i] = static_cast<int>(
                std::round(vector_length[i] / lattice.vectors[i].norm())
            );
            // make sure it's positive, non-zero
            if (size[i] <= 0)
                size[i] = 1;
        }
    }

    return size;
}

void Foundation::init_positions(Cartesian center) {
    auto origin = [&]{
        Cartesian width = Cartesian::Zero();
        for (auto i = 0u; i < lattice.vectors.size(); ++i) {
            width += static_cast<float>(size[i] - 1) * lattice.vectors[i];
        }
        return static_cast<Cartesian>(center - width / 2);
    }();

    positions.resize(num_sites);
    for_each_site([&](Site site) {
        positions[site.i] = calculate_position(site, origin);
    });
}

void Foundation::init_neighbor_count() {
    neighbour_count.resize(num_sites);

    for_each_site([&](Site site) {
        auto const& sublattice = lattice[site.sublattice];
        auto num_neighbors = static_cast<int16_t>(sublattice.hoppings.size());

        // Reduce the neighbor count for sites on the edges
        for (auto const& hopping : sublattice.hoppings) {
            auto const index = (site.index + hopping.relative_index).array();
            if (any_of(index < 0) || any_of(index >= size.array()))
                num_neighbors -= 1;
        }

        neighbour_count[site.i] = num_neighbors;
    });
}

void Foundation::trim_edges() {
    for_each_site([&](Site site) {
        if (!site.is_valid())
            clear_neighbors(site);
    });
}

void Foundation::apply(Symmetry const& symmetry) {
    auto symmetry_area = symmetry.build_for(*this);

    for_each_site([&](Site site) {
        site.set_valid(site.is_valid() && symmetry_area.contains(site.index));
    });
}

Cartesian Foundation::calculate_position(Site const& site, Cartesian origin) const {
    Cartesian position = origin;
    // + unit cell position (Bravais lattice)
    for (std::size_t i = 0; i < lattice.vectors.size(); ++i) {
        position += static_cast<float>(site.index[i]) * lattice.vectors[i];
    }
    // + sublattice offset
    position += lattice[site.sublattice].offset;
    return position;
}

void Foundation::clear_neighbors(Site& site) {
    if (site.num_neighbors() == 0)
        return;

    for_each_neighbour(site, [&](Site neighbour, Hopping) {
        if (!neighbour.is_valid())
            return;

        neighbour.set_neighbors(neighbour.num_neighbors() - 1);
        if (neighbour.num_neighbors() < lattice.min_neighbours) {
            neighbour.set_valid(false);
            clear_neighbors(neighbour); // recursive call... but it will not be very deep
        }
    });

    site.set_neighbors(0);
}

int Foundation::finalize()
{
    neighbour_count.resize(0); // not needed any more

    // Count the number of valid sites and assign a Hamiltonian index to them
    hamiltonian_indices = ArrayX<int32_t>::Constant(num_sites, -1);
    auto num_valid_sites = 0;
    for (int i = 0; i < num_sites; ++i) {
        if (is_valid[i])
            hamiltonian_indices[i] = num_valid_sites++;
    }

    return num_valid_sites;
}
