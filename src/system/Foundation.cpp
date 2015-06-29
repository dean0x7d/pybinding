#include "system/Foundation.hpp"
#include "system/Shape.hpp"
#include "system/Symmetry.hpp"
using namespace tbm;

Foundation::Foundation(const Lattice& lattice, const Shape& shape)
    : size_n(static_cast<int>(lattice.sublattices.size())), lattice(lattice)
{
    Cartesian vector_length = shape.length_for(lattice);

    // from: length [nanometers] in each lattice unit vector direction
    // to:   size - number of lattice sites
    for (std::size_t i = 0; i < lattice.vectors.size(); ++i) {
        if (shape.has_nice_edges) {
            // integer number of lattice vectors, plus one site (fencepost error otherwise)
            size[i] = (int)std::ceil(vector_length[i] / lattice.vectors[i].norm()) + 1;
            // make sure it's an odd number, so that (size-1)/2 is an integer
            size[i] += !(size[i] % 2);
        }
        else {
            // primitive shape, just round
            size[i] = (int)std::round(vector_length[i] / lattice.vectors[i].norm());
            // make sure it's positive, non-zero
            if (size[i] <= 0)
                size[i] = 1;
        }
    }

    // the foundation is a Bravais lattice
    Cartesian width = Cartesian::Zero();
    for (std::size_t i = 0; i < lattice.vectors.size(); ++i)
        width += (size[i] - 1) * lattice.vectors[i];
    origin = shape.center() - width / 2;

    // The total number of site states also includes the sublattices
    num_sites = size.prod() * size_n;

    positions.resize(num_sites);
    for_each_site([&](Site site) {
        positions[site.i] = calculate_position(site);
    });

    is_valid = ArrayX<bool>::Constant(num_sites, true);
    neighbour_count = ArrayX<int16_t>::Zero(num_sites);
}

void Foundation::cut_down_to(const Shape& shape)
{
    // Visit each foundation site and invalidate those that are not within the shape
    for_each_site([&](Site site) {
        if (!shape.contains(site.position())) { // this check is the most expensive part of this loop
            invalidate(site); // it's not within the shape
        }
        else if (shape.has_nice_edges) {
            // Within, but we still need to count the number of valid neighbours
            for_each_neighbour(site, [&](Site neighbour, const Hopping&) {
                if (neighbour.is_valid())
                    site.add_neighbour();
            });

            // There may not be enough
            if (site.num_neighbours() < lattice.min_neighbours)
                invalidate(site);
        }
    });
}

void Foundation::cut_down_to(const Symmetry& symmetry)
{
    auto symmetry_area = symmetry.build_for(*this);

    for_each_site([&](Site site) {
        site.set_valid(site.is_valid() && symmetry_area.contains(site.index));
    });
}

Cartesian Foundation::calculate_position(const Site& site) const
{
    Cartesian position = origin;
    // + unit cell position (Bravais lattice)
    for (std::size_t i = 0; i < lattice.vectors.size(); ++i) {
        position += site.index[i] * lattice.vectors[i];
    }
    // + sublattice offset
    position += lattice[site.sublattice].offset;
    return position;
}

void Foundation::invalidate(Site& site)
{
    if (!site.is_valid())
        return;
    site.set_valid(false);

    // visit the neighbours and lower *their* neighbour count
    for_each_neighbour(site, [&](Site neighbour, const Hopping&) {
        if (!neighbour.is_valid() || neighbour.num_neighbours() <= 0)
            return; // skip invalid neighbours

        neighbour.remove_neighbour();
        // if this takes the neighbour below the min threshold, invalidate it as well
        if (neighbour.num_neighbours() < lattice.min_neighbours)
            invalidate(neighbour); // recursive call... but it will not be very deep
    });
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
