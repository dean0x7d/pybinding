#include "system/Lattice.hpp"

namespace tbm {

void Lattice::add_vector(Cartesian primitive_vector) {
    if (vectors.size() >= 3)
        throw std::logic_error{"Lattices with more that 3 dimensions are not supported."};
    
    vectors.push_back(primitive_vector);
}

sub_id Lattice::create_sublattice(Cartesian offset, float onsite_potential, sub_id alias) {
    auto const sublattice_id = static_cast<sub_id>(sublattices.size());
    if (sublattice_id == std::numeric_limits<sub_id>::max())
        throw std::logic_error{"Cannot create more sublattices: " + std::to_string(sublattice_id)};

    sublattices.push_back({
        offset,
        onsite_potential,
        (alias < 0) ? sublattice_id : alias,
        {} // create an empty slot for this sublattice's hoppings
    });
    
    if (onsite_potential != 0)
        has_onsite_potential = true;

    return sublattice_id;
}

void Lattice::add_hopping(Index3D const& relative_index, sub_id from_sub,
                          sub_id to_sub, float hop_energy) {
    // sanity checks
    if (from_sub == to_sub && relative_index == Index3D::Zero()) {
        throw std::logic_error{
            "Hoppings from/to the same sublattice must have a non-zero relative "
            "index in at least one direction. Don't define onsite potential here."
        };
    }

    auto const num_sublattices = static_cast<sub_id>(sublattices.size());
    if (from_sub >= num_sublattices || to_sub >= num_sublattices)
        throw std::logic_error{"The specified sublattice does not exist."};

    // the other sublattice has an opposite relative index
    sublattices[from_sub].hoppings.push_back({relative_index, to_sub, hop_energy});
    sublattices[to_sub].hoppings.push_back({-relative_index, from_sub, hop_energy});
}

int Lattice::max_hoppings() const {
    auto max_size = 0;
    for (auto& sub : sublattices) {
        auto const size = static_cast<int>(sub.hoppings.size());
        if (size > max_size)
            max_size = size;
    }
    
    return max_size;
}

} // namespace tbm
