#include "system/Lattice.hpp"
using namespace tbm;


void Lattice::add_vector(const Cartesian& primitive_vector)
{
    if (vectors.size() >= 3)
        throw std::logic_error{"Lattices with more that 3 dimensions are not supported."};
    
    vectors.push_back(primitive_vector);
}

short Lattice::create_sublattice(const Cartesian& offset, float onsite_potential, short alias)
{
    const auto sublattice_id = static_cast<short>(sublattices.size());
    if (sublattice_id == std::numeric_limits<short>::max())
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

void Lattice::add_hopping(const Index3D& relative_index,
                          short from_sub, short to_sub, float hop_energy)
{
    // sanity checks
    if (from_sub == to_sub && relative_index == Index3D::Zero()) {
        throw std::logic_error{"Hoppings from/to the same sublattice must have a non-zero relative "
                          "index in at least one direction. Don't define onsite potential here."};
    }
    const auto num_sublattices = static_cast<short>(sublattices.size());
    if (from_sub >= num_sublattices || to_sub > num_sublattices)
        throw std::logic_error{"The specified sublattie does not exist."};

    // the other sublattice has an opposite relative index
    sublattices[from_sub].hoppings.push_back({relative_index, to_sub, hop_energy});
    sublattices[to_sub].hoppings.push_back({-relative_index, from_sub, hop_energy});
}

int Lattice::max_hoppings() const
{
    auto max_size = 0;
    for (const auto& sub : sublattices) {
        auto const size = static_cast<int>(sub.hoppings.size());
        if (size > max_size)
            max_size = size;
    }
    
    return max_size;
}
