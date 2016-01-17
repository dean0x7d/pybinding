#include "system/Lattice.hpp"

namespace tbm {

Lattice::Lattice(Cartesian v1, Cartesian v2, Cartesian v3) {
    vectors.push_back(v1);
    if (v2 != Cartesian::Zero()) vectors.push_back(v2);
    if (v3 != Cartesian::Zero()) vectors.push_back(v3);

    vectors.shrink_to_fit();
}

sub_id Lattice::add_sublattice(Cartesian offset, double onsite_energy, sub_id alias) {
    auto const sublattice_id = static_cast<sub_id>(sublattices.size());
    if (sublattice_id == std::numeric_limits<sub_id>::max())
        throw std::logic_error{"Cannot create more sublattices: " + std::to_string(sublattice_id)};

    sublattices.push_back({
        offset,
        onsite_energy,
        (alias < 0) ? sublattice_id : alias,
        {} // create an empty slot for this sublattice's hoppings
    });
    
    if (onsite_energy != .0)
        has_onsite_energy = true;

    return sublattice_id;
}

hop_id Lattice::add_hopping(Index3D rel_index, sub_id from_sub, sub_id to_sub,
                            std::complex<double> energy) {
    auto const hopping_id = [&] {
        auto const it = std::find(hopping_energies.begin(), hopping_energies.end(), energy);
        if (it != hopping_energies.end())
            return static_cast<hop_id>(it - hopping_energies.begin());
        else
            return register_hopping_energy(energy);
    }();

    add_registered_hopping(rel_index, from_sub, to_sub, hopping_id);
    return hopping_id;
}

hop_id Lattice::register_hopping_energy(std::complex<double> energy) {
    auto const hopping_id = static_cast<hop_id>(hopping_energies.size());
    if (hopping_id == std::numeric_limits<hop_id>::max())
        throw std::logic_error{"Can't create any more hoppings: " + std::to_string(hopping_id)};

    if (energy.imag() != .0)
        has_complex_hopping = true;

    hopping_energies.push_back(energy);
    return hopping_id;
}

void Lattice::add_registered_hopping(Index3D relative_index, sub_id from_sub,
                                       sub_id to_sub, hop_id hopping_id) {
    if (from_sub == to_sub && relative_index == Index3D::Zero()) {
        throw std::logic_error{
            "Hoppings from/to the same sublattice must have a non-zero relative "
            "index in at least one direction. Don't define onsite energy here."
        };
    }

    auto const num_sublattices = static_cast<sub_id>(sublattices.size());
    if (from_sub >= num_sublattices || to_sub >= num_sublattices)
        throw std::logic_error{"The specified sublattice does not exist."};

    if (hopping_id >= static_cast<hop_id>(hopping_energies.size()))
        throw std::logic_error{"The specified hopping does not exist."};

    // the other sublattice has an opposite relative index
    sublattices[from_sub].add_hopping(relative_index, to_sub, hopping_id, /*is_conjugate*/false);
    sublattices[to_sub].add_hopping(-relative_index, from_sub, hopping_id, /*is_conjugate*/true);
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

void Sublattice::add_hopping(Index3D relative_index, sub_id to_sub, hop_id hop, bool is_conj) {
    bool already_exists = std::any_of(hoppings.begin(), hoppings.end(), [&](Hopping const& h) {
        return h.relative_index == relative_index && h.to_sublattice == to_sub;
    });

    if (already_exists)
        throw std::logic_error{"The specified hopping already exists."};

    hoppings.push_back({relative_index, to_sub, hop, is_conj});
}

Cartesian Lattice::calc_position(Index3D index, Cartesian origin, sub_id sub) const {
    auto position = origin;
    // Bravais lattice position
    for (auto i = 0u; i < vectors.size(); ++i) {
        position += static_cast<float>(index[i]) * vectors[i];
    }
    if (sub >= 0) {
        position += sublattices[sub].offset;
    }
    return position;
}

} // namespace tbm
