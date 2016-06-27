#include "system/Lattice.hpp"

#include <Eigen/Dense>  // for `colPivHouseholderQr()`

namespace cpb {

void Sublattice::add_hopping(Index3D relative_index, sub_id to_sub, hop_id hop, bool is_conj) {
    bool already_exists = std::any_of(hoppings.begin(), hoppings.end(), [&](Hopping const& h) {
        return h.relative_index == relative_index && h.to_sublattice == to_sub;
    });

    if (already_exists) {
        throw std::logic_error("The specified hopping already exists.");
    }

    hoppings.push_back({relative_index, to_sub, hop, is_conj});
}

Lattice::Lattice(Cartesian a1, Cartesian a2, Cartesian a3) {
    vectors.push_back(a1);
    if (!a2.isZero()) { vectors.push_back(a2); }
    if (!a3.isZero()) { vectors.push_back(a3); }
    vectors.shrink_to_fit();
}

sub_id Lattice::add_sublattice(std::string const& name, Cartesian offset,
                               double onsite_energy, sub_id alias) {
    auto const sublattice_id = static_cast<sub_id>(sublattices.size());
    if (sublattice_id == std::numeric_limits<sub_id>::max()) {
        throw std::logic_error("Cannot create more sublattices: " + std::to_string(sublattice_id));
    }

    auto const is_unique_name = sub_name_map.emplace(name, sublattice_id).second;
    if (!is_unique_name) {
        throw std::logic_error("Sublattice '" + name + "' already exists");
    }

    sublattices.push_back({
        offset,
        onsite_energy,
        (alias < 0) ? sublattice_id : alias,
        {} // create an empty slot for this sublattice's hoppings
    });

    if (onsite_energy != .0) {
        has_onsite_energy = true;
    }

    return sublattice_id;
}

hop_id Lattice::add_hopping(Index3D rel_index, sub_id from_sub, sub_id to_sub,
                            std::complex<double> energy) {
    auto const hopping_id = [&] {
        auto const it = std::find(hopping_energies.begin(), hopping_energies.end(), energy);
        if (it != hopping_energies.end())
            return static_cast<hop_id>(it - hopping_energies.begin());
        else
            return register_hopping_energy({}, energy);
    }();

    add_registered_hopping(rel_index, from_sub, to_sub, hopping_id);
    return hopping_id;
}

hop_id Lattice::register_hopping_energy(std::string const& name, std::complex<double> energy) {
    auto const hopping_id = static_cast<hop_id>(hopping_energies.size());
    if (hopping_id == std::numeric_limits<hop_id>::max()) {
        throw std::logic_error("Can't create any more hoppings: " + std::to_string(hopping_id));
    }

    if (!name.empty()) {
        auto const is_unique_name = hop_name_map.emplace(name, hopping_id).second;
        if (!is_unique_name) {
            throw std::logic_error("Hopping '" + name + "' already exists");
        }
    }

    hopping_energies.push_back(energy);
    if (energy.imag() != .0) {
        has_complex_hopping = true;
    }

    return hopping_id;
}

void Lattice::add_registered_hopping(Index3D relative_index, sub_id from_sub,
                                     sub_id to_sub, hop_id hopping_id) {
    if (from_sub == to_sub && relative_index == Index3D::Zero()) {
        throw std::logic_error(
            "Hoppings from/to the same sublattice must have a non-zero relative "
            "index in at least one direction. Don't define onsite energy here."
        );
    }

    auto const num_sublattices = static_cast<sub_id>(sublattices.size());
    if (from_sub < 0 || from_sub >= num_sublattices || to_sub < 0 || to_sub >= num_sublattices) {
        throw std::logic_error("The specified sublattice does not exist.");
    }
    if (hopping_id < 0 || hopping_id >= static_cast<hop_id>(hopping_energies.size())) {
        throw std::logic_error("The specified hopping does not exist.");
    }

    // the other sublattice has an opposite relative index
    sublattices[from_sub].add_hopping(relative_index, to_sub, hopping_id, /*is_conjugate*/false);
    sublattices[to_sub].add_hopping(-relative_index, from_sub, hopping_id, /*is_conjugate*/true);
}

void Lattice::set_offset(Cartesian position) {
    if (any_of(translate_coordinates(position).array().abs() > 0.55f)) {
        throw std::logic_error("Lattice origin must not be moved by more than "
                               "half the length of a primitive lattice vector.");
    }
    offset = position;
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

Cartesian Lattice::calc_position(Index3D index, sub_id sub) const {
    auto position = offset;
    // Bravais lattice position
    for (auto i = size_t{0}, size = vectors.size(); i < size; ++i) {
        position += static_cast<float>(index[i]) * vectors[i];
    }
    if (sub >= 0) {
        position += sublattices[sub].offset;
    }
    return position;
}

Vector3f Lattice::translate_coordinates(Cartesian position) const {
    auto const size = ndim();
    auto const lattice_matrix = [&]{
        auto m = Eigen::MatrixXf(size, size);
        for (auto i = 0; i < size; ++i) {
            m.col(i) = vectors[i].head(size);
        }
        return m;
    }();

    // Solve `lattice_matrix * v = p`
    auto const& p = position.head(size);
    auto v = Vector3f(0, 0, 0);
    v.head(size) = lattice_matrix.colPivHouseholderQr().solve(p);
    return v;
}

Lattice Lattice::with_offset(Cartesian position) const {
    auto new_lattice = *this;
    new_lattice.set_offset(position);
    return new_lattice;
}

Lattice Lattice::with_min_neighbors(int number) const {
    auto new_lattice = *this;
    new_lattice.min_neighbors = number;
    return new_lattice;
}

} // namespace cpb
