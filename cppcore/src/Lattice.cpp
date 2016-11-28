#include "Lattice.hpp"

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

sub_id Lattice::add_sublattice(std::string const& name, Cartesian position,
                               double onsite_energy, sub_id alias) {
    constexpr auto max_size = static_cast<size_t>(std::numeric_limits<sub_id>::max());
    if (sites.structure.size() > max_size) {
        throw std::logic_error("Exceeded maximum number of unique sublattices: "
                               + std::to_string(max_size));
    }

    auto const id = static_cast<sub_id>(sites.structure.size());
    auto const is_unique_name = sites.id.emplace(name, id).second;
    if (!is_unique_name) { throw std::logic_error("Sublattice '" + name + "' already exists"); }

    sites.structure.push_back({position, (alias < 0) ? id : alias, {}});
    sites.energy.push_back(onsite_energy);
    return id;
}

hop_id Lattice::add_hopping(Index3D rel_index, sub_id from_sub, sub_id to_sub,
                            std::complex<double> energy) {
    auto const id = [&] {
        auto const it = std::find(hoppings.energy.begin(), hoppings.energy.end(), energy);
        if (it != hoppings.energy.end())
            return static_cast<hop_id>(it - hoppings.energy.begin());
        else
            return register_hopping_energy({}, energy);
    }();

    add_registered_hopping(rel_index, from_sub, to_sub, id);
    return id;
}

hop_id Lattice::register_hopping_energy(std::string const& name, std::complex<double> energy) {
    constexpr auto max_size = static_cast<size_t>(std::numeric_limits<hop_id>::max());
    if (hoppings.energy.size() > max_size) {
        throw std::logic_error("Exceeded maximum number of unique hoppings energies: "
                               + std::to_string(max_size));
    }

    auto const id = static_cast<hop_id>(hoppings.energy.size());
    if (!name.empty()) {
        auto const is_unique_name = hoppings.id.emplace(name, id).second;
        if (!is_unique_name) {
            throw std::logic_error("Hopping '" + name + "' already exists");
        }
    }

    hoppings.energy.push_back(energy);
    return id;
}

void Lattice::add_registered_hopping(Index3D relative_index, sub_id from_sub,
                                     sub_id to_sub, hop_id hopping_id) {
    if (from_sub == to_sub && relative_index == Index3D::Zero()) {
        throw std::logic_error(
            "Hoppings from/to the same sublattice must have a non-zero relative "
            "index in at least one direction. Don't define onsite energy here."
        );
    }

    auto const max_sub = static_cast<sub_id>(nsub());
    if (from_sub < 0 || from_sub >= max_sub || to_sub < 0 || to_sub >= max_sub) {
        throw std::logic_error("The specified sublattice does not exist.");
    }
    if (hopping_id < 0 || static_cast<std::size_t>(hopping_id) > hoppings.energy.size()) {
        throw std::logic_error("The specified hopping does not exist.");
    }

    // the other sublattice has an opposite relative index
    sites.structure[from_sub].add_hopping(relative_index, to_sub, hopping_id, /*is_conjugate*/false);
    sites.structure[to_sub].add_hopping(-relative_index, from_sub, hopping_id, /*is_conjugate*/true);
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
    for (auto& sub : sites.structure) {
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
        position += sites.structure[sub].position;
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

bool Lattice::has_onsite_energy() const {
    return std::any_of(sites.energy.begin(), sites.energy.end(),
                       [](double e) { return e != .0; });
}

bool Lattice::has_complex_hoppings() const {
    return std::any_of(hoppings.energy.begin(), hoppings.energy.end(),
                       [](std::complex<double> e) { return e.imag() != .0; });
}

} // namespace cpb
