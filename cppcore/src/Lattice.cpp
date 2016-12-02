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

sub_id Lattice::Sites::id_lookup(std::string const& name) const {
    auto const it = id.find(name);
    if (it == id.end()) { throw std::out_of_range("There is no sublattice named '"+ name + "'"); }
    return it->second;
}

hop_id Lattice::Hoppings::id_lookup(std::string const& name) const {
    auto const it = id.find(name);
    if (it == id.end()) { throw std::out_of_range("There is no hopping named '" + name + "'"); }
    return it->second;
}

Lattice::Lattice(Cartesian a1, Cartesian a2, Cartesian a3) {
    vectors.push_back(a1);
    if (!a2.isZero()) { vectors.push_back(a2); }
    if (!a3.isZero()) { vectors.push_back(a3); }
    vectors.shrink_to_fit();
}

void Lattice::add_sublattice(std::string const& name, Cartesian position,
                             double onsite_energy, std::string const& alias) {
    if (name.empty()) { throw std::logic_error("Sublattice name can't be blank"); }

    constexpr auto max_size = static_cast<size_t>(std::numeric_limits<sub_id>::max());
    if (sites.structure.size() > max_size) {
        throw std::logic_error("Exceeded maximum number of unique sublattices: "
                               + std::to_string(max_size));
    }

    auto const id = static_cast<sub_id>(sites.structure.size());
    auto const is_unique_name = sites.id.emplace(name, id).second;
    if (!is_unique_name) { throw std::logic_error("Sublattice '" + name + "' already exists"); }

    sites.structure.push_back({position, alias.empty() ? id : sites.id_lookup(alias), {}});
    sites.energy.push_back(onsite_energy);
}

void Lattice::register_hopping_energy(std::string const& name, std::complex<double> energy) {
    if (name.empty()) { throw std::logic_error("Hopping name can't be blank"); }

    constexpr auto max_size = static_cast<size_t>(std::numeric_limits<hop_id>::max());
    if (hoppings.energy.size() > max_size) {
        throw std::logic_error("Exceeded maximum number of unique hoppings energies: "
                               + std::to_string(max_size));
    }

    auto const id = static_cast<hop_id>(hoppings.energy.size());
    auto const is_unique_name = hoppings.id.emplace(name, id).second;
    if (!is_unique_name) { throw std::logic_error("Hopping '" + name + "' already exists"); }

    hoppings.energy.push_back(energy);
}

void Lattice::add_registered_hopping(Index3D relative_index, std::string const& from_sub,
                                     std::string const& to_sub, std::string const& hopping) {
    if (from_sub == to_sub && relative_index == Index3D::Zero()) {
        throw std::logic_error(
            "Hoppings from/to the same sublattice must have a non-zero relative "
            "index in at least one direction. Don't define onsite energy here."
        );
    }

    auto const from_id = sites.id_lookup(from_sub);
    auto const to_id = sites.id_lookup(to_sub);
    auto const hopping_id = hoppings.id_lookup(hopping);

    // the other sublattice has an opposite relative index
    sites.structure[from_id].add_hopping(relative_index, to_id, hopping_id, /*is_conjugate*/false);
    sites.structure[to_id].add_hopping(-relative_index, from_id, hopping_id, /*is_conjugate*/true);
}

void Lattice::add_hopping(Index3D rel_index, std::string const& from_sub,
                          std::string const& to_sub, std::complex<double> energy) {
    auto const hopping_name = [&] {
        // Look for an existing hopping ID with the same energy
        auto const it = std::find(hoppings.energy.begin(), hoppings.energy.end(), energy);
        if (it != hoppings.energy.end()) {
            auto const id = static_cast<hop_id>(it - hoppings.energy.begin());
            for (auto const& p : hoppings.id) {
                if (p.second == id) { return p.first; }
            }
            return std::string("This should never happen.");
        } else {
            auto const name = "__anonymous__" + std::to_string(hoppings.energy.size());
            register_hopping_energy(name, energy);
            return name;
        }
    }();

    add_registered_hopping(rel_index, from_sub, to_sub, hopping_name);
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

Cartesian Lattice::calc_position(Index3D index, std::string const& sub) const {
    auto position = offset;
    // Bravais lattice position
    for (auto i = size_t{0}, size = vectors.size(); i < size; ++i) {
        position += static_cast<float>(index[i]) * vectors[i];
    }
    if (!sub.empty()) {
        auto const id = sites.id_lookup(sub);
        position += sites.structure[id].position;
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
