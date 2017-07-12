#include "Model.hpp"
#include "system/Foundation.hpp"

#include "support/format.hpp"

namespace cpb {

void Model::add(Primitive new_primitive) {
    primitive = new_primitive;
    for (auto i = lattice.ndim(); i < 3; ++i) {
        if (primitive.size[i] != 1) {
            throw std::logic_error("Primitive shape has more dimensions than the lattice");
        }
    }
    clear_structure();
}

void Model::set_wave_vector(Cartesian const& new_wave_vector) {
    if (wave_vector != new_wave_vector) {
        wave_vector = new_wave_vector;
        clear_hamiltonian();
    }
}

void Model::add(Shape const& new_shape) {
    shape = new_shape;
    lattice.set_offset(lattice.get_offset() + shape.lattice_offset);
    clear_structure();
}

void Model::add(TranslationalSymmetry const& translational_symmetry) {
    symmetry = translational_symmetry;
    clear_structure();
}

void Model::attach_lead(int direction, Shape const& shape) {
    if (direction == 0) {
        throw std::logic_error("Lead direction must be one of: 1, 2, 3, -1, -2, -3");
    } else if (lattice.ndim() == 1) {
        throw std::logic_error("Attaching leads to 1D lattices is not supported");
    } else if (std::abs(direction) > lattice.ndim()) {
        throw std::logic_error(fmt::format("Direction {} is not valid for a {}D lattice",
                                           direction, lattice.ndim()));
    }

    _leads.add(direction, shape);
    clear_structure();
}

void Model::add(SiteStateModifier const& m) {
    system_modifiers.state.push_back(m);
    clear_structure();
}

void Model::add(PositionModifier const& m) {
    system_modifiers.position.push_back(m);
    clear_structure();
}

void Model::add(OnsiteModifier const& m) {
    hamiltonian_modifiers.onsite.push_back(m);
    clear_hamiltonian();
}

void Model::add(HoppingModifier const& m) {
    hamiltonian_modifiers.hopping.push_back(m);
    clear_hamiltonian();
}

void Model::add(HoppingGenerator const& g) {
    hopping_generators.push_back(g);
    lattice.register_hopping_energy(g.name, g.energy);
    clear_structure();
}

bool Model::is_double() const {
    return hamiltonian_modifiers.any_double();
}

bool Model::is_complex() const {
    return lattice.has_complex_hoppings() || hamiltonian_modifiers.any_complex()
           || symmetry || complex_override;
}

std::shared_ptr<System const> const& Model::system() const {
    if (!_system) {
        system_build_time.timeit([&]{
            _system = make_system();
        });
    }
    return _system;
}

Hamiltonian const& Model::hamiltonian() const {
    system();
    if (!_hamiltonian) {
        hamiltonian_build_time.timeit([&]{
            _hamiltonian = make_hamiltonian();
        });
    }
    return _hamiltonian;
}

Leads const& Model::leads() const {
    system();
    hamiltonian();
    _leads.make_hamiltonian(hamiltonian_modifiers, is_double(), is_complex());
    return _leads;
}

Model const& Model::eval() const {
    system();
    hamiltonian();
    leads();
    return *this;
}

std::string Model::report() {
    auto const num_sites = fmt::with_suffix(static_cast<double>(system()->num_sites()));
    auto const nnz = fmt::with_suffix(static_cast<double>(hamiltonian().non_zeros()));

    return fmt::format("Built system with {} lattice sites, {}\n"
                       "The Hamiltonian has {} non-zero values, {}",
                       num_sites, system_build_time, nnz, hamiltonian_build_time);
}

std::shared_ptr<System> Model::make_system() const {
    auto foundation = shape ? Foundation(lattice, shape)
                            : Foundation(lattice, primitive);
    if (symmetry)
        symmetry.apply(foundation);

    if (!system_modifiers.empty()) {
        for (auto const& site_state_modifier : system_modifiers.state) {
            for (auto const& pair : lattice.get_sublattices()) {
                auto slice = foundation[pair.second.unique_id];
                site_state_modifier.apply(slice.get_states(), slice.get_positions(), pair.first);
            }

            if (site_state_modifier.min_neighbors > 0) {
                remove_dangling(foundation, site_state_modifier.min_neighbors);
            }
        }

        for (auto const& position_modifier : system_modifiers.position) {
            for (auto const& pair : lattice.get_sublattices()) {
                auto slice = foundation[pair.second.unique_id];
                position_modifier.apply(slice.get_positions(), pair.first);
            }
        }
    }

    _leads.create_attachment_area(foundation);
    _leads.make_structure(foundation);

    return std::make_shared<System>(foundation, symmetry, hopping_generators);
}

Hamiltonian Model::make_hamiltonian() const {
    auto const& built_system = *system();
    auto const& modifiers = hamiltonian_modifiers;
    auto const& k = wave_vector;
    auto const simple_build = hopping_generators.empty();

    if (!is_complex()) {
        try {
            if (!is_double()) {
                return ham::make<float>(built_system, modifiers, k, simple_build);
            } else {
                return ham::make<double>(built_system, modifiers, k, simple_build);
            }
        } catch (ComplexOverride const&) {
            complex_override = true;
        }
    }

    if (!is_double()) {
        return ham::make<std::complex<float>>(built_system, modifiers, k, simple_build);
    } else {
        return ham::make<std::complex<double>>(built_system, modifiers, k, simple_build);
    }
}

void Model::clear_structure() {
    _system.reset();
    _leads.clear_structure();
    clear_hamiltonian();
}

void Model::clear_hamiltonian() {
    _hamiltonian.reset();
    _leads.clear_hamiltonian();
}

} // namespace cpb
