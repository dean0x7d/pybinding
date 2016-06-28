#include "Model.hpp"
#include "system/Foundation.hpp"

#include "support/format.hpp"

namespace cpb {

void Model::set_primitive(Primitive new_primitive) {
    primitive = new_primitive;
    clear_structure();
}

void Model::set_wave_vector(Cartesian const& new_wave_vector) {
    if (wave_vector != new_wave_vector) {
        wave_vector = new_wave_vector;
        clear_hamiltonian();
    }
}

void Model::set_shape(Shape const& new_shape) {
    shape = new_shape;
    lattice.set_offset(lattice.offset + shape.lattice_offset);
    clear_structure();
}

void Model::set_symmetry(TranslationalSymmetry const& translational_symmetry) {
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

void Model::add_site_state_modifier(SiteStateModifier const& m) {
    system_modifiers.state.push_back(m);
    clear_structure();
}

void Model::add_position_modifier(PositionModifier const& m) {
    system_modifiers.position.push_back(m);
    clear_structure();
}

void Model::add_onsite_modifier(OnsiteModifier const& m) {
    hamiltonian_modifiers.onsite.push_back(m);
    clear_hamiltonian();
}

void Model::add_hopping_modifier(HoppingModifier const& m) {
    hamiltonian_modifiers.hopping.push_back(m);
    clear_hamiltonian();
}

void Model::add_hopping_family(HoppingGenerator const& g) {
    hopping_generators.push_back(g);
    lattice.register_hopping_energy(g.name, g.energy);
    clear_structure();
}

bool Model::is_double() const {
    return hamiltonian_modifiers.any_double();
}

bool Model::is_complex() const {
    return lattice.has_complex_hopping || hamiltonian_modifiers.any_complex() || symmetry;
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
    if (!_hamiltonian) {
        hamiltonian_build_time.timeit([&]{
            _hamiltonian = make_hamiltonian();
        });
    }
    return _hamiltonian;
}

Leads const& Model::leads() const {
    system();
    _leads.make_hamiltonian(hamiltonian_modifiers, is_double(), is_complex());
    return _leads;
}

std::string Model::report() {
    auto const& built_system = *system();
    auto report = fmt::format("Built system with {} lattice sites, {}\n",
                              fmt::with_suffix(built_system.num_sites()), system_build_time);

    auto const& built_hamiltonian = hamiltonian();
    report += fmt::format("The Hamiltonian has {} non-zero values, {}",
                          fmt::with_suffix(built_hamiltonian.non_zeros()), hamiltonian_build_time);

    return report;
}

std::shared_ptr<System> Model::make_system() const {
    auto foundation = shape ? Foundation(lattice, shape)
                            : Foundation(lattice, primitive);
    if (symmetry)
        symmetry.apply(foundation);

    if (!system_modifiers.empty()) {
        auto const sublattices = detail::make_sublattice_ids(foundation);

        for (auto const& site_state_modifier : system_modifiers.state) {
            site_state_modifier.apply(foundation.get_states(), foundation.get_positions(),
                                       {sublattices, lattice.sub_name_map});
            if (site_state_modifier.min_neighbors > 0) {
                remove_dangling(foundation, site_state_modifier.min_neighbors);
            }
        }
        for (auto const& position_modifier : system_modifiers.position) {
            position_modifier.apply(foundation.get_positions(),
                                     {sublattices, lattice.sub_name_map});
        }
    }

    _leads.create_attachment_area(foundation);

    auto const hamiltonian_indices = HamiltonianIndices(foundation);
    _leads.make_structure(foundation, hamiltonian_indices);
    return std::make_shared<System>(foundation, hamiltonian_indices, symmetry, hopping_generators);
}

Hamiltonian Model::make_hamiltonian() const {
    auto const& built_system = *system();

    if (is_double()) {
        if (is_complex()) {
            return ham::make<std::complex<double>>(built_system, hamiltonian_modifiers, wave_vector);
        } else {
            return ham::make<double>(built_system, hamiltonian_modifiers, wave_vector);
        }
    } else {
        if (is_complex()) {
            return ham::make<std::complex<float>>(built_system, hamiltonian_modifiers, wave_vector);
        } else {
            return ham::make<float>(built_system, hamiltonian_modifiers, wave_vector);
        }
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
