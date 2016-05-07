#include "Model.hpp"

#include "system/Foundation.hpp"
#include "hamiltonian/Hamiltonian.hpp"

#include "support/format.hpp"

namespace tbm {

void Model::set_primitive(Primitive new_primitive) {
    primitive = new_primitive;
}

void Model::set_wave_vector(const Cartesian& new_wave_vector)
{
    if (wave_vector != new_wave_vector) {
        wave_vector = new_wave_vector;
        _hamiltonian.reset();
    }
}

void Model::set_shape(Shape const& new_shape) {
    shape = new_shape;
    _system.reset();
    _hamiltonian.reset();
}

void Model::set_symmetry(TranslationalSymmetry const& translational_symmetry) {
    symmetry = translational_symmetry;
    _system.reset();
    _hamiltonian.reset();
}

void Model::attach_lead(int direction, Shape const& shape) {
    _leads.add(direction, shape);
}

void Model::add_site_state_modifier(SiteStateModifier const& m) {
    system_modifiers.state.push_back(m);
    _system.reset();
    _hamiltonian.reset();
}

void Model::add_position_modifier(PositionModifier const& m) {
    system_modifiers.position.push_back(m);
    _system.reset();
    _hamiltonian.reset();
}

void Model::add_onsite_modifier(OnsiteModifier const& m) {
    hamiltonian_modifiers.onsite.push_back(m);
    _hamiltonian.reset();
}

void Model::add_hopping_modifier(HoppingModifier const& m) {
    hamiltonian_modifiers.hopping.push_back(m);
    _hamiltonian.reset();
}

void Model::add_hopping_family(HoppingGenerator const& g) {
    hopping_generators.push_back(g);
    lattice.register_hopping_energy(g.name, g.energy);
    _system.reset();
    _hamiltonian.reset();
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

} // namespace tbm
