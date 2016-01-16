#include "Model.hpp"
#include "system/Foundation.hpp"
#include "hamiltonian/Hamiltonian.hpp"
#include "support/format.hpp"
#include "utils/Chrono.hpp"

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

void Model::set_symmetry(Symmetry const& new_symmetry) {
    symmetry = new_symmetry;
    _system.reset();
    _hamiltonian.reset();
}

void Model::add_site_state_modifier(SiteStateModifier const& m) {
    if (system_modifiers.add_unique(m)) {
        _system.reset();
        _hamiltonian.reset();
    }
}

void Model::add_position_modifier(PositionModifier const& m) {
    if (system_modifiers.add_unique(m)) {
        _system.reset();
        _hamiltonian.reset();
    }
}

void Model::add_onsite_modifier(OnsiteModifier const& m) {
    if (hamiltonian_modifiers.add_unique(m))
        _hamiltonian.reset();
}

void Model::add_hopping_modifier(HoppingModifier const& m) {
    if (hamiltonian_modifiers.add_unique(m))
        _hamiltonian.reset();
}

std::shared_ptr<const System> Model::system() const {
    if (!_system) {
        auto build_time = Chrono();
        _system = make_system();
        build_report = fmt::format("Built system with {} lattice sites, {}",
                                   fmt::with_suffix(_system->num_sites()), build_time.toc());
    }

    return _system;
}

std::shared_ptr<const Hamiltonian> Model::hamiltonian() const {
    if (!_hamiltonian) {
        // create a new Hamiltonian of suitable type
        if (hamiltonian_modifiers.any_complex() ||
            system()->lattice.has_complex_hopping ||
            !system()->boundaries.empty())
            _hamiltonian = std::make_shared<HamiltonianT<std::complex<float>>>(*system(), hamiltonian_modifiers, wave_vector);
        else
            _hamiltonian = std::make_shared<HamiltonianT<float>>(*system(), hamiltonian_modifiers, wave_vector);
    }
    
    return _hamiltonian;
}

std::string Model::report() {
    system();
    // this could be a single line, but GCC 4.8 produces a runtime error otherwise
    auto ret = build_report + '\n';
    ret += hamiltonian()->report;
    return ret;
}

std::shared_ptr<System> Model::make_system() const {
    auto foundation = shape ? Foundation(lattice, shape)
                            : Foundation(lattice, primitive);
    if (symmetry)
        symmetry.apply(foundation);

    if (!system_modifiers.empty()) {
        auto const sublattices_ids = detail::make_sublattice_ids(foundation);

        for (auto const& site_state_modifier : system_modifiers.state) {
            site_state_modifier->apply(foundation.get_states(), foundation.get_positions(),
                                       sublattices_ids);
        }
        for (auto const& position_modifier : system_modifiers.position) {
            position_modifier->apply(foundation.get_positions(), sublattices_ids);
        }
    }

    return std::make_shared<System>(foundation, symmetry);
}

} // namespace tbm
