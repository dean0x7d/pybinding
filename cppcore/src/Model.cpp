#include "Model.hpp"
#include "system/Foundation.hpp"
#include "hamiltonian/Hamiltonian.hpp"
#include "support/format.hpp"
#include "utils/Chrono.hpp"

namespace tbm {

void Model::set_lattice(const std::shared_ptr<Lattice>& new_lattice)
{
    if (_lattice == new_lattice)
        return;

    if (!new_lattice || new_lattice->sublattices.size() < 1)
        throw std::logic_error{"At least 1 sublattice must be specified."};
    if (new_lattice->vectors.size() < 1)
        throw std::logic_error{"At least 1 lattice vector must be specified."};

    _lattice = new_lattice;
    _system.reset();
    _hamiltonian.reset();
}

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

void Model::set_shape(const std::shared_ptr<Shape>& new_shape)
{
    if (_shape != new_shape) {
        _shape = new_shape;
        _system.reset();
        _hamiltonian.reset();
    }
}

void Model::set_symmetry(const std::shared_ptr<Symmetry>& new_symmetry)
{
    if (_symmetry != new_symmetry) {
        _symmetry = new_symmetry;
        _system.reset();
        _hamiltonian.reset();
    }
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
        // check for all the required parameters
        if (!_lattice)
            throw std::logic_error{"A lattice must be defined."};

        auto build_time = Chrono{};

        auto foundation = _shape ? Foundation(*_lattice, *_shape)
                                 : Foundation(*_lattice, primitive);
        _system = build_system(foundation, system_modifiers, _symmetry.get());

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

} // namespace tbm
