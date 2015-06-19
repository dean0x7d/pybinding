#include "Model.hpp"
#include "hamiltonian/Hamiltonian.hpp"
using namespace tbm;

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

void Model::add_site_state_modifier(const std::shared_ptr<SiteStateModifier>& m)
{
    if (system_modifiers.add_unique(m)) {
        _system.reset();
        _hamiltonian.reset();
    }
}

void Model::add_position_modifier(const std::shared_ptr<PositionModifier>& m)
{
    if (system_modifiers.add_unique(m)) {
        _system.reset();
        _hamiltonian.reset();
    }
}

void Model::add_onsite_modifier(const std::shared_ptr<OnsiteModifier>& m)
{
    if (hamiltonian_modifiers.add_unique(m))
        _hamiltonian.reset();
}

void Model::add_hopping_modifier(const std::shared_ptr<HoppingModifier>& m)
{
    if (hamiltonian_modifiers.add_unique(m))
        _hamiltonian.reset();
}

std::shared_ptr<const System> Model::system() const {
    if (!_system) {
        // check for all the required parameters
        if (!_lattice)
            throw std::logic_error{"A lattice must be defined."};
        if (!_shape)
            _shape = std::make_shared<Primitive>();
        
        _system = std::make_shared<System>(*_lattice, *_shape, _symmetry.get(), system_modifiers);
    }
    
    return _system;
}

std::shared_ptr<const Hamiltonian> Model::hamiltonian() const {
    if (!_hamiltonian) {
        // create a new Hamiltonian of suitable type
        if (hamiltonian_modifiers.any_complex() || _symmetry)
            _hamiltonian = std::make_shared<HamiltonianT<std::complex<float>>>(*system(), hamiltonian_modifiers, wave_vector);
        else
            _hamiltonian = std::make_shared<HamiltonianT<float>>(*system(), hamiltonian_modifiers, wave_vector);
    }
    
    return _hamiltonian;
}

std::string Model::build_report()
{
    // this could be a single line, but GCC 4.8 produces a runtime error otherwise
    auto ret = system()->report + '\n';
    ret += hamiltonian()->report;
    return ret;
}
