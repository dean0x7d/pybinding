#include "Model.hpp"
#include "system/Foundation.hpp"

#include "support/format.hpp"

namespace cpb {

Model::Model(Lattice const& lattice)
    : lattice(lattice),
      site_registry(lattice.site_registry()),
      hopping_registry(lattice.hopping_registry()) {}

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
    structure_modifiers.emplace_back(m);
    clear_structure();
}

void Model::add(PositionModifier const& m) {
    structure_modifiers.emplace_back(m);
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

void Model::add(SiteGenerator const& g) {
    structure_modifiers.emplace_back(g);
    site_registry.register_family(g.name, g.energy);
    clear_structure();
}

void Model::add(HoppingGenerator const& g) {
    structure_modifiers.emplace_back(g);
    hopping_registry.register_family(g.name, g.energy);
    clear_structure();
}

bool Model::is_multiorbital() const {
    return site_registry.has_multiple_orbitals() || hopping_registry.has_multiple_orbitals();
}

bool Model::is_double() const {
    return hamiltonian_modifiers.any_double();
}

bool Model::is_complex() const {
    return site_registry.any_complex_terms() || hopping_registry.any_complex_terms()
           || hamiltonian_modifiers.any_complex() || symmetry || complex_override;
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
    _leads.make_hamiltonian(lattice, hamiltonian_modifiers, is_double(), is_complex());
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
    if (symmetry) {
        symmetry.apply(foundation);
    }

    auto const it = std::find_if(structure_modifiers.begin(), structure_modifiers.end(),
                                 [](StructureModifier const& m) { return requires_system(m); });
    auto const foundation_modifiers = make_range(structure_modifiers.begin(), it);
    auto const system_modifiers = make_range(it, structure_modifiers.end());

    for (auto const& modifier : foundation_modifiers) {
        apply(modifier, foundation);
    }

    _leads.create_attachment_area(foundation);
    _leads.make_structure(foundation);

    auto sys = std::make_shared<System>(site_registry, hopping_registry);
    detail::populate_system(*sys, foundation);
    if (symmetry) {
        detail::populate_boundaries(*sys, foundation, symmetry);
    }

    for (auto const& modifier : system_modifiers) {
        apply(modifier, *sys);
    }

    detail::remove_invalid(*sys);

    if (sys->num_sites() == 0) { throw std::runtime_error{"Impossible system: 0 sites"}; }

    return sys;
}

Hamiltonian Model::make_hamiltonian() const {
    auto const& built_system = *system();
    auto const& modifiers = hamiltonian_modifiers;
    auto const& k = wave_vector;
    auto const simple_build = std::none_of(
        structure_modifiers.begin(), structure_modifiers.end(),
        [](StructureModifier const& m) { return is_generator(m); }
    );

    if (!is_complex()) {
        try {
            if (!is_double()) {
                return ham::make<float>(built_system, lattice, modifiers, k, simple_build);
            } else {
                return ham::make<double>(built_system, lattice, modifiers, k, simple_build);
            }
        } catch (ComplexOverride const&) {
            complex_override = true;
        }
    }

    if (!is_double()) {
        return ham::make<std::complex<float>>(built_system, lattice, modifiers, k, simple_build);
    } else {
        return ham::make<std::complex<double>>(built_system, lattice, modifiers, k, simple_build);
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
