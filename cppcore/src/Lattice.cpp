#include "Lattice.hpp"

#include <Eigen/Dense>  // for `colPivHouseholderQr()`

#include <support/format.hpp>
using namespace fmt::literals;

namespace cpb {

namespace {
    /// Visit all hopping terms in all families
    template<class F>
    void for_each_term(Lattice::Hoppings const& hoppings, F lambda) {
        for (auto const& pair : hoppings) {
            auto const& family = pair.second;
            for (auto const& term : family.terms) {
                lambda(family, term);
            }
        }
    }
} // anonymous namespace

Lattice::Lattice(Cartesian a1, Cartesian a2, Cartesian a3) {
    vectors.push_back(a1);
    if (!a2.isZero()) { vectors.push_back(a2); }
    if (!a3.isZero()) { vectors.push_back(a3); }
    vectors.shrink_to_fit();
}

void Lattice::add_sublattice(string_view name, Cartesian position, double onsite_energy) {
    add_sublattice(name, position, MatrixXcd::Constant(1, 1, onsite_energy).eval());
}

void Lattice::add_sublattice(string_view name, Cartesian position, VectorXd const& onsite_energy) {
    auto const size = onsite_energy.size();
    auto mat = MatrixXcd::Zero(size, size).eval();
    mat.diagonal() = onsite_energy.cast<std::complex<double>>();
    add_sublattice(name, position, mat);
}

void Lattice::add_sublattice(string_view name, Cartesian position,
                             MatrixXcd const& onsite_energy) {
    if (onsite_energy.rows() != onsite_energy.cols()) {
        throw std::logic_error("The onsite hopping term must be a real vector or a square matrix");
    }
    if (onsite_energy.rows() == 0) {
        throw std::logic_error("The onsite hopping term can't be zero-dimensional");
    }
    if (!onsite_energy.diagonal().imag().isZero()) {
        throw std::logic_error("The main diagonal of the onsite hopping term must be real");
    }
    if (!onsite_energy.isUpperTriangular() && onsite_energy != onsite_energy.adjoint()) {
        throw std::logic_error("The onsite hopping matrix must be upper triangular or Hermitian");
    }

    auto const hermitian_view = onsite_energy.selfadjointView<Eigen::Upper>();
    auto const unique_id = make_unique_sublattice_id(name);
    auto const alias_id = SubAliasID(unique_id);
    sublattices[name] = {position, hermitian_view, unique_id, alias_id};
}

void Lattice::add_alias(string_view alias_name, string_view original_name, Cartesian position) {
    auto const& original = sublattice(original_name);
    auto const alias_id = SubAliasID(original.unique_id);
    auto const unique_id = make_unique_sublattice_id(alias_name);
    sublattices[alias_name] = {position, original.energy, unique_id, alias_id};
}

void Lattice::register_hopping_energy(std::string const& name, std::complex<double> energy) {
    register_hopping_energy(name, MatrixXcd::Constant(1, 1, energy));
}

void Lattice::register_hopping_energy(std::string const& name, MatrixXcd const& energy) {
    if (name.empty()) { throw std::logic_error("Hopping name can't be blank"); }

    if (energy.rows() == 0 || energy.cols() == 0) {
        throw std::logic_error("Hoppings can't be zero-dimensional");
    }

    auto const unique_id = HopID(hoppings.size());
    auto const is_unique_name = hoppings.insert({name, {energy, unique_id, {}}}).second;
    if (!is_unique_name) { throw std::logic_error("Hopping '" + name + "' already exists"); }
}

void Lattice::add_hopping(Index3D relative_index, string_view from_sub, string_view to_sub,
                          string_view hopping_family_name) {
    if (from_sub == to_sub && relative_index == Index3D::Zero()) {
        throw std::logic_error(
            "Hoppings from/to the same sublattice must have a non-zero relative "
            "index in at least one direction. Don't define onsite energy here."
        );
    }

    auto const& from = sublattice(from_sub);
    auto const& to = sublattice(to_sub);
    auto const& hop_matrix = hopping_family(hopping_family_name).energy;

    if (from.energy.rows() != hop_matrix.rows() || to.energy.cols() != hop_matrix.cols()) {
        throw std::logic_error(
            "Hopping size mismatch: from '{}' ({}) to '{}' ({}) with matrix '{}' ({}, {})"_format(
                from_sub, from.energy.rows(), to_sub, to.energy.cols(),
                hopping_family_name, hop_matrix.rows(), hop_matrix.cols()
            )
        );
    }

    auto const candidate = HoppingTerm{relative_index, from.unique_id, to.unique_id};
    for_each_term(hoppings, [&](HoppingFamily const&, HoppingTerm const& existing) {
        if (candidate == existing) {
            throw std::logic_error("The specified hopping already exists.");
        }
    });

    hoppings[hopping_family_name].terms.push_back(candidate);
}

void Lattice::add_hopping(Index3D relative_index, string_view from_sub, string_view to_sub,
                          std::complex<double> energy) {
    add_hopping(relative_index, from_sub, to_sub, MatrixXcd::Constant(1, 1, energy));
}

void Lattice::add_hopping(Index3D relative_index, string_view from_sub, string_view to_sub,
                          MatrixXcd const& energy) {
    auto const hopping_name = [&] {
        // Look for an existing hopping ID with the same energy
        auto const it = std::find_if(hoppings.begin(), hoppings.end(), [&](Hoppings::reference r) {
            auto const& e = r.second.energy;
            return e.rows() == energy.rows() && e.cols() == energy.cols() && e == energy;
        });

        if (it != hoppings.end()) {
            return it->first;
        } else {
            auto const name = "__anonymous__{}"_format(hoppings.size());
            register_hopping_energy(name, energy);
            return name;
        }
    }();

    add_hopping(relative_index, from_sub, to_sub, hopping_name);
}

void Lattice::set_offset(Cartesian position) {
    if (any_of(translate_coordinates(position).array().abs() > 0.55f)) {
        throw std::logic_error("Lattice origin must not be moved by more than "
                               "half the length of a primitive lattice vector.");
    }
    offset = position;
}

Lattice::Sublattice const& Lattice::sublattice(std::string const& name) const {
    auto const it = sublattices.find(name);
    if (it == sublattices.end()) {
        throw std::out_of_range("There is no sublattice named '{}'"_format(name));
    }
    return it->second;
}

Lattice::Sublattice const& Lattice::sublattice(SubID id) const {
    using Pair = Sublattices::value_type;
    auto const it = std::find_if(sublattices.begin(), sublattices.end(),
                                 [&](Pair const& p) { return p.second.unique_id == id; });
    if (it == sublattices.end()) {
        throw std::out_of_range("There is no sublattice with ID = {}"_format(id.value()));
    }
    return it->second;
}

Lattice::HoppingFamily const& Lattice::hopping_family(std::string const& name) const {
    auto const it = hoppings.find(name);
    if (it == hoppings.end()) {
        throw std::out_of_range("There is no hopping named '{}'"_format(name));
    }
    return it->second;
}

Lattice::HoppingFamily const& Lattice::hopping_family(HopID id) const {
    using Pair = Hoppings::value_type;
    auto const it = std::find_if(hoppings.begin(), hoppings.end(),
                                 [&](Pair const& p) { return p.second.family_id == id; });
    if (it == hoppings.end()) {
        throw std::out_of_range("There is no hopping with ID = {}"_format(id.value()));
    }
    return it->second;
}

string_view Lattice::sublattice_name(SubID id) const {
    using Pair = Sublattices::value_type;
    auto const it = std::find_if(sublattices.begin(), sublattices.end(),
                                 [&](Pair const& p) { return p.second.unique_id == id; });
    if (it == sublattices.end()) {
        throw std::out_of_range("There is no sublattice with ID = {}"_format(id.value()));
    }
    return it->first;
}

string_view Lattice::hopping_family_name(HopID id) const {
    using Pair = Hoppings::value_type;
    auto const it = std::find_if(hoppings.begin(), hoppings.end(),
                                 [&](Pair const& p) { return p.second.family_id == id; });
    if (it == hoppings.end()) {
        throw std::out_of_range("There is no hopping with ID = {}"_format(id.value()));
    }
    return it->first;
}

int Lattice::max_hoppings() const {
    auto result = idx_t{0};
    for (auto const& pair : sublattices) {
        auto const& sub = pair.second;

        // Include hoppings in onsite matrix (-1 for diagonal value which is not a hopping)
        auto num_scalar_hoppings = sub.energy.cols() - 1;

        // Conjugate term counts rows instead of columns
        for_each_term(hoppings, [&](HoppingFamily const& family, HoppingTerm const& term) {
            if (term.from == sub.unique_id) { num_scalar_hoppings += family.energy.cols(); }
            if (term.to == sub.unique_id)   { num_scalar_hoppings += family.energy.rows(); }
        });

        result = std::max(result, num_scalar_hoppings);
    }
    return static_cast<int>(result);
}

Cartesian Lattice::calc_position(Index3D index, string_view sublattice_name) const {
    auto position = offset;
    // Bravais lattice position
    for (auto i = 0, size = ndim(); i < size; ++i) {
        position += static_cast<float>(index[i]) * vectors[i];
    }
    if (!sublattice_name.empty()) {
        position += sublattice(sublattice_name).position;
    }
    return position;
}

Vector3f Lattice::translate_coordinates(Cartesian position) const {
    auto const size = ndim();
    auto const lattice_matrix = [&]{
        auto m = ColMajorMatrixX<float>(size, size);
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

bool Lattice::has_diagonal_terms() const {
    return std::any_of(sublattices.begin(), sublattices.end(), [](Sublattices::const_reference r) {
        return !r.second.energy.diagonal().isZero();
    });
}

bool Lattice::has_onsite_energy() const {
    return std::any_of(sublattices.begin(), sublattices.end(), [](Sublattices::const_reference r) {
        return !r.second.energy.isZero();
    });
}

bool Lattice::has_multiple_orbitals() const {
    return std::any_of(sublattices.begin(), sublattices.end(), [](Sublattices::const_reference r) {
        return r.second.energy.cols() != 1;
    });
}

bool Lattice::has_complex_hoppings() const {
    return std::any_of(hoppings.begin(), hoppings.end(), [](Hoppings::const_reference r) {
        return !r.second.energy.imag().isZero();
    }) || std::any_of(sublattices.begin(), sublattices.end(), [](Sublattices::const_reference r) {
        return !r.second.energy.imag().isZero();
    });
}

OptimizedUnitCell Lattice::optimized_unit_cell() const {
    return OptimizedUnitCell(*this);
}

NameMap Lattice::sub_name_map() const {
    auto map = NameMap();
    for (auto const& p : sublattices) {
        map[p.first] = p.second.unique_id.value();
    }
    return map;
}

NameMap Lattice::hop_name_map() const {
    auto map = NameMap();
    for (auto const& p : hoppings) {
        map[p.first] = p.second.family_id.value();
    }
    return map;
}

SubID Lattice::make_unique_sublattice_id(string_view name) {
    if (name.empty()) { throw std::logic_error("Sublattice name can't be blank"); }

    if (sublattices.find(name) != sublattices.end()) {
        throw std::logic_error("Sublattice '" + name + "' already exists");
    }

    return SubID(sublattices.size());
}

OptimizedUnitCell::OptimizedUnitCell(Lattice const& lattice) : sites(lattice.nsub()) {
    // Populate sites in ascending unique_id order
    for (auto const& pair : lattice.get_sublattices()) {
        auto const& sub = pair.second;
        auto const idx = sub.unique_id.as<size_t>();
        sites[idx] = {sub.position, /*norb*/static_cast<storage_idx_t>(sub.energy.cols()),
                      sub.unique_id, sub.alias_id, /*hoppings*/{}};
    }

    // Sites with equal `alias_id` will be merged in the final system. Stable sort
    // ensures that the ascending unique_id order is preserved within alias groups.
    std::stable_sort(sites.begin(), sites.end(), [](Site const& a, Site const& b) {
        return a.alias_id < b.alias_id;
    });

    // Sort by number of orbitals, but make sure alias ordering from the previous step
    // is preserved (stable sort). Aliases have the same `norb` so these sites will
    // remain as consecutive elements in the final sorted vector.
    std::stable_sort(sites.begin(), sites.end(), [](Site const& a, Site const& b) {
        return a.norb < b.norb;
    });

    // Find the index in `sites` of a site with the given unique ID
    auto find_index = [&](SubID unique_id) {
        auto const it = std::find_if(sites.begin(), sites.end(), [&](Site const& s) {
            return s.unique_id == unique_id;
        });
        assert(it != sites.end());
        return static_cast<storage_idx_t>(it - sites.begin());
    };

    for_each_term(lattice.get_hoppings(), [&](Lattice::HoppingFamily const& hopping_family,
                                              Lattice::HoppingTerm const& term) {
        auto const idx1 = find_index(term.from);
        auto const idx2 = find_index(term.to);
        auto const id = hopping_family.family_id;

        // The other sublattice has an opposite relative index (conjugate)
        sites[idx1].hoppings.push_back({ term.relative_index, idx2, id, /*is_conjugate*/false});
        sites[idx2].hoppings.push_back({-term.relative_index, idx1, id, /*is_conjugate*/true });
    });
}

} // namespace cpb
