#include "system/Registry.hpp"

#include "support/format.hpp"
using namespace fmt::literals;

namespace cpb {

namespace {

void check_energy(SiteRegistry const&, MatrixXcd const& energy) {
    detail::check_onsite_energy(energy);
}

void check_energy(HoppingRegistry const&, MatrixXcd const& energy) {
    detail::check_hopping_energy(energy);
}

template<class ID>
constexpr char const* kind() { return std::is_same<ID, SiteID>() ? "Site" : "Hopping"; }

} // anonymous namespace

template<class ID>
Registry<ID>::Registry(std::vector<MatrixXcd> energies, std::vector<std::string> names)
    : energies(std::move(energies)), names(std::move(names)) {
    assert(energies.size() == names.size());
}

template<class ID>
void Registry<ID>::register_family(std::string const& name, MatrixXcd const& energy) {
    if (name.empty()) {
        throw std::logic_error("{} family name can't be blank"_format(kind<ID>()));
    }
    check_energy(*this, energy);

    auto const not_unique = std::find(names.begin(), names.end(), name) != names.end();
    if (not_unique) {
        throw std::logic_error("{} family '{}' already exists"_format(kind<ID>(), name));
    }

    names.push_back(name);
    energies.push_back(energy);
}

template<class ID>
NameMap Registry<ID>::name_map() const {
    auto map = NameMap();
    for (auto i = size_t{0}; i < names.size(); ++i) {
        map[names[i]] = static_cast<storage_idx_t>(i);
    }
    return map;
}

template<class ID>
string_view Registry<ID>::name(ID id) const {
    auto const index = id.value();
    if (index >= size()) {
        throw std::out_of_range("There is no {} with ID = {}"_format(kind<ID>(), index));
    }
    return names[index];
}

template<class ID>
MatrixXcd const& Registry<ID>::energy(ID id) const {
    auto const index = id.value();
    if (index >= size()) {
        throw std::out_of_range("There is no {} with ID = {}"_format(kind<ID>(), index));
    }
    return energies[index];
}

template<class ID>
ID Registry<ID>::id(string_view name) const {
    auto const it = std::find(names.begin(), names.end(), name);
    if (it == names.end()) {
        throw std::out_of_range("There is no {} named '{}'"_format(kind<ID>(), name));
    }
    return ID(std::distance(names.begin(), it));
}

template<class ID>
bool Registry<ID>::has_nonzero_energy() const {
    return std::any_of(energies.begin(), energies.end(), [](MatrixXcd const& energy) {
        return !energy.isZero();
    });
}

template<class ID>
bool Registry<ID>::any_complex_terms() const {
    return std::any_of(energies.begin(), energies.end(), [](MatrixXcd const& energy) {
        return !energy.imag().isZero();
    });
}

template<class ID>
bool Registry<ID>::has_multiple_orbitals() const {
    return std::any_of(energies.begin(), energies.end(), [](MatrixXcd const& energy) {
        return energy.size() != 1;
    });
}

template class Registry<SiteID>;
template class Registry<HopID>;

namespace detail {

void check_onsite_energy(MatrixXcd const& energy) {
    if (energy.rows() != energy.cols()) {
        throw std::logic_error("The onsite hopping term must be a real vector or a square matrix");
    }
    if (energy.rows() == 0) {
        throw std::logic_error("The onsite hopping term can't be zero-dimensional");
    }
    if (!energy.diagonal().imag().isZero()) {
        throw std::logic_error("The main diagonal of the onsite hopping term must be real");
    }
    if (!energy.isUpperTriangular() && energy != energy.adjoint()) {
        throw std::logic_error("The onsite hopping matrix must be upper triangular or Hermitian");
    }
}

MatrixXcd canonical_onsite_energy(std::complex<double> energy) {
    return MatrixXcd::Constant(1, 1, energy);
}

MatrixXcd canonical_onsite_energy(VectorXd const& energy) {
    auto const size = energy.size();
    auto result = MatrixXcd::Zero(size, size).eval();
    result.diagonal() = energy.cast<std::complex<double>>();
    return result;
}

void check_hopping_energy(MatrixXcd const& energy) {
    if (energy.rows() == 0 || energy.cols() == 0) {
        throw std::logic_error("Hoppings can't be zero-dimensional");
    }
}

MatrixXcd canonical_hopping_energy(std::complex<double> energy) {
    return MatrixXcd::Constant(1, 1, energy);
}

} // namespace detail
} // namespace cpb
