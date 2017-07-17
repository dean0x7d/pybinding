#pragma once
#include "numeric/dense.hpp"
#include "detail/opaque_alias.hpp"

#include <string>
#include <vector>
#include <unordered_map>

namespace cpb {

// TODO: replace with proper string_view
using string_view = std::string const&;

/// Sublattice and hopping ID data types
using SubID = detail::OpaqueIntegerAlias<class SubIDTag>;
using SubAliasID = detail::OpaqueIntegerAlias<class SubAliasIDTag>;
using SiteID = detail::OpaqueIntegerAlias<class SiteIDTag>;
using HopID = detail::OpaqueIntegerAlias<class HopIDTag>;

/// Map from friendly sublattice/hopping name to numeric ID
using NameMap = std::unordered_map<std::string, storage_idx_t>;

template<class ID>
class Registry {
public:
    Registry(std::vector<MatrixXcd> energies, std::vector<std::string> names);

    std::vector<MatrixXcd> const& get_energies() const { return energies; }
    std::vector<std::string> const& get_names() const { return names; }

    void register_family(string_view name, MatrixXcd const& energy);

    idx_t size() const { return static_cast<idx_t>(names.size()); }

    /// Mapping from friendly names to unique IDs
    NameMap name_map() const;

    string_view name(ID id) const;
    MatrixXcd const& energy(ID id) const;
    ID id(string_view name) const;

    /// Is at least one energy term not equal to zero?
    bool has_nonzero_energy() const;
    /// Is at least one energy term complex?
    bool any_complex_terms() const;
    /// Is at least one energy term a matrix?
    bool has_multiple_orbitals() const;

private:
    std::vector<MatrixXcd> energies;
    std::vector<std::string> names;
};

using SiteRegistry = Registry<SiteID>;
using HoppingRegistry = Registry<HopID>;

extern template class Registry<SiteID>;
extern template class Registry<HopID>;

namespace detail {

/// Check that the onsite energy matrix satisfies all the requirements
void check_onsite_energy(MatrixXcd const& energy);
/// Convert the onsite energy into the canonical format
MatrixXcd canonical_onsite_energy(std::complex<double> energy);
MatrixXcd canonical_onsite_energy(VectorXd const& energy);
inline MatrixXcd canonical_onsite_energy(MatrixXcd const& energy) { return energy; }

/// Check that the hopping energy matrix satisfies all the requirements
void check_hopping_energy(MatrixXcd const& energy);
/// Convert the hopping energy into the canonical format
MatrixXcd canonical_hopping_energy(std::complex<double> energy);
inline MatrixXcd canonical_hopping_energy(MatrixXcd const& energy) { return energy; }

} // namespace detail
} // namespace cpb
