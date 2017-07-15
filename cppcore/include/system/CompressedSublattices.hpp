#pragma once
#include "Lattice.hpp"

namespace cpb {

/**
 Stores the sublattice IDs for all sites in a system in a compressed format

 Since sublattices with the same ID are always arranges as consecutive elements, this
 data is easily compressed using RLE (run-length encoding). The same IDs are consecutive
 and appear in one block. But the IDs and not necessarily sorted:

 E.g. possible sublattice IDs are: 111000002222 --> encoded as: [1, 0, 2], [3, 5, 4]
 */
class CompressedSublattices {
public:
    struct Element {
        SiteID id; ///< the alias ID of each sublattice (unique among elements)
        storage_idx_t num_sites; ///< number of sublattice sites in the final system
        storage_idx_t num_orbitals; ///< number of orbitals on this sublattice
    };

    class It {
    public:
        using iterator_category = std::input_iterator_tag;
        using difference_type = std::ptrdiff_t;
        using value_type = It;
        using reference = value_type const&;
        using pointer = value_type const*;

        It(std::vector<Element>::const_iterator it) : it(it) {}

        /// Directly correspond to fields in `Element`
        SiteID id() const { return it->id; }
        idx_t num_sites() const { return it->num_sites; }
        idx_t num_orbitals() const { return it->num_orbitals; }

        /// The starting system index for sites of this sublattice
        idx_t sys_start() const { return sys_idx; }
        /// The past end system site index (== sys_start + num_sites)
        idx_t sys_end() const { return sys_idx + it->num_sites; }
        /// The starting hamiltonian index (>= sys_start due to multiple orbitals)
        idx_t ham_start() const { return ham_idx; }
        /// The past end hamiltonian index
        idx_t ham_end() const { return ham_idx + ham_size(); }
        /// The number of Hamiltonian matrix elements for this sublattice
        idx_t ham_size() const { return it->num_sites * it->num_orbitals; }

        reference operator*() const { return *this; }
        pointer operator->() const { return this; }

        It& operator++() {
            sys_idx += it->num_sites;
            ham_idx += it->num_sites * it->num_orbitals;
            ++it;
            return *this;
        }

        friend bool operator==(It const& a, It const& b) { return a.it == b.it; }
        friend bool operator!=(It const& a, It const& b) { return !(a == b); }

    private:
        std::vector<Element>::const_iterator it;
        idx_t sys_idx = 0;
        idx_t ham_idx = 0;
    };

    CompressedSublattices() = default;
    CompressedSublattices(ArrayXi const& alias_ids, ArrayXi const& site_counts,
                          ArrayXi const& orbital_counts);

    /// Start a new sublattice block or increment the site count for the existing block
    void add(SiteID id, idx_t norb, idx_t count = 1);

    /// Remove sites for which `keep == false`
    void filter(VectorX<bool> const& keep);

    /// Verify that the stored data is correct: `sum(site_counts) == num_sites`
    void verify(idx_t num_sites) const;

    /// Return the index of the first site with the given number of orbitals
    idx_t start_index(idx_t num_orbitals) const;

    /// Total size if decompressed (sum of the number of sites in all sublattices)
    idx_t decompressed_size() const;
    /// Return the full uncompressed array of IDs
    ArrayX<storage_idx_t> decompressed() const;

    It begin() const { return data.begin(); }
    It end() const { return data.end(); }

    /// Assess raw data
    ArrayXi alias_ids() const;
    ArrayXi site_counts() const;
    ArrayXi orbital_counts() const;

private:
    std::vector<Element> data; ///< sorted by `num_orbitals` (not `alias_id`)
};

} // namespace cpb
