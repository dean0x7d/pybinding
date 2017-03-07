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
        sub_id alias_id; ///< the alias ID of each sublattice (unique among elements)
        storage_idx_t num_sites; ///< number of sublattice sites in the final system
        storage_idx_t num_orbitals; ///< number of orbitals on this sublattice
    };

    CompressedSublattices() = default;
    CompressedSublattices(ArrayXi const& alias_ids, ArrayXi const& site_counts,
                          ArrayXi const& orbital_counts);

    /// Start a new sublattice block or increment the site count for the existing block
    void add(sub_id id, idx_t norb);
    /// Verify that the stored data is correct: `sum(site_counts) == num_sites`
    void verify(idx_t num_sites) const;

    /// Return the index of the first site with the given number of orbitals
    idx_t start_index(idx_t num_orbitals) const;

    /// Total size if decompressed (sum of the number of sites in all sublattices)
    idx_t decompressed_size() const;
    /// Return the full uncompressed array of IDs
    ArrayX<sub_id> decompress() const;

    std::vector<Element>::const_iterator begin() const { return data.begin(); }
    std::vector<Element>::const_iterator end() const { return data.end(); }

    /// Assess raw data
    ArrayXi alias_ids() const;
    ArrayXi site_counts() const;
    ArrayXi orbital_counts() const;

private:
    std::vector<Element> data; ///< sorted by `num_orbitals` (not `alias_id`)
};

} // namespace cpb
