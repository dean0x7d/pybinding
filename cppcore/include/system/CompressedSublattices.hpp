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
    struct Iterator {
        using iterator_category = std::input_iterator_tag;
        using difference_type = std::ptrdiff_t;
        using value_type = Iterator;
        using reference = value_type const&;
        using pointer = value_type const*;

        using It = std::vector<storage_idx_t>::const_iterator;
        It alias_it;
        It count_it;

        Iterator(It alias_it, It site_count_it) : alias_it(alias_it), count_it(site_count_it) {}

        sub_id alias_id() const { return static_cast<sub_id>(*alias_it); }
        storage_idx_t site_count() const { return *count_it; }

        reference operator*() { return *this; }
        pointer operator->() { return this; }
        Iterator& operator++() { ++alias_it; ++count_it; return *this; }

        friend bool operator==(Iterator const& a, Iterator const& b) {
            return std::tie(a.alias_it, a.count_it) == std::tie(b.alias_it, b.count_it);
        }
        friend bool operator!=(Iterator const& a, Iterator const& b) { return !(a == b); }
    };

public:
    CompressedSublattices() = default;
    CompressedSublattices(std::vector<storage_idx_t> alias_ids,
                          std::vector<storage_idx_t> site_counts)
        : alias_ids(std::move(alias_ids)), site_counts(std::move(site_counts)) {}

    /// Start a new sublattice block or increment the site count for the existing block
    void add(sub_id id);
    /// Verify that the stored data is correct: `sum(site_counts) == num_sites`
    void verify(idx_t num_sites) const;
    /// Return the full uncompressed array of IDs
    ArrayX<sub_id> decompress() const;

    Iterator begin() const { return {alias_ids.begin(), site_counts.begin()}; }
    Iterator end() const { return {alias_ids.end(), site_counts.end()}; }

    std::vector<storage_idx_t> const& get_alias_ids() const { return alias_ids; }
    std::vector<storage_idx_t> const& get_site_counts() const { return site_counts; }

private:
    std::vector<storage_idx_t> alias_ids;
    std::vector<storage_idx_t> site_counts;
};

} // namespace cpb
