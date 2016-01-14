#pragma once
#include "system/Lattice.hpp"

#include "detail/slice.hpp"
#include "support/dense.hpp"

#include <array>
#include <vector>
#include <cstdint>

namespace tbm {

class Primitive;
class Shape;
class Site;
class FoundationIterator;
class SliceIterator;

/**
 The foundation class creates a lattice-vector-aligned set of sites. The number of sites is high
 enough to encompass the given shape. After creation, the foundation can be cut down to the shape.
 */
class Foundation {
public:
    struct Slice {
        Foundation* const foundation;
        SliceIndex3D index;

        SliceIterator begin();
        SliceIterator end();
    };

public:
    Foundation(Lattice const& lattice, Primitive const& shape);
    Foundation(Lattice const& lattice, Shape const& shape);

public:
    FoundationIterator begin();
    FoundationIterator end();

    Slice operator[](SliceIndex3D const& index) { return {this, index}; }

private:
    /// Return the lower and upper bound of the shape in lattice vector coordinates
    static std::pair<Index3D, Index3D> find_bounds(Lattice const& lattice, Shape const& shape);
    /// Initialize the coordinates for each site
    void init_positions(Cartesian origin);
    /// Initialize the neighbor count for each site
    void init_neighbor_count();
    /// Remove edge sites which have a neighbor count lower than the lattice minimum
    void trim_edges();

public:
    /// Calculate the spacial position of a lattice site
    Cartesian calculate_position(Site const& site, Cartesian origin) const;
    /// Reduce this site's neighbor count to zero and inform its neighbors of the change
    void clear_neighbors(Site& site);
    /// Assign Hamiltonian indices to all valid sites. Returns final number of valid sites.
    int finalize();

    /// Loop over all neighbours of a site
    template<class Fn>
    void for_each_neighbour(const Site& site, Fn lambda);
    
public:
    Site make_site(Index3D index, int sublattice);

public:
    Index3D size; ///< number of unit cells in each lattice vector direction
    int size_n; ///< sublattice size (number of sites in each unit cell)
    int num_sites; ///< total number of sites: product of all sizes (3D and sublattice)

    // arrays of length num_sites which track various properties
    CartesianArray positions; ///< coordinates
    ArrayX<bool> is_valid; ///< indicates if the site should be included in the final system
    ArrayX<int16_t> neighbour_count; ///< number sites which have a hopping to this one
    ArrayX<int32_t> hamiltonian_indices; ///< Hamiltonian index (single number) for the final system

    Lattice const& lattice; ///< lattice specification
};

/// Describes a site on the lattice foundation
class Site {
public:
    Foundation* const foundation; ///< the site's parent foundation
    Index3D index; ///< unit cell index in 3 directions
    int sublattice; ///< sublattice index
    int idx; ///< absolute single number index

    Cartesian position() const { return foundation->positions[idx]; }
    bool is_valid() const { return foundation->is_valid[idx]; }
    int16_t num_neighbors() const { return foundation->neighbour_count[idx]; }
    int32_t hamiltonian_index() const { return foundation->hamiltonian_indices[idx]; }

    void set_valid(bool state) { foundation->is_valid[idx] = state; }
    void set_neighbors(int16_t n) { foundation->neighbour_count[idx] = n; }

    Site shift(Index3D shft) const { return foundation->make_site(index + shft, sublattice); }
};

class FoundationIterator {
    Site site;

public:
    FoundationIterator(Foundation* foundation, int idx) : site{foundation, {0, 0, 0}, 0, idx} {}

    Site& operator*() { return site; }

    FoundationIterator& operator++() {
        ++site.idx;
        ++site.sublattice;
        if (site.sublattice == site.foundation->size_n) {
            ++site.index[2];
            if (site.index[2] == site.foundation->size[2]) {
                ++site.index[1];
                if (site.index[1] == site.foundation->size[1]) {
                    ++site.index[0];
                    site.index[1] = 0;
                }
                site.index[2] = 0;
            }
            site.sublattice = 0;
        }
        return *this;
    }

    FoundationIterator operator++(int) {
        auto const copy = *this;
        this->operator++();
        return copy;
    }

    friend bool operator==(FoundationIterator const& l, FoundationIterator const& r) {
        return l.site.idx == r.site.idx;
    }
    friend bool operator!=(FoundationIterator const& l, FoundationIterator const& r) {
        return !(l == r);
    }
};

class SliceIterator {
    Site site;
    SliceIndex3D slice_index;

    static int calc_idx(Site const& site) {
        auto const& index = site.index;
        auto const& size = site.foundation->size;
        auto const& size_n = site.foundation->size_n;
        return ((index[0] * size[1] + index[1]) * size[2] + index[2]) * size_n + site.sublattice;
    }

    static SliceIndex3D normalize(SliceIndex3D const& index, Index3D const& size) {
        auto ret = index;
        for (auto i = 0u; i < 3; ++i) {
            if (ret[i].end < 0) {
                ret[i].end = size[i];
            }
        }
        return ret;
    }

public:
    SliceIterator(Foundation* foundation) : site{foundation, {}, 0, -1} {}
    SliceIterator(Foundation* foundation, SliceIndex3D index)
        : site(foundation->make_site({index[0].start, index[1].start, index[2].start}, 0)),
          slice_index(normalize(index, foundation->size)) {}

    Site& operator*() { return site; }

    SliceIterator& operator++() {
        ++site.sublattice;
        if (site.sublattice == site.foundation->size_n) {
            ++site.index[2];
            if (site.index[2] == slice_index[2].end) {
                ++site.index[1];
                if (site.index[1] == slice_index[1].end) {
                    ++site.index[0];
                    if (site.index[0] == slice_index[0].end) {
                        site.idx = -1;
                        return *this;
                    }
                    site.index[1] = slice_index[1].start;
                }
                site.index[2] = slice_index[2].start;
            }
            site.sublattice = 0;
        }

        site.idx = calc_idx(site);
        return *this;
    }

    SliceIterator operator++(int) {
        auto const copy = *this;
        this->operator++();
        return copy;
    }

    friend bool operator==(SliceIterator const& l, SliceIterator const& r) {
        return l.site.idx == r.site.idx;
    }
    friend bool operator!=(SliceIterator const& l, SliceIterator const& r) { return !(l == r); }
};

inline Site Foundation::make_site(Index3D index, int sublattice) {
    auto i = ((index[0]*size[1] + index[1])*size[2] + index[2])*size_n + sublattice;
    return {this, index, sublattice, i};
}

template<class Fn>
void Foundation::for_each_neighbour(const Site& site, Fn lambda) {
    // loop over all hoppings from this site's sublattice
    for (const auto& hopping : lattice[site.sublattice].hoppings) {
        const auto neighbour_index = (site.index + hopping.relative_index).array();
        if (any_of(neighbour_index < 0) || any_of(neighbour_index >= size.array()))
            continue; // out of bounds

        lambda(make_site(neighbour_index, hopping.to_sublattice), hopping);
    }
}

} // namespace tbm
